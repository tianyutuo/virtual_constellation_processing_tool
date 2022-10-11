"""
逐像元计算高分数据角度信息
"""

from utils.array_to_raster import array2Raster
from utils.geo_rpc import *
from osgeo import gdal
import numpy as np
import xml.dom.minidom
import math
from math import cos, acos, sin, asin
import scipy
from scipy import ndimage
from meanDEM import MeanDEM
import sys
from glob import iglob
import shutil


def calculate_view_angle(img_file, sensor_type='WFV'):
    """
    逐像素计算高分数据观测天顶角（view zenith angle，VZA）和观测方位角（view azimuth angle，VAA）
    考虑了地球曲率的影响，参考文献：https://doi.org/10.3390/rs10030478
    :param img_file:
    :return:
    """
    ds = gdal.Open(img_file)
    cols = ds.RasterXSize
    rows = ds.RasterYSize
    # 中心像元的行列号

    h1 = 101
    h2 = 100

    if sensor_type == 'WFV':
        factor = int(1000 / 16)
    else:
        factor = int(1000 / 8)

    if rows % factor == 0:
        init_row = int(rows / factor)
    else:
        init_row = int(rows / factor) + 1
    if cols % factor == 0:
        init_col = int(cols / factor)
    else:
        init_col = int(cols / factor) + 1

    vza = np.zeros((init_row, init_col), dtype=np.float32)
    vaa = np.zeros((init_row, init_col), dtype=np.float32)

    rpc = rpc_from_geotiff(img_file)

    row = 0
    for i in range(0, rows, factor):
        col = 0
        for j in range(0, cols, factor):
            print(f'{i * cols + j + 1}/{rows * cols}')
            # 由RPC参数计算中心像素经纬度
            lon1, lat1 = rpc.localization(i, j, h1)
            lon2, lat2 = rpc.localization(i, j, h2)

            wgs84_x1, wgs84_y1, wgs84_z1 = wgs84blh2xyz(lat1, lon1, h1)
            wgs84_x2, wgs84_y2, wgs84_z2 = wgs84blh2xyz(lat2, lon2, h2)

            wgs84_x, wgs84_y, wgs84_z = (wgs84_x1 - wgs84_x2), (wgs84_y1 - wgs84_y2), (wgs84_z1 - wgs84_z2)

            # print(f'WGS84地心坐标系下的视线向量为{wgs84_x, wgs84_y, wgs84_z}')

            x, y, z = wgs84geocentric2local(lat1, lon1, wgs84_x, wgs84_y, wgs84_z)

            # print(f'站心坐标系下的视线向量为{x, y, z}')

            vaa[row][col] = math.atan2(x, y) * 180 / math.pi
            vza[row][col] = math.atan2(z, math.sqrt(x * x + y * y)) * 180 / math.pi
            if vaa[row][col] < 0:
                vaa[row][col] += 360

            col += 1
        row += 1

    row_factor = rows * 1.0 / init_row
    col_factor = cols * 1.0 / init_col

    # bilinear resample
    vaa = scipy.ndimage.zoom(vaa, (row_factor, col_factor), order=1)
    vza = scipy.ndimage.zoom(vza, (row_factor, col_factor), order=1)

    array2Raster(vaa, img_file[0:-5] + '_resampled_VAA.tif', refImg=img_file)
    array2Raster(vza, img_file[0:-5] + '_resampled_VZA.tif', refImg=img_file)

    # 为角度数据赋值rpb文件，方便对角度数据进行相同的正射校正
    product_path = os.path.dirname(os.path.abspath(img_file))
    rpb_file = sorted(list(iglob(os.path.join(product_path, '**', img_file[:-5]+'.rpb'), recursive=True)))[0]
    shutil.copy(rpb_file, rpb_file[:-4] + '_resampled_VAA.rpb')
    shutil.copy(rpb_file, rpb_file[:-4] + '_resampled_VZA.rpb')


def calculate_solar_angle(img_file, metadata, sensor_type='WFV'):
    """
    逐像素计算高分数据太阳天顶角（solar zenith angle，SZA）和太阳方位角（solar azimuth angle，VAA）
    :param img_file:
    :param metadata:
    :return:
    """

    ds = gdal.Open(img_file, gdal.GA_ReadOnly)

    # 读取影像行列数
    cols = ds.RasterXSize
    rows = ds.RasterYSize

    # 为加快计算速度，将1km×1km的像素块聚合为1个像素计算角度
    # 考虑到不同数据的分辨率不同，不应以像元为单位聚合，以距离为尺度聚合更有意义，如1km×1km
    if sensor_type == 'WFV':
        factor = int(1000 / 16)
    else:
        factor = int(1000 / 8)

    if rows % 100 == 0:
        init_row = int(rows / factor)
    else:
        init_row = int(rows / factor) + 1
    if cols % 100 == 0:
        init_col = int(cols / factor)
    else:
        init_col = int(cols / factor) + 1
    sza = np.zeros((init_row, init_col), dtype=np.float32)
    saa = np.zeros((init_row, init_col), dtype=np.float32)

    # 读取元数据
    dom = xml.dom.minidom.parse(metadata)
    # 年、月、日、时、分、秒
    DateTimeparm = dom.getElementsByTagName('CenterTime')[0].firstChild.data
    DateTime = DateTimeparm.split(' ')
    Date = DateTime[0].split('-')
    year = int(Date[0])
    month = int(Date[1])
    day = int(Date[2])
    time = DateTime[1].split(':')
    hour = int(time[0])
    minute = int(time[1])
    second = int(time[2])
    # 中心像元经纬度
    TopLeftLat = float(dom.getElementsByTagName('TopLeftLatitude')[0].firstChild.data)
    TopLeftLon = float(dom.getElementsByTagName('TopLeftLongitude')[0].firstChild.data)
    TopRightLat = float(dom.getElementsByTagName('TopRightLatitude')[0].firstChild.data)
    TopRightLon = float(dom.getElementsByTagName('TopRightLongitude')[0].firstChild.data)
    BottomRightLat = float(dom.getElementsByTagName('BottomRightLatitude')[0].firstChild.data)
    BottomRightLon = float(dom.getElementsByTagName('BottomRightLongitude')[0].firstChild.data)
    BottomLeftLat = float(dom.getElementsByTagName('BottomLeftLatitude')[0].firstChild.data)
    BottomLeftLon = float(dom.getElementsByTagName('BottomLeftLongitude')[0].firstChild.data)

    # 读取RPC系数，用于将影像行列数坐标转换为经纬度
    rpc = rpc_from_geotiff(img_file)

    # 由研究区域的范围求DEM高度
    pointUL = dict()
    pointDR = dict()
    pointUL["lat"] = max(TopLeftLat, TopRightLat, BottomRightLat, BottomLeftLat)
    pointUL["lon"] = min(TopLeftLon, TopRightLon, BottomRightLon, BottomLeftLon)
    pointDR["lat"] = min(TopLeftLat, TopRightLat, BottomRightLat, BottomLeftLat)
    pointDR["lon"] = max(TopLeftLon, TopRightLon, BottomRightLon, BottomLeftLon)
    h = (MeanDEM(pointUL, pointDR)) * 0.001

    row = 0
    for i in range(0, rows, factor):
        col = 0
        for j in range(0, cols, factor):
            print(f'{i * cols + j + 1}/{rows * cols}')
            # 由RPC参数计算经纬度
            lon, lat = rpc.localization(i, j, h)

            zenithAngle, azimuthAngle = solar_zenith_azimuth(lon, lat, year, month, day, hour, minute, second)

            sza[row][col] = zenithAngle
            saa[row][col] = azimuthAngle

            col += 1
        row += 1

    row_factor = rows * 1.0 / init_row
    col_factor = cols * 1.0 / init_col

    # bilinear resample
    saa = scipy.ndimage.zoom(saa, (row_factor, col_factor), order=1)
    sza = scipy.ndimage.zoom(sza, (row_factor, col_factor), order=1)

    array2Raster(saa, img_file[0:-5] + '_resampled_SAA.tif', refImg=img_file)
    array2Raster(sza, img_file[0:-5] + '_resampled_SZA.tif', refImg=img_file)

    # 为角度数据赋值rpb文件，方便对角度数据进行相同的正射校正
    product_path = os.path.dirname(os.path.abspath(img_file))
    rpb_file = sorted(list(iglob(os.path.join(product_path, '**', img_file[:-5] + '.rpb'), recursive=True)))[0]
    shutil.copy(rpb_file, rpb_file[:-4] + '_resampled_SAA.rpb')
    shutil.copy(rpb_file, rpb_file[:-4] + '_resampled_SZA.rpb')


def solar_zenith_azimuth(lon, lat, year, month, day, hour, minute, second):
    """
    根据经纬度和时间信息计算太阳天顶角，方位角
    :param lon:
    :param lat:
    :param year:
    :param month:
    :param day:
    :param hour:
    :param minute:
    :param second:
    :return:
    """
    # 儒略日 Julian day
    JD0 = int(365.25 * (year - 1)) + int(30.6001 * (1 + 13)) + 1 + hour / 24 + 1720981.5

    if month <= 2:
        JD2 = int(365.25 * (year - 1)) + int(30.6001 * (month + 13)) + day + hour / 24 + 1720981.5
    else:
        JD2 = int(365.25 * year) + int(30.6001 * (month + 1)) + day + hour / 24 + 1720981.5

    # 年积日 Day of year
    DOY = JD2 - JD0 + 1

    # 计算太阳时 Solar time
    B = (DOY - 1) * 360 / 365   # 注意此处的单位为度
    BArc = B * math.pi / 180
    E = 229 * (0.000075 + 0.001868 * cos(BArc) - 0.032077 * sin(BArc)
               - 0.014615 * cos(2*BArc) - 0.04089 * sin(2*BArc))
    lt = hour + minute / 60.0 + second / 3600.0     # 地方时
    st = lt - 4 * (120.0 - lon) / 60 + E / 60       # 注意此处的单位应将分换成24小时制

    # 计算太阳时角
    timeAngle = (st - 12) * 15
    timeAngleArc = timeAngle * math.pi / 180

    # 计算太阳赤纬
    # TODO：此处为粗略计算的公式，待后续改进
    ED = -23.44 * cos(2 * math.pi * (DOY + 10) / 365)   # 注意此处的单位为度
    EDArc = ED * math.pi / 180

    # 计算太阳高度角
    latArc = lat * math.pi / 180
    sinAltitudeAngle = sin(latArc) * sin(EDArc) + cos(latArc) * cos(EDArc) * cos(timeAngleArc)
    altitudeAngleArc = asin(sinAltitudeAngle)
    altitudeAngle = altitudeAngleArc * 180 / math.pi

    # 计算太阳天顶角
    zenithAngle = 90 - altitudeAngle

    # 计算太阳方位角
    cosAzimuthAngle = (sin(altitudeAngleArc) * sin(latArc) - sin(EDArc)) / (cos(altitudeAngleArc) * cos(latArc))
    azimuthAngleArc = acos(cosAzimuthAngle)
    azimuthAngle = azimuthAngleArc * 180 / math.pi

    if timeAngle < 0:
        azimuthAngle = 180 - azimuthAngle
    else:
        azimuthAngle = 180 + azimuthAngle

    return zenithAngle, azimuthAngle


if __name__ == '__main__':
    import os
    import time
    from geometry.rpcOrthorectification import RPCrect

    a = time.time()

    os.chdir(r'F:\Experiment\BRDF Correction\GF1PMS\GF1_PMS2_E113.9_N30.5_20180417_L1A0003127222')

    img_file = 'GF1_PMS2_E113.9_N30.5_20180417_L1A0003127222-MSS2.tiff'
    meta_data = 'GF1_PMS2_E113.9_N30.5_20180417_L1A0003127222-MSS2.xml'

    calculate_solar_angle(
        img_file,
        meta_data
    )
    calculate_view_angle(img_file)

    # 对角度数据进行正射校正
    product_path = r'F:\Experiment\BRDF Correction\GF1PMS\GF1_PMS2_E113.9_N30.5_20180417_L1A0003127222'
    angle_files = sorted(list(iglob(os.path.join(product_path, '**', '*resampled*.tif'), recursive=True)))

    for file in angle_files:
        output = file[:-4] + '_ORT.tif'
        RPCrect(file, output)

    b = time.time()
    print(f'共耗时{b-a}s')

if __name__ == '__main__1':
    """
    以高分数据中心像元来验证太阳天顶角、方位角的计算结果
    """
    import os

    os.chdir(r'F:\Experiment\BRDF Correction\GF2_PMS1_E114.8_N30.8_20180821_L1A0003404413')

    img_file = 'GF2_PMS1_E114.8_N30.8_20180821_L1A0003404413-MSS1.tiff'
    metadata = 'GF2_PMS1_E114.8_N30.8_20180821_L1A0003404413-MSS1.xml'

    ds = gdal.Open(img_file)
    cols = ds.RasterXSize
    rows = ds.RasterYSize
    center_col = int(cols / 2)
    center_row = int(rows / 2)

    # 读取元数据
    dom = xml.dom.minidom.parse(metadata)
    # 中心经纬度
    TopLeftLat = float(dom.getElementsByTagName('TopLeftLatitude')[0].firstChild.data)
    TopLeftLon = float(dom.getElementsByTagName('TopLeftLongitude')[0].firstChild.data)
    TopRightLat = float(dom.getElementsByTagName('TopRightLatitude')[0].firstChild.data)
    TopRightLon = float(dom.getElementsByTagName('TopRightLongitude')[0].firstChild.data)
    BottomRightLat = float(dom.getElementsByTagName('BottomRightLatitude')[0].firstChild.data)
    BottomRightLon = float(dom.getElementsByTagName('BottomRightLongitude')[0].firstChild.data)
    BottomLeftLat = float(dom.getElementsByTagName('BottomLeftLatitude')[0].firstChild.data)
    BottomLeftLon = float(dom.getElementsByTagName('BottomLeftLongitude')[0].firstChild.data)

    ImageCenterLat = (TopLeftLat + TopRightLat + BottomRightLat + BottomLeftLat) / 4
    ImageCenterLon = (TopLeftLon + TopRightLon + BottomRightLon + BottomLeftLon) / 4
    # 年、月、日、时、分、秒
    DateTimeparm = dom.getElementsByTagName('CenterTime')[0].firstChild.data
    DateTime = DateTimeparm.split(' ')
    Date = DateTime[0].split('-')
    year = int(Date[0])
    month = int(Date[1])
    day = int(Date[2])
    time = DateTime[1].split(':')
    hour = int(time[0])
    minute = int(time[1])
    second = int(time[2])

    # 由研究区域的范围求DEM高度
    pointUL = dict()
    pointDR = dict()
    pointUL["lat"] = max(TopLeftLat, TopRightLat, BottomRightLat, BottomLeftLat)
    pointUL["lon"] = min(TopLeftLon, TopRightLon, BottomRightLon, BottomLeftLon)
    pointDR["lat"] = min(TopLeftLat, TopRightLat, BottomRightLat, BottomLeftLat)
    pointDR["lon"] = max(TopLeftLon, TopRightLon, BottomRightLon, BottomLeftLon)
    meanDEM = (MeanDEM(pointUL, pointDR)) * 0.001
    rpc = rpc_from_geotiff(img_file)
    center_lon, center_lat = rpc.localization(center_col, center_row, meanDEM)

    zenithAngle, azimuthAngle = solar_zenith_azimuth(center_lon, center_lat, year, month, day, hour, minute, second)
    print(f'SZA: {zenithAngle}, SAA: {azimuthAngle}')

if __name__ == '__main__1':
    """
    太阳方位角、天顶角数据计算验证
    参考：https://blog.csdn.net/Wanglin6328/article/details/109724797?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_baidulandingword~default-1-109724797-blog-42468281.pc_relevant_vip_default&spm=1001.2101.3001.4242.1&utm_relevant_index=4
    """
    lat = 40.34924
    lon = 115.78388
    year = 2019
    month = 10
    day = 8
    hour = 10
    minute = 20
    second = 0

    zenithAngle, azimuthAngle = solar_zenith_azimuth(lon, lat, year, month, day, hour, minute, second)
    print(f'SZA: {zenithAngle}, SAA: {azimuthAngle}')
