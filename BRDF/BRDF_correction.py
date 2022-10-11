"""
BRDF校正
参考文献：
"""

from osgeo import gdal
import math
from math import sin, cos, tan, acos, sqrt
import numpy as np
from utils.array_to_raster import array2Raster
import matplotlib.pyplot as plt


def directional_reflectance(sza, saa, vza, vaa, iso, vol, geo):
    """
    根据角度数据模拟方向反射率
    """
    # ---------------------------------------------------------------------------------------------------
    # 计算RossThick Kernel
    # ---------------------------------------------------------------------------------------------------
    raa = vaa - saa
    # raa = np.abs(saa - vaa)  # 相对方位角
    # 相对方位角的范围应在0-180
    # if raa > 180:
    #     raa = 360 - 180
    cosPhaseAngle = cos(sza * math.pi / 180) * cos(vza * math.pi / 180) + \
                    sin(sza * math.pi / 180) * sin(vza * math.pi / 180) * cos(raa * math.pi / 180)
    phaseAngleArc = acos(cosPhaseAngle)
    k_vol = ((math.pi / 2 - phaseAngleArc) * cos(phaseAngleArc) + sin(phaseAngleArc)) / \
            (cos(sza * math.pi / 180) + cos(vza * math.pi / 180)) - math.pi / 4
    # ---------------------------------------------------------------------------------------------------
    # 计算LiSparse Reciprocal Kernel
    # ---------------------------------------------------------------------------------------------------
    D = sqrt(tan(sza * math.pi / 180) ** 2 + tan(vza * math.pi / 180) ** 2 \
        - 2 * tan(sza * math.pi / 180) * tan(vza * math.pi / 180) * cos(raa * math.pi / 180))
    cost = 2 * sqrt(D ** 2 + (tan(sza * math.pi / 180) * tan(vza * math.pi / 180) * sin(raa * math.pi / 180))**2) / \
           (1/cos(sza * math.pi / 180) + 1/cos(vza * math.pi / 180))
    # TODO：cost的值域超出了[-1, 1]
    tArc = acos(np.clip(cost, -1, 1))
    O = 1/math.pi * (tArc - sin(tArc) * cos(tArc)) * (1/cos(sza * math.pi / 180) + 1/cos(vza * math.pi / 180))
    k_geo = O - 1/cos(sza * math.pi / 180) - 1/cos(vza * math.pi / 180) + \
            1/2 * (1 + cos(phaseAngleArc)) * 1/cos(sza * math.pi / 180) * 1/cos(vza * math.pi / 180)

    # 计算模拟方向反射率
    sr = iso * 0.001 + vol * 0.001 * k_vol + geo * 0.001 * k_geo

    return sr


def pix2map(xpixel, ypixel, geoTransform):
    """
    像素坐标转换为地理坐标
    """
    xgeo = geoTransform[0] + geoTransform[1] * xpixel + ypixel * geoTransform[2]
    ygeo = geoTransform[3] + geoTransform[4] * xpixel + ypixel * geoTransform[5]

    xgeo = math.ceil(xgeo)
    ygeo = math.ceil(ygeo)

    return xgeo, ygeo


def map2pix(xgeo, ygeo, geoTransform):
    """
    地理坐标转换为像素坐标
    """
    ypixel = ((ygeo - geoTransform[3] - geoTransform[4] / geoTransform[1] * xgeo + geoTransform[4] / geoTransform[1] *
               geoTransform[
                   0]) / (geoTransform[5] - geoTransform[4] / geoTransform[1] * geoTransform[2]))
    xpixel = (xgeo - geoTransform[0] - ypixel * geoTransform[2]) / geoTransform[1]

    ypixel = math.ceil(ypixel)
    xpixel = math.ceil(xpixel)

    return ypixel, xpixel


def plot1(x, y, coef, intercept, n, rmse, title):
    title_font = {
        'family': 'serif',
        'color': 'darkred',
        'weight': 'normal',
        'size': 12,
    }
    font = {
        'family': 'serif',
        'color': 'darkred',
        'weight': 'normal',
        'size': 9,
    }
    plt.scatter(x * 0.0001, y * 0.0001, c='r', s=0.5)
    plt.plot(np.arange(0, 1, 0.001),
             np.arange(0, 1, 0.001) * coef + intercept * 0.0001, color='darkred',
             linewidth=0.8)
    plt.title(title, fontdict=title_font)
    plt.xlabel(f'Simulated reflectance of GF1-WFV', fontdict=font)
    plt.ylabel(f'MODIS NBAR', fontdict=font)
    plt.text(0.1, 0.90, f'n={n}', fontdict=font)
    plt.text(0.1, 0.80, f'RMSE: {round(rmse, 5)}', fontdict=font)
    plt.text(0.1, 0.70,
             f'y = {round(coef, 5)}x + {round(intercept * 0.0001, 5)}',
             fontdict=font)
    plt.axis([0, 1, 0, 1])


def brdf_correction(gf_sr_file, gf_sza_file, gf_saa_file, gf_vza_file, gf_vaa_file,
                    mcd43a1_band1_file, mcd43a1_band2_file, mcd43a1_band3_file, mcd43a1_band4_file,
                    mcd43a4_band1_file, mcd43a4_band2_file, mcd43a4_band3_file, mcd43a4_band4_file,
                    sensor_type='WFV', cv_threshold=0.1):
    # ---------------------------------------------------------------------------------------------------
    # 读取数据
    # ---------------------------------------------------------------------------------------------------
    # GF-1反射率数据
    gf_sr = gdal.Open(gf_sr_file)
    gf_band1 = gf_sr.GetRasterBand(1).ReadAsArray()  # 读取蓝色波段
    gf_band2 = gf_sr.GetRasterBand(2).ReadAsArray()  # 读取绿色波段
    gf_band3 = gf_sr.GetRasterBand(3).ReadAsArray()  # 读取红色波段
    gf_band4 = gf_sr.GetRasterBand(4).ReadAsArray()  # 读取近红外波段
    gf_rows = gf_sr.RasterYSize
    gf_cols = gf_sr.RasterXSize
    gf_geoTransform = gf_sr.GetGeoTransform()

    # GF-1角度数据
    gf_sza_ds = gdal.Open(gf_sza_file)
    gf_sza = gf_sza_ds.GetRasterBand(1).ReadAsArray()
    gf_saa_ds = gdal.Open(gf_saa_file)
    gf_saa = gf_saa_ds.GetRasterBand(1).ReadAsArray()
    gf_vza_ds = gdal.Open(gf_vza_file)
    gf_vza = gf_vza_ds.GetRasterBand(1).ReadAsArray()
    gf_vaa_ds = gdal.Open(gf_vaa_file)
    gf_vaa = gf_vaa_ds.GetRasterBand(1).ReadAsArray()
    angle_geoTransform = gf_sza_ds.GetGeoTransform()

    # MODIS BRDF参数
    mcd43a1_ds1 = gdal.Open(mcd43a1_band1_file)
    red_iso = mcd43a1_ds1.GetRasterBand(1).ReadAsArray()
    red_vol = mcd43a1_ds1.GetRasterBand(2).ReadAsArray()
    red_geo = mcd43a1_ds1.GetRasterBand(3).ReadAsArray()
    mcd43a1_ds2 = gdal.Open(mcd43a1_band2_file)
    nir_iso = mcd43a1_ds2.GetRasterBand(1).ReadAsArray()
    nir_vol = mcd43a1_ds2.GetRasterBand(2).ReadAsArray()
    nir_geo = mcd43a1_ds2.GetRasterBand(3).ReadAsArray()
    mcd43a1_ds3 = gdal.Open(mcd43a1_band3_file)
    blue_iso = mcd43a1_ds3.GetRasterBand(1).ReadAsArray()
    blue_vol = mcd43a1_ds3.GetRasterBand(2).ReadAsArray()
    blue_geo = mcd43a1_ds3.GetRasterBand(3).ReadAsArray()
    mcd43a1_ds4 = gdal.Open(mcd43a1_band4_file)
    green_iso = mcd43a1_ds4.GetRasterBand(1).ReadAsArray()
    green_vol = mcd43a1_ds4.GetRasterBand(2).ReadAsArray()
    green_geo = mcd43a1_ds4.GetRasterBand(3).ReadAsArray()
    mcd43a1_rows = mcd43a1_ds1.RasterYSize
    mcd43a1_cols = mcd43a1_ds1.RasterXSize

    # MODIS NBAR数据
    mcd43a4_ds1 = gdal.Open(mcd43a4_band1_file)
    mcd43a4_ds2 = gdal.Open(mcd43a4_band2_file)
    mcd43a4_ds3 = gdal.Open(mcd43a4_band3_file)
    mcd43a4_ds4 = gdal.Open(mcd43a4_band4_file)
    nbar_b1 = mcd43a4_ds1.GetRasterBand(1).ReadAsArray()
    nbar_b2 = mcd43a4_ds2.GetRasterBand(1).ReadAsArray()
    nbar_b3 = mcd43a4_ds3.GetRasterBand(1).ReadAsArray()
    nbar_b4 = mcd43a4_ds4.GetRasterBand(1).ReadAsArray()
    modis_geoTransform = mcd43a4_ds1.GetGeoTransform()
    # ---------------------------------------------------------------------------------------------------
    # 模拟GF-1的方向反射率 （500m）
    # ---------------------------------------------------------------------------------------------------
    # 根据传感器类型判断MODIS数据缩小的范围和其对应的像素块
    if sensor_type == 'WFV':
        reduce_pixels = 120
        window_pixels = 15
    else:
        reduce_pixels = 20
        window_pixels = 31
    sr_red_gf = np.zeros((mcd43a1_rows, mcd43a1_cols), dtype=np.int16)  # 存储模拟的GF-1的方向反射率
    sr_nir_gf = np.zeros((mcd43a1_rows, mcd43a1_cols), dtype=np.int16)
    sr_blue_gf = np.zeros((mcd43a1_rows, mcd43a1_cols), dtype=np.int16)
    sr_green_gf = np.zeros((mcd43a1_rows, mcd43a1_cols), dtype=np.int16)
    for i in range(reduce_pixels, mcd43a1_rows - reduce_pixels):  # 防止MODIS像元在分辨率较高的高分数据上溢出，此处缩小范围
        for j in range(reduce_pixels, mcd43a1_cols - reduce_pixels):
            x, y = pix2map(i, j, modis_geoTransform)  # 将MODIS行列坐标转换为平面坐标
            row, col = map2pix(x, y, angle_geoTransform)  # 将MODIS的平面坐标转换为GF角度数据行列号
            row = math.ceil(row)  # 行列号带有小数点，此处需要向上取整
            col = math.ceil(col)
            # 依据角度数据的分辨率，对应500m分辨率的MODIS数据建立相应大小的窗口，并取窗口均值为该区域的角度数据
            sample_sza = gf_sza[row - window_pixels: row + window_pixels, col - window_pixels:col + window_pixels]
            mean_sza = sample_sza.mean()
            sample_saa = gf_saa[row - window_pixels: row + window_pixels, col - window_pixels:col + window_pixels]
            mean_saa = sample_saa.mean()
            sample_vza = gf_vza[row - window_pixels: row + window_pixels, col - window_pixels:col + window_pixels]
            # TODO:卫星天顶角确定
            mean_vza = 90 - sample_vza.mean()
            sample_vaa = gf_vaa[row - window_pixels: row + window_pixels, col - window_pixels:col + window_pixels]
            mean_vaa = sample_vaa.mean()
            # BRDF parameters
            iso_red = red_iso[i, j]
            vol_red = red_vol[i, j]
            geo_red = red_geo[i, j]
            iso_nir = nir_iso[i, j]
            vol_nir = nir_vol[i, j]
            geo_nir = nir_geo[i, j]
            iso_blue = blue_iso[i, j]
            vol_blue = blue_vol[i, j]
            geo_blue = blue_geo[i, j]
            iso_green = green_iso[i, j]
            vol_green = green_vol[i, j]
            geo_green = green_geo[i, j]

            # 方便数据存储，此处将模拟的反射率扩大10000倍
            sr_red_gf[i, j] = directional_reflectance(mean_sza, mean_saa, mean_vza, mean_vaa, iso_red, vol_red,
                                                      geo_red) * 10000
            sr_nir_gf[i, j] = directional_reflectance(mean_sza, mean_saa, mean_vza, mean_vaa, iso_nir, vol_nir,
                                                      geo_nir) * 10000
            sr_blue_gf[i, j] = directional_reflectance(mean_sza, mean_saa, mean_vza, mean_vaa, iso_blue, vol_blue,
                                                       geo_blue) * 10000
            sr_green_gf[i, j] = directional_reflectance(mean_sza, mean_saa, mean_vza, mean_vaa, iso_green, vol_green,
                                                        geo_green) * 10000

    array2Raster([sr_blue_gf, sr_green_gf, sr_red_gf, sr_nir_gf], 'simulated_directional_sr.tif',
                 refImg=mcd43a4_file1, noDataValue=0)
    # ---------------------------------------------------------------------------------------------------
    # 选取纯净像元
    # ---------------------------------------------------------------------------------------------------
    # 以模拟的500米待校正数据进行遍历，找对应的16米像元建立窗口判断是否为均一像元
    pure_red = []  # 记录纯净像元的索引
    pure_nir = []
    pure_blue = []
    pure_green = []
    delete = []  # 记录BRDF参数的NaN
    for i in range(reduce_pixels, mcd43a1_rows - reduce_pixels):
        for j in range(reduce_pixels, mcd43a1_cols - reduce_pixels):
            x, y = pix2map(i, j, modis_geoTransform)
            row, col = map2pix(x, y, gf_geoTransform)
            row = math.ceil(row)
            col = math.ceil(col)
            sample_gf_band3 = gf_band3[row - window_pixels: row + window_pixels, col - window_pixels:col + window_pixels]
            cv_sample_gf_band3 = sample_gf_band3.std() / sample_gf_band3.mean()

            sample_gf_band4 = gf_band4[row - window_pixels: row + window_pixels, col - window_pixels:col + window_pixels]
            cv_sample_gf_band4 = sample_gf_band4.std() / sample_gf_band4.mean()

            sample_gf_band1 = gf_band1[row - window_pixels: row + window_pixels, col - window_pixels:col + window_pixels]
            cv_sample_gf_band1 = sample_gf_band1.std() / sample_gf_band1.mean()

            sample_gf_band2 = gf_band2[row - window_pixels: row + window_pixels, col - window_pixels:col + window_pixels]
            cv_sample_gf_band2 = sample_gf_band2.std() / sample_gf_band2.mean()

            if cv_sample_gf_band3 < cv_threshold:  # TODO：此处变异系数的阈值设置为多少合适
                pure_red.append([i, j])

            if cv_sample_gf_band4 < cv_threshold:
                pure_nir.append([i, j])

            if cv_sample_gf_band1 < cv_threshold:
                pure_blue.append([i, j])

            if cv_sample_gf_band2 < cv_threshold:
                pure_green.append([i, j])

            if nbar_b1[i, j] == 0:
                delete.append([i, j])

    pure_pixel_location = [val for val in pure_red if val in pure_nir]  # 在所有波段上取交集
    pure_pixel_location = [val for val in pure_pixel_location if val in pure_blue]
    pure_pixel_location = [val for val in pure_pixel_location if val in pure_green]
    pure_pixel_location = [val for val in pure_pixel_location if val not in delete]  # 去除BRDF参数为NaN的索引

    # 计算NDVI将每个波段上的纯净像元分类
    np.seterr(divide='ignore', invalid='ignore')
    ndvi_500m = (sr_nir_gf - sr_red_gf) * 1.0 / (sr_nir_gf + sr_red_gf) * 1.0
    array2Raster(ndvi_500m, 'ndvi_500m.tif', refImg=mcd43a4_file1, noDataValue=0)

    # 计算NDVI，将地物分为五个类别
    ndvi_class1 = []
    ndvi_class2 = []
    ndvi_class3 = []
    ndvi_class4 = []

    for i in range(reduce_pixels, mcd43a1_rows - reduce_pixels):
        for j in range(reduce_pixels, mcd43a1_cols - reduce_pixels):
            if ndvi_500m[i, j] <= 0.3:
                ndvi_class1.append([i, j])
            elif ndvi_500m[i, j] <= 0.4:
                ndvi_class2.append([i, j])
            elif ndvi_500m[i, j] <= 0.5:
                ndvi_class3.append([i, j])
            else:
                ndvi_class4.append([i, j])

    # 按类别提取纯净像元的位置
    pure_pixel_location_class1 = [val for val in pure_pixel_location if val in ndvi_class1]
    pure_pixel_location_class2 = [val for val in pure_pixel_location if val in ndvi_class2]
    pure_pixel_location_class3 = [val for val in pure_pixel_location if val in ndvi_class3]
    pure_pixel_location_class4 = [val for val in pure_pixel_location if val in ndvi_class4]

    # 输出存储纯净像元位置的影像
    pure_pixel_location_image = np.array(np.zeros((mcd43a1_rows, mcd43a1_cols), dtype=np.int16))
    pure_pixel_location_image[
        np.array(pure_pixel_location_class1)[:, 0], np.array(pure_pixel_location_class1)[:, 1]] = 1
    pure_pixel_location_image[
        np.array(pure_pixel_location_class2)[:, 0], np.array(pure_pixel_location_class2)[:, 1]] = 2
    pure_pixel_location_image[
        np.array(pure_pixel_location_class3)[:, 0], np.array(pure_pixel_location_class3)[:, 1]] = 3
    pure_pixel_location_image[
        np.array(pure_pixel_location_class4)[:, 0], np.array(pure_pixel_location_class4)[:, 1]] = 4

    array2Raster(pure_pixel_location_image, 'pure_pixel_location_image.tif', refImg=mcd43a4_file1, noDataValue=0)
    # ---------------------------------------------------------------------------------------------------
    # 按波段提取不同类别不同方向的反射率，高分数据和MODIS数据的反射率为了方便存储都扩大了10000倍
    # ---------------------------------------------------------------------------------------------------
    # 红色波段
    sr_gf_red_class1 = [sr_red_gf[pure_pixel_location_class1[i][0], pure_pixel_location_class1[i][1]] for i in
                        range(len(pure_pixel_location_class1))]
    sr_gf_red_class2 = [sr_red_gf[pure_pixel_location_class2[i][0], pure_pixel_location_class2[i][1]] for i in
                        range(len(pure_pixel_location_class2))]
    sr_gf_red_class3 = [sr_red_gf[pure_pixel_location_class3[i][0], pure_pixel_location_class3[i][1]] for i in
                        range(len(pure_pixel_location_class3))]
    sr_gf_red_class4 = [sr_red_gf[pure_pixel_location_class4[i][0], pure_pixel_location_class4[i][1]] for i in
                        range(len(pure_pixel_location_class4))]
    nbar_b1_class1 = [nbar_b1[pure_pixel_location_class1[i][0], pure_pixel_location_class1[i][1]] for i in
                      range(len(pure_pixel_location_class1))]
    nbar_b1_class2 = [nbar_b1[pure_pixel_location_class2[i][0], pure_pixel_location_class2[i][1]] for i in
                      range(len(pure_pixel_location_class2))]
    nbar_b1_class3 = [nbar_b1[pure_pixel_location_class3[i][0], pure_pixel_location_class3[i][1]] for i in
                      range(len(pure_pixel_location_class3))]
    nbar_b1_class4 = [nbar_b1[pure_pixel_location_class4[i][0], pure_pixel_location_class4[i][1]] for i in
                      range(len(pure_pixel_location_class4))]

    # 近红外波段
    sr_gf_nir_class1 = [sr_nir_gf[pure_pixel_location_class1[i][0], pure_pixel_location_class1[i][1]] for i in
                        range(len(pure_pixel_location_class1))]
    sr_gf_nir_class2 = [sr_nir_gf[pure_pixel_location_class2[i][0], pure_pixel_location_class2[i][1]] for i in
                        range(len(pure_pixel_location_class2))]
    sr_gf_nir_class3 = [sr_nir_gf[pure_pixel_location_class3[i][0], pure_pixel_location_class3[i][1]] for i in
                        range(len(pure_pixel_location_class3))]
    sr_gf_nir_class4 = [sr_nir_gf[pure_pixel_location_class4[i][0], pure_pixel_location_class4[i][1]] for i in
                        range(len(pure_pixel_location_class4))]
    nbar_b2_class1 = [nbar_b2[pure_pixel_location_class1[i][0], pure_pixel_location_class1[i][1]] for i in
                      range(len(pure_pixel_location_class1))]
    nbar_b2_class2 = [nbar_b2[pure_pixel_location_class2[i][0], pure_pixel_location_class2[i][1]] for i in
                      range(len(pure_pixel_location_class2))]
    nbar_b2_class3 = [nbar_b2[pure_pixel_location_class3[i][0], pure_pixel_location_class3[i][1]] for i in
                      range(len(pure_pixel_location_class3))]
    nbar_b2_class4 = [nbar_b2[pure_pixel_location_class4[i][0], pure_pixel_location_class4[i][1]] for i in
                      range(len(pure_pixel_location_class4))]

    # 蓝色波段
    sr_gf_blue_class1 = [sr_blue_gf[pure_pixel_location_class1[i][0], pure_pixel_location_class1[i][1]] for i in
                         range(len(pure_pixel_location_class1))]
    sr_gf_blue_class2 = [sr_blue_gf[pure_pixel_location_class2[i][0], pure_pixel_location_class2[i][1]] for i in
                         range(len(pure_pixel_location_class2))]
    sr_gf_blue_class3 = [sr_blue_gf[pure_pixel_location_class3[i][0], pure_pixel_location_class3[i][1]] for i in
                         range(len(pure_pixel_location_class3))]
    sr_gf_blue_class4 = [sr_blue_gf[pure_pixel_location_class4[i][0], pure_pixel_location_class4[i][1]] for i in
                         range(len(pure_pixel_location_class4))]
    nbar_b3_class1 = [nbar_b3[pure_pixel_location_class1[i][0], pure_pixel_location_class1[i][1]] for i in
                      range(len(pure_pixel_location_class1))]
    nbar_b3_class2 = [nbar_b3[pure_pixel_location_class2[i][0], pure_pixel_location_class2[i][1]] for i in
                      range(len(pure_pixel_location_class2))]
    nbar_b3_class3 = [nbar_b3[pure_pixel_location_class3[i][0], pure_pixel_location_class3[i][1]] for i in
                      range(len(pure_pixel_location_class3))]
    nbar_b3_class4 = [nbar_b3[pure_pixel_location_class4[i][0], pure_pixel_location_class4[i][1]] for i in
                      range(len(pure_pixel_location_class4))]

    # 绿色波段
    sr_gf_green_class1 = [sr_green_gf[pure_pixel_location_class1[i][0], pure_pixel_location_class1[i][1]] for i in
                          range(len(pure_pixel_location_class1))]
    sr_gf_green_class2 = [sr_green_gf[pure_pixel_location_class2[i][0], pure_pixel_location_class2[i][1]] for i in
                          range(len(pure_pixel_location_class2))]
    sr_gf_green_class3 = [sr_green_gf[pure_pixel_location_class3[i][0], pure_pixel_location_class3[i][1]] for i in
                          range(len(pure_pixel_location_class3))]
    sr_gf_green_class4 = [sr_green_gf[pure_pixel_location_class4[i][0], pure_pixel_location_class4[i][1]] for i in
                          range(len(pure_pixel_location_class4))]
    nbar_b4_class1 = [nbar_b4[pure_pixel_location_class1[i][0], pure_pixel_location_class1[i][1]] for i in
                      range(len(pure_pixel_location_class1))]
    nbar_b4_class2 = [nbar_b4[pure_pixel_location_class2[i][0], pure_pixel_location_class2[i][1]] for i in
                      range(len(pure_pixel_location_class2))]
    nbar_b4_class3 = [nbar_b4[pure_pixel_location_class3[i][0], pure_pixel_location_class3[i][1]] for i in
                      range(len(pure_pixel_location_class3))]
    nbar_b4_class4 = [nbar_b4[pure_pixel_location_class4[i][0], pure_pixel_location_class4[i][1]] for i in
                      range(len(pure_pixel_location_class4))]
    # ---------------------------------------------------------------------------------------------------
    # 建模，计算BRDF校正系数
    # ---------------------------------------------------------------------------------------------------
    # 红色波段
    coefficients_red_class1 = np.polyfit(sr_gf_red_class1, nbar_b1_class1, deg=1)
    coefficients_red_class2 = np.polyfit(sr_gf_red_class2, nbar_b1_class2, deg=1)
    coefficients_red_class3 = np.polyfit(sr_gf_red_class3, nbar_b1_class3, deg=1)
    coefficients_red_class4 = np.polyfit(sr_gf_red_class4, nbar_b1_class4, deg=1)
    # 近红外波段
    coefficients_nir_class1 = np.polyfit(sr_gf_nir_class1, nbar_b2_class1, deg=1)
    coefficients_nir_class2 = np.polyfit(sr_gf_nir_class2, nbar_b2_class2, deg=1)
    coefficients_nir_class3 = np.polyfit(sr_gf_nir_class3, nbar_b2_class3, deg=1)
    coefficients_nir_class4 = np.polyfit(sr_gf_nir_class4, nbar_b2_class4, deg=1)
    # 蓝色波段
    coefficients_blue_class1 = np.polyfit(sr_gf_blue_class1, nbar_b3_class1, deg=1)
    coefficients_blue_class2 = np.polyfit(sr_gf_blue_class2, nbar_b3_class2, deg=1)
    coefficients_blue_class3 = np.polyfit(sr_gf_blue_class3, nbar_b3_class3, deg=1)
    coefficients_blue_class4 = np.polyfit(sr_gf_blue_class4, nbar_b3_class4, deg=1)
    # 绿色波段
    coefficients_green_class1 = np.polyfit(sr_gf_green_class1, nbar_b4_class1, deg=1)
    coefficients_green_class2 = np.polyfit(sr_gf_green_class2, nbar_b4_class2, deg=1)
    coefficients_green_class3 = np.polyfit(sr_gf_green_class3, nbar_b4_class3, deg=1)
    coefficients_green_class4 = np.polyfit(sr_gf_green_class4, nbar_b4_class4, deg=1)
    # ---------------------------------------------------------------------------------------------------
    # 对校正模型画图
    # ---------------------------------------------------------------------------------------------------
    before_blue_class1_rmse = np.sqrt(
        ((np.array(sr_gf_red_class1) * 0.0001 - np.array(nbar_b1_class1) * 0.0001) ** 2).sum() / len(
            pure_pixel_location_class1))
    before_blue_class2_rmse = np.sqrt(
        ((np.array(sr_gf_red_class2) * 0.0001 - np.array(nbar_b1_class2) * 0.0001) ** 2).sum() / len(
            pure_pixel_location_class2))
    before_blue_class3_rmse = np.sqrt(
        ((np.array(sr_gf_red_class3) * 0.0001 - np.array(nbar_b1_class3) * 0.0001) ** 2).sum() / len(
            pure_pixel_location_class3))
    before_blue_class4_rmse = np.sqrt(
        ((np.array(sr_gf_red_class4) * 0.0001 - np.array(nbar_b1_class4) * 0.0001) ** 2).sum() / len(
            pure_pixel_location_class4))

    before_green_class1_rmse = np.sqrt(
        ((np.array(sr_gf_green_class1) * 0.0001 - np.array(nbar_b4_class1) * 0.0001) ** 2).sum() / len(
            pure_pixel_location_class1))
    before_green_class2_rmse = np.sqrt(
        ((np.array(sr_gf_green_class2) * 0.0001 - np.array(nbar_b4_class2) * 0.0001) ** 2).sum() / len(
            pure_pixel_location_class2))
    before_green_class3_rmse = np.sqrt(
        ((np.array(sr_gf_green_class3) * 0.0001 - np.array(nbar_b4_class3) * 0.0001) ** 2).sum() / len(
            pure_pixel_location_class3))
    before_green_class4_rmse = np.sqrt(
        ((np.array(sr_gf_green_class4) * 0.0001 - np.array(nbar_b4_class4) * 0.0001) ** 2).sum() / len(
            pure_pixel_location_class4))

    before_red_class1_rmse = np.sqrt(
        ((np.array(sr_gf_red_class1) * 0.0001 - np.array(nbar_b1_class1) * 0.0001) ** 2).sum() / len(
            pure_pixel_location_class1))
    before_red_class2_rmse = np.sqrt(
        ((np.array(sr_gf_red_class2) * 0.0001 - np.array(nbar_b1_class2) * 0.0001) ** 2).sum() / len(
            pure_pixel_location_class2))
    before_red_class3_rmse = np.sqrt(
        ((np.array(sr_gf_red_class3) * 0.0001 - np.array(nbar_b1_class3) * 0.0001) ** 2).sum() / len(
            pure_pixel_location_class3))
    before_red_class4_rmse = np.sqrt(
        ((np.array(sr_gf_red_class4) * 0.0001 - np.array(nbar_b1_class4) * 0.0001) ** 2).sum() / len(
            pure_pixel_location_class4))

    before_nir_class1_rmse = np.sqrt(
        ((np.array(sr_gf_nir_class1) * 0.0001 - np.array(nbar_b2_class1) * 0.0001) ** 2).sum() / len(
            pure_pixel_location_class1))
    before_nir_class2_rmse = np.sqrt(
        ((np.array(sr_gf_nir_class2) * 0.0001 - np.array(nbar_b2_class2) * 0.0001) ** 2).sum() / len(
            pure_pixel_location_class2))
    before_nir_class3_rmse = np.sqrt(
        ((np.array(sr_gf_nir_class3) * 0.0001 - np.array(nbar_b2_class3) * 0.0001) ** 2).sum() / len(
            pure_pixel_location_class3))
    before_nir_class4_rmse = np.sqrt(
        ((np.array(sr_gf_nir_class4) * 0.0001 - np.array(nbar_b2_class4) * 0.0001) ** 2).sum() / len(
            pure_pixel_location_class4))

    plt.figure(figsize=(15, 18), dpi=80)
    # 红色波段
    fig1 = plt.subplot(4, 4, 1)
    plot1(np.array(sr_gf_red_class1), np.array(nbar_b1_class1), coefficients_red_class1[0], coefficients_red_class1[1],
          len(pure_pixel_location_class1), rmse=before_red_class1_rmse, title='Red Class 1')
    fig2 = plt.subplot(4, 4, 2)
    plot1(np.array(sr_gf_red_class2), np.array(nbar_b1_class2), coefficients_red_class2[0], coefficients_red_class2[1],
          len(pure_pixel_location_class2), rmse=before_red_class2_rmse, title='Red Class 2')
    fig3 = plt.subplot(4, 4, 3)
    plot1(np.array(sr_gf_red_class3), np.array(nbar_b1_class3), coefficients_red_class3[0], coefficients_red_class3[1],
          len(pure_pixel_location_class3), rmse=before_red_class3_rmse, title='Red Class 3')
    fig4 = plt.subplot(4, 4, 4)
    plot1(np.array(sr_gf_red_class4), np.array(nbar_b1_class4), coefficients_red_class4[0], coefficients_red_class4[1],
          len(pure_pixel_location_class4), rmse=before_red_class4_rmse, title='Red Class 4')
    # 绿色波段
    fig5 = plt.subplot(4, 4, 5)
    plot1(np.array(sr_gf_green_class1), np.array(nbar_b4_class1),
          coefficients_green_class1[0], coefficients_green_class1[1],
          len(pure_pixel_location_class1), rmse=before_green_class1_rmse, title='Green Class 1')
    fig6 = plt.subplot(4, 4, 6)
    plot1(np.array(sr_gf_green_class2), np.array(nbar_b4_class2),
          coefficients_green_class2[0], coefficients_green_class2[1],
          len(pure_pixel_location_class2), rmse=before_green_class2_rmse, title='Green Class 2')
    fig7 = plt.subplot(4, 4, 7)
    plot1(np.array(sr_gf_green_class3), np.array(nbar_b4_class3),
          coefficients_green_class3[0], coefficients_green_class3[1],
          len(pure_pixel_location_class3), rmse=before_green_class3_rmse, title='Green Class 3')
    fig8 = plt.subplot(4, 4, 8)
    plot1(np.array(sr_gf_green_class4), np.array(nbar_b4_class4),
          coefficients_green_class4[0], coefficients_green_class4[1],
          len(pure_pixel_location_class4), rmse=before_green_class4_rmse, title='Green Class 4')
    # 蓝色波段
    fig9 = plt.subplot(4, 4, 9)
    plot1(np.array(sr_gf_blue_class1), np.array(nbar_b3_class1),
          coefficients_blue_class1[0], coefficients_blue_class1[1],
          len(pure_pixel_location_class1), rmse=before_blue_class1_rmse, title='Blue Class 1')
    fig10 = plt.subplot(4, 4, 10)
    plot1(np.array(sr_gf_blue_class2), np.array(nbar_b3_class2),
          coefficients_blue_class2[0], coefficients_blue_class2[1],
          len(pure_pixel_location_class2), rmse=before_blue_class2_rmse, title='Blue Class 2')
    fig11 = plt.subplot(4, 4, 11)
    plot1(np.array(sr_gf_blue_class3), np.array(nbar_b3_class3),
          coefficients_blue_class3[0], coefficients_blue_class3[1],
          len(pure_pixel_location_class3), rmse=before_blue_class3_rmse, title='Blue Class 3')
    fig12 = plt.subplot(4, 4, 12)
    plot1(np.array(sr_gf_blue_class4), np.array(nbar_b3_class4),
          coefficients_blue_class4[0], coefficients_blue_class4[1],
          len(pure_pixel_location_class4), rmse=before_blue_class4_rmse, title='Blue Class 4')
    # 近红外波段
    fig13 = plt.subplot(4, 4, 13)
    plot1(np.array(sr_gf_nir_class1), np.array(nbar_b2_class1),
          coefficients_nir_class1[0], coefficients_nir_class1[1],
          len(pure_pixel_location_class1), rmse=before_nir_class1_rmse, title='NIR Class 1')
    fig14 = plt.subplot(4, 4, 14)
    plot1(np.array(sr_gf_nir_class2), np.array(nbar_b2_class2),
          coefficients_nir_class2[0], coefficients_nir_class2[1],
          len(pure_pixel_location_class2), rmse=before_nir_class2_rmse, title='NIR Class 2')
    fig15 = plt.subplot(4, 4, 15)
    plot1(np.array(sr_gf_nir_class3), np.array(nbar_b2_class3),
          coefficients_nir_class3[0], coefficients_nir_class3[1],
          len(pure_pixel_location_class3), rmse=before_nir_class3_rmse, title='NIR Class 3')
    fig16 = plt.subplot(4, 4, 16)
    plot1(np.array(sr_gf_nir_class4), np.array(nbar_b2_class4),
          coefficients_nir_class4[0], coefficients_nir_class4[1],
          len(pure_pixel_location_class4), rmse=before_nir_class4_rmse, title='NIR Class 4')

    plt.tight_layout()
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.7)

    plt.savefig(f'SimulatedGFSR_MODISNBAR.png', dpi=300)
    plt.show()
    # ---------------------------------------------------------------------------------------------------
    # 进行BRDF校正
    # ---------------------------------------------------------------------------------------------------
    ndvi_30m = (gf_band4 - gf_band3) * 1.0 / (gf_band4 + gf_band3) * 1.0
    array2Raster(ndvi_30m, 'ndvi_30m.tif', refImg=gf_sr_file)
    # 存储BRDF校正后的结果
    simulated_directional_sr_red = np.zeros((gf_rows, gf_cols), dtype=np.int16)
    simulated_directional_sr_nir = np.zeros((gf_rows, gf_cols), dtype=np.int16)
    simulated_directional_sr_blue = np.zeros((gf_rows, gf_cols), dtype=np.int16)
    simulated_directional_sr_green = np.zeros((gf_rows, gf_cols), dtype=np.int16)

    ndvi_30m_class1_index = np.argwhere(ndvi_30m <= 0.3)
    ndvi_30m_class2_index = np.argwhere((ndvi_30m > 0.3) & (ndvi_30m <= 0.4))
    ndvi_30m_class3_index = np.argwhere((ndvi_30m > 0.4) & (ndvi_30m <= 0.6))
    ndvi_30m_class4_index = np.argwhere(ndvi_30m > 0.6)

    # 校正红色波段
    simulated_directional_sr_red[ndvi_30m_class1_index[:, 0], ndvi_30m_class1_index[:, 1]] = \
        gf_band3[ndvi_30m_class1_index[:, 0], ndvi_30m_class1_index[:, 1]] * coefficients_red_class1[0] \
        + coefficients_red_class1[1]
    simulated_directional_sr_red[ndvi_30m_class2_index[:, 0], ndvi_30m_class2_index[:, 1]] = \
        gf_band3[ndvi_30m_class2_index[:, 0], ndvi_30m_class2_index[:, 1]] * coefficients_red_class2[0] \
        + coefficients_red_class2[1]
    simulated_directional_sr_red[ndvi_30m_class3_index[:, 0], ndvi_30m_class3_index[:, 1]] = \
        gf_band3[ndvi_30m_class3_index[:, 0], ndvi_30m_class3_index[:, 1]] * coefficients_red_class3[0] \
        + coefficients_red_class3[1]
    simulated_directional_sr_red[ndvi_30m_class4_index[:, 0], ndvi_30m_class4_index[:, 1]] = \
        gf_band3[ndvi_30m_class4_index[:, 0], ndvi_30m_class4_index[:, 1]] * coefficients_red_class4[0] \
        + coefficients_red_class4[1]
    # 校正绿色波段
    simulated_directional_sr_green[ndvi_30m_class1_index[:, 0], ndvi_30m_class1_index[:, 1]] = \
        gf_band2[ndvi_30m_class1_index[:, 0], ndvi_30m_class1_index[:, 1]] * coefficients_green_class1[0] \
        + coefficients_green_class1[1]
    simulated_directional_sr_green[ndvi_30m_class2_index[:, 0], ndvi_30m_class2_index[:, 1]] = \
        gf_band2[ndvi_30m_class2_index[:, 0], ndvi_30m_class2_index[:, 1]] * coefficients_green_class2[0] \
        + coefficients_green_class2[1]
    simulated_directional_sr_green[ndvi_30m_class3_index[:, 0], ndvi_30m_class3_index[:, 1]] = \
        gf_band2[ndvi_30m_class3_index[:, 0], ndvi_30m_class3_index[:, 1]] * coefficients_green_class3[0] \
        + coefficients_green_class3[1]
    simulated_directional_sr_green[ndvi_30m_class4_index[:, 0], ndvi_30m_class4_index[:, 1]] = \
        gf_band2[ndvi_30m_class4_index[:, 0], ndvi_30m_class4_index[:, 1]] * coefficients_green_class4[0] \
        + coefficients_green_class4[1]
    # 校正蓝色波段
    simulated_directional_sr_blue[ndvi_30m_class1_index[:, 0], ndvi_30m_class1_index[:, 1]] = \
        gf_band1[ndvi_30m_class1_index[:, 0], ndvi_30m_class1_index[:, 1]] * coefficients_blue_class1[0] \
        + coefficients_blue_class1[1]
    simulated_directional_sr_blue[ndvi_30m_class2_index[:, 0], ndvi_30m_class2_index[:, 1]] = \
        gf_band1[ndvi_30m_class2_index[:, 0], ndvi_30m_class2_index[:, 1]] * coefficients_blue_class2[0] \
        + coefficients_blue_class2[1]
    simulated_directional_sr_blue[ndvi_30m_class3_index[:, 0], ndvi_30m_class3_index[:, 1]] = \
        gf_band1[ndvi_30m_class3_index[:, 0], ndvi_30m_class3_index[:, 1]] * coefficients_blue_class3[0] \
        + coefficients_blue_class3[1]
    simulated_directional_sr_blue[ndvi_30m_class4_index[:, 0], ndvi_30m_class4_index[:, 1]] = \
        gf_band1[ndvi_30m_class4_index[:, 0], ndvi_30m_class4_index[:, 1]] * coefficients_blue_class4[0] \
        + coefficients_blue_class4[1]
    # 校正近红外波段
    simulated_directional_sr_nir[ndvi_30m_class1_index[:, 0], ndvi_30m_class1_index[:, 1]] = \
        gf_band4[ndvi_30m_class1_index[:, 0], ndvi_30m_class1_index[:, 1]] * coefficients_nir_class1[0] \
        + coefficients_nir_class1[1]
    simulated_directional_sr_nir[ndvi_30m_class2_index[:, 0], ndvi_30m_class2_index[:, 1]] = \
        gf_band4[ndvi_30m_class2_index[:, 0], ndvi_30m_class2_index[:, 1]] * coefficients_nir_class2[0] \
        + coefficients_nir_class2[1]
    simulated_directional_sr_nir[ndvi_30m_class3_index[:, 0], ndvi_30m_class3_index[:, 1]] = \
        gf_band4[ndvi_30m_class3_index[:, 0], ndvi_30m_class3_index[:, 1]] * coefficients_nir_class3[0] \
        + coefficients_nir_class3[1]
    simulated_directional_sr_nir[ndvi_30m_class4_index[:, 0], ndvi_30m_class4_index[:, 1]] = \
        gf_band4[ndvi_30m_class4_index[:, 0], ndvi_30m_class4_index[:, 1]] * coefficients_nir_class4[0] \
        + coefficients_nir_class4[1]

    # 结果输出
    array2Raster([simulated_directional_sr_blue, simulated_directional_sr_green,
                  simulated_directional_sr_red, simulated_directional_sr_nir],
                 gf_sr_file[:-5]+'_BRDFCorrected.tif',
                 refImg=gf_sr_file, noDataValue=0)
    # ---------------------------------------------------------------------------------------------------
    # BRDF校正结果评价
    # ---------------------------------------------------------------------------------------------------
    # 将均一像元的值提取出来
    true_sr_band3_gf_30 = np.zeros((len(pure_pixel_location)), dtype=np.int16)
    true_sr_band4_gf_30 = np.zeros((len(pure_pixel_location)), dtype=np.int16)
    simulated_sr_b3 = np.zeros((len(pure_pixel_location)), dtype=np.int16)
    simulated_sr_b4 = np.zeros((len(pure_pixel_location)), dtype=np.int16)
    nbar_band1 = np.zeros((len(pure_pixel_location)), dtype=np.int16)
    nbar_band2 = np.zeros((len(pure_pixel_location)), dtype=np.int16)
    for i in range(len(pure_pixel_location)):
        x_30, y_30 = pix2map(pure_pixel_location[i][0], pure_pixel_location[i][0], modis_geoTransform)
        row_30, col_30 = map2pix(x_30, y_30, gf_geoTransform)
        row_30 = math.ceil(row_30)
        col_30 = math.ceil(col_30)
        # 校正前的反射率
        true_sr_band3_gf_30[i] = gf_band3[row_30, col_30]
        true_sr_band4_gf_30[i] = gf_band4[row_30, col_30]
        # 校正后的反射率
        simulated_sr_b3[i] = simulated_directional_sr_red[row_30, col_30]
        simulated_sr_b4[i] = simulated_directional_sr_nir[row_30, col_30]
        # NBAR
        nbar_band1[i] = nbar_b1[pure_pixel_location[i][0], pure_pixel_location[i][0]]
        nbar_band2[i] = nbar_b2[pure_pixel_location[i][0], pure_pixel_location[i][0]]
    # 计算校正前和校正后的EVI2
    evi2_nbar = 2.5 * (nbar_band2 - nbar_band1) / (nbar_band2 + nbar_band1)
    evi2_before = 2.5 * (true_sr_band4_gf_30 - true_sr_band3_gf_30) / (true_sr_band4_gf_30 + true_sr_band3_gf_30)
    evi2_after = 2.5 * (simulated_sr_b4 - simulated_sr_b3) / (simulated_sr_b4 + simulated_sr_b3)
    before_evi2_difference = evi2_before - evi2_nbar
    after_evi2_difference = evi2_after - evi2_nbar
    # 去除nan
    before_evi2_difference = before_evi2_difference[~np.isnan(before_evi2_difference)]
    after_evi2_difference = after_evi2_difference[~np.isnan(after_evi2_difference)]
    before_rmse = (before_evi2_difference ** 2).sum() / len(pure_pixel_location)
    after_rmse = (after_evi2_difference ** 2).sum() / len(pure_pixel_location)

    print(f'Before RMSE: {before_rmse} After RMSE: {after_rmse}')

    # 画图
    xxx = evi2_nbar
    yyy1 = evi2_after
    yyy2 = evi2_before

    font = {
        'family': 'serif',
        'color': 'darkred',
        'weight': 'normal',
        'size': 16,
    }
    plt.figure(figsize=(10, 5), dpi=80)
    plt.subplot(1, 2, 1)
    plt.scatter(xxx, yyy1, c='k')
    plt.xlabel('MODIS NBAR EVI2', fontdict=font)
    plt.ylabel('GF-1 EVI2', fontdict=font)
    plt.text(0.02, 0.95, f'RMSE: {round(before_rmse, 5)}')
    plt.axis([0, 1, 0, 1])

    plt.subplot(1, 2, 2)
    plt.scatter(xxx, yyy2, c='k')
    plt.xlabel('MODIS NBAR EVI2', fontdict=font)
    plt.ylabel('BRDF Corrected GF-1 EVI2', fontdict=font)
    plt.text(0.02, 0.95, f'RMSE: {round(after_rmse, 5)}')
    plt.axis([0, 1, 0, 1])

    plt.tight_layout()

    plt.show()


if __name__ == '__main__':
    import os
    import time

    a = time.time()

    # GF1_PMS1_E113.5_N30.3_20180417_L1A0003127123
    # os.chdir(r'F:\Experiment\BRDF Correction\GF1_PMS1_E113.5_N30.3_20180417_L1A0003127123')
    # # 反射率数据
    # gf_sr_file = 'GF1_PMS1_E113.5_N30.3_20180417_L1A0003127123-MSS1_RC_FLAASH_ORT.tiff'
    # # 角度数据
    # gf_sza_file = 'GF1_PMS1_E113.5_N30.3_20180417_L1A0003127123-MSS1_resampled_SZA_ORT.tif'
    # gf_saa_file = 'GF1_PMS1_E113.5_N30.3_20180417_L1A0003127123-MSS1_resampled_SAA_ORT.tif'
    # gf_vza_file = 'GF1_PMS1_E113.5_N30.3_20180417_L1A0003127123-MSS1_resampled_VZA_ORT.tif'
    # gf_vaa_file = 'GF1_PMS1_E113.5_N30.3_20180417_L1A0003127123-MSS1_resampled_VAA_ORT.tif'
    # # BRDF参数数据
    # mcd43a1_file1 = r'F:\Experiment\BRDF Correction\BRDF model parameters\MCD43A1_2018_04_17_band1.tif'  # 红波段
    # mcd43a1_file2 = r'F:\Experiment\BRDF Correction\BRDF model parameters\MCD43A1_2018_04_17_band2.tif'  # 近红外
    # mcd43a1_file3 = r'F:\Experiment\BRDF Correction\BRDF model parameters\MCD43A1_2018_04_17_band3.tif'  # 蓝波段
    # mcd43a1_file4 = r'F:\Experiment\BRDF Correction\BRDF model parameters\MCD43A1_2018_04_17_band4.tif'  # 绿波段
    # # NBAR数据
    # mcd43a4_file1 = r'F:\Experiment\BRDF Correction\BRDF model parameters\MCD43A4_2018_04_17_band1.tif'
    # mcd43a4_file2 = r'F:\Experiment\BRDF Correction\BRDF model parameters\MCD43A4_2018_04_17_band2.tif'
    # mcd43a4_file3 = r'F:\Experiment\BRDF Correction\BRDF model parameters\MCD43A4_2018_04_17_band3.tif'
    # mcd43a4_file4 = r'F:\Experiment\BRDF Correction\BRDF model parameters\MCD43A4_2018_04_17_band4.tif'

    # GF1_WFV1_E112.6_N29.6_20220502_L1A0006459246
    # os.chdir(r'F:\Experiment\BRDF Correction\GF1_WFV1_E112.6_N29.6_20220502_L1A0006459246')
    #
    # gf_sr_file = 'GF1_WFV1_E112.6_N29.6_20220502_L1A0006459246_RC_FLAASH_ORT.tiff'
    #
    # gf_sza_file = 'GF1_WFV1_E112.6_N29.6_20220502_L1A0006459246_resampled_SZA_ORT.tif'
    # gf_saa_file = 'GF1_WFV1_E112.6_N29.6_20220502_L1A0006459246_resampled_SAA_ORT.tif'
    # gf_vza_file = 'GF1_WFV1_E112.6_N29.6_20220502_L1A0006459246_resampled_VZA_ORT.tif'
    # gf_vaa_file = 'GF1_WFV1_E112.6_N29.6_20220502_L1A0006459246_resampled_VAA_ORT.tif'
    #
    # mcd43a1_file1 = r'F:\Experiment\BRDF Correction\BRDF model parameters\MCD43A1_2022_05_02_band1.tif'  # 红波段
    # mcd43a1_file2 = r'F:\Experiment\BRDF Correction\BRDF model parameters\MCD43A1_2022_05_02_band2.tif'  # 近红外
    # mcd43a1_file3 = r'F:\Experiment\BRDF Correction\BRDF model parameters\MCD43A1_2022_05_02_band3.tif'  # 蓝波段
    # mcd43a1_file4 = r'F:\Experiment\BRDF Correction\BRDF model parameters\MCD43A1_2022_05_02_band4.tif'  # 绿波段
    #
    # mcd43a4_file1 = r'F:\Experiment\BRDF Correction\BRDF model parameters\MCD43A4_2022_05_02_band1.tif'
    # mcd43a4_file2 = r'F:\Experiment\BRDF Correction\BRDF model parameters\MCD43A4_2022_05_02_band2.tif'
    # mcd43a4_file3 = r'F:\Experiment\BRDF Correction\BRDF model parameters\MCD43A4_2022_05_02_band3.tif'
    # mcd43a4_file4 = r'F:\Experiment\BRDF Correction\BRDF model parameters\MCD43A4_2022_05_02_band4.tif'

    # GF1_PMS1_E114.2_N30.0_20180729_L1A0003357062
    # os.chdir(r'F:\Experiment\BRDF Correction\GF1PMS\GF1_PMS1_E114.2_N30.0_20180729_L1A0003357062')
    # # 反射率数据
    # gf_sr_file = 'GF1_PMS1_E114.2_N30.0_20180729_L1A0003357062-MSS1_RC_FLAASH_ORT.tiff'
    # # 角度数据
    # gf_sza_file = 'GF1_PMS1_E114.2_N30.0_20180729_L1A0003357062-MSS1_resampled_SZA_ORT.tif'
    # gf_saa_file = 'GF1_PMS1_E114.2_N30.0_20180729_L1A0003357062-MSS1_resampled_SAA_ORT.tif'
    # gf_vza_file = 'GF1_PMS1_E114.2_N30.0_20180729_L1A0003357062-MSS1_resampled_VZA_ORT.tif'
    # gf_vaa_file = 'GF1_PMS1_E114.2_N30.0_20180729_L1A0003357062-MSS1_resampled_VAA_ORT.tif'
    # # BRDF参数数据
    # mcd43a1_file1 = r'F:\Experiment\BRDF Correction\BRDF model parameters\MCD43A1_2018_07_29_band1.tif'  # 红波段
    # mcd43a1_file2 = r'F:\Experiment\BRDF Correction\BRDF model parameters\MCD43A1_2018_07_29_band2.tif'  # 近红外
    # mcd43a1_file3 = r'F:\Experiment\BRDF Correction\BRDF model parameters\MCD43A1_2018_07_29_band3.tif'  # 蓝波段
    # mcd43a1_file4 = r'F:\Experiment\BRDF Correction\BRDF model parameters\MCD43A1_2018_07_29_band4.tif'  # 绿波段
    # # NBAR数据
    # mcd43a4_file1 = r'F:\Experiment\BRDF Correction\BRDF model parameters\MCD43A4_2018_07_29_band1.tif'
    # mcd43a4_file2 = r'F:\Experiment\BRDF Correction\BRDF model parameters\MCD43A4_2018_07_29_band2.tif'
    # mcd43a4_file3 = r'F:\Experiment\BRDF Correction\BRDF model parameters\MCD43A4_2018_07_29_band3.tif'
    # mcd43a4_file4 = r'F:\Experiment\BRDF Correction\BRDF model parameters\MCD43A4_2018_07_29_band4.tif'

    # GF1_PMS1_E114.2_N30.3_20190729_L1A0004147581
    # os.chdir(r'F:\Experiment\BRDF Correction\GF1PMS\GF1_PMS1_E114.2_N30.3_20190729_L1A0004147581')
    # # 反射率数据
    # gf_sr_file = 'GF1_PMS1_E114.2_N30.3_20190729_L1A0004147581-MSS1_RC_FLAASH_ORT.tiff'
    # # 角度数据
    # gf_sza_file = 'GF1_PMS1_E114.2_N30.3_20190729_L1A0004147581-MSS1_resampled_SZA_ORT.tif'
    # gf_saa_file = 'GF1_PMS1_E114.2_N30.3_20190729_L1A0004147581-MSS1_resampled_SAA_ORT.tif'
    # gf_vza_file = 'GF1_PMS1_E114.2_N30.3_20190729_L1A0004147581-MSS1_resampled_VZA_ORT.tif'
    # gf_vaa_file = 'GF1_PMS1_E114.2_N30.3_20190729_L1A0004147581-MSS1_resampled_VAA_ORT.tif'
    # # BRDF参数数据
    # mcd43a1_file1 = r'F:\Experiment\BRDF Correction\BRDF model parameters\MCD43A1_2019_07_29_band1.tif'  # 红波段
    # mcd43a1_file2 = r'F:\Experiment\BRDF Correction\BRDF model parameters\MCD43A1_2019_07_29_band2.tif'  # 近红外
    # mcd43a1_file3 = r'F:\Experiment\BRDF Correction\BRDF model parameters\MCD43A1_2019_07_29_band3.tif'  # 蓝波段
    # mcd43a1_file4 = r'F:\Experiment\BRDF Correction\BRDF model parameters\MCD43A1_2019_07_29_band4.tif'  # 绿波段
    # # NBAR数据
    # mcd43a4_file1 = r'F:\Experiment\BRDF Correction\BRDF model parameters\MCD43A4_2019_07_29_band1.tif'
    # mcd43a4_file2 = r'F:\Experiment\BRDF Correction\BRDF model parameters\MCD43A4_2019_07_29_band2.tif'
    # mcd43a4_file3 = r'F:\Experiment\BRDF Correction\BRDF model parameters\MCD43A4_2019_07_29_band3.tif'
    # mcd43a4_file4 = r'F:\Experiment\BRDF Correction\BRDF model parameters\MCD43A4_2019_07_29_band4.tif'

    # GF1_PMS2_E113.9_N30.5_20180417_L1A0003127222
    os.chdir(r'F:\Experiment\BRDF Correction\GF1PMS\GF1_PMS2_E113.9_N30.5_20180417_L1A0003127222')
    # 反射率数据
    gf_sr_file = 'GF1_PMS2_E113.9_N30.5_20180417_L1A0003127222-MSS2_RC_FLAASH_ORT.tiff'
    # 角度数据
    gf_sza_file = 'GF1_PMS2_E113.9_N30.5_20180417_L1A0003127222-MSS2_resampled_SZA_ORT.tif'
    gf_saa_file = 'GF1_PMS2_E113.9_N30.5_20180417_L1A0003127222-MSS2_resampled_SAA_ORT.tif'
    gf_vza_file = 'GF1_PMS2_E113.9_N30.5_20180417_L1A0003127222-MSS2_resampled_VZA_ORT.tif'
    gf_vaa_file = 'GF1_PMS2_E113.9_N30.5_20180417_L1A0003127222-MSS2_resampled_VAA_ORT.tif'
    # BRDF参数数据
    mcd43a1_file1 = r'F:\Experiment\BRDF Correction\BRDF model parameters\MCD43A1_2018_04_17_band1_2.tif'  # 红波段
    mcd43a1_file2 = r'F:\Experiment\BRDF Correction\BRDF model parameters\MCD43A1_2018_04_17_band2_2.tif'  # 近红外
    mcd43a1_file3 = r'F:\Experiment\BRDF Correction\BRDF model parameters\MCD43A1_2018_04_17_band3_2.tif'  # 蓝波段
    mcd43a1_file4 = r'F:\Experiment\BRDF Correction\BRDF model parameters\MCD43A1_2018_04_17_band4_2.tif'  # 绿波段
    # NBAR数据
    mcd43a4_file1 = r'F:\Experiment\BRDF Correction\BRDF model parameters\MCD43A4_2018_04_17_band1_2.tif'
    mcd43a4_file2 = r'F:\Experiment\BRDF Correction\BRDF model parameters\MCD43A4_2018_04_17_band2_2.tif'
    mcd43a4_file3 = r'F:\Experiment\BRDF Correction\BRDF model parameters\MCD43A4_2018_04_17_band3_2.tif'
    mcd43a4_file4 = r'F:\Experiment\BRDF Correction\BRDF model parameters\MCD43A4_2018_04_17_band4_2.tif'

    brdf_correction(gf_sr_file, gf_sza_file, gf_saa_file, gf_vza_file, gf_vaa_file,
                    mcd43a1_file1, mcd43a1_file2, mcd43a1_file3, mcd43a1_file4,
                    mcd43a4_file1, mcd43a4_file2, mcd43a4_file3, mcd43a4_file4,
                    sensor_type='PMS', cv_threshold=0.4)

    b = time.time()
    print(f'共耗时{b-a}s')
