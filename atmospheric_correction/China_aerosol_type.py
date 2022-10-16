"""
计算国内不同区域的气溶胶类型
即沙尘性（dust）、水溶性（water）、海洋性（oceanic）和煤烟性（soot）粒子各自的占比
"""

from Py6S import *
import numpy as np
from itertools import product
from tqdm import *


aerosol_proportion = {
    # 范娇等，2015的研究结果袁发现无论是中国内陆地区还是沿海地区,其气溶胶水溶性粒子所占比重最大,均高于40%
    # 海洋性和煤烟性粒子组分最少,均低于10%
    'dust': np.arange(0, 0.46, 0.01),
    'water': np.arange(0.4, 1.01, 0.01),
    'oceanic': np.arange(0, 0.11, 0.01)
}


def day_of_year(year, month, day):
    hour = 0
    # 儒略日 Julian day
    JD0 = int(365.25 * (year - 1)) + int(30.6001 * (1 + 13)) + 1 + hour / 24 + 1720981.5

    if month <= 2:
        JD2 = int(365.25 * (year - 1)) + int(30.6001 * (month + 13)) + day + hour / 24 + 1720981.5
    else:
        JD2 = int(365.25 * year) + int(30.6001 * (month + 1)) + day + hour / 24 + 1720981.5

    # 年积日 Day of year
    DOY = JD2 - JD0 + 1

    return DOY


def permutate_aerosol_invars(invars):
    """
    将输入的参数排列组合
    """
    return list(product(invars['dust'],
                        invars['water'],
                        invars['oceanic']))


def define_aerosol_type(aeronet_aod675, aeronet_aod440, sixs_param):
    # 李兆麟等取0.47μm、0.66μm两个波段的气溶胶光学厚度的平均值得到550nm的AOD
    # TODO:范娇等，2015，基于MODIS数据的杭州地区气溶胶光学厚度反演：可依据Angstrom波长指数公式将太阳光度计的观测波段转换到550nm的AOD
    aod550 = (aeronet_aod675 + aeronet_aod440)

    # 根据MODIS数据，设置6S模型参数
    s = SixS()
    # 1.Geometries
    s.geometry = Geometry.User()
    s.geometry.month = sixs_param['month']
    s.geometry.day = sixs_param['day']
    s.geometry.solar_a = sixs_param['saa']
    s.geometry.solar_z = sixs_param['sza']
    s.geometry.view_a = sixs_param['vaa']
    s.geometry.view_z = sixs_param['vza']
    # 2.Atmospheric Profiles
    s.atmos_profile = AtmosProfile.UserWaterAndOzone(water=sixs_param['water'],
                                                     ozone=sixs_param['o3'])
    # 大气模式类型
    # 3.AOD550
    s.aot550 = aod550
    # 4.Wavelength
    s.wavelength = Wavelength(PredefinedWavelengths.ACCURATE_MODIS_TERRA_2)
    # 5.Altitudes
    s.altitudes.set_sensor_satellite_level()
    s.altitudes.set_target_custom_altitude(0)   # 海拔对大气校正参数影响不大，暂时设置为0
    # 6.大气校正设置
    # 大气校正模式对校正系数结果无影响，设置为均一下垫面绿色植被
    s.atmos_corr = AtmosCorr.AtmosCorrLambertianFromReflectance(GroundReflectance.GreenVegetation)

    radiance = sixs_param['radiance']
    sr = sixs_param['sr']

    perms = permutate_aerosol_invars(aerosol_proportion)
    pbar = tqdm(total=len(perms))
    outputs = []
    for perm in perms:
        dust_proportion = perm[0]
        water_proportion = perm[1]
        oceanic_proportion = perm[2]
        if (dust_proportion + water_proportion + oceanic_proportion) > 1:
            pbar.update(1)
            continue
        soot_proportion = 1 - (dust_proportion + water_proportion + oceanic_proportion)

        # 7.自定义设置气溶胶模式
        s.aero_profile = AeroProfile.User(dust=dust_proportion,
                                          water=water_proportion,
                                          oceanic=oceanic_proportion,
                                          soot=soot_proportion)
        # 运行6S模型
        s.run()

        xa = s.outputs.coef_xa
        xb = s.outputs.coef_xb
        xc = s.outputs.coef_xc

        y = xa * radiance - xb
        corrected_sr = y / (1 + xc * y)
        outputs.append((corrected_sr, sr, (corrected_sr-sr)**2,
                        dust_proportion, water_proportion, oceanic_proportion, soot_proportion))

        pbar.update(1)

    return outputs


if __name__ == '__main__1':
    """
    复现李兆麟等，2019，利用6S模型的自定义气溶胶类型反演北京地区气溶胶光学厚度
    确定北京地区气溶胶类型的实验
    """
    import os
    import pickle
    import gdal
    from pyproj import CRS, Transformer
    from BRDF.BRDF_correction import map2pix

    os.chdir(r'F:\Experiment\China Aerosol Type\MOD021KM')

    # 角度数据
    mod021km_file = 'MOD021KM.A2017261.0320.061.2017261205630_Radiance_Georeferenced_geotiff.tif'
    mod09_file = 'MOD09.A2017261.0320.006.2017263021726_500mSR_Georeferenced_geotiff.tif'
    geometries_file = 'MOD021KM.A2017261.0320.061.2017261205630_Geometries_Resampled_Georeferenced_geotiff.tif'
    mod05_file = 'MOD05_Swath_2D_1_georef_geotiff.tif'
    mod07_file = 'MOD07_Swath_2D_1_georef_geotiff.tif'
    mod021km_ds = gdal.Open(mod021km_file)
    mod09_ds = gdal.Open(mod09_file)
    geometries_ds = gdal.Open(geometries_file)
    mod05_ds = gdal.Open(mod05_file)
    mod07_ds = gdal.Open(mod07_file)
    # 读取影像
    mod021km_red_img = mod021km_ds.GetRasterBand(1).ReadAsArray()   # 红色波段
    mod021km_nir_img = mod021km_ds.GetRasterBand(2).ReadAsArray()   # 近红外波段
    mod09_red_img = mod09_ds.GetRasterBand(1).ReadAsArray()
    mod09_nir_img = mod09_ds.GetRasterBand(2).ReadAsArray()
    vza_img = geometries_ds.GetRasterBand(1).ReadAsArray()
    vaa_img = geometries_ds.GetRasterBand(2).ReadAsArray()
    sza_img = geometries_ds.GetRasterBand(3).ReadAsArray()
    saa_img = geometries_ds.GetRasterBand(4).ReadAsArray()
    water_img = mod05_ds.GetRasterBand(1).ReadAsArray()
    o3_img = mod07_ds.GetRasterBand(1).ReadAsArray()

    # 读取仿射变换系数
    mod021km_geotransform = mod021km_ds.GetGeoTransform()
    mod09_geotransform = mod09_ds.GetGeoTransform()
    geometries_geotransform = geometries_ds.GetGeoTransform()
    mod05_geotransform = mod05_ds.GetGeoTransform()
    mod07_geotransform = mod07_ds.GetGeoTransform()

    # 选取2017年9月18日Beijing-CAMS测站对应的AOD
    # AERONET的观测数据是空间某一点的时间序列，Terra卫星的过境时间大约为当地时间上午10:30，对数据进行时空匹配
    # AOD数据选取卫星过境时刻前后0.5h内的地基观测平均值
    Beijing_CAMS_aod675 = 0.142300333
    Beijing_CAMS_aod440 = 0.229117389
    # Beijing_CAMS测站的经纬度
    Beijing_CAMS_longitude = 116.317
    Beijing_CAMS_latitude = 39.933
    # 将经纬度坐标转换为UTM-WGS84 51N投影坐标
    geo_crs = CRS.from_epsg(4326)
    pro_crs = CRS.from_epsg(32651)
    transformer = Transformer.from_crs(geo_crs, pro_crs)
    x, y = transformer.transform(Beijing_CAMS_latitude, Beijing_CAMS_longitude)
    # 将地理坐标转换为影像行列数
    mod021km_row, mod021km_col = map2pix(x, y, mod021km_geotransform)
    mod09_row, mod09_col = map2pix(x, y, mod09_geotransform)
    geometries_row, geometries_col = map2pix(x, y, geometries_geotransform)
    mod05_row, mod05_col = map2pix(x, y, mod05_geotransform)
    mod07_row, mod07_col = map2pix(x, y, mod07_geotransform)
    # 根据对应行列数读取影像
    red_radiance = mod021km_red_img[mod021km_row, mod021km_col]
    nir_radiance = mod021km_nir_img[mod021km_row, mod021km_col]
    red_sr = mod09_red_img[mod09_row, mod09_col] / 1e8
    nir_sr = mod09_nir_img[mod09_row, mod09_col] / 1e8
    vza = vza_img[geometries_row, geometries_col] / 100
    vaa = vaa_img[geometries_row, geometries_col] / 100
    sza = sza_img[geometries_row, geometries_col] / 100
    saa = saa_img[geometries_row, geometries_col] / 100
    water = water_img[mod05_row, mod05_col]
    o3 = o3_img[mod07_row, mod07_col] / 1000

    # 记录MODIS数据6S模型参数设置
    modis_config = {'radiance': nir_radiance, 'sr': nir_sr, 'vza': vza, 'vaa': vaa, 'sza': sza,
                    'saa': saa, 'water': water, 'o3': o3, 'month': 9, 'day': 18}

    res = define_aerosol_type(Beijing_CAMS_aod675, Beijing_CAMS_aod440, modis_config)
    # 存储结果
    pickle.dump(res, open(r'F:\Experiment\China Aerosol Type\Beijing_CAMS_20170918_nir_band.txt', 'wb'))


if __name__ == '__main__':
    import os
    import pickle

    os.chdir(r'F:\Experiment\China Aerosol Type')
    red_band_file = 'Beijing_CAMS_20170918_red_band.txt'
    nir_band_file = 'Beijing_CAMS_20170918_nir_band.txt'

    with open(red_band_file, "rb") as red_band:
        red_band_res = pickle.load(red_band)

    with open(nir_band_file, "rb") as nir_band:
        nir_band_res = pickle.load(nir_band)

    res = []
    pbar = tqdm(total=len(red_band_res))
    for i in range(len(red_band_res)):
        err = red_band_res[i][2] + (nir_band_res[i][0] - 0.1357945999431777) ** 2
        res.append((err, red_band_res[i][3:]))
        pbar.update(1)
    # 按照误差的大小排序
    res.sort(key=lambda x: x[0], reverse=False)
    print(res[0])
