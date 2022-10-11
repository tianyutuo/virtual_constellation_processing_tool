"""
使用6S模型建立查找表
"""
import os.path

from Py6S import *
from itertools import product
import sys
import time
import math
from tqdm import *
import pickle
import json
import numpy as np
from scipy import interpolate
from band_conversion.calculate_SBAF import read_SRF
import matplotlib.pyplot as plt


def input_variables(build_type):
    """
    定义查找表参数
    参数包括：
    - 太阳天顶角（Solar Zenith Angle，SAA，单位为度）
    - 观测天顶角（View Zenith Angle，VAA，单位为度）
    - 相对方位角（Relative Azimuth Angle，RAA，单位为度）
    - 气溶胶光学厚度（Aerosol Optical Depth，AOD）
    - 水蒸气（Water Vapour Colum，单位为g/m2）
    - 臭氧（Ozone Column，单位为cm-atm）
    - 海拔（altitude，单位为km）
    """
    fixed = {
        'sza': [10],
        'vza': [5],
        'raa': [60],
        'aod': [0.2],
        'h2o': [1],
        'o3': [0.4],
        'alt': [1]
    }

    sza_sensitivity_analysis = {
        'sza': np.arange(0, 81, 1),
        'vza': [5],
        'raa': [60],
        'aod': [0.2],
        'h2o': [1],
        'o3': [0.4],
        'alt': [1]
    }

    sza_big_angle_step10 = {
        'sza': [60, 70],
        'vza': [5, 10],
        'raa': [30, 60],
        'aod': [0.2, 0.3],
        'h2o': [1, 1.5],
        'o3': [0.4, 0.8],
        'alt': [1, 3]
    }

    sza_big_angle_step5 = {
        'sza': [60, 65, 70],
        'vza': [5, 10],
        'raa': [30, 60],
        'aod': [0.2, 0.3],
        'h2o': [1, 1.5],
        'o3': [0.4, 0.8],
        'alt': [1, 3]
    }

    sza_big_angle_step2 = {
        'sza': [60, 62, 64, 66, 68, 70],
        'vza': [5, 10],
        'raa': [30, 60],
        'aod': [0.2, 0.3],
        'h2o': [1, 1.5],
        'o3': [0.4, 0.8],
        'alt': [1, 3]
    }

    sza_big_angle_validate = {
        'sza': [63],
        'vza': [6],
        'raa': [60],
        'aod': [0.3],
        'h2o': [1],
        'o3': [0.4],
        'alt': [1]
    }

    sza_small_angle_step20 = {
        'sza': [0, 20],
        'vza': [5, 10],
        'raa': [30, 60],
        'aod': [0.2, 0.3],
        'h2o': [1, 1.5],
        'o3': [0.4, 0.8],
        'alt': [1, 3]
    }

    sza_small_angle_step10 = {
        'sza': [0, 10, 20],
        'vza': [5, 10],
        'raa': [30, 60],
        'aod': [0.2, 0.3],
        'h2o': [1, 1.5],
        'o3': [0.4, 0.8],
        'alt': [1, 3]
    }

    sza_small_angle_validate = {
        'sza': [13],
        'vza': [6],
        'raa': [60],
        'aod': [0.3],
        'h2o': [1],
        'o3': [0.4],
        'alt': [1]
    }

    vza_sensitivity_analysis = {
        'sza': [10],
        'vza': np.arange(0, 81, 1),
        'raa': [60],
        'aod': [0.2],
        'h2o': [1],
        'o3': [0.4],
        'alt': [1]
    }

    vza_big_angle_step10 = {
        'sza': [5, 10],
        'vza': [60, 70],
        'raa': [30, 60],
        'aod': [0.2, 0.3],
        'h2o': [1, 1.5],
        'o3': [0.4, 0.8],
        'alt': [1, 3]
    }

    vza_big_angle_step5 = {
        'sza': [5, 10],
        'vza': [60, 65, 70],
        'raa': [30, 60],
        'aod': [0.2, 0.3],
        'h2o': [1, 1.5],
        'o3': [0.4, 0.8],
        'alt': [1, 3]
    }

    vza_big_angle_step2 = {
        'sza': [5, 10],
        'vza': [60, 62, 64, 66, 68, 70],
        'raa': [30, 60],
        'aod': [0.2, 0.3],
        'h2o': [1, 1.5],
        'o3': [0.4, 0.8],
        'alt': [1, 3]
    }

    vza_big_angle_validate = {
        'sza': [6],
        'vza': [63],
        'raa': [60],
        'aod': [0.3],
        'h2o': [1],
        'o3': [0.4],
        'alt': [1]
    }

    vza_small_angle_step20 = {
        'sza': [5, 10],
        'vza': [0, 20],
        'raa': [30, 60],
        'aod': [0.2, 0.3],
        'h2o': [1, 1.5],
        'o3': [0.4, 0.8],
        'alt': [1, 3]
    }

    vza_small_angle_step10 = {
        'sza': [5, 10],
        'vza': [0, 10, 20],
        'raa': [30, 60],
        'aod': [0.2, 0.3],
        'h2o': [1, 1.5],
        'o3': [0.4, 0.8],
        'alt': [1, 3]
    }

    vza_small_angle_validate = {
        'sza': [6],
        'vza': [13],
        'raa': [60],
        'aod': [0.3],
        'h2o': [1],
        'o3': [0.4],
        'alt': [1]
    }

    raa_sensitivity_analysis = {
        'sza': [10],
        'vza': [10],
        'raa': np.arange(0, 361, 1),
        'aod': [0.2],
        'h2o': [1],
        'o3': [0.4],
        'alt': [1]
    }

    o3_sensitivity_analysis = {
        'sza': [10],
        'vza': [5],
        'raa': [60],
        'aod': [0.2],
        'h2o': [1],
        'o3': np.arange(0, 0.9, 0.1),
        'alt': [1]
    }

    h2o_sensitivity_analysis = {
        'sza': [10],
        'vza': [5],
        'raa': [60],
        'aod': [0.2],
        'h2o': np.arange(0, 8.6, 0.1),
        'o3': [0.4],
        'alt': [1]
    }

    aod_sensitivity_analysis = {
        'sza': [10],
        'vza': [5],
        'raa': [60],
        'aod': np.arange(0, 3.1, 0.1),
        'h2o': [1],
        'o3': [0.4],
        'alt': [1]
    }

    small_aod_step025 = {
        'sza': [5, 10],
        'vza': [5, 10],
        'raa': [30, 60],
        'aod': [0.3, 0.55, 0.8],
        'h2o': [1, 1.5],
        'o3': [0.4, 0.8],
        'alt': [1, 3]
    }

    small_aod_step01 = {
        'sza': [5, 10],
        'vza': [5, 10],
        'raa': [30, 60],
        'aod': [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        'h2o': [1, 1.5],
        'o3': [0.4, 0.8],
        'alt': [1, 3]
    }

    small_aod_validate = {
        'sza': [6],
        'vza': [6],
        'raa': [60],
        'aod': [0.45],
        'h2o': [1],
        'o3': [0.4],
        'alt': [1]
    }

    big_aod_step025 = {
        'sza': [5, 10],
        'vza': [5, 10],
        'raa': [30, 60],
        'aod': [2.0, 2.25, 2.5],
        'h2o': [1, 1.5],
        'o3': [0.4, 0.8],
        'alt': [1, 3]
    }

    big_aod_step01 = {
        'sza': [5, 10],
        'vza': [5, 10],
        'raa': [30, 60],
        'aod': [2.0, 2.1, 2.2, 2.3, 2.4, 2.5],
        'h2o': [1, 1.5],
        'o3': [0.4, 0.8],
        'alt': [1, 3]
    }

    big_aod_step005 = {
        'sza': [5, 10],
        'vza': [5, 10],
        'raa': [30, 60],
        'aod': [2.0, 2.05, 2.1, 2.15, 2.2, 2.25, 2.3, 2.35, 2.4, 2.45, 2.5],
        'h2o': [1, 1.5],
        'o3': [0.4, 0.8],
        'alt': [1, 3]
    }

    big_aod_validate = {
        'sza': [6],
        'vza': [6],
        'raa': [60],
        'aod': [2.35],
        'h2o': [1],
        'o3': [0.4],
        'alt': [1]
    }

    alt_sensitivity_analysis = {
        'sza': [10],
        'vza': [5],
        'raa': [60],
        'aod': [0.2],
        'h2o': [1],
        'o3': [0.4],
        'alt': np.arange(0, 8.00, 0.25)
    }

    full = {
        'sza': [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65],
        'vza': [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65],
        'raa': [0, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120, 132, 144, 156, 168, 180],
        'aod': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0, 3.0],
        'h2o': [0, 0.25, 0.5, 1, 1.5, 2, 3, 5, 8.5],
        'o3': [0.0, 0.8],
        'alt': [0, 1, 4, 7.75]
    }

    build_type_selector = {
        'fixed': fixed,
        'sza_sensitivity_analysis': sza_sensitivity_analysis,
        'sza_big_angle_step10': sza_big_angle_step10,
        'sza_big_angle_step5': sza_big_angle_step5,
        'sza_big_angle_step2': sza_big_angle_step2,
        'sza_big_angle_validate': sza_big_angle_validate,
        'sza_small_angle_step20': sza_small_angle_step20,
        'sza_small_angle_step10': sza_small_angle_step10,
        'sza_small_angle_validate': sza_small_angle_validate,
        'vza_sensitivity_analysis': vza_sensitivity_analysis,
        'vza_big_angle_step10': vza_big_angle_step10,
        'vza_big_angle_step5': vza_big_angle_step5,
        'vza_big_angle_step2': vza_big_angle_step2,
        'vza_big_angle_validate': vza_big_angle_validate,
        'vza_small_angle_step20': vza_small_angle_step20,
        'vza_small_angle_step10': vza_small_angle_step10,
        'vza_small_angle_validate': vza_small_angle_validate,
        'raa_sensitivity_analysis': raa_sensitivity_analysis,
        'o3_sensitivity_analysis': o3_sensitivity_analysis,
        'h2o_sensitivity_analysis': h2o_sensitivity_analysis,
        'aod_sensitivity_analysis': aod_sensitivity_analysis,
        'small_aod_step025': small_aod_step025,
        'small_aod_step01': small_aod_step01,
        'small_aod_validate': small_aod_validate,
        'big_aod_step025': big_aod_step025,
        'big_aod_step01': big_aod_step01,
        'big_aod_step005': big_aod_step005,
        'big_aod_validate': big_aod_validate,
        'alt_sensitivity_analysis': alt_sensitivity_analysis,
        'full': full
    }

    return build_type_selector[build_type]


def permutate_invars(invars):
    """
    将输入的参数排列组合
    """
    return list(product(invars['sza'],
                        invars['vza'],
                        invars['raa'],
                        invars['aod'],
                        invars['h2o'],
                        invars['o3'],
                        invars['alt']))


def creat_LUT(config):
    """
    构建6S辐射传输模型查找表（Look Up Table，LUT）
    """
    s = SixS()
    # ---------------------------------------------------------------------------------------------------
    # 设置固定的参数
    # ---------------------------------------------------------------------------------------------------
    # 传感器类型用户自定义（需定义SZA、SAA、VZA、VAA、影像获取的月份、天数）
    s.geometry = Geometry.User()
    # 参考：https://sourcegraph.com/github.com/samsammurphy/6S_emulator@master/-/blob/LUT_build.py
    # 此处将月份、天数设置为近日点
    # s.geometry.month = 1
    # s.geometry.day = 4
    s.geometry.month = config['month']
    s.geometry.day = config['day']
    # 气溶胶模式
    s.aero_profile = AeroProfile.__dict__[config['aerosol_profile']]
    # 将高度设置为星载传感器的级别
    s.altitudes.set_sensor_satellite_level()
    # TODO:参考https://sourcegraph.com/github.com/samsammurphy/6S_emulator@master/-/blob/LUT_build.py未设置下垫面类型和大气校正设置
    # 下垫面类型
    # 默认设置为：ground_reflectance = GroundReflectance.HomogeneousLambertian(0.3)
    s.ground_reflectance = GroundReflectance.HomogeneousLambertian(0.3)
    # s.ground_reflectance = GroundReflectance.HomogeneousMODISBRDF(0.035, 0.024, 0.013)
    # 大气校正设置
    # 默认设置为：self.atmos_corr = AtmosCorr.NoAtmosCorr()
    if config['atmo_corr'] == 'BRDFFromRadiance':
        s.atmos_corr = AtmosCorr.AtmosCorrBRDFFromRadiance(config['radiance'])
    elif config['atmo_corr'] == 'BRDFFromReflectance':
        s.atmos_corr = AtmosCorr.AtmosCorrBRDFFromReflectance(config['reflectance'])
    elif config['atmo_corr'] == 'LambertianFromRadiance':
        s.atmos_corr = AtmosCorr.AtmosCorrLambertianFromRadiance(config['radiance'])
    elif config['atmo_corr'] == 'LambertianFromReflectance':
        s.atmos_corr = AtmosCorr.AtmosCorrLambertianFromReflectance(config['reflectance'])
    elif config['atmo_corr'] == 'NotAtmosCorr':
        s.atmos_corr = AtmosCorr.NoAtmosCorr()
    # ---------------------------------------------------------------------------------------------------
    # 依照其他参数设置的组合排列构建查找表
    # ---------------------------------------------------------------------------------------------------
    # 计算其他参数的组合排列
    perms = permutate_invars(config['invars'])

    outputs = []
    pbar = tqdm(total=len(perms))   # 进度条
    for perm in perms:
        print('{0}: SZA = {1[0]:02}, VZA = {1[1]:02}, RAA = {1[2]:03}, AOT = {1[3]:.2f}, '
              'H2O = {1[4]:.2f}, O3 = {1[5]:.1f},  alt = {1[6]:.2f}'.format(config['filename'], perm))
        # 设置角度信息
        s.geometry.solar_z = perm[0]
        s.geometry.view_z = perm[1]
        # TODO:相对方位角如何设置
        # 两个方位角的绝对值之差不变结果相同，绝对值之差结果互补结果相同
        s.geometry.solar_a = perm[2]
        s.geometry.view_a = 0
        # 设置臭氧和水汽
        s.atmos_profile = AtmosProfile.UserWaterAndOzone(perm[4], perm[5])
        # 设置550nm气溶胶光学厚度
        s.aot550 = perm[3]
        # 设置海拔高度
        s.altitudes.set_target_custom_altitude(perm[6])
        # 设置传感器光谱信息
        s.wavelength = config['spectrum']

        # 运行6S模型
        s.run()
        # print(s.outputs.fulltext)

        if config['atmo_corr'] == 'NotAtmosCorr':
            # 太阳辐照度 Solar irradiance
            Edir = s.outputs.direct_solar_irradiance  # direct solar irradiance
            Edif = s.outputs.diffuse_solar_irradiance  # diffuse solar irradiance

            E = Edir + Edif  # total solar irradiance
            # transmissivity
            absorb = s.outputs.trans['global_gas'].upward  # absorption transmissivity
            scatter = s.outputs.trans['total_scattering'].upward  # scattering transmissivity
            tau2 = absorb * scatter  # transmissivity (from surface to sensor)
            # path radiance
            Lp = s.outputs.atmospheric_intrinsic_radiance  # path radiance

            # correction coefficients for this configuration
            # i.e. surface_reflectance = (L - a) / b,
            #      where, L is at-sensor radiance
            a = Lp
            b = (tau2 * E) / math.pi
            outputs.append((a, b))
        else:
            # 输出大气校正参数
            # TODO:Py6S不设置大气校正设置无法输出大气校正系数xa、xb、xc
            xa = s.outputs.coef_xa
            xb = s.outputs.coef_xb
            xc = s.outputs.coef_xc

            outputs.append((xa, xb, xc))
        pbar.update(1)

    # 建立查找表
    LUT = {'config': config, 'outputs': outputs}
    pickle.dump(LUT, open(config['filepath'], 'wb'))

    return outputs


def IO_handler(config, channel, wavelength, spectral_filter, sensor_name):
    """
    设置查找表输出文件名和路径
    """
    # 输出文件名
    filename = []
    if wavelength:
        filename.append('wavelength')
        if len(wavelength) == 2:
            filename.append('_%.2f_%.2f' % (wavelength[0], wavelength[1]))
        else:
            filename.append('_%f' % wavelength[0])
    if spectral_filter:
        filename.append('_f')
    # 增加build_type的类型
    filename.append('_%s' % config['build_type'])
    filename = ''.join(filename)

    if channel:
        filename = channel

    if channel:
        sensor_name = '_'.join(filename.split('_')[:-1])
    elif sensor_name is None:
        sensor_name = 'user-defined-sensor'

    # 输出路径
    base_path = os.path.dirname(os.path.abspath(__file__))
    outdir = os.path.join(base_path, 'LUTs', sensor_name, config['aerosol_profile'])
    if not os.path.exists(outdir):
        print('创建输出路径\n'+outdir+'\n')
        os.makedirs(outdir)
    os.chdir(outdir)

    # 更新设置文件
    config['outdir'] = outdir
    config['filename'] = filename + '.lut'
    config['filepath'] = os.path.join(outdir, filename + '.lut')


def main(channel=None, wavelength=None, spectral_filter=None, aerosol_profile=None, build_type=None, sensor_name=None,
         atmo_corr=None, radiance=None, reflectance=None, month=None, day=None):
    # Py6S定义传感器光谱信息的方法有四种
    # 1.只给出一个波长值，模拟只在这个波长上进行，例Wavelength(0.43)
    # 2.给出波段范围，例Wavelength(0.43, 0.50)
    # 3.给出波段范围和相应的光谱响应函数，注意此处光谱响应函数的间隔为2.5nm，例Wavelength(0.400, 0.410, [0.7, 0.9, 1.0, 0.3])
    # 4.使用Py6S预定义的传感器波段，例Wavelength(PredefinedWavelengths.LANDSAT_TM_B1)
    if wavelength:
        if len(wavelength) > 2:
            print('波长指定错误，只能是单个数值或开始和结束的波长，给定的波长为： ', wavelength)
            sys.exit(1)

        start_wavelength = float(wavelength[0])

        if len(wavelength) == 2:
            end_wavelength = float(wavelength[1])

            # (可选) 提供光谱响应函数
            if spectral_filter:
                # 注意光谱响应函数的间隔应为2.5nm
                n = (end_wavelength - start_wavelength) / 0.0025 + 1
                l = len(spectral_filter)
                if abs(l - n) > 1e-6:
                    print('光谱响应函数的间隔应为2.5nm, 应为 {} 个值, 输入 {}个值'.format(round(n), l))
                    sys.exit(1)
                else:
                    spectrum = Wavelength(start_wavelength, end_wavelength=end_wavelength, filter=spectral_filter)
            else:
                spectrum = Wavelength(start_wavelength, end_wavelength=end_wavelength)
        else:
            # 此处只设定一个波长值
            spectrum = Wavelength(start_wavelength)

    # Py6S预定义的传感器光谱设置, 详见Py6S 'PredefinedWavelengths':
    # https://github.com/robintw/Py6S/blob/master/Py6S/Params/wavelength.py
    if channel:
        try:
            spectrum = Wavelength(PredefinedWavelengths.__dict__[channel])
        except:
            print('未能识别预定义的传感器波段: ', channel)
            sys.exit(1)

    # 确保传感器光谱参数的设置
    try:
        spectrum
    except NameError:
        print('must define wavelength(s) or sensor channel, returning..')
        sys.exit(1)

    # aerosol profile（默认为Continental）
    if aerosol_profile:
        try:
            test = AeroProfile.__dict__[aerosol_profile]
        except:
            print('未能识别该气溶胶模式： ', aerosol_profile,
                  'Py6S提供了7中标准模式类型：无气溶胶（NoAerosols）、大陆型（Continental）、海洋型（Maritime）、'
                  '城市型（Urban）、沙漠型（Desert）、生物燃烧型（BiomassBurning）、平流层气溶胶模式（Stratospheric）')
            sys.exit(1)
    else:
        aerosol_profile = 'Continental'

    # bulid type 查找表建立模式（默认为test）
    if build_type:
        if build_type not in ['fixed', 'sza_sensitivity_analysis', 'vza_sensitivity_analysis',
                              'sza_big_angle_step10', 'sza_big_angle_step5', 'sza_big_angle_step2',
                              'sza_big_angle_validate', 'sza_small_angle_step20', 'sza_small_angle_step10',
                              'sza_small_angle_validate', 'vza_big_angle_step10', 'vza_big_angle_step5',
                              'vza_big_angle_step2', 'vza_big_angle_validate', 'vza_small_angle_step20',
                              'vza_small_angle_step10', 'vza_small_angle_validate', 'small_aod_step025',
                              'small_aod_step01', 'small_aod_validate', 'big_aod_step005',
                              'big_aod_step025', 'big_aod_step01', 'big_aod_validate',
                              'raa_sensitivity_analysis', 'o3_sensitivity_analysis', 'h2o_sensitivity_analysis',
                              'aod_sensitivity_analysis', 'alt_sensitivity_analysis', 'full']:
            print('查找表模式未识别：', build_type)
            sys.exit(1)
    else:
        print('未定义查找表构建模式，将使用test模式')
        build_type = 'test'

    # atmo_corr 大气校正模式（默认为NotAtmosCorr()）
    if atmo_corr:
        if atmo_corr not in ['BRDFFromRadiance', 'BRDFFromReflectance',
                             'LambertianFromRadiance', 'LambertianFromReflectance',
                             'NotAtmosCorr']:
            print('大气校正模式未识别：', atmo_corr)
            sys.exit(1)
    else:
        print('未定义大气校正模式，将使用NotAtmosCorr()')
        atmo_corr = 'NotAtmosCorr'

    # 下垫面类型和大气校正模式radiance和reflectance的参数设置
    if radiance:
        if radiance < 0:
            print('辐射率设置错误：', radiance)
            sys.exit(1)
    else:
        radiance = 30
    if reflectance:
        if reflectance < 0 or reflectance > 1:
            print('反射率设置错误：', reflectance)
            sys.exit(1)
    else:
        reflectance = 0.1

    # 影像获取的日期
    if month:
        if month < 0 or month > 12:
            print('影像获取月份设置错误：', month)
            sys.exit(1)
    else:
        month = 1
    if day:
        if day < 0 or day > 31:
            print('影像获取日期设置错误：', day)
            sys.exit(1)
    else:
        day = 4

    config = {
        'spectrum': spectrum,
        'aerosol_profile': aerosol_profile,
        'build_type': build_type,
        'atmo_corr': atmo_corr,
        'radiance': radiance,
        'reflectance': reflectance,
        'month': month,
        'day': day,
        'invars': input_variables(build_type)
    }

    # 设置查找表文件的输出
    IO_handler(config, channel, wavelength, spectral_filter, sensor_name)

    # 记录查找表构建用时
    a = time.time()

    # 构建查找表
    if os.path.isfile(config['filepath']):
        print('查找表已存在，停止构建查找表：' + config['filepath'])
        sys.exit(1)
    else:
        print('开始构建查找表：' + config['filepath'])
        res = creat_LUT(config)

    b = time.time()
    T = b - a
    print('time: {:.1f} secs, {:.1f} mins,{:.1f} hours'.format(T, T / 60, T / 3600))

    return res


# 读取GF1-WFV1四个波段光谱响应函数（此文件只记录了固定的波长范围，如蓝色波段450~520nm）
radiometricCorrectionParameter_file = r'E:\code\atmospheric_correction\RadiometricCorrectionParameter.json'
radiometricCorrectionParameter = json.load(open(radiometricCorrectionParameter_file))
GF1_WFV_Blue_SRF = radiometricCorrectionParameter['Parameter']['GF1']['WFV1']["SRF"]["1"]
GF1_WFV_Green_SRF = radiometricCorrectionParameter['Parameter']['GF1']['WFV1']["SRF"]["2"]
GF1_WFV_Red_SRF = radiometricCorrectionParameter['Parameter']['GF1']['WFV1']["SRF"]["3"]
GF1_WFV_NIR_SRF = radiometricCorrectionParameter['Parameter']['GF1']['WFV1']["SRF"]["4"]


if __name__ == '__main__1':
    """
    高分数据波段设置差异（1.只设置固定的波长范围（如蓝色波段450~520nm） 2.设置400~1000nm的响应函数
    """
    # 直接从光谱响应函数文件里读取，波段范围为400~1000nm（GF数据波段设置不止在波长范围内有响应）
    srf_file = r'F:\Experiment\Band Conversion\Spectral Response Function\高分1号光谱响应函数.xlsx'
    sensor_type = 'PMS1'
    srf_blue, srf_green, srf_red, srf_nir = read_SRF(srf_file, sensor_type)
    # 光谱响应函数的波段间隔为1nm，此处需重采样为2.5nm
    wl = [i for i in np.arange(0.4, 1.001, 0.001)]
    new_wl = [i for i in np.arange(0.4, 1.0025, 0.0025)]
    blue_fun = interpolate.splrep(wl, srf_blue)
    interpolated_blue_srf = list(interpolate.splev(new_wl, blue_fun))
    green_fun = interpolate.splrep(wl, srf_green)
    interpolated_green_srf = list(interpolate.splev(new_wl, green_fun))
    red_fun = interpolate.splrep(wl, srf_red)
    interpolated_red_srf = list(interpolate.splev(new_wl, red_fun))
    nir_fun = interpolate.splrep(wl, srf_nir)
    interpolated_nir_srf = list(interpolate.splev(new_wl, nir_fun))

    # 1.只设置固定的波长范围
    # 蓝色波段
    outputs1 = main(channel=None,
                    wavelength=(0.450, 0.520),
                    spectral_filter=GF1_WFV_Blue_SRF,
                    aerosol_profile='Continental',
                    build_type='fixed',
                    atmo_corr='LambertianFromRadiance',
                    radiance=30,
                    sensor_name='GF1-PMS1')
    # 绿色波段
    outputs2 = main(channel=None,
                    wavelength=(0.520, 0.590),
                    spectral_filter=GF1_WFV_Green_SRF,
                    aerosol_profile='Continental',
                    build_type='fixed',
                    atmo_corr='LambertianFromRadiance',
                    radiance=30,
                    sensor_name='GF1-PMS1')
    # 红色波段
    outputs3 = main(channel=None,
                    wavelength=(0.630, 0.690),
                    spectral_filter=GF1_WFV_Red_SRF,
                    aerosol_profile='Continental',
                    build_type='fixed',
                    atmo_corr='LambertianFromRadiance',
                    radiance=30,
                    sensor_name='GF1-PMS1')
    # 近红外波段
    outputs4 = main(channel=None,
                    wavelength=(0.770, 0.890),
                    spectral_filter=GF1_WFV_NIR_SRF,
                    aerosol_profile='Continental',
                    build_type='fixed',
                    atmo_corr='LambertianFromRadiance',
                    radiance=30,
                    sensor_name='GF1-PMS1')

    # 2.设置400~1000nm的响应函数
    # 蓝色波段
    outputs5 = main(channel=None,
                    wavelength=(0.400, 1.000),
                    spectral_filter=interpolated_blue_srf,
                    aerosol_profile='Continental',
                    build_type='fixed',
                    atmo_corr='LambertianFromRadiance',
                    radiance=30,
                    sensor_name='GF1-PMS1')
    # 绿色波段
    outputs6 = main(channel=None,
                    wavelength=(0.400, 1.000),
                    spectral_filter=interpolated_green_srf,
                    aerosol_profile='Continental',
                    build_type='fixed',
                    atmo_corr='LambertianFromRadiance',
                    radiance=30,
                    sensor_name='GF1-PMS1')
    # 红色波段
    outputs7 = main(channel=None,
                    wavelength=(0.400, 1.000),
                    spectral_filter=interpolated_red_srf,
                    aerosol_profile='Continental',
                    build_type='fixed',
                    atmo_corr='LambertianFromRadiance',
                    radiance=30,
                    sensor_name='GF1-PMS1')
    # 近红外波段
    output8 = main(channel=None,
                   wavelength=(0.400, 1.000),
                   spectral_filter=interpolated_nir_srf,
                   aerosol_profile='Continental',
                   build_type='fixed',
                   atmo_corr='LambertianFromRadiance',
                   radiance=30,
                   sensor_name='GF1-PMS1')


if __name__ == '__main__1':
    """
    比较两种大气校正系数
    """
    # 直接从光谱响应函数文件里读取，波段范围为400~1000nm（GF数据波段设置不止在波长范围内有响应）
    srf_file = r'F:\Experiment\Band Conversion\Spectral Response Function\高分1号光谱响应函数.xlsx'
    sensor_type = 'WFV1'
    srf_blue, srf_green, srf_red, srf_nir = read_SRF(srf_file, sensor_type)
    # 光谱响应函数的波段间隔为1nm，此处需重采样为2.5nm
    wl = [i for i in np.arange(0.4, 1.001, 0.001)]
    new_wl = [i for i in np.arange(0.4, 1.0025, 0.0025)]
    blue_fun = interpolate.splrep(wl, srf_blue)
    interpolated_blue_srf = list(interpolate.splev(new_wl, blue_fun))
    green_fun = interpolate.splrep(wl, srf_green)
    interpolated_green_srf = list(interpolate.splev(new_wl, green_fun))
    red_fun = interpolate.splrep(wl, srf_red)
    interpolated_red_srf = list(interpolate.splev(new_wl, red_fun))
    nir_fun = interpolate.splrep(wl, srf_nir)
    interpolated_nir_srf = list(interpolate.splev(new_wl, nir_fun))

    outputs1 = main(channel=None,
                    wavelength=(0.400, 1.000),
                    spectral_filter=interpolated_nir_srf,
                    aerosol_profile='Continental',
                    build_type='fixed',
                    atmo_corr='LambertianFromReflectance',
                    radiance=30,
                    sensor_name='GF1-PMS1')
    outputs2 = main(channel=None,
                    wavelength=(0.400, 1.000),
                    spectral_filter=interpolated_nir_srf,
                    aerosol_profile='Continental',
                    build_type='fixed',
                    atmo_corr='NotAtmosCorr',
                    sensor_name='GF1-PMS1')
    sensor_radiance = 160
    y = outputs1[0][0] * sensor_radiance - outputs1[0][1]
    sr1 = y / (1 + outputs1[0][2] * y)
    doy = 4
    elliptical_orbit_correction = 0.03275104 * math.cos(doy / 59.66638337) + 0.96804905
    a = outputs2[0][0] * elliptical_orbit_correction
    b = outputs2[0][1] * elliptical_orbit_correction
    sr2 = (sensor_radiance - a) / b
    print(outputs1)
    print(sr1)
    print(outputs2)
    print(sr2)


if __name__ == '__main__1':
    """
    参数敏感性分析——臭氧
    """
    # 直接从光谱响应函数文件里读取，波段范围为400~1000nm（GF数据波段设置不止在波长范围内有响应）
    srf_file = r'F:\Experiment\Band Conversion\Spectral Response Function\高分1号光谱响应函数.xlsx'
    sensor_type = 'WFV1'
    srf_blue, srf_green, srf_red, srf_nir = read_SRF(srf_file, sensor_type)
    # 光谱响应函数的波段间隔为1nm，此处需重采样为2.5nm
    wl = [i for i in np.arange(0.4, 1.001, 0.001)]
    new_wl = [i for i in np.arange(0.4, 1.0025, 0.0025)]
    blue_fun = interpolate.splrep(wl, srf_blue)
    interpolated_blue_srf = list(interpolate.splev(new_wl, blue_fun))
    green_fun = interpolate.splrep(wl, srf_green)
    interpolated_green_srf = list(interpolate.splev(new_wl, green_fun))
    red_fun = interpolate.splrep(wl, srf_red)
    interpolated_red_srf = list(interpolate.splev(new_wl, red_fun))
    nir_fun = interpolate.splrep(wl, srf_nir)
    interpolated_nir_srf = list(interpolate.splev(new_wl, nir_fun))

    # 蓝色波段
    outputs1 = main(channel=None,
                    wavelength=(0.400, 1.000),
                    spectral_filter=interpolated_blue_srf,
                    aerosol_profile='Continental',
                    build_type='o3_sensitivity_analysis',
                    atmo_corr='LambertianFromReflectance',
                    radiance=30,
                    sensor_name='GF1-PMS1')
    # 绿色波段
    outputs2 = main(channel=None,
                    wavelength=(0.400, 1.000),
                    spectral_filter=interpolated_green_srf,
                    aerosol_profile='Continental',
                    build_type='o3_sensitivity_analysis',
                    atmo_corr='LambertianFromReflectance',
                    radiance=30,
                    sensor_name='GF1-PMS1')
    # 红色波段
    outputs3 = main(channel=None,
                    wavelength=(0.400, 1.000),
                    spectral_filter=interpolated_red_srf,
                    aerosol_profile='Continental',
                    build_type='o3_sensitivity_analysis',
                    atmo_corr='LambertianFromReflectance',
                    radiance=30,
                    sensor_name='GF1-PMS1')
    # 近红外波段
    outputs4 = main(channel=None,
                    wavelength=(0.400, 1.000),
                    spectral_filter=interpolated_nir_srf,
                    aerosol_profile='Continental',
                    build_type='o3_sensitivity_analysis',
                    atmo_corr='LambertianFromReflectance',
                    radiance=30,
                    sensor_name='GF1-PMS1')
    o3 = np.arange(0, 0.9, 0.1)
    xa1s = [val[0] for val in outputs1]
    xb1s = [val[1] for val in outputs1]
    xc1s = [val[2] for val in outputs1]
    xa2s = [val[0] for val in outputs2]
    xb2s = [val[1] for val in outputs2]
    xc2s = [val[2] for val in outputs2]
    xa3s = [val[0] for val in outputs3]
    xb3s = [val[1] for val in outputs3]
    xc3s = [val[2] for val in outputs3]
    xa4s = [val[0] for val in outputs4]
    xb4s = [val[1] for val in outputs4]
    xc4s = [val[2] for val in outputs4]

    # 画图
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
    plt.figure(figsize=(10, 10), dpi=80)
    fig1 = plt.subplot(2, 2, 1)
    plt.plot(o3, xa1s)
    plt.plot(o3, xb1s)
    plt.plot(o3, xc1s)
    plt.legend(['xa', 'xb', 'xc'])
    plt.axis([0, 0.8, 0, 0.6])
    plt.xlabel(f'O3', fontdict=font)
    plt.ylabel(f'Blue band Correction Coefficients', fontdict=font)
    fig2 = plt.subplot(2, 2, 2)
    plt.plot(o3, xa2s)
    plt.plot(o3, xb2s)
    plt.plot(o3, xc2s)
    plt.legend(['xa', 'xb', 'xc'])
    plt.axis([0, 0.8, 0, 0.6])
    plt.xlabel(f'O3', fontdict=font)
    plt.ylabel(f'Green band Correction Coefficients', fontdict=font)
    fig3 = plt.subplot(2, 2, 3)
    plt.plot(o3, xa3s)
    plt.plot(o3, xb3s)
    plt.plot(o3, xc3s)
    plt.legend(['xa', 'xb', 'xc'])
    plt.axis([0, 0.8, 0, 0.6])
    plt.xlabel(f'O3', fontdict=font)
    plt.ylabel(f'Red band Correction Coefficients', fontdict=font)
    fig4 = plt.subplot(2, 2, 4)
    plt.plot(o3, xa4s)
    plt.plot(o3, xb4s)
    plt.plot(o3, xc4s)
    plt.legend(['xa', 'xb', 'xc'])
    plt.axis([0, 0.8, 0, 0.6])
    plt.xlabel(f'O3', fontdict=font)
    plt.ylabel(f'NIR band Correction Coefficients', fontdict=font)

    plt.tight_layout()
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)

    plt.savefig(f'Sensitivity analysis of O3.png', dpi=300)
    plt.show()


if __name__ == '__main__1':
    """
    参数敏感性分析——水汽
    """
    # 蓝色波段
    outputs1 = main(channel=None,
                    wavelength=(0.450, 0.520),
                    spectral_filter=GF1_WFV_Blue_SRF,
                    aerosol_profile='Continental',
                    build_type='h2o_sensitivity_analysis',
                    atmo_corr='LambertianFromReflectance',
                    radiance=30,
                    sensor_name='GF1-WFV1')
    # 绿色波段
    outputs2 = main(channel=None,
                    wavelength=(0.520, 0.590),
                    spectral_filter=GF1_WFV_Green_SRF,
                    aerosol_profile='Continental',
                    build_type='h2o_sensitivity_analysis',
                    atmo_corr='LambertianFromReflectance',
                    radiance=30,
                    sensor_name='GF1-WFV1')
    # 红色波段
    outputs3 = main(channel=None,
                    wavelength=(0.630, 0.690),
                    spectral_filter=GF1_WFV_Red_SRF,
                    aerosol_profile='Continental',
                    build_type='h2o_sensitivity_analysis',
                    atmo_corr='LambertianFromReflectance',
                    radiance=30,
                    sensor_name='GF1-WFV1')
    # 近红外波段
    outputs4 = main(channel=None,
                    wavelength=(0.770, 0.890),
                    spectral_filter=GF1_WFV_NIR_SRF,
                    aerosol_profile='Continental',
                    build_type='h2o_sensitivity_analysis',
                    atmo_corr='LambertianFromReflectance',
                    radiance=30,
                    sensor_name='GF1-WFV1')
    h2o = np.arange(0, 8.6, 0.1)
    xa1s = [val[0] for val in outputs1]
    xb1s = [val[1] for val in outputs1]
    xc1s = [val[2] for val in outputs1]
    xa2s = [val[0] for val in outputs2]
    xb2s = [val[1] for val in outputs2]
    xc2s = [val[2] for val in outputs2]
    xa3s = [val[0] for val in outputs3]
    xb3s = [val[1] for val in outputs3]
    xc3s = [val[2] for val in outputs3]
    xa4s = [val[0] for val in outputs4]
    xb4s = [val[1] for val in outputs4]
    xc4s = [val[2] for val in outputs4]

    # 画图
    font = {
        'family': 'serif',
        'color': 'darkred',
        'weight': 'normal',
        'size': 9,
    }
    plt.figure(figsize=(10, 10), dpi=80)
    fig1 = plt.subplot(2, 2, 1)
    plt.plot(h2o, xa1s)
    plt.plot(h2o, xb1s)
    plt.plot(h2o, xc1s)
    plt.legend(['xa', 'xb', 'xc'])
    plt.axis([0, 8.5, 0, 0.6])
    plt.xlabel(f'H2O', fontdict=font)
    plt.ylabel(f'Blue band Correction Coefficients', fontdict=font)
    fig2 = plt.subplot(2, 2, 2)
    plt.plot(h2o, xa2s)
    plt.plot(h2o, xb2s)
    plt.plot(h2o, xc2s)
    plt.legend(['xa', 'xb', 'xc'])
    plt.axis([0, 8.5, 0, 0.6])
    plt.xlabel(f'H2O', fontdict=font)
    plt.ylabel(f'Green band Correction Coefficients', fontdict=font)
    fig3 = plt.subplot(2, 2, 3)
    plt.plot(h2o, xa3s)
    plt.plot(h2o, xb3s)
    plt.plot(h2o, xc3s)
    plt.legend(['xa', 'xb', 'xc'])
    plt.axis([0, 8.5, 0, 0.6])
    plt.xlabel(f'H2O', fontdict=font)
    plt.ylabel(f'Red band Correction Coefficients', fontdict=font)
    fig4 = plt.subplot(2, 2, 4)
    plt.plot(h2o, xa4s)
    plt.plot(h2o, xb4s)
    plt.plot(h2o, xc4s)
    plt.legend(['xa', 'xb', 'xc'])
    plt.axis([0, 8.5, 0, 0.6])
    plt.xlabel(f'H2O', fontdict=font)
    plt.ylabel(f'NIR band Correction Coefficients', fontdict=font)

    plt.tight_layout()
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)

    plt.savefig(f'Sensitivity analysis of H2O.png', dpi=300)
    plt.show()


if __name__ == '__main__1':
    """
    参数敏感性分析——气溶胶模式
    """
    predefined_aerosol_files = ['NoAerosols', 'Continental', 'Maritime',
                                'Urban', 'Desert', 'BiomassBurning', 'Stratospheric']
    xa1s = []
    xb1s = []
    xc1s = []
    xa2s = []
    xb2s = []
    xc2s = []
    xa3s = []
    xb3s = []
    xc3s = []
    xa4s = []
    xb4s = []
    xc4s = []
    for predefined_aerosol_file in predefined_aerosol_files:
        # 蓝色波段
        outputs1 = main(channel=None,
                        wavelength=(0.450, 0.520),
                        spectral_filter=GF1_WFV_Blue_SRF,
                        aerosol_profile=predefined_aerosol_file,
                        build_type='fixed',
                        atmo_corr='LambertianFromReflectance',
                        radiance=30,
                        sensor_name='GF1-WFV1')
        # 绿色波段
        outputs2 = main(channel=None,
                        wavelength=(0.520, 0.590),
                        spectral_filter=GF1_WFV_Green_SRF,
                        aerosol_profile=predefined_aerosol_file,
                        build_type='fixed',
                        atmo_corr='LambertianFromReflectance',
                        radiance=30,
                        sensor_name='GF1-WFV1')
        # 红色波段
        outputs3 = main(channel=None,
                        wavelength=(0.630, 0.690),
                        spectral_filter=GF1_WFV_Red_SRF,
                        aerosol_profile=predefined_aerosol_file,
                        build_type='fixed',
                        atmo_corr='LambertianFromReflectance',
                        radiance=30,
                        sensor_name='GF1-WFV1')
        # 近红外波段
        outputs4 = main(channel=None,
                        wavelength=(0.770, 0.890),
                        spectral_filter=GF1_WFV_NIR_SRF,
                        aerosol_profile=predefined_aerosol_file,
                        build_type='fixed',
                        atmo_corr='LambertianFromReflectance',
                        radiance=30,
                        sensor_name='GF1-WFV1')
        xa1s.append(outputs1[0][0])
        xb1s.append(outputs1[0][1])
        xc1s.append(outputs1[0][2])
        xa2s.append(outputs2[0][0])
        xb2s.append(outputs2[0][1])
        xc2s.append(outputs2[0][2])
        xa3s.append(outputs3[0][0])
        xb3s.append(outputs3[0][1])
        xc3s.append(outputs3[0][2])
        xa4s.append(outputs4[0][0])
        xb4s.append(outputs4[0][1])
        xc4s.append(outputs4[0][2])

    # 画图
    aerosol_files = np.arange(0, 7, 1)
    font = {
        'family': 'serif',
        'color': 'darkred',
        'weight': 'normal',
        'size': 9,
    }
    plt.figure(figsize=(15, 10), dpi=80)
    fig1 = plt.subplot(2, 2, 1)
    plt.plot(aerosol_files, xa1s)
    plt.plot(aerosol_files, xb1s)
    plt.plot(aerosol_files, xc1s)
    plt.legend(['xa', 'xb', 'xc'])
    plt.axis([0, 6, 0, 0.6])
    plt.ylabel(f'Blue band Correction Coefficients', fontdict=font)
    plt.xticks(aerosol_files, predefined_aerosol_files, rotation=70)
    fig2 = plt.subplot(2, 2, 2)
    plt.plot(aerosol_files, xa2s)
    plt.plot(aerosol_files, xb2s)
    plt.plot(aerosol_files, xc2s)
    plt.legend(['xa', 'xb', 'xc'])
    plt.axis([0, 6, 0, 0.6])
    plt.ylabel(f'Green band Correction Coefficients', fontdict=font)
    plt.xticks(aerosol_files, predefined_aerosol_files, rotation=70)
    fig3 = plt.subplot(2, 2, 3)
    plt.plot(aerosol_files, xa3s)
    plt.plot(aerosol_files, xb3s)
    plt.plot(aerosol_files, xc3s)
    plt.legend(['xa', 'xb', 'xc'])
    plt.axis([0, 6, 0, 0.6])
    plt.ylabel(f'Red band Correction Coefficients', fontdict=font)
    plt.xticks(aerosol_files, predefined_aerosol_files, rotation=70)
    fig4 = plt.subplot(2, 2, 4)
    plt.plot(aerosol_files, xa4s)
    plt.plot(aerosol_files, xb4s)
    plt.plot(aerosol_files, xc4s)
    plt.legend(['xa', 'xb', 'xc'])
    plt.axis([0, 6, 0, 0.6])
    plt.ylabel(f'NIR band Correction Coefficients', fontdict=font)
    plt.xticks(aerosol_files, predefined_aerosol_files, rotation=70)

    plt.tight_layout()
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)

    plt.savefig(f'Sensitivity analysis of Aerosol profiles.png', dpi=300)
    plt.show()

if __name__ == '__main__1':
    """
    参数敏感性分析——AOD
    """
    # 蓝色波段
    outputs1 = main(channel=None,
                    wavelength=(0.450, 0.520),
                    spectral_filter=GF1_WFV_Blue_SRF,
                    aerosol_profile='Continental',
                    build_type='aod_sensitivity_analysis',
                    atmo_corr='LambertianFromReflectance',
                    radiance=30,
                    sensor_name='GF1-WFV1')
    # 绿色波段
    outputs2 = main(channel=None,
                    wavelength=(0.520, 0.590),
                    spectral_filter=GF1_WFV_Green_SRF,
                    aerosol_profile='Continental',
                    build_type='aod_sensitivity_analysis',
                    atmo_corr='LambertianFromReflectance',
                    radiance=30,
                    sensor_name='GF1-WFV1')
    # 红色波段
    outputs3 = main(channel=None,
                    wavelength=(0.630, 0.690),
                    spectral_filter=GF1_WFV_Red_SRF,
                    aerosol_profile='Continental',
                    build_type='aod_sensitivity_analysis',
                    atmo_corr='LambertianFromReflectance',
                    radiance=30,
                    sensor_name='GF1-WFV1')
    # 近红外波段
    outputs4 = main(channel=None,
                    wavelength=(0.770, 0.890),
                    spectral_filter=GF1_WFV_NIR_SRF,
                    aerosol_profile='Continental',
                    build_type='aod_sensitivity_analysis',
                    atmo_corr='LambertianFromReflectance',
                    radiance=30,
                    sensor_name='GF1-WFV1')
    aod = np.arange(0, 3.1, 0.1)
    xa1s = [val[0] for val in outputs1]
    xb1s = [val[1] for val in outputs1]
    xc1s = [val[2] for val in outputs1]
    xa2s = [val[0] for val in outputs2]
    xb2s = [val[1] for val in outputs2]
    xc2s = [val[2] for val in outputs2]
    xa3s = [val[0] for val in outputs3]
    xb3s = [val[1] for val in outputs3]
    xc3s = [val[2] for val in outputs3]
    xa4s = [val[0] for val in outputs4]
    xb4s = [val[1] for val in outputs4]
    xc4s = [val[2] for val in outputs4]

    # 画图
    font = {
        'family': 'serif',
        'color': 'darkred',
        'weight': 'normal',
        'size': 9,
    }
    plt.figure(figsize=(10, 10), dpi=80)
    fig1 = plt.subplot(2, 2, 1)
    plt.plot(aod, xa1s)
    plt.plot(aod, xb1s)
    plt.plot(aod, xc1s)
    plt.legend(['xa', 'xb', 'xc'])
    plt.axis([0, 3.0, 0, 2])
    plt.xlabel(f'AOD', fontdict=font)
    plt.ylabel(f'Blue band Correction Coefficients', fontdict=font)
    fig2 = plt.subplot(2, 2, 2)
    plt.plot(aod, xa2s)
    plt.plot(aod, xb2s)
    plt.plot(aod, xc2s)
    plt.legend(['xa', 'xb', 'xc'])
    plt.axis([0, 3.0, 0, 2])
    plt.xlabel(f'AOD', fontdict=font)
    plt.ylabel(f'Green band Correction Coefficients', fontdict=font)
    fig3 = plt.subplot(2, 2, 3)
    plt.plot(aod, xa3s)
    plt.plot(aod, xb3s)
    plt.plot(aod, xc3s)
    plt.legend(['xa', 'xb', 'xc'])
    plt.axis([0, 3.0, 0, 2])
    plt.xlabel(f'AOD', fontdict=font)
    plt.ylabel(f'Red band Correction Coefficients', fontdict=font)
    fig4 = plt.subplot(2, 2, 4)
    plt.plot(aod, xa4s)
    plt.plot(aod, xb4s)
    plt.plot(aod, xc4s)
    plt.legend(['xa', 'xb', 'xc'])
    plt.axis([0, 3.0, 0, 2])
    plt.xlabel(f'AOD', fontdict=font)
    plt.ylabel(f'NIR band Correction Coefficients', fontdict=font)

    plt.tight_layout()
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)

    plt.savefig(f'Sensitivity analysis of AOD.png', dpi=300)
    plt.show()


if __name__ == '__main__1':
    """
    参数敏感性分析——SZA
    """
    # 蓝色波段
    outputs1 = main(channel=None,
                    wavelength=(0.450, 0.520),
                    spectral_filter=GF1_WFV_Blue_SRF,
                    aerosol_profile='Continental',
                    build_type='sza_sensitivity_analysis',
                    atmo_corr='LambertianFromReflectance',
                    radiance=30,
                    sensor_name='GF1-WFV1')
    # 绿色波段
    outputs2 = main(channel=None,
                    wavelength=(0.520, 0.590),
                    spectral_filter=GF1_WFV_Green_SRF,
                    aerosol_profile='Continental',
                    build_type='sza_sensitivity_analysis',
                    atmo_corr='LambertianFromReflectance',
                    radiance=30,
                    sensor_name='GF1-WFV1')
    # 红色波段
    outputs3 = main(channel=None,
                    wavelength=(0.630, 0.690),
                    spectral_filter=GF1_WFV_Red_SRF,
                    aerosol_profile='Continental',
                    build_type='sza_sensitivity_analysis',
                    atmo_corr='LambertianFromReflectance',
                    radiance=30,
                    sensor_name='GF1-WFV1')
    # 近红外波段
    outputs4 = main(channel=None,
                    wavelength=(0.770, 0.890),
                    spectral_filter=GF1_WFV_NIR_SRF,
                    aerosol_profile='Continental',
                    build_type='sza_sensitivity_analysis',
                    atmo_corr='LambertianFromReflectance',
                    radiance=30,
                    sensor_name='GF1-WFV1')
    sza = np.arange(0, 81, 1)
    xa1s = [val[0] for val in outputs1]
    xb1s = [val[1] for val in outputs1]
    xc1s = [val[2] for val in outputs1]
    xa2s = [val[0] for val in outputs2]
    xb2s = [val[1] for val in outputs2]
    xc2s = [val[2] for val in outputs2]
    xa3s = [val[0] for val in outputs3]
    xb3s = [val[1] for val in outputs3]
    xc3s = [val[2] for val in outputs3]
    xa4s = [val[0] for val in outputs4]
    xb4s = [val[1] for val in outputs4]
    xc4s = [val[2] for val in outputs4]

    # 画图
    font = {
        'family': 'serif',
        'color': 'darkred',
        'weight': 'normal',
        'size': 9,
    }
    plt.figure(figsize=(10, 10), dpi=80)
    fig1 = plt.subplot(2, 2, 1)
    plt.plot(sza, xa1s)
    plt.plot(sza, xb1s)
    plt.plot(sza, xc1s)
    plt.legend(['xa', 'xb', 'xc'])
    plt.axis([0, 80, 0, 1])
    plt.xlabel(f'SZA', fontdict=font)
    plt.ylabel(f'Blue band Correction Coefficients', fontdict=font)
    fig2 = plt.subplot(2, 2, 2)
    plt.plot(sza, xa2s)
    plt.plot(sza, xb2s)
    plt.plot(sza, xc2s)
    plt.legend(['xa', 'xb', 'xc'])
    plt.axis([0, 80, 0, 1])
    plt.xlabel(f'SZA', fontdict=font)
    plt.ylabel(f'Green band Correction Coefficients', fontdict=font)
    fig3 = plt.subplot(2, 2, 3)
    plt.plot(sza, xa3s)
    plt.plot(sza, xb3s)
    plt.plot(sza, xc3s)
    plt.legend(['xa', 'xb', 'xc'])
    plt.axis([0, 80, 0, 1])
    plt.xlabel(f'SZA', fontdict=font)
    plt.ylabel(f'Red band Correction Coefficients', fontdict=font)
    fig4 = plt.subplot(2, 2, 4)
    plt.plot(sza, xa4s)
    plt.plot(sza, xb4s)
    plt.plot(sza, xc4s)
    plt.legend(['xa', 'xb', 'xc'])
    plt.axis([0, 80, 0, 1])
    plt.xlabel(f'SZA', fontdict=font)
    plt.ylabel(f'NIR band Correction Coefficients', fontdict=font)

    plt.tight_layout()
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)

    plt.savefig(f'Sensitivity analysis of SZA.png', dpi=300)
    plt.show()


if __name__ == '__main__1':
    """
    参数敏感性分析——VZA
    """
    # 蓝色波段
    outputs1 = main(channel=None,
                    wavelength=(0.450, 0.520),
                    spectral_filter=GF1_WFV_Blue_SRF,
                    aerosol_profile='Continental',
                    build_type='vza_sensitivity_analysis',
                    atmo_corr='LambertianFromReflectance',
                    radiance=30,
                    sensor_name='GF1-WFV1')
    # 绿色波段
    outputs2 = main(channel=None,
                    wavelength=(0.520, 0.590),
                    spectral_filter=GF1_WFV_Green_SRF,
                    aerosol_profile='Continental',
                    build_type='vza_sensitivity_analysis',
                    atmo_corr='LambertianFromReflectance',
                    radiance=30,
                    sensor_name='GF1-WFV1')
    # 红色波段
    outputs3 = main(channel=None,
                    wavelength=(0.630, 0.690),
                    spectral_filter=GF1_WFV_Red_SRF,
                    aerosol_profile='Continental',
                    build_type='vza_sensitivity_analysis',
                    atmo_corr='LambertianFromReflectance',
                    radiance=30,
                    sensor_name='GF1-WFV1')
    # 近红外波段
    outputs4 = main(channel=None,
                    wavelength=(0.770, 0.890),
                    spectral_filter=GF1_WFV_NIR_SRF,
                    aerosol_profile='Continental',
                    build_type='vza_sensitivity_analysis',
                    atmo_corr='LambertianFromReflectance',
                    radiance=30,
                    sensor_name='GF1-WFV1')
    vza = np.arange(0, 81, 1)
    xa1s = [val[0] for val in outputs1]
    xb1s = [val[1] for val in outputs1]
    xc1s = [val[2] for val in outputs1]
    xa2s = [val[0] for val in outputs2]
    xb2s = [val[1] for val in outputs2]
    xc2s = [val[2] for val in outputs2]
    xa3s = [val[0] for val in outputs3]
    xb3s = [val[1] for val in outputs3]
    xc3s = [val[2] for val in outputs3]
    xa4s = [val[0] for val in outputs4]
    xb4s = [val[1] for val in outputs4]
    xc4s = [val[2] for val in outputs4]

    # 画图
    font = {
        'family': 'serif',
        'color': 'darkred',
        'weight': 'normal',
        'size': 9,
    }
    plt.figure(figsize=(10, 10), dpi=80)
    fig1 = plt.subplot(2, 2, 1)
    plt.plot(vza, xa1s)
    plt.plot(vza, xb1s)
    plt.plot(vza, xc1s)
    plt.legend(['xa', 'xb', 'xc'])
    plt.axis([0, 80, 0, 1])
    plt.xlabel(f'VZA', fontdict=font)
    plt.ylabel(f'Blue band Correction Coefficients', fontdict=font)
    fig2 = plt.subplot(2, 2, 2)
    plt.plot(vza, xa2s)
    plt.plot(vza, xb2s)
    plt.plot(vza, xc2s)
    plt.legend(['xa', 'xb', 'xc'])
    plt.axis([0, 80, 0, 1])
    plt.xlabel(f'VZA', fontdict=font)
    plt.ylabel(f'Green band Correction Coefficients', fontdict=font)
    fig3 = plt.subplot(2, 2, 3)
    plt.plot(vza, xa3s)
    plt.plot(vza, xb3s)
    plt.plot(vza, xc3s)
    plt.legend(['xa', 'xb', 'xc'])
    plt.axis([0, 80, 0, 1])
    plt.xlabel(f'VZA', fontdict=font)
    plt.ylabel(f'Red band Correction Coefficients', fontdict=font)
    fig4 = plt.subplot(2, 2, 4)
    plt.plot(vza, xa4s)
    plt.plot(vza, xb4s)
    plt.plot(vza, xc4s)
    plt.legend(['xa', 'xb', 'xc'])
    plt.axis([0, 80, 0, 1])
    plt.xlabel(f'VZA', fontdict=font)
    plt.ylabel(f'NIR band Correction Coefficients', fontdict=font)

    plt.tight_layout()
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)

    plt.savefig(f'Sensitivity analysis of VZA.png', dpi=300)
    plt.show()


if __name__ == '__main__1':
    """
    参数敏感性分析——RAA
    """
    # 蓝色波段
    outputs1 = main(channel=None,
                    wavelength=(0.450, 0.520),
                    spectral_filter=GF1_WFV_Blue_SRF,
                    aerosol_profile='Continental',
                    build_type='raa_sensitivity_analysis',
                    atmo_corr='LambertianFromReflectance',
                    radiance=30,
                    sensor_name='GF1-WFV1')
    # 绿色波段
    outputs2 = main(channel=None,
                    wavelength=(0.520, 0.590),
                    spectral_filter=GF1_WFV_Green_SRF,
                    aerosol_profile='Continental',
                    build_type='raa_sensitivity_analysis',
                    atmo_corr='LambertianFromReflectance',
                    radiance=30,
                    sensor_name='GF1-WFV1')
    # 红色波段
    outputs3 = main(channel=None,
                    wavelength=(0.630, 0.690),
                    spectral_filter=GF1_WFV_Red_SRF,
                    aerosol_profile='Continental',
                    build_type='raa_sensitivity_analysis',
                    atmo_corr='LambertianFromReflectance',
                    radiance=30,
                    sensor_name='GF1-WFV1')
    # 近红外波段
    outputs4 = main(channel=None,
                    wavelength=(0.770, 0.890),
                    spectral_filter=GF1_WFV_NIR_SRF,
                    aerosol_profile='Continental',
                    build_type='raa_sensitivity_analysis',
                    atmo_corr='LambertianFromReflectance',
                    radiance=30,
                    sensor_name='GF1-WFV1')
    raa = np.arange(0, 361, 1)
    xa1s = [val[0] for val in outputs1]
    xb1s = [val[1] for val in outputs1]
    xc1s = [val[2] for val in outputs1]
    xa2s = [val[0] for val in outputs2]
    xb2s = [val[1] for val in outputs2]
    xc2s = [val[2] for val in outputs2]
    xa3s = [val[0] for val in outputs3]
    xb3s = [val[1] for val in outputs3]
    xc3s = [val[2] for val in outputs3]
    xa4s = [val[0] for val in outputs4]
    xb4s = [val[1] for val in outputs4]
    xc4s = [val[2] for val in outputs4]

    # 画图
    font = {
        'family': 'serif',
        'color': 'darkred',
        'weight': 'normal',
        'size': 9,
    }
    plt.figure(figsize=(10, 10), dpi=80)
    fig1 = plt.subplot(2, 2, 1)
    plt.plot(raa, xa1s)
    plt.plot(raa, xb1s)
    plt.plot(raa, xc1s)
    plt.legend(['xa', 'xb', 'xc'])
    plt.axis([0, 360, 0, 1])
    plt.xlabel(f'RAA', fontdict=font)
    plt.ylabel(f'Blue band Correction Coefficients', fontdict=font)
    fig2 = plt.subplot(2, 2, 2)
    plt.plot(raa, xa2s)
    plt.plot(raa, xb2s)
    plt.plot(raa, xc2s)
    plt.legend(['xa', 'xb', 'xc'])
    plt.axis([0, 360, 0, 1])
    plt.xlabel(f'RAA', fontdict=font)
    plt.ylabel(f'Green band Correction Coefficients', fontdict=font)
    fig3 = plt.subplot(2, 2, 3)
    plt.plot(raa, xa3s)
    plt.plot(raa, xb3s)
    plt.plot(raa, xc3s)
    plt.legend(['xa', 'xb', 'xc'])
    plt.axis([0, 360, 0, 1])
    plt.xlabel(f'RAA', fontdict=font)
    plt.ylabel(f'Red band Correction Coefficients', fontdict=font)
    fig4 = plt.subplot(2, 2, 4)
    plt.plot(raa, xa4s)
    plt.plot(raa, xb4s)
    plt.plot(raa, xc4s)
    plt.legend(['xa', 'xb', 'xc'])
    plt.axis([0, 360, 0, 1])
    plt.xlabel(f'RAA', fontdict=font)
    plt.ylabel(f'NIR band Correction Coefficients', fontdict=font)

    plt.tight_layout()
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)

    plt.savefig(f'Sensitivity analysis of RAA.png', dpi=300)
    plt.show()


if __name__ == '__main__1':
    """
    参数敏感性分析——海拔
    """
    # 蓝色波段
    outputs1 = main(channel=None,
                    wavelength=(0.450, 0.520),
                    spectral_filter=GF1_WFV_Blue_SRF,
                    aerosol_profile='Continental',
                    build_type='alt_sensitivity_analysis',
                    atmo_corr='LambertianFromReflectance',
                    radiance=30,
                    sensor_name='GF1-WFV1')
    # 绿色波段
    outputs2 = main(channel=None,
                    wavelength=(0.520, 0.590),
                    spectral_filter=GF1_WFV_Green_SRF,
                    aerosol_profile='Continental',
                    build_type='alt_sensitivity_analysis',
                    atmo_corr='LambertianFromReflectance',
                    radiance=30,
                    sensor_name='GF1-WFV1')
    # 红色波段
    outputs3 = main(channel=None,
                    wavelength=(0.630, 0.690),
                    spectral_filter=GF1_WFV_Red_SRF,
                    aerosol_profile='Continental',
                    build_type='alt_sensitivity_analysis',
                    atmo_corr='LambertianFromReflectance',
                    radiance=30,
                    sensor_name='GF1-WFV1')
    # 近红外波段
    outputs4 = main(channel=None,
                    wavelength=(0.770, 0.890),
                    spectral_filter=GF1_WFV_NIR_SRF,
                    aerosol_profile='Continental',
                    build_type='alt_sensitivity_analysis',
                    atmo_corr='LambertianFromReflectance',
                    radiance=30,
                    sensor_name='GF1-WFV1')
    alt = np.arange(0, 8.00, 0.25)
    xa1s = [val[0] for val in outputs1]
    xb1s = [val[1] for val in outputs1]
    xc1s = [val[2] for val in outputs1]
    xa2s = [val[0] for val in outputs2]
    xb2s = [val[1] for val in outputs2]
    xc2s = [val[2] for val in outputs2]
    xa3s = [val[0] for val in outputs3]
    xb3s = [val[1] for val in outputs3]
    xc3s = [val[2] for val in outputs3]
    xa4s = [val[0] for val in outputs4]
    xb4s = [val[1] for val in outputs4]
    xc4s = [val[2] for val in outputs4]

    # 画图
    font = {
        'family': 'serif',
        'color': 'darkred',
        'weight': 'normal',
        'size': 9,
    }
    plt.figure(figsize=(10, 10), dpi=80)
    fig1 = plt.subplot(2, 2, 1)
    plt.plot(alt, xa1s)
    plt.plot(alt, xb1s)
    plt.plot(alt, xc1s)
    plt.legend(['xa', 'xb', 'xc'])
    plt.axis([0, 7.75, 0, 1])
    plt.xlabel(f'Altitude', fontdict=font)
    plt.ylabel(f'Blue band Correction Coefficients', fontdict=font)
    fig2 = plt.subplot(2, 2, 2)
    plt.plot(alt, xa2s)
    plt.plot(alt, xb2s)
    plt.plot(alt, xc2s)
    plt.legend(['xa', 'xb', 'xc'])
    plt.axis([0, 7.75, 0, 1])
    plt.xlabel(f'Altitude', fontdict=font)
    plt.ylabel(f'Green band Correction Coefficients', fontdict=font)
    fig3 = plt.subplot(2, 2, 3)
    plt.plot(alt, xa3s)
    plt.plot(alt, xb3s)
    plt.plot(alt, xc3s)
    plt.legend(['xa', 'xb', 'xc'])
    plt.axis([0, 7.75, 0, 1])
    plt.xlabel(f'Altitude', fontdict=font)
    plt.ylabel(f'Red band Correction Coefficients', fontdict=font)
    fig4 = plt.subplot(2, 2, 4)
    plt.plot(alt, xa4s)
    plt.plot(alt, xb4s)
    plt.plot(alt, xc4s)
    plt.legend(['xa', 'xb', 'xc'])
    plt.axis([0, 7.75, 0, 1])
    plt.xlabel(f'Altitude', fontdict=font)
    plt.ylabel(f'NIR band Correction Coefficients', fontdict=font)

    plt.tight_layout()
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)

    plt.savefig(f'Sensitivity analysis of Altitude.png', dpi=300)
    plt.show()


if __name__ == '__main__1':
    """
    参数敏感性分析——影像获取日期
    """
    xa1s = []
    xb1s = []
    xc1s = []
    xa2s = []
    xb2s = []
    xc2s = []
    xa3s = []
    xb3s = []
    xc3s = []
    xa4s = []
    xb4s = []
    xc4s = []
    for i in range(1, 13):
        if i in [1, 3, 5, 7, 8, 10, 12]:
            for j in range(1, 32):
                # 蓝色波段
                outputs1 = main(channel=None,
                                wavelength=(0.450, 0.520),
                                spectral_filter=GF1_WFV_Blue_SRF,
                                aerosol_profile='Continental',
                                build_type='fixed',
                                atmo_corr='LambertianFromReflectance',
                                radiance=30,
                                month=i,
                                day=j,
                                sensor_name='GF1-WFV1')
                # 绿色波段
                outputs2 = main(channel=None,
                                wavelength=(0.520, 0.590),
                                spectral_filter=GF1_WFV_Green_SRF,
                                aerosol_profile='Continental',
                                build_type='fixed',
                                atmo_corr='LambertianFromReflectance',
                                radiance=30,
                                month=i,
                                day=j,
                                sensor_name='GF1-WFV1')
                # 红色波段
                outputs3 = main(channel=None,
                                wavelength=(0.630, 0.690),
                                spectral_filter=GF1_WFV_Red_SRF,
                                aerosol_profile='Continental',
                                build_type='fixed',
                                atmo_corr='LambertianFromReflectance',
                                radiance=30,
                                month=i,
                                day=j,
                                sensor_name='GF1-WFV1')
                # 近红外波段
                outputs4 = main(channel=None,
                                wavelength=(0.770, 0.890),
                                spectral_filter=GF1_WFV_NIR_SRF,
                                aerosol_profile='Continental',
                                build_type='fixed',
                                atmo_corr='LambertianFromReflectance',
                                radiance=30,
                                month=i,
                                day=j,
                                sensor_name='GF1-WFV1')
                xa1s.append(outputs1[0][0])
                xb1s.append(outputs1[0][1])
                xc1s.append(outputs1[0][2])
                xa2s.append(outputs2[0][0])
                xb2s.append(outputs2[0][1])
                xc2s.append(outputs2[0][2])
                xa3s.append(outputs3[0][0])
                xb3s.append(outputs3[0][1])
                xc3s.append(outputs3[0][2])
                xa4s.append(outputs4[0][0])
                xb4s.append(outputs4[0][1])
                xc4s.append(outputs4[0][2])
        elif i == 2:
            for j in range(1, 29):
                # 蓝色波段
                outputs1 = main(channel=None,
                                wavelength=(0.450, 0.520),
                                spectral_filter=GF1_WFV_Blue_SRF,
                                aerosol_profile='Continental',
                                build_type='fixed',
                                atmo_corr='LambertianFromReflectance',
                                radiance=30,
                                month=i,
                                day=j,
                                sensor_name='GF1-WFV1')
                # 绿色波段
                outputs2 = main(channel=None,
                                wavelength=(0.520, 0.590),
                                spectral_filter=GF1_WFV_Green_SRF,
                                aerosol_profile='Continental',
                                build_type='fixed',
                                atmo_corr='LambertianFromReflectance',
                                radiance=30,
                                month=i,
                                day=j,
                                sensor_name='GF1-WFV1')
                # 红色波段
                outputs3 = main(channel=None,
                                wavelength=(0.630, 0.690),
                                spectral_filter=GF1_WFV_Red_SRF,
                                aerosol_profile='Continental',
                                build_type='fixed',
                                atmo_corr='LambertianFromReflectance',
                                radiance=30,
                                month=i,
                                day=j,
                                sensor_name='GF1-WFV1')
                # 近红外波段
                outputs4 = main(channel=None,
                                wavelength=(0.770, 0.890),
                                spectral_filter=GF1_WFV_NIR_SRF,
                                aerosol_profile='Continental',
                                build_type='fixed',
                                atmo_corr='LambertianFromReflectance',
                                radiance=30,
                                month=i,
                                day=j,
                                sensor_name='GF1-WFV1')
                xa1s.append(outputs1[0][0])
                xb1s.append(outputs1[0][1])
                xc1s.append(outputs1[0][2])
                xa2s.append(outputs2[0][0])
                xb2s.append(outputs2[0][1])
                xc2s.append(outputs2[0][2])
                xa3s.append(outputs3[0][0])
                xb3s.append(outputs3[0][1])
                xc3s.append(outputs3[0][2])
                xa4s.append(outputs4[0][0])
                xb4s.append(outputs4[0][1])
                xc4s.append(outputs4[0][2])
        else:
            for j in range(1, 31):
                # 蓝色波段
                outputs1 = main(channel=None,
                                wavelength=(0.450, 0.520),
                                spectral_filter=GF1_WFV_Blue_SRF,
                                aerosol_profile='Continental',
                                build_type='fixed',
                                atmo_corr='LambertianFromReflectance',
                                radiance=30,
                                month=i,
                                day=j,
                                sensor_name='GF1-WFV1')
                # 绿色波段
                outputs2 = main(channel=None,
                                wavelength=(0.520, 0.590),
                                spectral_filter=GF1_WFV_Green_SRF,
                                aerosol_profile='Continental',
                                build_type='fixed',
                                atmo_corr='LambertianFromReflectance',
                                radiance=30,
                                month=i,
                                day=j,
                                sensor_name='GF1-WFV1')
                # 红色波段
                outputs3 = main(channel=None,
                                wavelength=(0.630, 0.690),
                                spectral_filter=GF1_WFV_Red_SRF,
                                aerosol_profile='Continental',
                                build_type='fixed',
                                atmo_corr='LambertianFromReflectance',
                                radiance=30,
                                month=i,
                                day=j,
                                sensor_name='GF1-WFV1')
                # 近红外波段
                outputs4 = main(channel=None,
                                wavelength=(0.770, 0.890),
                                spectral_filter=GF1_WFV_NIR_SRF,
                                aerosol_profile='Continental',
                                build_type='fixed',
                                atmo_corr='LambertianFromReflectance',
                                radiance=30,
                                month=i,
                                day=j,
                                sensor_name='GF1-WFV1')
                xa1s.append(outputs1[0][0])
                xb1s.append(outputs1[0][1])
                xc1s.append(outputs1[0][2])
                xa2s.append(outputs2[0][0])
                xb2s.append(outputs2[0][1])
                xc2s.append(outputs2[0][2])
                xa3s.append(outputs3[0][0])
                xb3s.append(outputs3[0][1])
                xc3s.append(outputs3[0][2])
                xa4s.append(outputs4[0][0])
                xb4s.append(outputs4[0][1])
                xc4s.append(outputs4[0][2])

    # 画图
    date = np.arange(1, 366, 1)
    font = {
        'family': 'serif',
        'color': 'darkred',
        'weight': 'normal',
        'size': 9,
    }
    plt.figure(figsize=(10, 10), dpi=80)
    fig1 = plt.subplot(2, 2, 1)
    plt.plot(date, xa1s)
    plt.plot(date, xb1s)
    plt.plot(date, xc1s)
    plt.legend(['xa', 'xb', 'xc'])
    plt.axis([1, 365, 0, 1])
    plt.xlabel(f'Date', fontdict=font)
    plt.ylabel(f'Blue band Correction Coefficients', fontdict=font)
    fig2 = plt.subplot(2, 2, 2)
    plt.plot(date, xa2s)
    plt.plot(date, xb2s)
    plt.plot(date, xc2s)
    plt.legend(['xa', 'xb', 'xc'])
    plt.axis([1, 365, 0, 1])
    plt.xlabel(f'Date', fontdict=font)
    plt.ylabel(f'Green band Correction Coefficients', fontdict=font)
    fig3 = plt.subplot(2, 2, 3)
    plt.plot(date, xa3s)
    plt.plot(date, xb3s)
    plt.plot(date, xc3s)
    plt.legend(['xa', 'xb', 'xc'])
    plt.axis([1, 365, 0, 1])
    plt.xlabel(f'Date', fontdict=font)
    plt.ylabel(f'Red band Correction Coefficients', fontdict=font)
    fig4 = plt.subplot(2, 2, 4)
    plt.plot(date, xa4s)
    plt.plot(date, xb4s)
    plt.plot(date, xc4s)
    plt.legend(['xa', 'xb', 'xc'])
    plt.axis([1, 365, 0, 1])
    plt.xlabel(f'Date', fontdict=font)
    plt.ylabel(f'NIR band Correction Coefficients', fontdict=font)

    plt.tight_layout()
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)

    plt.savefig(f'Sensitivity analysis of Date.png', dpi=300)
    plt.show()


if __name__ == '__main__1':
    """
    参数敏感性——Ground Reflectance
    """
    outputs1 = main(channel=None,
                    wavelength=(0.450, 0.520),
                    spectral_filter=GF1_WFV_Blue_SRF,
                    aerosol_profile='Continental',
                    build_type='fixed',
                    atmo_corr='LambertianFromReflectance',
                    radiance=30,
                    sensor_name='GF1-WFV1')


if __name__ == '__main__1':
    """
    SZA大角度不同步长设置精度
    """
    # 蓝色波段
    outputs1 = main(channel=None,
                    wavelength=(0.450, 0.520),
                    spectral_filter=GF1_WFV_Blue_SRF,
                    aerosol_profile='Continental',
                    build_type='sza_small_angle_step10',
                    atmo_corr='LambertianFromRadiance',
                    radiance=30,
                    sensor_name='GF1-WFV1')
    # 绿色波段
    outputs2 = main(channel=None,
                    wavelength=(0.520, 0.590),
                    spectral_filter=GF1_WFV_Green_SRF,
                    aerosol_profile='Continental',
                    build_type='sza_small_angle_step10',
                    atmo_corr='LambertianFromRadiance',
                    radiance=30,
                    sensor_name='GF1-WFV1')
    # 红色波段
    outputs3 = main(channel=None,
                    wavelength=(0.630, 0.690),
                    spectral_filter=GF1_WFV_Red_SRF,
                    aerosol_profile='Continental',
                    build_type='sza_small_angle_step10',
                    atmo_corr='LambertianFromRadiance',
                    radiance=30,
                    sensor_name='GF1-WFV1')
    # 近红外波段
    outputs4 = main(channel=None,
                    wavelength=(0.770, 0.890),
                    spectral_filter=GF1_WFV_NIR_SRF,
                    aerosol_profile='Continental',
                    build_type='sza_small_angle_step10',
                    atmo_corr='LambertianFromRadiance',
                    radiance=30,
                    sensor_name='GF1-WFV1')


if __name__ == '__main__1':
    """
    验证
    """
    sensor_radiance = 160

    # 蓝色波段
    outputs1 = main(channel=None,
                    wavelength=(0.450, 0.520),
                    spectral_filter=GF1_WFV_Blue_SRF,
                    aerosol_profile='Continental',
                    build_type='big_aod_validate',
                    atmo_corr='LambertianFromRadiance',
                    radiance=30,
                    sensor_name='GF1-WFV1')
    y1 = outputs1[0][0] * sensor_radiance - outputs1[0][1]
    sr1 = y1 / (1 + outputs1[0][2] * y1)
    print(outputs1)
    print(sr1)
    # 绿色波段
    outputs2 = main(channel=None,
                    wavelength=(0.520, 0.590),
                    spectral_filter=GF1_WFV_Green_SRF,
                    aerosol_profile='Continental',
                    build_type='big_aod_validate',
                    atmo_corr='LambertianFromRadiance',
                    radiance=30,
                    sensor_name='GF1-WFV1')
    y2 = outputs2[0][0] * sensor_radiance - outputs2[0][1]
    sr2 = y2 / (1 + outputs2[0][2] * y2)
    print(outputs2)
    print(sr2)
    # 红色波段
    outputs3 = main(channel=None,
                    wavelength=(0.630, 0.690),
                    spectral_filter=GF1_WFV_Red_SRF,
                    aerosol_profile='Continental',
                    build_type='big_aod_validate',
                    atmo_corr='LambertianFromRadiance',
                    radiance=30,
                    sensor_name='GF1-WFV1')
    y3 = outputs3[0][0] * sensor_radiance - outputs3[0][1]
    sr3 = y3 / (1 + outputs3[0][2] * y3)
    print(outputs3)
    print(sr3)
    # 近红外波段
    outputs4 = main(channel=None,
                    wavelength=(0.770, 0.890),
                    spectral_filter=GF1_WFV_NIR_SRF,
                    aerosol_profile='Continental',
                    build_type='big_aod_validate',
                    atmo_corr='LambertianFromRadiance',
                    radiance=30,
                    sensor_name='GF1-WFV1')
    y4 = outputs4[0][0] * sensor_radiance - outputs4[0][1]
    sr4 = y4 / (1 + outputs4[0][2] * y4)
    print(outputs4)
    print(sr4)


if __name__ == '__main__1':
    """
    VZA大角度不同步长设置精度
    """
    # 蓝色波段
    outputs1 = main(channel=None,
                    wavelength=(0.450, 0.520),
                    spectral_filter=GF1_WFV_Blue_SRF,
                    aerosol_profile='Continental',
                    build_type='vza_big_angle_step2',
                    atmo_corr='LambertianFromRadiance',
                    radiance=30,
                    sensor_name='GF1-WFV1')
    # 绿色波段
    outputs2 = main(channel=None,
                    wavelength=(0.520, 0.590),
                    spectral_filter=GF1_WFV_Green_SRF,
                    aerosol_profile='Continental',
                    build_type='vza_big_angle_step2',
                    atmo_corr='LambertianFromRadiance',
                    radiance=30,
                    sensor_name='GF1-WFV1')
    # 红色波段
    outputs3 = main(channel=None,
                    wavelength=(0.630, 0.690),
                    spectral_filter=GF1_WFV_Red_SRF,
                    aerosol_profile='Continental',
                    build_type='vza_big_angle_step2',
                    atmo_corr='LambertianFromRadiance',
                    radiance=30,
                    sensor_name='GF1-WFV1')
    # 近红外波段
    outputs4 = main(channel=None,
                    wavelength=(0.770, 0.890),
                    spectral_filter=GF1_WFV_NIR_SRF,
                    aerosol_profile='Continental',
                    build_type='vza_big_angle_step2',
                    atmo_corr='LambertianFromRadiance',
                    radiance=30,
                    sensor_name='GF1-WFV1')


if __name__ == '__main__':
    """
    AOD不同步长设置精度
    """
    # 蓝色波段
    outputs1 = main(channel=None,
                    wavelength=(0.450, 0.520),
                    spectral_filter=GF1_WFV_Blue_SRF,
                    aerosol_profile='Continental',
                    build_type='big_aod_step005',
                    atmo_corr='LambertianFromRadiance',
                    radiance=30,
                    sensor_name='GF1-WFV1')
    # 绿色波段
    outputs2 = main(channel=None,
                    wavelength=(0.520, 0.590),
                    spectral_filter=GF1_WFV_Green_SRF,
                    aerosol_profile='Continental',
                    build_type='big_aod_step005',
                    atmo_corr='LambertianFromRadiance',
                    radiance=30,
                    sensor_name='GF1-WFV1')
    # 红色波段
    outputs3 = main(channel=None,
                    wavelength=(0.630, 0.690),
                    spectral_filter=GF1_WFV_Red_SRF,
                    aerosol_profile='Continental',
                    build_type='big_aod_step005',
                    atmo_corr='LambertianFromRadiance',
                    radiance=30,
                    sensor_name='GF1-WFV1')
    # 近红外波段
    outputs4 = main(channel=None,
                    wavelength=(0.770, 0.890),
                    spectral_filter=GF1_WFV_NIR_SRF,
                    aerosol_profile='Continental',
                    build_type='big_aod_step005',
                    atmo_corr='LambertianFromRadiance',
                    radiance=30,
                    sensor_name='GF1-WFV1')
