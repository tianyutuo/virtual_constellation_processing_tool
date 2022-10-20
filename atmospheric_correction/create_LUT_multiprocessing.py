"""
利用CPU多核，实现多进行计算查找表
"""

import os.path

from Py6S import *
from itertools import product
import sys
import time
from tqdm import *
import pickle
import numpy as np
import json
import math
from glob import iglob
import re
from multiprocessing import *


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
    sza_angle_step10 = {
        'sza': np.arange(0, 90, 10),
        'vza': [5, 10],
        'raa': [30, 60],
        'aod': [0.2, 0.3],
        'h2o': [1, 1.5],
        'o3': [0.4, 0.8],
        'alt': [1, 3]
    }

    fixed = {
        'sza': [50],
        'vza': [48],
        'raa': [48],
        'aod': [0.3],
        'h2o': [1],
        'o3': [0.4],
        'alt': [1]
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
        'sza_angle_step10': sza_angle_step10,
        'fixed': fixed,
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
        print('创建输出路径\n' + outdir + '\n')
        os.makedirs(outdir)
    os.chdir(outdir)

    # 更新设置文件
    config['outdir'] = outdir
    config['filename'] = filename + '.lut'
    config['filepath'] = os.path.join(outdir, filename + '.lut')


def creat_LUT_partly(config, start, end, part_file_path, que):
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
    s.geometry.month = 1
    s.geometry.day = 4
    # 气溶胶模式
    s.aero_profile = AeroProfile.__dict__[config['aerosol_profile']]
    # 将高度设置为星载传感器的级别
    s.altitudes.set_sensor_satellite_level()
    # 下垫面类型
    s.ground_reflectance = GroundReflectance.HomogeneousLambertian(GroundReflectance.GreenVegetation)
    # 大气校正设置
    s.atmos_corr = AtmosCorr.AtmosCorrLambertianFromReflectance(GroundReflectance.GreenVegetation)

    # ---------------------------------------------------------------------------------------------------
    # 依照其他参数设置的组合排列构建查找表
    # ---------------------------------------------------------------------------------------------------
    # 其他参数的组合排列
    perms = config['perms'][start:end]

    outputs = []
    for i in range(len(perms)):
        perm = perms[i]
        # print('{0}: SZA = {1[0]:02}, VZA = {1[1]:02}, RAA = {1[2]:03}, AOT = {1[3]:.2f}, '
        #       'H2O = {1[4]:.2f}, O3 = {1[5]:.1f},  alt = {1[6]:.2f}'.format(config['filename'], perm))
        # 设置角度信息
        s.geometry.solar_z = perm[0]
        s.geometry.view_z = perm[1]
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

        # 输出大气校正参数
        xa = s.outputs.coef_xa
        xb = s.outputs.coef_xb
        xc = s.outputs.coef_xc

        outputs.append((xa, xb, xc))

        # 每循环10次或最后1次，更新1次进度条
        mod = (i + 1) % 10
        if mod == 0 or (i + 1) == len(perms):
            r = 10 if mod == 0 else mod
            que.put(r)

    # 建立查找表
    LUT = {'config': config, 'perms': perms, 'outputs': outputs}
    part_outdir = os.path.dirname(part_file_path)
    if not os.path.exists(part_outdir):
        print('创建输出路径\n' + part_outdir + '\n')
        os.makedirs(part_outdir)
    pickle.dump(LUT, open(part_file_path, 'wb'))

    # print(f"Get {part_file_path}")

    que.put(-1)  # 队列中此进程结束的标志

    return outputs


def main(channel=None, wavelength=None, spectral_filter=None, aerosol_profile=None, build_type=None, sensor_name=None):
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
        if build_type not in ['sza_angle_step10', 'fixed', 'full']:
            print('查找表模式未识别：', build_type)
            sys.exit(1)
    else:
        print('未定义查找表构建模式，将使用test模式')
        build_type = 'test'

    config = {
        'spectrum': spectrum,
        'aerosol_profile': aerosol_profile,
        'build_type': build_type,
        'invars': input_variables(build_type)
    }

    # 设置查找表文件的输出
    IO_handler(config, channel, wavelength, spectral_filter, sensor_name)

    config['perms'] = permutate_invars(config['invars'])

    # ---------------------------------------------------------------------------------------------------
    # 多线程并行构建查找表
    # ---------------------------------------------------------------------------------------------------
    # 记录构建查找表用时
    a = time.time()

    cpu_num = cpu_count()  # 获取cpu的内核数量
    n = len(config['perms'])
    if cpu_num > n:
        process_num = n
    else:
        process_num = cpu_num
    size = math.floor(n / process_num)  # 向下取整

    queues = [Queue() for i in range(process_num)]  # 为每个进程创建一个队列
    finished = [False for i in range(process_num)]  # 用于标识每个进程是否结束

    bar = tqdm(total=n, desc=f'Processing bar')  # 创建总进程的进度条
    processes = []
    for i in range(process_num):
        start_size = size * i
        if i < cpu_num - 1:
            end_size = size * (i + 1)
        else:
            end_size = n

        # 创建线程
        p = Process(target=creat_LUT_partly,
                    name=f'Process{i + 1}',
                    kwargs={'config': config,
                            'start': start_size,
                            'end': end_size,
                            'part_file_path':
                                os.path.join(config['outdir'], 'temp', config['filename'][:-4]+f'_part{i+1}.lut'),
                            'que': queues[i]}
                    )
        # 守护线程
        p.daemon = True

        processes.append(p)

    for proc in processes:
        proc.start()

    while True:
        for i in range(process_num):
            q = queues[i]
            try:
                res = q.get_nowait()  # 从队列中获取数据
                if res == -1:
                    finished[i] = True
                    continue
                bar.update(res)
            except Exception as e:
                continue
        if all(finished):
            break

    for proc in processes:
        proc.join()

    # 合并所有part_files
    part_files = sorted(list(iglob(os.path.join(os.path.join(config['outdir'], 'temp'), '**',
                                                config['filename'][0:-4] + f'_part*.lut'), recursive=True)))
    if len(part_files) != process_num:
        print('合并查找表失败')
        exit(1)
    part_files = sorted(part_files, key=lambda x: int(re.findall(r'part\d+', x)[0][4:]))  # 使part_files按顺序排列
    final_outputs = []
    for part_file in part_files:
        LUT = pickle.load(open(part_file, 'rb'))
        outputs = LUT['outputs']
        final_outputs.extend(outputs)

    final_LUT = {'config': config, 'outputs': final_outputs}
    pickle.dump(final_LUT, open(config['filepath'], 'wb'))

    b = time.time()
    T = b - a
    print('time: {:.1f} secs, {:.1f} mins,{:.1f} hours'.format(T, T / 60, T / 3600))


if __name__ == '__main__':
    # 读取GF1-WFV1四个波段光谱响应函数（此文件只记录了固定的波长范围，如蓝色波段450~520nm）
    radiometricCorrectionParameter_file = r'E:\code\atmospheric_correction\RadiometricCorrectionParameter.json'
    radiometricCorrectionParameter = json.load(open(radiometricCorrectionParameter_file))
    GF1_WFV_Blue_SRF = radiometricCorrectionParameter['Parameter']['GF1']['WFV1']["SRF"]["1"]
    GF1_WFV_Green_SRF = radiometricCorrectionParameter['Parameter']['GF1']['WFV1']["SRF"]["2"]
    GF1_WFV_Red_SRF = radiometricCorrectionParameter['Parameter']['GF1']['WFV1']["SRF"]["3"]
    GF1_WFV_NIR_SRF = radiometricCorrectionParameter['Parameter']['GF1']['WFV1']["SRF"]["4"]
    # 蓝色波段
    main(channel=None,
         wavelength=(0.450, 0.520),
         spectral_filter=GF1_WFV_Blue_SRF,
         aerosol_profile='Continental',
         build_type='sza_angle_step10',
         sensor_name='GF1-WFV1')
