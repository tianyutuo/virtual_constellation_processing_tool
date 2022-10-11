# -*- coding: utf-8 -*-
"""
读取查找表文件（.lut）并依据输入的参数进行插值
"""
import os
import pickle
import time
import re
import sys
import glob
from scipy.interpolate import LinearNDInterpolator

from create_LUT import permutate_invars


def create_interpolator(filename):
    # 读取查找表
    LUT = pickle.load(open(filename, 'rb'))

    # 读取设置的各个参数（SZA、VZA、RAA、AOD）
    inputs = permutate_invars(LUT['config']['invars'])
    # 读取输出，即大气校正系数
    outputs = LUT['outputs']

    # 线性插值
    t = time.time()
    interpolator = LinearNDInterpolator(inputs, outputs)
    print('插值耗时{:.2f}s'.format(time.time() - t))

    # 插值结果检验
    i = 0
    true = (outputs[i][0], outputs[i][1])
    interp = interpolator(inputs[i][0], inputs[i][1], inputs[i][2], inputs[i][3],
                          inputs[i][4], inputs[i][5], inputs[i][6])
    print('true   = {0[0]:.2f} {0[1]:.2f}'.format(true))
    print('interp = {0[0]:.2f} {0[1]:.2f}'.format(interp))

    return interpolator


def main(LUT_path):
    try:
        os.chdir(LUT_path)
    except:
        print('无效路径：', LUT_path)
        sys.exit(-1)

    # 创建插值后查找表新路径
    match = re.search('LUTs', LUT_path)
    base_path = LUT_path[0:match.start()]
    end_path = LUT_path[match.end():]
    iLUT_path = base_path + 'iLUTs' + end_path

    # 查找表名称
    fnames = glob.glob('*.lut')
    fnames.sort()
    if len(fnames) == 0:
        print('该路径未找到查找表文件（.lut）：', LUT_path)
        sys.exit(1)

    if not os.path.exists(iLUT_path):
        os.makedirs(iLUT_path)

    # 对查找表文件进行插值
    for fname in fnames:
        fid, ext = os.path.splitext(fname)
        iLUT_filepath = os.path.join(iLUT_path, fid + '.ilut')

        if os.path.isfile(iLUT_filepath):
            print('插值查找表（.ilut）已存在：{}'.format(os.path.basename(iLUT_filepath)))
        else:
            print('开始对查找表进行插值：' + fname)
            interpolator = create_interpolator(fname)

            # 输出插值后查找表
            pickle.dump(interpolator, open(iLUT_filepath, 'wb'))


if __name__ == '__main__1':
    main(r'E:\code\remote_sensing_data_harmonization\atmospheric_correction\LUTs\GF1-WFV1\Continental')


if __name__ == '__main__':
    """
    读取插值后的查找表
    """
    import pickle
    import os

    os.chdir(r'E:\code\remote_sensing_data_harmonization\atmospheric_correction\iLUTs\GF1-WFV1\Continental')

    fpath = 'wavelength_0.77_0.89_f_big_aod_step005.ilut'

    with open(fpath, "rb") as ilut_file:
        iLUT = pickle.load(ilut_file)

    xa, xb, xc = iLUT(6, 6, 60, 2.35, 1, 0.4, 1)

    sensor_radiance = 160

    y = xa * sensor_radiance - xb
    sr = y / (1 + xc * y)

    print(xa, xb, xc)
    print(sr)
