"""
根据传感器光谱响应函数和ASTER光谱库数据计算光谱匹配因子（spectral band adjustment factor，SBAF）
"""


import os
import xlrd
from glob import iglob
import numpy as np
from scipy import interpolate
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


def read_aster_spectral_library(spectral_file):
    """
    读取ASTER光谱库数据
    """
    spectral_curve = []
    with open(spectral_file, 'r', encoding='gb18030', errors='ignore') as f:
        temp = 20   # 记录数据开始的行数
        content = f.readlines()
        for line in content[19:]:    # ASTER光谱库数据前面描述性文字至少有19行
            temp += 1
            if 'Additional information' in line.split(':') or 'Additional Information' in line.split(':'):
                break
        for line in content[temp:]:
            if len(line.split()) >= 2:
                spectral_curve.append([float(line.split()[0]), float(line.split()[1])])

        # 此处确保波长从小到大排列，且响应度与其一一对应
        sorted_spectral_curve = [value for index, value in sorted(enumerate(spectral_curve), key=lambda d: d[1])]
        wave_length = [sorted_spectral_curve[i][0] for i in range(len(sorted_spectral_curve))]
        spectral_response = [sorted_spectral_curve[i][1] for i in range(len(sorted_spectral_curve))]

        return wave_length, spectral_response


def read_SRF(SRF_file, sensor_type):
    """
    根据传感器名称读取其光谱响应函数
    """
    srf_xls = xlrd.open_workbook(SRF_file)

    if sensor_type[:2] == 'S2':
        srf = srf_xls.sheet_by_name(sensor_type)
        srf_blue = srf.col_values(2, start_rowx=101, end_rowx=702)
        srf_green = srf.col_values(3, start_rowx=101, end_rowx=702)
        srf_red = srf.col_values(4, start_rowx=101, end_rowx=702)
        srf_nir = srf.col_values(9, start_rowx=101, end_rowx=702)
    elif sensor_type[:3] == 'OLI':
        # 注意Landsat 8 OLI传感器的光谱响应函数文件记录形式与哨兵2号和高分数据不同，只记录了有响应的波长
        srf_blue = srf_xls.sheet_by_name('Blue').col_values(1, start_rowx=1, end_rowx=None)
        srf_green = srf_xls.sheet_by_name('Green').col_values(1, start_rowx=1, end_rowx=None)
        srf_red = srf_xls.sheet_by_name('Red').col_values(1, start_rowx=1, end_rowx=None)
        srf_nir = srf_xls.sheet_by_name('NIR').col_values(1, start_rowx=1, end_rowx=None)
        srf_blue_wl = srf_xls.sheet_by_name('Blue').col_values(0, start_rowx=1, end_rowx=None)
        srf_green_wl = srf_xls.sheet_by_name('Green').col_values(0, start_rowx=1, end_rowx=None)
        srf_red_wl = srf_xls.sheet_by_name('Red').col_values(0, start_rowx=1, end_rowx=None)
        srf_nir_wl = srf_xls.sheet_by_name('NIR').col_values(0, start_rowx=1, end_rowx=None)
        # 将Landsat 8 OLI传感器的光谱响应函数的波长范围扩充到400-1000nm
        srf_blue = [0.0] * int(srf_blue_wl[0] - 400) + srf_blue + [0.0] * int(1000 - srf_blue_wl[-1])
        srf_green = [0.0] * int(srf_green_wl[0] - 400) + srf_green + [0.0] * int(
            1000 - srf_green_wl[-1])
        srf_red = [0.0] * int(srf_red_wl[0] - 400) + srf_red + [0.0] * int(1000 - srf_red_wl[-1])
        srf_nir = [0.0] * int(srf_nir_wl[0] - 400) + srf_nir + [0.0] * int(1000 - srf_nir_wl[-1])
    elif sensor_type == 'PMS':
        srf = srf_xls.sheet_by_name(sensor_type)
        srf_blue = srf.col_values(2, start_rowx=2, end_rowx=None)
        srf_green = srf.col_values(3, start_rowx=2, end_rowx=None)
        srf_red = srf.col_values(4, start_rowx=2, end_rowx=None)
        srf_nir = srf.col_values(5, start_rowx=2, end_rowx=None)
    elif sensor_type[:3] == 'PMS':
        srf = srf_xls.sheet_by_name(sensor_type)
        srf_blue = srf.col_values(2, start_rowx=1, end_rowx=None)
        srf_green = srf.col_values(3, start_rowx=1, end_rowx=None)
        srf_red = srf.col_values(4, start_rowx=1, end_rowx=None)
        srf_nir = srf.col_values(5, start_rowx=1, end_rowx=None)
    elif sensor_type == 'WFV':
        srf = srf_xls.sheet_by_name(sensor_type)
        srf_blue = srf.col_values(1, start_rowx=2, end_rowx=None)
        srf_green = srf.col_values(2, start_rowx=2, end_rowx=None)
        srf_red = srf.col_values(3, start_rowx=2, end_rowx=None)
        srf_nir = srf.col_values(4, start_rowx=2, end_rowx=None)
    else:
        srf = srf_xls.sheet_by_name(sensor_type)
        srf_blue = srf.col_values(1, start_rowx=1, end_rowx=None)
        srf_green = srf.col_values(2, start_rowx=1, end_rowx=None)
        srf_red = srf.col_values(3, start_rowx=1, end_rowx=None)
        srf_nir = srf.col_values(4, start_rowx=1, end_rowx=None)

    return srf_blue, srf_green, srf_red, srf_nir


def linear_regression(ref_simulated_sr_list, tar_simulated_sr_list):
    reg = LinearRegression(fit_intercept=True)  # 设置无截距
    # 去除NaN数据
    ref_simulated_sr_list = [val for val in ref_simulated_sr_list if val == val]
    tar_simulated_sr_list = [val for val in tar_simulated_sr_list if val == val]
    x_data = np.array(ref_simulated_sr_list).reshape(-1, 1)
    y_data = np.array(tar_simulated_sr_list)
    res = reg.fit(x_data, y_data)
    coef = res.coef_
    intercept = reg.intercept_

    return coef[0], intercept


def calculate_SBAF(spectral_library_dir, ref_SRF_file, ref_sensor_type, tar_SRF_file, tar_sensor_type):
    """
    根据光谱库数据和光谱响应函数模拟地物反射率计算光谱匹配因子
    """
    # 读取光谱响应函数（Spectral Response Function，SRF）
    ref_srf_blue, ref_srf_green, ref_srf_red, ref_srf_nir = read_SRF(ref_SRF_file, ref_sensor_type)
    tar_srf_blue, tar_srf_green, tar_srf_red, tar_srf_nir = read_SRF(tar_SRF_file, tar_sensor_type)

    # 读取ASTER光谱库地物光谱曲线数据，计算模拟的地物地表反射率
    ref_blue_simulated_sr_list = []
    ref_green_simulated_sr_list = []
    ref_red_simulated_sr_list = []
    ref_nir_simulated_sr_list = []
    tar_blue_simulated_sr_list = []
    tar_green_simulated_sr_list = []
    tar_red_simulated_sr_list = []
    tar_nir_simulated_sr_list = []
    spectral_files = sorted(list(iglob(os.path.join(spectral_library_dir, '**', '*spectrum.txt'), recursive=True)))
    for spectral_file in spectral_files:
        wl, sr = read_aster_spectral_library(spectral_file)
        # 线性插值使地物光谱曲线波长间隔为1nm
        new_wl = [i for i in np.arange(0.4, 1.001, 0.001)]
        fun = interpolate.splrep(wl, sr)
        interpolated_sr = interpolate.splev(new_wl, fun)

        # 模拟地物在某一波段的地表反射率
        ref_blue_simulated_sr = np.trapz(ref_srf_blue * interpolated_sr) / np.trapz(ref_srf_blue)
        ref_green_simulated_sr = np.trapz(ref_srf_green * interpolated_sr) / np.trapz(ref_srf_green)
        ref_red_simulated_sr = np.trapz(ref_srf_red * interpolated_sr) / np.trapz(ref_srf_red)
        ref_nir_simulated_sr = np.trapz(ref_srf_nir * interpolated_sr) / np.trapz(ref_srf_nir)

        tar_blue_simulated_sr = np.trapz(tar_srf_blue * interpolated_sr) / np.trapz(tar_srf_blue)
        tar_green_simulated_sr = np.trapz(tar_srf_green * interpolated_sr) / np.trapz(tar_srf_green)
        tar_red_simulated_sr = np.trapz(tar_srf_red * interpolated_sr) / np.trapz(tar_srf_red)
        tar_nir_simulated_sr = np.trapz(tar_srf_nir * interpolated_sr) / np.trapz(tar_srf_nir)

        if 100 > ref_blue_simulated_sr > 0:
            ref_blue_simulated_sr_list.append(ref_blue_simulated_sr)
            ref_green_simulated_sr_list.append(ref_green_simulated_sr)
            ref_red_simulated_sr_list.append(ref_red_simulated_sr)
            ref_nir_simulated_sr_list.append(ref_nir_simulated_sr)
            tar_blue_simulated_sr_list.append(tar_blue_simulated_sr)
            tar_green_simulated_sr_list.append(tar_green_simulated_sr)
            tar_red_simulated_sr_list.append(tar_red_simulated_sr)
            tar_nir_simulated_sr_list.append(tar_nir_simulated_sr)

    # 根据地物模拟反射率计算光谱匹配因子SBAF
    blue_coef, blue_intercept = linear_regression(ref_blue_simulated_sr_list, tar_blue_simulated_sr_list)
    green_coef, green_intercept = linear_regression(ref_green_simulated_sr_list, tar_green_simulated_sr_list)
    red_coef, red_intercept = linear_regression(ref_red_simulated_sr_list, tar_red_simulated_sr_list)
    nir_coef, nir_intercept = linear_regression(ref_nir_simulated_sr_list, tar_nir_simulated_sr_list)

    # 计算RMSE，验证模型精度
    before_blue_rmse = ((np.array(ref_blue_simulated_sr_list) - np.array(tar_blue_simulated_sr_list)) ** 2).sum() / len(
        tar_blue_simulated_sr_list)
    after_blue_rmse = (((np.array(ref_blue_simulated_sr_list) * blue_coef + blue_intercept) - np.array(
        tar_blue_simulated_sr_list)) ** 2).sum() / len(
        tar_blue_simulated_sr_list)
    before_green_rmse = ((np.array(ref_green_simulated_sr_list) - np.array(
        tar_green_simulated_sr_list)) ** 2).sum() / len(
        tar_green_simulated_sr_list)
    after_green_rmse = (((np.array(ref_green_simulated_sr_list) * green_coef + green_intercept) - np.array(
        tar_green_simulated_sr_list)) ** 2).sum() / len(
        tar_green_simulated_sr_list)
    before_red_rmse = ((np.array(ref_red_simulated_sr_list) - np.array(
        tar_red_simulated_sr_list)) ** 2).sum() / len(
        tar_red_simulated_sr_list)
    after_red_rmse = (((np.array(ref_red_simulated_sr_list) * red_coef + red_intercept) - np.array(
        tar_red_simulated_sr_list)) ** 2).sum() / len(
        tar_red_simulated_sr_list)
    before_nir_rmse = ((np.array(ref_nir_simulated_sr_list) - np.array(
        tar_nir_simulated_sr_list)) ** 2).sum() / len(
        tar_nir_simulated_sr_list)
    after_nir_rmse = (((np.array(ref_nir_simulated_sr_list) * nir_coef + nir_intercept) - np.array(
        tar_nir_simulated_sr_list)) ** 2).sum() / len(
        tar_nir_simulated_sr_list)

    print(f'{ref_sensor_type} to {tar_sensor_type} SBAF:')
    print(f'Blue:\t{blue_coef}\t{blue_intercept}\t{after_blue_rmse}')
    print(f'Green:\t{green_coef}\t{green_intercept}\t{after_green_rmse}')
    print(f'Red:\t{red_coef}\t{red_intercept}\t{after_red_rmse}')
    print(f'NIR:\t{nir_coef}\t{nir_intercept}\t{after_nir_rmse}')

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
    plt.figure(figsize=(8, 15), dpi=80)
    fig1 = plt.subplot(4, 2, 1)
    plt.title('Blue', fontdict=title_font)
    plt.scatter(np.array(ref_blue_simulated_sr_list), np.array(tar_blue_simulated_sr_list), c='r', s=0.5)
    plt.xlabel(f'Simulated reflectance of {ref_sensor_type}', fontdict=font)
    plt.ylabel(f'Simulated reflectance of {tar_sensor_type}', fontdict=font)
    plt.text(10, 85, f'RMSE: {round(before_blue_rmse, 5)}', fontdict=font)
    plt.axis([0, 100, 0, 100])
    fig2 = plt.subplot(4, 2, 2)
    plt.title('Blue', fontdict=title_font)
    plt.scatter(np.array(ref_blue_simulated_sr_list) * nir_coef + nir_intercept, np.array(tar_blue_simulated_sr_list), c='r', s=0.5)
    plt.xlabel(f'Corrected simulated reflectance of {ref_sensor_type}', fontdict=font)
    plt.ylabel(f'Simulated reflectance of {tar_sensor_type}', fontdict=font)
    plt.text(10, 85, f'RMSE: {round(after_blue_rmse, 5)}', fontdict=font)
    plt.axis([0, 100, 0, 100])
    fig3 = plt.subplot(4, 2, 3)
    plt.title('Green', fontdict=title_font)
    plt.scatter(np.array(ref_green_simulated_sr_list), np.array(tar_green_simulated_sr_list), c='r', s=0.5)
    plt.xlabel(f'Simulated reflectance of {ref_sensor_type}', fontdict=font)
    plt.ylabel(f'Simulated reflectance of {tar_sensor_type}', fontdict=font)
    plt.text(10, 85, f'RMSE: {round(before_green_rmse, 5)}', fontdict=font)
    plt.axis([0, 100, 0, 100])
    fig4 = plt.subplot(4, 2, 4)
    plt.title('Green', fontdict=title_font)
    plt.scatter(np.array(ref_green_simulated_sr_list) * green_coef - green_intercept, np.array(tar_green_simulated_sr_list), c='r', s=0.5)
    plt.xlabel(f'Corrected simulated reflectance of {ref_sensor_type}', fontdict=font)
    plt.ylabel(f'Simulated reflectance of {tar_sensor_type}', fontdict=font)
    plt.text(10, 85, f'RMSE: {round(after_green_rmse, 5)}', fontdict=font)
    plt.axis([0, 100, 0, 100])
    fig5 = plt.subplot(4, 2, 5)
    plt.title('Red', fontdict=title_font)
    plt.scatter(np.array(ref_red_simulated_sr_list), np.array(tar_red_simulated_sr_list), c='r', s=0.5)
    plt.xlabel(f'Simulated reflectance of {ref_sensor_type}', fontdict=font)
    plt.ylabel(f'Simulated reflectance of {tar_sensor_type}', fontdict=font)
    plt.text(10, 85, f'RMSE: {round(before_red_rmse, 5)}', fontdict=font)
    plt.axis([0, 100, 0, 100])
    fig6 = plt.subplot(4, 2, 6)
    plt.title('Red', fontdict=title_font)
    plt.scatter(np.array(ref_red_simulated_sr_list) * red_coef - red_intercept,
                np.array(tar_red_simulated_sr_list), c='r', s=0.5)
    plt.xlabel(f'Corrected simulated reflectance of {ref_sensor_type}', fontdict=font)
    plt.ylabel(f'Simulated reflectance of {tar_sensor_type}', fontdict=font)
    plt.text(10, 85, f'RMSE: {round(after_red_rmse, 5)}', fontdict=font)
    plt.axis([0, 100, 0, 100])
    fig7 = plt.subplot(4, 2, 7)
    plt.title('NIR', fontdict=title_font)
    plt.scatter(np.array(ref_nir_simulated_sr_list), np.array(tar_nir_simulated_sr_list), c='r', s=0.5)
    plt.xlabel(f'Simulated reflectance of {ref_sensor_type}', fontdict=font)
    plt.ylabel(f'Simulated reflectance of {tar_sensor_type}', fontdict=font)
    plt.text(10, 85, f'RMSE: {round(before_nir_rmse, 5)}', fontdict=font)
    plt.axis([0, 100, 0, 100])
    fig8 = plt.subplot(4, 2, 8)
    plt.title('NIR', fontdict=title_font)
    plt.scatter(np.array(ref_nir_simulated_sr_list) * nir_coef - nir_intercept,
                np.array(tar_nir_simulated_sr_list), c='r', s=0.5)
    plt.xlabel(f'Corrected simulated reflectance of {ref_sensor_type}', fontdict=font)
    plt.ylabel(f'Simulated reflectance of {tar_sensor_type}', fontdict=font)
    plt.text(10, 85, f'RMSE: {round(after_nir_rmse, 5)}', fontdict=font)
    plt.axis([0, 100, 0, 100])

    plt.tight_layout()
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)

    # plt.savefig(f'{ref_sensor_type}to{tar_sensor_type}.png', dpi=300)
    # plt.show()


if __name__ == '__main__1':
    os.chdir(r'F:\Experiment\Band Conversion\Aster spectral library\Aster2.0\selected\jhu_soil')

    spectral_file = 'jhu.becknic.soil.lunar.highlands.fine.60051.spectrum.txt'
    read_aster_spectral_library(spectral_file)


if __name__ == '__main__':
    os.chdir(r'F:\Experiment\Band Conversion\Spectral Response Function')
    spectral_library_dir = r'F:\Experiment\Band Conversion\Aster spectral library\Aster2.0\selected'
    # ref_srf_file = '高分6号光谱响应函数.xlsx'
    # ref_sensor_type = 'WFV'
    # tar_srf_file = 'S2-SRF_COPE-GSEG-EOPG-TN-15-0007_3.1.xlsx'
    # tar_sensor_type = 'S2A'
    ref_srf_file = 'S2-SRF_COPE-GSEG-EOPG-TN-15-0007_3.1.xlsx'
    ref_sensor_type = 'S2B'
    tar_srf_file = 'Ball_BA_RSR.v1.1-1.xlsx'
    tar_sensor_type = 'OLI'
    calculate_SBAF(spectral_library_dir,
                   ref_srf_file, ref_sensor_type,
                   tar_srf_file, tar_sensor_type)
