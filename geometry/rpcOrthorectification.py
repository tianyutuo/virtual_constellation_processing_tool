import os
from osgeo import gdal
from osgeo import osr


# 注意：在使用高分数据进行正射校正时，rpb文件与tif必须同名，才能正确识别
def RPCrect(input, output, nodata=0):
    ds = gdal.Open(input)

    for i in range(1, ds.RasterCount + 1):
        # set the nodata value of the band
        ds.GetRasterBand(i).SetNoDataValue(nodata)

    # 根据文件名判断UTM Zone
    isNorth = 1 if os.path.basename(input).split('_')[3][0] == 'N' else 0
    zone = str(int(float(os.path.basename(input).split('_')[2][1:]) / 6) + 31)
    zone = int('326' + zone) if isNorth else int('327' + zone)
    dstSRS = osr.SpatialReference()
    dstSRS.ImportFromEPSG(zone)

    # 根据文件名判断分辨率
    filename_split = os.path.basename(input).split('_')
    satelliteID = filename_split[0]
    sensorID = filename_split[1]
    if satelliteID == 'GF1':
        if sensorID[0:3] == 'PMS':
            res = 8
        else:
            res = 16
    elif satelliteID == 'GF2':
        res = 4
    elif satelliteID == 'GF6':
        if sensorID[0:3] == 'PMS':
            res = 8
        else:
            res = 16
    else:
        print('数据卫星类型有误')
        return 0

    dst = gdal.Warp(output, ds,
                    dstSRS=dstSRS,
                    xRes=res,
                    yRes=res,
                    resampleAlg=gdal.GRIORA_Bilinear,
                    rpc=True,
                    transformerOptions=['../data/GMTED2km.tif'])

    ds = None
    print(f'{os.path.basename(input)} RPC Orthorectification done')


def set_nodata(file, nodata):
    ds = gdal.Open(file)

    for i in range(1, ds.RasterCount + 1):
        # set the nodata value of the band
        ds.GetRasterBand(i).SetNoDataValue(nodata)
        ds.GetRasterBand(i).FlushCache()

    ds = None
    print(f'{os.path.basename(file)} set nodata value {nodata}')


if __name__ == '__main__1':
    """
    对计算得到的角度数据进行RPC正射校正
    """
    from glob import iglob

    product_path = r'F:\Experiment\BRDF Correction\Clear sky\GF1_WFV1_E81.2_N46.3_20220908_L1A0006733833'
    angle_files = sorted(list(iglob(os.path.join(product_path, '**', '*resampled*.tif'), recursive=True)))

    for file in angle_files:
        output = file[:-4] + '_ORT.tif'
        RPCrect(file, output)


if __name__ == '__main__1':
    """
    对ENVI预处理后的高分数据设置no data
    """
    from glob import iglob

    product_path = r'F:\Experiment\BRDF Correction'
    files = sorted(list(iglob(os.path.join(product_path, '**', '*RC_FLAASH_ORT.tiff'), recursive=True)))

    for file in files:
        set_nodata(file, 0)
