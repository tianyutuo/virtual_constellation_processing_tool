from osgeo import gdal, gdal_array
import numpy as np
from typing import Tuple, Union, List
from pathlib import Path


def array2Raster(array: Union[np.ndarray, List], outputPath: str, refImg=None, geoTransform=None, crs=None, gType=None,
                 noDataValue=None, driverName='GTiff'):
    '''
    :param array: np.array(rows * cols) or np.array(rows * cols * band_num) or List[np.array(rows * cols)]
    :param outputPath: 输出路径
    :param refImg: 参考影像，提供geoTransform, crs和nodata, 如果后面提供了对应参数，该参数会被覆盖
    :param geoTransform: geoTransform[0]：左上角像素经度
                        geoTransform[1]：影像宽度方向上的分辨率(经度范围/像素个数)
                        geoTransform[2]：旋转, 0表示上面为北方
                        geoTransform[3]：左上角像素纬度
                        geoTransform[4]：旋转, 0表示上面为北方
                        geoTransform[5]：影像宽度方向上的分辨率(纬度范围/像素个数)
    :param crs: 坐标系
    :param gType: 像元类型，默认与array相同
    :param noDataValue:
    :param driverName: gdal driver name
    :return:
    '''

    if isinstance(array, list):
        array = np.array(array).transpose(1, 2, 0)

    refDs = None
    refTransform = None
    refCrs = None
    refNoDataValue = None

    if refImg:
        if isinstance(refImg, str) and Path(refImg).is_file():
            refDs = gdal.Open(refImg)
        elif isinstance(refImg, gdal.Dataset):
            refDs = refImg

    if refDs:
        refTransform = refDs.GetGeoTransform()
        refCrs = refDs.GetProjection()
        refNoDataValue = refDs.GetRasterBand(1).GetNoDataValue()

    if not geoTransform:
        geoTransform = refTransform

    if not crs:
        crs = refCrs

    if not gType:
        gType = gdal_array.NumericTypeCodeToGDALTypeCode(array.dtype)

    if noDataValue is None:
        noDataValue = refNoDataValue

    # 全部转为rows * cols * band_num
    cols = array.shape[1]
    rows = array.shape[0]

    if array.ndim == 2:
        array = array.reshape((rows, cols, 1))
    bandNum = array.shape[2]

    driver: gdal.Driver = gdal.GetDriverByName(driverName)
    outDs: gdal.Dataset = driver.Create(outputPath, cols, rows, bandNum, gType)
    outDs.SetGeoTransform(geoTransform)
    outDs.SetProjection(crs)

    for i in range(0, bandNum):
        band: gdal.Band = outDs.GetRasterBand(i + 1)  # 在GDAL中, band是从1开始索引的
        band.WriteArray(array[..., i])
        if noDataValue is not None:
            band.SetNoDataValue(noDataValue)
        band.FlushCache()
    print(f'Get {outputPath}')


def set_nodata(file, nodata):
    dataset = gdal.Open(file, gdal.GA_Update)

    if dataset is not None:
        band = dataset.GetRasterBand(1)
        band.SetNoDataValue(nodata)
        band.FlushCache()

        band = None
        dataset = None
        print('%f nodata value is set to %s' % (nodata, file))


if __name__ == '__main__':
    import os

    os.chdir(r'D:\co-registration\Flaash-master\ipsETM\Gaofen')

    ds = gdal.Open('GF2_PMS2_E115.0_N31.0_20190623_L1A0004073433-MSS2_RC_FLAASH_ORT.tiff')

    b = ds.GetRasterBand(1).ReadAsArray()
    g = ds.GetRasterBand(2).ReadAsArray()
    r = ds.GetRasterBand(3).ReadAsArray()
    nir = ds.GetRasterBand(4).ReadAsArray()

    new_ar = [b.astype(np.uint16), g.astype(np.uint16), r.astype(np.uint16), nir.astype(np.uint16)]

    array2Raster(new_ar, 'GF2_PMS2_E115.0_N31.0_20190623_L1A0004073433-MSS2_RC_FLAASH_ORT_1.tiff',
                 refImg='GF2_PMS2_E115.0_N31.0_20190623_L1A0004073433-MSS2_RC_FLAASH_ORT.tiff')
