"""
从GEE上下载 MODIS BRDF_Albedo_Parameters数据
"""

import geemap
import ee


geemap.set_proxy('33210', 'http://127.0.0.1')


def ee_download_mcd43(start, end, roi, out_dir):
    """
    :param start:
    :param end:
    :param roi:
    :return:
    """
    mcd43a1 = (
        ee.ImageCollection('MODIS/061/MCD43A1')
        .filterBounds(roi)
        .filterDate(start, end)
        .select(['BRDF_Albedo_Parameters_Band1_iso',
                 'BRDF_Albedo_Parameters_Band1_vol',
                 'BRDF_Albedo_Parameters_Band1_geo'])
        .first()
    )
    print(mcd43a1.id().getInfo())
    geemap.download_ee_image(image=mcd43a1,
                             filename='test.tif',
                             scale=500)


if __name__ == '__main__':
    ee.Initialize()

    start = '2018-04-17'
    end = '2018-04-18'

    roi = ee.Geometry.Polygon(
        [[[113.335, 30.4432],
          [113.74, 30.3769],
          [113.661, 30.0361],
          [113.258, 30.1023]]], None, False)

    out_dir = r'F:\Experiment\BRDF Correction\BRDF model parameters'

    ee_download_mcd43(start, end, roi, out_dir)
