"""
RPC模型读取及正反投影坐标计算
参考：
https://blog.csdn.net/weishaodong/article/details/121363600?spm=1001.2101.3001.6650.1&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-1-121363600-blog-95253177.pc_relevant_paycolumn_v3&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-1-121363600-blog-95253177.pc_relevant_paycolumn_v3
"""
import numpy as np
from osgeo import gdal
import math


# 最大迭代次数超过则报错
class MaxLocalizationIterationsError(Exception):
    """
    Custom rpcm Exception.
    """
    pass


def apply_poly(poly, x, y, z):
    """
    Evaluates a 3-variables polynom of degree 3 on a triplet of numbers.
    将三次多项式的统一模式构建为一个单独的函数
    Args:
        poly: list of the 20 coefficients of the 3-variate degree 3 polynom,
            ordered following the RPC convention.
        x, y, z: triplet of floats. They may be numpy arrays of same length.

    Returns:
        the value(s) of the polynom on the input point(s).
    """
    out = 0
    out += poly[0]
    out += poly[1]*y + poly[2]*x + poly[3]*z
    out += poly[4]*y*x + poly[5]*y*z +poly[6]*x*z
    out += poly[7]*y*y + poly[8]*x*x + poly[9]*z*z
    out += poly[10]*x*y*z
    out += poly[11]*y*y*y
    out += poly[12]*y*x*x + poly[13]*y*z*z + poly[14]*y*y*x
    out += poly[15]*x*x*x
    out += poly[16]*x*z*z + poly[17]*y*y*z + poly[18]*x*x*z
    out += poly[19]*z*z*z
    return out


def apply_rfm(num, den, x, y, z):
    """
    Evaluates a Rational Function Model (rfm), on a triplet of numbers.
    执行20个参数的分子和20个参数的除法
    Args:
        num: list of the 20 coefficients of the numerator
        den: list of the 20 coefficients of the denominator
            All these coefficients are ordered following the RPC convention.
        x, y, z: triplet of floats. They may be numpy arrays of same length.

    Returns:
        the value(s) of the rfm on the input point(s).
    """
    return apply_poly(num, x, y, z) / apply_poly(den, x, y, z)


def rpc_from_geotiff(geotiff_path):
    """
    Read the RPC coefficients from a GeoTIFF file and return an RPCModel object.
    该函数返回影像的Gdal格式的影像和RPCmodel
    Args:
        geotiff_path (str): path or url to a GeoTIFF file

    Returns:
        instance of the rpc_model.RPCModel class
    """
    # with rasterio.open(geotiff_path, 'r') as src:
    #
    dataset = gdal.Open(geotiff_path, gdal.GA_ReadOnly)
    rpc_dict = dataset.GetMetadata("RPC")

    del dataset

    return RPCModel(rpc_dict, 'geotiff')


def read_rpc_file(rpc_file):
    """
    Read RPC from a RPC_txt file and return a RPCmodel
    从TXT中直接单独读取RPC模型
    Args:
        rpc_file: RPC sidecar file path

    Returns:
        dictionary read from the RPC file, or an empty dict if fail

    """
    with open(rpc_file) as f:
        rpc_content = f.read()
    rpc = read_rpc(rpc_content)
    return RPCModel(rpc)


def read_rpc(rpc_content):
    """
    Read RPC file assuming the ikonos format
    解析RPC参数
    Args:
        rpc_content: content of RPC sidecar file path read as a string

    Returns:
        dictionary read from the RPC file

    """
    import re

    lines = rpc_content.split('\n')

    dict_rpc = {}
    for l in lines:
        ll = l.split()
        if len(ll) > 1:
            k = re.sub(r"[^a-zA-Z0-9_]", "", ll[0])
            dict_rpc[k] = ll[1]

    def parse_coeff(dic, prefix, indices):
        """ helper function"""
        return ' '.join([dic["%s_%s" % (prefix, str(x))] for x in indices])

    dict_rpc['SAMP_NUM_COEFF'] = parse_coeff(dict_rpc, "SAMP_NUM_COEFF", range(1, 21))
    dict_rpc['SAMP_DEN_COEFF'] = parse_coeff(dict_rpc, "SAMP_DEN_COEFF", range(1, 21))
    dict_rpc['LINE_NUM_COEFF'] = parse_coeff(dict_rpc, "LINE_NUM_COEFF", range(1, 21))
    dict_rpc['LINE_DEN_COEFF'] = parse_coeff(dict_rpc, "LINE_DEN_COEFF", range(1, 21))

    return dict_rpc


class RPCModel:
    def __init__(self, d, dict_format="geotiff"):
        """
        Args:
            d (dict): dictionary read from a geotiff file with
                rasterio.open('/path/to/file.tiff', 'r').tags(ns='RPC'),
                or from the .__dict__ of an RPCModel object.
            dict_format (str): format of the dictionary passed in `d`.
                Either "geotiff" if read from the tags of a geotiff file,
                or "rpcm" if read from the .__dict__ of an RPCModel object.
        """
        if dict_format == "geotiff":
            self.row_offset = float(d['LINE_OFF'][0:d['LINE_OFF'].rfind(' ')])
            self.col_offset = float(d['SAMP_OFF'][0:d['SAMP_OFF'].rfind(' ')])
            self.lat_offset = float(d['LAT_OFF'][0:d['LAT_OFF'].rfind(' ')])
            self.lon_offset = float(d['LONG_OFF'][0:d['LONG_OFF'].rfind(' ')])
            self.alt_offset = float(d['HEIGHT_OFF'][0:d['HEIGHT_OFF'].rfind(' ')])

            self.row_scale = float(d['LINE_SCALE'][0:d['LINE_SCALE'].rfind(' ')])
            self.col_scale = float(d['SAMP_SCALE'][0:d['SAMP_SCALE'].rfind(' ')])
            self.lat_scale = float(d['LAT_SCALE'][0:d['LAT_SCALE'].rfind(' ')])
            self.lon_scale = float(d['LONG_SCALE'][0:d['LONG_SCALE'].rfind(' ')])
            self.alt_scale = float(d['HEIGHT_SCALE'][0:d['HEIGHT_SCALE'].rfind(' ')])

            self.row_num = list(map(float, d['LINE_NUM_COEFF'].split()))
            self.row_den = list(map(float, d['LINE_DEN_COEFF'].split()))
            self.col_num = list(map(float, d['SAMP_NUM_COEFF'].split()))
            self.col_den = list(map(float, d['SAMP_DEN_COEFF'].split()))

            if 'LON_NUM_COEFF' in d:
                self.lon_num = list(map(float, d['LON_NUM_COEFF'].split()))
                self.lon_den = list(map(float, d['LON_DEN_COEFF'].split()))
                self.lat_num = list(map(float, d['LAT_NUM_COEFF'].split()))
                self.lat_den = list(map(float, d['LAT_DEN_COEFF'].split()))

        elif dict_format == "rpcm":
            self.__dict__ = d

        else:
            raise ValueError(
                "dict_format '{}' not supported. "
                "Should be {{'geotiff','rpcm'}}".format(dict_format)
            )

    def projection(self, lon, lat, alt):
        """
        Convert geographic coordinates of 3D points into image coordinates.
        正投影：从地理坐标到图像坐标
        Args:
            lon (float or list): longitude(s) of the input 3D point(s)
            lat (float or list): latitude(s) of the input 3D point(s)
            alt (float or list): altitude(s) of the input 3D point(s)

        Returns:
            float or list: horizontal image coordinate(s) (column index, ie x)
            float or list: vertical image coordinate(s) (row index, ie y)
        """
        nlon = (np.asarray(lon) - self.lon_offset) / self.lon_scale
        nlat = (np.asarray(lat) - self.lat_offset) / self.lat_scale
        nalt = (np.asarray(alt) - self.alt_offset) / self.alt_scale

        col = apply_rfm(self.col_num, self.col_den, nlat, nlon, nalt)
        row = apply_rfm(self.row_num, self.row_den, nlat, nlon, nalt)

        col = col * self.col_scale + self.col_offset
        row = row * self.row_scale + self.row_offset

        return col, row

    def localization(self, col, row, alt, return_normalized=False):
        """
        Convert image coordinates plus altitude into geographic coordinates.
        反投影：从图像坐标到地理坐标
        Args:
            col (float or list): x image coordinate(s) of the input point(s)
            row (float or list): y image coordinate(s) of the input point(s)
            alt (float or list): altitude(s) of the input point(s)

        Returns:
            float or list: longitude(s)
            float or list: latitude(s)
        """
        ncol = (np.asarray(col) - self.col_offset) / self.col_scale
        nrow = (np.asarray(row) - self.row_offset) / self.row_scale
        nalt = (np.asarray(alt) - self.alt_offset) / self.alt_scale

        if not hasattr(self, 'lat_num'):
            lon, lat = self.localization_iterative(ncol, nrow, nalt)
        else:
            lon = apply_rfm(self.lon_num, self.lon_den, nrow, ncol, nalt)
            lat = apply_rfm(self.lat_num, self.lat_den, nrow, ncol, nalt)

        if not return_normalized:
            lon = lon * self.lon_scale + self.lon_offset
            lat = lat * self.lat_scale + self.lat_offset

        return lon, lat

    def localization_iterative(self, col, row, alt):
        """
        Iterative estimation of the localization function (image to ground),
        for a list of image points expressed in image coordinates.
        逆投影时的迭代函数
        Args:
            col, row: normalized image coordinates (between -1 and 1)
            alt: normalized altitude (between -1 and 1) of the corresponding 3D
                point

        Returns:
            lon, lat: normalized longitude and latitude

        Raises:
            MaxLocalizationIterationsError: if the while loop exceeds the max
                number of iterations, which is set to 100.
        """
        # target point: Xf (f for final)
        Xf = np.vstack([col, row]).T

        # use 3 corners of the lon, lat domain and project them into the image
        # to get the first estimation of (lon, lat)
        # EPS is 2 for the first iteration, then 0.1.
        lon = -col ** 0  # vector of ones
        lat = -col ** 0
        EPS = 2
        x0 = apply_rfm(self.col_num, self.col_den, lat, lon, alt)
        y0 = apply_rfm(self.row_num, self.row_den, lat, lon, alt)
        x1 = apply_rfm(self.col_num, self.col_den, lat, lon + EPS, alt)
        y1 = apply_rfm(self.row_num, self.row_den, lat, lon + EPS, alt)
        x2 = apply_rfm(self.col_num, self.col_den, lat + EPS, lon, alt)
        y2 = apply_rfm(self.row_num, self.row_den, lat + EPS, lon, alt)

        n = 0
        while not np.all((x0 - col) ** 2 + (y0 - row) ** 2 < 1e-18):

            if n > 100:
                raise MaxLocalizationIterationsError("Max localization iterations (100) exceeded")

            X0 = np.vstack([x0, y0]).T
            X1 = np.vstack([x1, y1]).T
            X2 = np.vstack([x2, y2]).T
            e1 = X1 - X0
            e2 = X2 - X0
            u = Xf - X0

            # project u on the base (e1, e2): u = a1*e1 + a2*e2
            # the exact computation is given by:
            #   M = np.vstack((e1, e2)).T
            #   a = np.dot(np.linalg.inv(M), u)
            # but I don't know how to vectorize this.
            # Assuming that e1 and e2 are orthogonal, a1 is given by
            # <u, e1> / <e1, e1>
            num = np.sum(np.multiply(u, e1), axis=1)
            den = np.sum(np.multiply(e1, e1), axis=1)
            a1 = np.divide(num, den).squeeze()

            num = np.sum(np.multiply(u, e2), axis=1)
            den = np.sum(np.multiply(e2, e2), axis=1)
            a2 = np.divide(num, den).squeeze()

            # use the coefficients a1, a2 to compute an approximation of the
            # point on the gound which in turn will give us the new X0
            lon += a1 * EPS
            lat += a2 * EPS

            # update X0, X1 and X2
            EPS = .1
            x0 = apply_rfm(self.col_num, self.col_den, lat, lon, alt)
            y0 = apply_rfm(self.row_num, self.row_den, lat, lon, alt)
            x1 = apply_rfm(self.col_num, self.col_den, lat, lon + EPS, alt)
            y1 = apply_rfm(self.row_num, self.row_den, lat, lon + EPS, alt)
            x2 = apply_rfm(self.col_num, self.col_den, lat + EPS, lon, alt)
            y2 = apply_rfm(self.row_num, self.row_den, lat + EPS, lon, alt)

            n += 1

        return lon, lat


def wgs84blh2xyz(lat, lon, h):
    """

    :param lat:
    :param lon:
    :param h:
    :return:
    """
    a = 6378137     # a为椭球的长半轴
    f = 1 / 298.257223563   # f为椭球扁率
    b = a * (1 - f)     # b为椭球的短半轴
    e = math.sqrt(a * a - b * b) / a    # e为椭球的第一偏心率
    N = a / math.sqrt(1 - e * e * math.sin(lat * math.pi / 180) * math.sin(lat * math.pi / 180))    # N为椭球面卯酉圈曲率半径
    x = (N + h) * math.cos(lat * math.pi / 180) * math.cos(lon * math.pi / 180)
    y = (N + h) * math.cos(lat * math.pi / 180) * math.sin(lon * math.pi / 180)
    z = (N * (1 - (e * e)) + h) * math.sin(lat * math.pi / 180)

    return x, y, z


def xyz2wgs84blh(x, y, z):
    a = 6378137
    f = 1 / 298.257223563  # f为椭球扁率
    b = a * (1 - f)  # b为椭球的短半轴
    e = math.sqrt(a * a - b * b) / a  # e为椭球的第一偏心率

    temp_x = x
    temp_y = y
    temp_z = z

    curB = 0
    N = 0
    calB = math.atan2(temp_z, math.sqrt(temp_x * temp_x + temp_y * temp_y))

    counter = 0

    while math.fabs(curB - calB) * 180 / math.pi > 1e-15 and counter < 25:
        curB = calB
        N = a / math.sqrt(1 - e * e * math.sin(curB) * math.sin(curB))
        calB = math.atan2(temp_z + N * e * e * math.sin(curB), math.sqrt(temp_x * temp_x + temp_y * temp_y))
        counter += 1

    l = math.atan2(temp_y, temp_x) * 180 / math.pi
    b = curB * 180 / math.pi
    h = temp_z / math.sin(curB) - N * (1 - e * e)

    return b, l, h


def wgs84geocentric2local(lat, lon, wgs84_x, wgs84_y, wgs84_z):
    x = -wgs84_x * math.sin(lon * math.pi / 180) + wgs84_y * math.cos(lon * math.pi / 180)
    y = -wgs84_x * math.sin(lat * math.pi / 180) * math.cos(lon * math.pi / 180)\
        - wgs84_y * math.sin(lat * math.pi / 180) * math.sin(lon * math.pi / 180) \
        + wgs84_z * math.cos(lat * math.pi / 180)
    z = wgs84_x * math.cos(lat * math.pi / 180) * math.cos(lon * math.pi / 180) \
        + wgs84_y * math.cos(lat * math.pi / 180) * math.sin(lon * math.pi / 180)\
        + wgs84_z * math.sin(lat * math.pi / 180)

    return x, y, z


def local2wgs84geocentric(lat, lon, x, y, z):
    wgs84_x = -x * math.sin(lon * math.pi / 180) \
              - y * math.sin(lat * math.pi / 180) * math.cos(lon * math.pi / 180)\
              + z * math.cos(lat * math.pi / 180) * math.cos(lon * math.pi / 180)
    wgs84_y = x * math.cos(lon * math.pi / 180) \
              - y * math.sin(lat * math.pi / 180) * math.sin(lon * math.pi / 180)\
              + z * math.cos(lat * math.pi / 180) * math.sin(lon * math.pi / 180)
    wgs84_z = y * math.cos(lat * math.pi / 180) + z * math.sin(lat * math.pi / 180)

    return wgs84_x, wgs84_y, wgs84_z


if __name__ == '__main__1':
    import os

    os.chdir(r'D:\虚拟星座\高分数据\武大遥感\2018.4-8\GF1_PMS1_E113.5_N30.3_20180417_L1A0003127123')

    file = 'GF1_PMS1_E113.5_N30.3_20180417_L1A0003127123-MSS1_rpcOC.tiff'
    rpc = rpc_from_geotiff(file)

    # 正投影
    # # 影像左上角经纬度
    # TopLeftLatitude = 30.4432
    # TopLeftLongitude = 113.335
    # h = 10
    # row, col = rpc.projection(TopLeftLongitude, TopLeftLatitude, h)
    # print('row and col', row, col)

    # 反投影
    lon, lat = rpc.localization(4901, 5981, 10)
    print('lon and lat', lon, lat)

    ds = gdal.Open(file)

    im = ds.GetRasterBand(1).ReadAsArray()

    print(im)


if __name__ == '__main__':
    x, y, z = wgs84blh2xyz(30.246070082249744, 113.50193006346346, 10)
    print(f'原大地经纬度坐标：{30.246070082249744, 113.50193006346346, 10}')
    print(f'地心地固直角坐标：{x, y, z}')
    b, l, h = xyz2wgs84blh(x, y, z)
    print(f'转回大地经纬度坐标：{b, l, h}')
    e, n, u = wgs84geocentric2local(b, l, x, y, z)
    print(f'站心坐标：{e, n, u}')
    x, y, z = local2wgs84geocentric(b, l, e, n, u)
    print(f'转回地心地固直角坐标：{x, y, z}')

