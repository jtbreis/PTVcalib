import numpy as np

from ..calibration_method import CalibrationMethod


def calculate_2D_rw2px_calibration_error(calib: CalibrationMethod, XYZ, xy_known):
    xy = calib.transform_to_pixel(XYZ)
    error_2d = xy_known - xy
    mean_2d = np.mean(np.linalg.norm(error_2d, axis=1))
    return {'error': error_2d, 'mean': mean_2d, 'xy': xy_known}


def calculate_2D_px2rw_calibration_error(calib: CalibrationMethod, xy, XYZ_known):
    error = np.empty(len(XYZ_known), dtype=object)
    for idx, XYlayer in enumerate(XYZ_known):
        XYZ = calib.transform_to_real_world(xy[idx])
        error_2d = XYlayer - XYZ[:, :, idx]
        mean_2d = np.mean(np.linalg.norm(error_2d, axis=1))
        error[idx] = {'error': error_2d, 'mean': mean_2d, 'xy': XYlayer}
    return error


def calculate_matching_error():
    return
