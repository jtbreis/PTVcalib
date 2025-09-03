import numpy as np

from ..calibration_method import CalibrationMethod


def calculate_2D_calibration_error(calib: CalibrationMethod, XYZ, xy_known):
    xy = calib.transform_to_pixel(XYZ)
    error_2d = xy_known - xy
    mean_2d = np.mean(np.linalg.norm(error_2d, axis=1))
    return {'error': error_2d, 'mean': mean_2d, 'xy': xy_known}


def calculate_matching_error():
    return
