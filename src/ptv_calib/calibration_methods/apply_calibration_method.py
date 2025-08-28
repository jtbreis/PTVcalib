import numpy as np

# TODO change implementation of soloff model
from .soloff import F
from scipy.optimize import least_squares

from .polynomial_4d_ptv import calibrate_4dptv_layer


def calibrate_camera(matches, calibration_method: str):

    if calibration_method == 'Soloff':
        cam_calibration = perform_soloff(matches)
    if calibration_method == '4d-ptv':
        cam_calibration = perform_4dptv_calibration(matches)
    else:
        raise ValueError(
            f"Unknown calibration method: {calibration_method}")

    return cam_calibration


def perform_soloff(matches):
    """
    xy: image coordinates
    XYZ: real world coordinates
    """
    points = np.vstack(matches)
    XYZ = points[:, :3]
    xy = points[:, 3:]

    N = 20

    sx = least_squares(lambda a: F(XYZ, a) -
                       xy[:, 0], np.zeros(N), method='trf').x
    sy = least_squares(lambda a: F(XYZ, a) -
                       xy[:, 1], np.zeros(N), method='trf').x

    return sx, sy


def perform_opencv_calibration(xy, XYZ):
    return


def perform_4dptv_calibration(matches):
    nlayers = matches.shape[0]
    calibration = np.empty(nlayers, dtype=object)
    for layer_idx, layer in enumerate(matches):
        layer_points = np.vstack(layer)
        XY = layer_points[:, :2]
        xy = layer_points[:, 3:]
        Z = layer_points[0, 2]
        calibration[layer_idx] = calibrate_4dptv_layer(XY, xy, Z)
    return calibration
