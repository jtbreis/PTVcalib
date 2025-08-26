import numpy as np

# TODO change implementation of soloff model
from .soloff import F
from scipy.optimize import least_squares


def calibrate(matches, calibration_method: str):
    XYZ = matches[:, :3]
    xy = matches[:, 3:]
    if calibration_method == 'Soloff':
        perform_soloff(xy, XYZ)
    else:
        raise ValueError(f"Unknown calibration method: {calibration_method}")


def perform_soloff(xy, XYZ):
    """
    xy: image coordinates
    XYZ: real world coordinates
    """
    N = 20

    sx = least_squares(lambda a: F(XYZ, a) -
                       xy[:, 0], np.zeros(N), method='trf').x
    sy = least_squares(lambda a: F(XYZ, a) -
                       xy[:, 1], np.zeros(N), method='trf').x

    return sx, sy
