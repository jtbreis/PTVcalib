import numpy as np

from .calibration_error import calculate_2D_calibration_error
from ..calibration_method import CalibrationMethod


def test_camera(calib: CalibrationMethod, n_layers, XYZ_grouped, xy_grouped):
    error_layer = np.empty(n_layers, dtype=object)
    for idx_layer, XYZ_layer in enumerate(XYZ_grouped):
        error_layer[idx_layer] = calculate_2D_calibration_error(
            calib, XYZ_layer, xy_grouped[idx_layer])

    return error_layer
