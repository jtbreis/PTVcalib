import numpy as np

from .calibration_error import calculate_2D_rw2px_calibration_error, calculate_2D_px2rw_calibration_error
from ..calibration_method import CalibrationMethod


def test_camera(calib: CalibrationMethod, n_layers, XYZ_grouped, xy_grouped):
    error_layer_rw2px = np.empty(n_layers, dtype=object)
    for idx_layer, XYZ_layer in enumerate(XYZ_grouped):
        error_layer_rw2px[idx_layer] = calculate_2D_rw2px_calibration_error(
            calib, XYZ_layer, xy_grouped[idx_layer])

    error_layer_px2rw = calculate_2D_px2rw_calibration_error(
        calib, xy_grouped, XYZ_grouped)

    return error_layer_rw2px, error_layer_px2rw
