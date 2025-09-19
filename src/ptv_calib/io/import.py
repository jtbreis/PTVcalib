import h5py
import numpy as np

from ..calibration_method import CalibrationMethod
from ..utils.structure import Folders


def import_calibration_method(folder):
    filename = folder + Folders.GENERAL.value + Folders.CALIBRATION.value
    calibration = []
    with h5py.File(filename, "r") as f:
        calibration = np.empty(len(f.keys), dtype=object)
        for cam_idx, camera in enumerate(f.keys()):
            grp = f[camera]
            data = {}
            for layer_key in grp.keys():
                subgrp = grp[layer_key]
                data[layer_key] = {
                    "posPlane": grp["posPlane"][()],
                    "T3rw2px": grp["T3rw2px"][()],
                    "T3px2rw": grp["T3px2rw"][()],
                }

            calib = CalibrationMethod('4d-ptv')
            calib.from_dict(data)
            calibration[cam_idx] = calib
