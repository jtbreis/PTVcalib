import h5py
import numpy as np

from ..calibration_method import CalibrationMethod
from ..utils.structure import Folders, Filenames


def import_calibration(folder):
    filename = folder + "/../.." + Folders.GENERAL.value + Filenames.CALIBRATION.value
    with h5py.File(filename, "r") as f:
        calibration = np.empty(len(f.keys()), dtype=object)
        for cam_idx, camera in enumerate(f.keys()):
            grp = f[camera]
            data = {}
            for layer_key in grp.keys():
                subgrp = grp[layer_key]
                data[layer_key] = {
                    "posPlane": subgrp["posPlane"][()],
                    "T3rw2px": subgrp["T3rw2px"][()],
                    "T3px2rw": subgrp["T3px2rw"][()],
                }

            calib = CalibrationMethod('4d-ptv')
            calib.from_dict(data)
            calibration[cam_idx] = calib

        return calibration
