import os
import h5py
import numpy as np

from ..calibration_method import CalibrationMethod
from ..utils.structure import Folders, Filenames


def _calibration_path_from_folder(folder):
    """Resolve path to calib.h5. Tries: folder/calib.h5, folder/Calibration/calib.h5, then parent/Calibration/calib.h5, etc."""
    rel = Folders.GENERAL.value + Filenames.CALIBRATION.value  # /Calibration/calib.h5
    candidates = [
        os.path.join(folder, "calib.h5"),  # calib directly in folder (e.g. Calibration_Before/calib.h5)
        os.path.normpath(folder + rel),    # folder/Calibration/calib.h5
    ]
    for n_up in range(1, 5):
        prefix = folder
        for _ in range(n_up):
            prefix = os.path.dirname(prefix)
        candidates.append(os.path.normpath(prefix + rel))  # parent/Calibration/calib.h5
    for filename in candidates:
        if os.path.isfile(filename):
            return filename
    raise FileNotFoundError(
        f"Calibration file calib.h5 not found. Tried:\n  " +
        "\n  ".join(candidates) +
        "\nRun the calibration step (e.g. 01_calibration.py) with output_path set to the experiment folder "
        "(e.g. data/julian/PTV_below) so that Calibration/calib.h5 is created, or place calib.h5 in the folder."
    )


def import_calibration(folder):
    filename = _calibration_path_from_folder(folder)
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
