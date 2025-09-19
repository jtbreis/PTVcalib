import h5py
import pandas as pd
import numpy as np
import scipy.io as sio


def write_h5_file(data, filename):
    with h5py.File(filename, 'w') as f:
        for key, value in data.items():
            f.create_dataset(key, data=value)


def write_h5_calibration(calibration, filename):
    with h5py.File(filename, 'w') as f:
        for cam_idx, calibration in enumerate(calibration):
            grp = f.create_group(f"Camera {cam_idx}")
            calib = calibration.to_dict()
            for layer, layer_calib in calib.items():
                subgrp = grp.create_group(layer)
                for key, value in layer_calib.items():
                    subgrp.create_dataset(key, data=value)


def write_h5_matches(matches, filename):
    with h5py.File(filename, 'w') as f:
        for cam_idx in range(matches.shape[0]):
            grp = f.create_group(f"Camera {cam_idx}")
            for layer_idx in range(matches.shape[1]):
                grp.create_dataset(
                    f"Layer {layer_idx}", data=matches[cam_idx, layer_idx])


def output_4dptv(calibration):

    sio.savemat('calib.mat', {

    })

    return
