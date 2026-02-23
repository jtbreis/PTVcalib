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


def read_h5_matches(filename):
    """Read matches from matches.h5. Returns array of shape (n_cameras, n_layers), dtype=object; each cell is (N, 5) array [X, Y, Z, x, y]."""
    with h5py.File(filename, 'r') as f:
        cam_keys = sorted([k for k in f.keys() if k.startswith("Camera ")], key=lambda x: int(x.split()[1]))
        n_cameras = len(cam_keys)
        n_layers = len([k for k in f[cam_keys[0]].keys() if k.startswith("Layer ")])
        matches = np.empty((n_cameras, n_layers), dtype=object)
        for cam_idx, ck in enumerate(cam_keys):
            grp = f[ck]
            layer_keys = sorted([k for k in grp.keys() if k.startswith("Layer ")], key=lambda x: int(x.split()[1]))
            for layer_idx, lk in enumerate(layer_keys):
                data = np.asarray(grp[lk])
                if data.size == 0:
                    data = np.empty((0, 5))
                elif data.ndim == 1:
                    data = data.reshape(1, -1)
                matches[cam_idx, layer_idx] = data
    return matches


def write_h5_matches(matches, filename):
    with h5py.File(filename, 'w') as f:
        for cam_idx in range(matches.shape[0]):
            grp = f.create_group(f"Camera {cam_idx}")
            for layer_idx in range(matches.shape[1]):
                data = matches[cam_idx, layer_idx]
                if hasattr(data, '__len__') and len(data) == 0:
                    data = np.empty((0, 5))
                elif isinstance(data, np.ndarray) and data.ndim == 1:
                    data = data.reshape(1, -1)
                grp.create_dataset(f"Layer {layer_idx}", data=data)


def write_h5_test_files(matches, folder):
    for cam_idx in range(matches.shape[0]):
        with h5py.File(folder+f'{cam_idx+1}.h5', 'w') as f:
            for layer_idx in range(matches.shape[1]):
                grp = f.create_group(f'frame{int(layer_idx):05d}')
                data = np.vstack(matches[cam_idx, layer_idx])
                grp.create_dataset(
                    'x', data=data[:, -2])
                grp.create_dataset(
                    'y', data=data[:, -1])
                grp.create_dataset(
                    'XYZ', data=data[:, 0:3])
