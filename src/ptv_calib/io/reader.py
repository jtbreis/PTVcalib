import os
import cv2
import pandas as pd


def read_images(folder_path: str, nplanes: int, extension: str = '.tif'):
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(
        ('.png', '.jpg', '.jpeg', '.bmp', '.tif'))]
    image_files.sort()
    image_files = [os.path.join(folder_path, fname) for fname in image_files]
    images = [cv2.imread(full_path,
                         cv2.IMREAD_GRAYSCALE) for full_path in image_files]

    if nplanes > len(images) or (len(images) - nplanes) % (nplanes-1) != 0:
        raise ValueError(
            f"Requested {nplanes} planes, but only {len(images)} images are available.")

    indices = [round(i * (len(images) - 1) / (nplanes - 1))
               for i in range(nplanes)] if nplanes > 1 else [0]
    image_files = [image_files[i] for i in indices]
    images = [images[i] for i in indices]

    return image_files, images


def load_calibration_target(path: str):
    grid_points = pd.read_csv(path).to_numpy()

    return grid_points
