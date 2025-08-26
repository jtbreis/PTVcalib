import os
import cv2
import pandas as pd


def read_images(folder_path: str, extension: str = '.tif'):
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(
        ('.png', '.jpg', '.jpeg', '.bmp', '.tif'))]
    image_files.sort()
    image_files = [os.path.join(folder_path, fname) for fname in image_files]
    images = [cv2.imread(full_path,
                         cv2.IMREAD_GRAYSCALE) for full_path in image_files]

    return image_files, images


def load_calibration_target(path: str):
    grid_points = pd.read_csv(path).to_numpy()

    return grid_points
