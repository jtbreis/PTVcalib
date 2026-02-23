import os
import cv2
import pandas as pd


def get_valid_plane_counts(num_images: int) -> list[int]:
    """
    Return list of valid n_planes for the given number of images.
    Valid n_planes allow evenly spaced sampling: (num_images - n_planes) % (n_planes - 1) == 0.
    """
    if num_images <= 0:
        return []
    valid = [1]
    if num_images >= 2:
        valid.append(2)
    for n in range(3, num_images + 1):
        if (num_images - n) % (n - 1) == 0:
            valid.append(n)
    return valid


def read_images(folder_path: str, nplanes: int, extension: str = '.tif'):
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(
        ('.png', '.jpg', '.jpeg', '.bmp', '.tif'))]
    image_files.sort()
    image_files = [os.path.join(folder_path, fname) for fname in image_files]
    images = [cv2.imread(full_path,
                         cv2.IMREAD_GRAYSCALE) for full_path in image_files]
    num_images = len(images)

    if nplanes > num_images or (nplanes > 1 and (num_images - nplanes) % (nplanes - 1) != 0):
        valid = get_valid_plane_counts(num_images)
        raise ValueError(
            f"Requested {nplanes} plane(s), but there are {num_images} images and that choice is invalid. "
            f"Valid number of layers for {num_images} images: {valid}. "
            f"Use one of these for n_planes (e.g. n_planes={valid[-1] if valid else '?'})."
        )

    indices = [round(i * (len(images) - 1) / (nplanes - 1))
               for i in range(nplanes)] if nplanes > 1 else [0]
    image_files = [image_files[i] for i in indices]
    images = [images[i] for i in indices]

    return image_files, images


def load_calibration_target(path: str):
    grid_points = pd.read_csv(path).to_numpy()

    return grid_points
