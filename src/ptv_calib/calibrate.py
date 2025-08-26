import numpy as np

from .io.reader import read_images, load_calibration_target
from .preprocessing.filter_images import fft_filter
from .preprocessing.point_detection import detect_target_points
from .grid_matching.match_target_points import perform_matching
from .utils.create_calibration_target import create_z_planes
from .calibration_methods.apply_calibration_method import calibrate


class Calibration:
    """
    Camera Calibration

    Attributes
    """

    # TODO: calculate grid spacing from the calibration grid file
    def __init__(self, cameras: list[int], folder_path: str, calibration_grid_path: str, grid_spacing: float, target_point_diameter: int, z_min: float, z_max: float, n_planes: int):
        self.plotting = 'None'
        self.cameras = cameras
        self.path = folder_path

        self.calibration_grid = calibration_grid_path
        self.calibration_grid_points = load_calibration_target(
            calibration_grid_path)
        self.grid_spacing = grid_spacing
        self.target_point_diameter = target_point_diameter

        self.z_planes = create_z_planes(z_min, z_max, n_planes)
        self.n_planes = n_planes

        self.image_points = np.empty(n_planes, dtype=object)
        self.matched_points = np.empty(n_planes, dtype=object)

    def preprocess_images(self, enhance_contrast: str = 'equalizeHist', filter_method: str = 'FFT', img_output_return: bool = False):
        self.image_files, images = read_images(self.path)

        for idx, img in enumerate(images):
            images[idx] = fft_filter(img, self.target_point_diameter,
                                     enhance_contrast, self.plotting)
            self.image_points[idx] = detect_target_points(
                images[idx], self.target_point_diameter, self.plotting)

        if img_output_return is True:
            return images

    def match_calibration_grid(self, center_find_method):

        for idx, img_path in enumerate(self.image_files):
            print(idx, img_path)
            if self.n_planes != 0:
                self.calibration_grid_points[:, 2] = self.z_planes[idx]
            self.matched_points = perform_matching(
                img_path, self.image_points[idx], self.calibration_grid_points, self.grid_spacing, self.target_point_diameter, center_find_method, self.plotting)

    def perform_calibration(self, calibration_method='Soloff'):
        matches = np.vstack(self.matched_points)
        self.calibration = calibrate(matches, calibration_method)

    def set_custom_zplanes(self, z_planes: list[float]):
        self.z_planes = z_planes

    def set_folder_path(self, path):
        self.path = path

    def set_calibration_grid(self, calibration_grid_path):
        self.calibration_grid = calibration_grid_path

    def set_plotting_mode(self, plotting: str = 'None'):
        self.plotting = plotting
