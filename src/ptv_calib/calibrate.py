import logging
import numpy as np
import os
from contextlib import nullcontext

from .io.reader import read_images, load_calibration_target
from .io.output import write_h5_matches, write_h5_calibration, write_h5_test_files
from .preprocessing.filter_images import fft_filter
from .preprocessing.point_detection import detect_target_points
from .grid_matching.match_target_points import perform_matching
from .utils.create_calibration_target import create_z_planes
from .utils.utils import group_matches_by_planes
from .utils.structure import create_folder_structure, Folders, Filenames
from .utils.timing import timed
from .calibration_method import CalibrationMethod

from .calibration_tests.test_camera_calibration import test_camera

from .visualization.plot_error import plot_2d_error, plot_2d_mean_error

logger = logging.getLogger(__name__)


def _timed_opt(log, detailed: bool, step_name: str, **kwargs):
    """Return a timing context manager if detailed is True, else a no-op."""
    return timed(log, step_name, **kwargs) if detailed else nullcontext()


class Calibration:
    """
    Camera Calibration

    Attributes
    """

    # TODO: calculate grid spacing from the calibration grid file
    def __init__(self, cameras: list[int], folder_path: str, output_path: str, calibration_grid_path: str, grid_spacing: float, target_point_diameter: int, z_min: float, z_max: float, n_planes: int, calibration_method: str = 'Soloff', detailed_timing: bool = False, **kwargs):
        self.plotting = 'None'
        self.detailed_timing = detailed_timing
        self.cameras = cameras
        self.ncameras = len(cameras)
        self.calibration_method = calibration_method
        self.path = folder_path
        self.output_path = output_path
        create_folder_structure(self.output_path)
        self.image_files = np.empty((self.ncameras), dtype=object)

        self.calibration_grid = calibration_grid_path
        self.calibration_grid_points = load_calibration_target(
            calibration_grid_path)
        self.grid_spacing = grid_spacing
        self.target_point_diameter = target_point_diameter

        self.z_planes = create_z_planes(z_min, z_max, n_planes)
        self.n_planes = n_planes

        logger.info(
            "Calibration setup: %d camera(s), %d plane(s), method=%s, grid=%s",
            self.ncameras, self.n_planes, self.calibration_method, calibration_grid_path,
        )

        self.image_points = np.empty((self.ncameras, n_planes), dtype=object)
        self.matched_points = np.empty((self.ncameras, n_planes), dtype=object)
        self.error2dpx2rw = np.empty(self.ncameras, dtype=object)
        self.error2drw2px = np.empty(self.ncameras, dtype=object)
        self.calibration = np.array([CalibrationMethod(
            self.calibration_method, **kwargs) for cam_idx in range(self.ncameras)], dtype=object)
        # Optional: store preloaded images to avoid re-reading in match_calibration_grid
        self._preloaded_images = np.empty((self.ncameras, n_planes), dtype=object)

    def preprocess_images(self, enhance_contrast: str = 'equalizeHist', filter_method: str = 'FFT', img_output_return: bool = False, denoise_method: str = 'nlmeans'):
        logger.info("Preprocessing images (contrast=%s, filter=%s, denoise=%s) ...", enhance_contrast, filter_method, denoise_method)
        with timed(logger, "preprocess_images (total)"):
            for cam_idx, cam in enumerate(self.cameras):
                camera_path = self.path + f'/Camera{cam}'
                logger.info("  Camera %d: reading %d planes from %s", cam, self.n_planes, camera_path)
                with _timed_opt(logger, self.detailed_timing, "  read_images", extra_msg=f"Camera {cam}"):
                    self.image_files[cam_idx], images = read_images(
                        camera_path, self.n_planes)

                for idx, img in enumerate(images):
                    with _timed_opt(logger, self.detailed_timing, "    fft_filter", extra_msg=f"Camera {cam} plane {idx}"):
                        images[idx] = fft_filter(img, self.target_point_diameter,
                                                 enhance_contrast, self.plotting, denoise_method=denoise_method)
                    with _timed_opt(logger, self.detailed_timing, "    detect_target_points", extra_msg=f"Camera {cam} plane {idx}"):
                        self.image_points[cam_idx, idx] = detect_target_points(
                            images[idx], self.target_point_diameter, self.plotting)
                    self._preloaded_images[cam_idx, idx] = images[idx]
                    n_pts = len(self.image_points[cam_idx, idx])
                    logger.debug("    Plane %d (z=%.2f): %d points detected", idx, self.z_planes[idx], n_pts)

                logger.info("  Camera %d: preprocessing done", cam)
                if img_output_return is True:
                    return images
        logger.info("Preprocessing complete.")

    def match_calibration_grid(self, center_find_method, use_preloaded_images: bool = True):
        logger.info("Matching calibration grid (center method=%s) ...", center_find_method)
        with timed(logger, "match_calibration_grid (total)"):
            for cam_idx, cam in enumerate(self.cameras):
                logger.info("  Camera %d: matching %d planes", cam, self.n_planes)
                for idx, img_path in enumerate(self.image_files[cam_idx]):
                    output_path = self.output_path + \
                        f'{Folders.ANNOTATIONS.value}/Camera{cam_idx}_{self.z_planes[idx]}.jpg'
                    if self.n_planes != 0:
                        self.calibration_grid_points[:, 2] = self.z_planes[idx]
                    preloaded = self._preloaded_images[cam_idx, idx] if use_preloaded_images else None
                    with _timed_opt(logger, self.detailed_timing, "    perform_matching", extra_msg=f"Camera {cam} plane {idx} (z={self.z_planes[idx]:.1f})"):
                        self.matched_points[cam_idx, idx] = perform_matching(
                            img_path, output_path, self.image_points[cam_idx, idx], self.calibration_grid_points, self.grid_spacing, self.target_point_diameter, center_find_method, self.plotting, image=preloaded)
                    n_matched = len(self.matched_points[cam_idx, idx])
                    logger.info("    Plane %d/%d (z=%.2f): %d matches — %s", idx + 1, self.n_planes, self.z_planes[idx], n_matched, os.path.basename(img_path))
                logger.info("  Camera %d: matching done", cam)
        logger.info("Grid matching complete.")

    def perform_calibration(self):
        logger.info("Performing calibration (method=%s) ...", self.calibration_method)
        with timed(logger, "perform_calibration (total)"):
            for cam_idx, cam in enumerate(self.cameras):
                cam_matches = np.vstack(self.matched_points[cam_idx, :])
                XYZ = cam_matches[:, :3]
                xy = cam_matches[:, 3:]
                n_pts = len(cam_matches)
                logger.info("  Camera %d: fitting %d points", cam, n_pts)
                with _timed_opt(logger, self.detailed_timing, "  fit", extra_msg=f"Camera {cam}, {n_pts} points"):
                    self.calibration[cam_idx].fit(XYZ, xy)
                logger.info("  Camera %d: fit complete", cam)
        logger.info("Calibration complete.")
        return self.calibration

    def set_custom_zplanes(self, z_planes: list[float]):
        self.z_planes = z_planes

    def set_folder_path(self, path):
        self.path = path

    def set_calibration_grid(self, calibration_grid_path):
        self.calibration_grid = calibration_grid_path

    def set_plotting_mode(self, plotting: str = 'None'):
        self.plotting = plotting

    def set_calibration_method(self, calibration_method='Soloff'):
        self.calibration = np.empty(self.ncameras, dtype=object)
        for cam_idx in range(self.ncameras):
            self.calibration[cam_idx]

    def check_calibrated_layers(self):
        logger.info("Checking calibrated layers ...")
        for cam_idx, cam in enumerate(self.cameras):
            cam_matches = np.vstack(self.matched_points[cam_idx, :])
            XYZ = cam_matches[:, :3]
            xy = cam_matches[:, 3:]

            _, XYZ_grouped, xy_grouped = group_matches_by_planes(
                XYZ=XYZ, xy=xy)

            self.error2drw2px[cam_idx], self.error2dpx2rw[cam_idx] = test_camera(self.calibration[cam_idx], n_layers=self.n_planes,
                                                                                 XYZ_grouped=XYZ_grouped, xy_grouped=xy_grouped)
            logger.info("  Camera %d: layer test done", cam)

        logger.info("Plotting 2D errors ...")
        plot_2d_error(self.error2drw2px)
        plot_2d_mean_error(self.error2drw2px)
        plot_2d_error(self.error2dpx2rw)
        plot_2d_mean_error(self.error2dpx2rw)

    def write_matches(self):
        path = self.output_path + Folders.MATCHES.value + Filenames.MATCHES.value
        logger.info("Writing matches to %s", path)
        with _timed_opt(logger, self.detailed_timing, "write_matches"):
            write_h5_matches(self.matched_points, path)

    def write_calibration_test_files(self):
        path = self.output_path + Folders.TESTS.value + Folders.CENTERS.value + Filenames.CAMERA.value
        logger.info("Writing calibration test files to %s", path)
        with _timed_opt(logger, self.detailed_timing, "write_calibration_test_files"):
            write_h5_test_files(self.matched_points, path)

    def write_calibration(self):
        path = self.output_path + Folders.GENERAL.value + Filenames.CALIBRATION.value
        logger.info("Writing calibration to %s", path)
        with _timed_opt(logger, self.detailed_timing, "write_calibration"):
            write_h5_calibration(self.calibration, path)
