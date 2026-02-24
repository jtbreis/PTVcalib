import logging
import os
from contextlib import nullcontext

import cv2
import numpy as np

from .io.reader import read_images, load_calibration_target
from .io.output import read_h5_matches, write_h5_matches, write_h5_calibration, write_h5_test_files
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
from .visualization.plotting import display_matched_points_from_path

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
                            images[idx], self.target_point_diameter, plot=(self.plotting == 'Debug'))
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

    def load_matches_from_file(self):
        """Load matched points from matches.h5 into self.matched_points."""
        path = self.output_path + Folders.MATCHES.value + Filenames.MATCHES.value
        logger.info("Loading matches from %s", path)
        self.matched_points = read_h5_matches(path)

    def remove_matched_points(self, points_to_remove, grid_tolerance_mm=0.5, z_plane_tolerance_mm=0.01, redraw=True):
        """
        Remove matched grid points by (camera_index, X, Y, Z) in mm.
        Z is the plane depth (real-world Z); the layer index is inferred from z_planes.
        Then write matches.h5, re-run calibration, write calib.h5, and optionally
        redraw affected layer images with remaining matches.
        points_to_remove: list of (camera_index, X, Y, Z).
        """
        if not points_to_remove:
            logger.info("No points to remove.")
            return
        modified_layers = set()
        for cam_idx, X, Y, Z in points_to_remove:
            xyz = np.array([float(X), float(Y), float(Z)])
            z_val = float(Z)
            layer_idx = int(np.argmin(np.abs(self.z_planes - z_val)))
            if np.abs(self.z_planes[layer_idx] - z_val) > z_plane_tolerance_mm:
                logger.warning(
                    "Z=%.3f mm does not match any plane (nearest: %.3f at layer %d)",
                    z_val, self.z_planes[layer_idx], layer_idx,
                )
            layer = self.matched_points[cam_idx, layer_idx]
            arr = np.asarray(layer)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            if len(arr) == 0:
                logger.warning("No match found for (cam=%d, Z=%.2f, XY=%s)", cam_idx, z_val, xyz[:2])
                continue
            dist = np.linalg.norm(arr[:, :3] - xyz, axis=1)
            idx = np.argmin(dist)
            if dist[idx] > grid_tolerance_mm:
                logger.warning("No point within tolerance for (cam=%d, Z=%.2f, XYZ=%s)", cam_idx, z_val, xyz)
                continue
            new_layer = np.delete(arr, idx, axis=0)
            self.matched_points[cam_idx, layer_idx] = new_layer
            modified_layers.add((cam_idx, layer_idx))
            logger.info("Removed point (cam=%d, Z=%.2f mm, XYZ=%s)", cam_idx, z_val, xyz)
        if not modified_layers:
            return
        if redraw and modified_layers:
            image_paths = self._get_calibration_image_paths()
            for (cam_idx, layer_idx) in modified_layers:
                img_path = image_paths[cam_idx][layer_idx]
                save_path = self.output_path + f'{Folders.ANNOTATIONS.value}/Camera{cam_idx}_{self.z_planes[layer_idx]}.jpg'
                title = f"Camera {cam_idx} Z={self.z_planes[layer_idx]:.1f} mm (after removal)"
                logger.info("Saving redrawn image to %s", save_path)
                print(f"Saving redrawn image to: {save_path}")
                display_matched_points_from_path(
                    img_path, self.matched_points[cam_idx, layer_idx], title=title, save_path=save_path
                )
        self.write_matches()
        self.perform_calibration()
        self.write_calibration()
        logger.info("Matches and calibration files updated.")

    def _get_calibration_image_paths(self):
        """Return image_paths[cam_idx][layer_idx] from folder_path/cameras/n_planes."""
        image_paths = np.empty((self.ncameras, self.n_planes), dtype=object)
        for cam_idx, cam in enumerate(self.cameras):
            camera_path = self.path + f'/Camera{cam}'
            files, _ = read_images(camera_path, self.n_planes)
            for layer_idx in range(self.n_planes):
                image_paths[cam_idx, layer_idx] = files[layer_idx]
        return image_paths
