from skimage import transform
import numpy as np

from .base import _CalibrationMethod


class Method4DPTV(_CalibrationMethod):
    def __init__(self, polynomial_order: int = 3):
        self.calibration = None
        self.polynomial_order = polynomial_order

    def fit(self, XYZ, xy):
        self.n_planes, XYZ_grouped, xy_grouped = group_matches_by_planes(
            XYZ=XYZ, xy=xy)
        self.calibration = np.empty(self.n_planes, dtype=object)

        for plane_idx, XYZ_plane in enumerate(XYZ_grouped):
            xy_plane = xy_grouped[plane_idx]
            XY_plane = XYZ_plane[:, :2]
            Z_plane = XYZ_plane[0, 2]
            self.calibration[plane_idx] = self.calibrate_layer(
                XY_plane, xy_plane, Z_plane)

    def transform_to_pixel(self, XYZ):
        Z = XYZ[:, 2]
        calib_planes = [calib['posPlane'] for calib in self.calibration]
        if not np.any(np.isclose(Z, calib_planes)):
            raise ValueError(
                "Z value is not one of the calibrated plane positions.")
        else:
            plane_idx = np.where(np.isclose(Z[0], calib_planes))[0][0]
            calib = self.calibration[plane_idx]
            xy = calib['T3rw2px'].transform(XYZ[:, :2])

        return xy

    def transform_to_real_world(self, points):
        # TODO requires a 3D matching algorithm
        return super().inverse_transform(points)

    def calibrate_layer(self, xy, XY, Z):
        T3rw2px = transform.estimate_transform(
            'polynomial', xy, XY, order=self.polynomial_order)
        T3px2rw = transform.estimate_transform(
            'polynomial', XY, xy, order=self.polynomial_order)
        calibration = {'posPlane': Z, 'T3rw2px': T3rw2px, 'T3px2rw': T3px2rw}
        return calibration


def group_matches_by_planes(XYZ, xy):
    Z_values = [plane[2] for plane in XYZ]
    unique_Z = np.unique(Z_values)
    n_planes = len(unique_Z)
    # Group XYZ and xy by unique Z values
    XYZ_grouped = []
    xy_grouped = []
    for Z in unique_Z:
        indices = [i for i, plane in enumerate(
            XYZ) if np.isclose(plane[2], Z)]
        XYZ_grouped.append(np.vstack([XYZ[i] for i in indices]))
        xy_grouped.append(np.vstack([xy[i] for i in indices]))

    return n_planes, XYZ_grouped, xy_grouped
