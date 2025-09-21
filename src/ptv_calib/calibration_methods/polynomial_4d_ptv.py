from skimage import transform
import numpy as np

from .base import _CalibrationMethod
from ..utils.utils import group_matches_by_planes


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
        xy = np.zeros_like(XYZ[:, :2])
        calib_planes = [calib['posPlane'] for calib in self.calibration]
        if not np.any(np.isclose(Z[0], calib_planes)):
            raise ValueError(
                "Z value is not one of the calibrated plane positions.")
        else:
            plane_idx = np.where(np.isclose(Z[0], calib_planes))[0][0]

        calib = self.calibration[plane_idx]
        xy = calib['T3rw2px']((XYZ[:, :2]))

        return xy

    def transform_to_real_world(self, points):
        XYZ = np.zeros([points.shape[0], points.shape[1]+1, self.n_planes])
        for layer_idx, layer_calib in enumerate(self.calibration):
            XYZ[:, :2, layer_idx] = layer_calib['T3px2rw'](points)
            XYZ[:, 2, layer_idx] = layer_calib['posPlane']
        return XYZ

    def calibrate_layer(self, xy, XY, Z):
        T3rw2px = transform.estimate_transform(
            'polynomial', xy, XY, order=self.polynomial_order)
        T3px2rw = transform.estimate_transform(
            'polynomial', XY, xy, order=self.polynomial_order)
        calibration = {'posPlane': Z, 'T3rw2px': T3rw2px, 'T3px2rw': T3px2rw}
        return calibration

    def from_dict(self, struct):
        self.n_planes = len(struct.keys())
        self.calibration = np.empty(self.n_planes, dtype=object)
        for idx, [layer_key, layer_data] in enumerate(struct.items()):
            # Rebuild forward transform (rw → px)
            T3rw2px = transform.PolynomialTransform()
            T3rw2px.params = np.array(layer_data["T3rw2px"], dtype=float)

            # Rebuild inverse transform (px → rw)
            T3px2rw = transform.PolynomialTransform()
            T3px2rw.params = np.array(layer_data["T3px2rw"], dtype=float)

            self.calibration[idx] = {
                "posPlane": layer_data["posPlane"],
                "T3rw2px": T3rw2px,
                "T3px2rw": T3px2rw,
            }

    def to_dict(self):
        output = {}
        for layer_idx, layer in enumerate(self.calibration):
            output[f'layer{layer_idx}'] = {
                'posPlane': layer['posPlane'],
                'T3rw2px': layer['T3rw2px'].params,
                'T3px2rw': layer['T3px2rw'].params
            }

        return output
