from skimage import transform
import numpy as np

from .calibration_method import _CalibrationMethod

class Method4DPTV(_CalibrationMethod):
    def __init__(self, nlayers:int, polynomial_order: int=3):
        self.calibration = np.empty(nlayers, dtype=object)
        self.nlayers = None
        self.polynomial_order=polynomial_order

    def fit(self, XYZ, xy):
        nlayers = XYZ.shape[0]
        
        for plane_idx, XYZ_plane in enumerate(XYZ):
            xy_plane = xy[plane_idx]
            XY_plane = XYZ_plane[:, :2]
            Z_plane = XYZ_plane[0, 2]
            self.calibration[plane_idx] = self.calibrate_layer(XY_plane, xy_plane, Z_plane)
    
    def transform_to_pixel(self, XYZ):
        Z = XYZ[:, 2]
        calib_planes = [calib['posPlane'] for calib in self.calibration]
        if not np.any(np.isclose(Z, calib_planes)):
            raise ValueError("Z value is not one of the calibrated plane positions.")
        else:
            plane_idx = np.where(np.isclose(Z[0], calib_planes))[0][0]
            calib = self.calibration[plane_idx]
            xy = calib['T3rw2px'].transform(XYZ[:, :2])

        return xy
    
    def transform_to_real_world(self, points):
        #TODO requires a 3D matching algorithm
        return super().inverse_transform(points)

    def calibrate_layer(self, xy, XY, Z):
        T3rw2px = transform.estimate_transform('polynomial', xy, XY, order=self.polynomial_order)
        T3px2rw = transform.estimate_transform('polynomial', XY, xy, order=self.polynomial_order)
        calibration = {'posPlane': Z, 'T3rw2px': T3rw2px, 'T3px2rw': T3px2rw}
        return calibration
