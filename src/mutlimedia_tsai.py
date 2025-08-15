import numpy as np
from scipy.optimize import least_squares

class MultimediaTsaiCalibrator:
    """
    Camera calibration using the Multimedia Tsai model.
    This model extends the classic Tsai camera calibration to account for refraction at a flat interface (e.g., underwater imaging).
    """

    def __init__(self, n_air=1.0, n_medium=1.333, d=0.0):
        """
        n_air: refractive index of air (default 1.0)
        n_medium: refractive index of medium (e.g., water, default 1.333)
        d: distance from camera center to interface (in camera z direction)
        """
        self.n_air = n_air
        self.n_medium = n_medium
        self.d = d
        self.params = None  # To be estimated

    def project(self, Xw, params):
        """
        Project 3D world point Xw (shape: Nx3) to 2D image using current parameters.
        params: [fx, fy, cx, cy, k1, rx, ry, rz, tx, ty, tz]
        """
        fx, fy, cx, cy, k1, rx, ry, rz, tx, ty, tz = params
        # Rotation vector to rotation matrix
        theta = np.linalg.norm([rx, ry, rz])
        if theta < 1e-8:
            R = np.eye(3)
        else:
            r = np.array([rx, ry, rz]) / theta
            K = np.array([[0, -r[2], r[1]],
                          [r[2], 0, -r[0]],
                          [-r[1], r[0], 0]])
            R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
        t = np.array([tx, ty, tz])

        # Transform world to camera coordinates
        Xc = (R @ Xw.T).T + t

        # Refraction at interface (z = d)
        Xc_refracted = self._refract_points(Xc, self.d, self.n_air, self.n_medium)

        # Perspective projection
        x = Xc_refracted[:, 0] / Xc_refracted[:, 2]
        y = Xc_refracted[:, 1] / Xc_refracted[:, 2]

        # Radial distortion
        r2 = x**2 + y**2
        x_dist = x * (1 + k1 * r2)
        y_dist = y * (1 + k1 * r2)

        # Pixel coordinates
        u = fx * x_dist + cx
        v = fy * y_dist + cy
        return np.stack([u, v], axis=1)

    def _refract_points(self, Xc, d, n1, n2):
        """
        Apply refraction at a flat interface at z=d.
        Xc: Nx3 points in camera coordinates.
        Returns Nx3 refracted points.
        """
        Xc = np.asarray(Xc)
        # For each point, compute intersection with interface plane z=d
        z = Xc[:, 2]
        scale = (d - 0) / (z - 0)
        xi = Xc[:, 0] * scale
        yi = Xc[:, 1] * scale
        zi = np.full_like(xi, d)
        Pi = np.stack([xi, yi, zi], axis=1)

        # Incident direction
        dirs = Xc - np.zeros(3)
        dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)

        # Refract direction using Snell's law
        n = np.array([0, 0, 1])  # normal to interface
        cos_theta1 = np.dot(dirs, n)
        sin2_theta2 = (n1 / n2)**2 * (1 - cos_theta1**2)
        # Total internal reflection not handled
        cos_theta2 = np.sqrt(1 - sin2_theta2)
        dirs_refr = (n1 / n2) * dirs + (cos_theta2 - (n1 / n2) * cos_theta1)[:, None] * n

        # Continue ray from interface into medium
        t = (Xc[:, 2] - d) / dirs_refr[:, 2]
        Xc_refr = Pi + dirs_refr * t[:, None]
        return Xc_refr

    def calibrate(self, Xw, uv, initial_params):
        """
        Estimate camera parameters using nonlinear least squares.
        Xw: Nx3 world points
        uv: Nx2 observed image points
        initial_params: initial guess for parameters
        """

        def residuals(params):
            uv_proj = self.project(Xw, params)
            return (uv_proj - uv).ravel()

        result = least_squares(residuals, initial_params, method='lm')
        self.params = result.x
        return result

# Example usage:
# Xw = np.array([[...], ...])  # Nx3 world points
# uv = np.array([[...], ...])  # Nx2 image points
# initial_params = [fx, fy, cx, cy, k1, rx, ry, rz, tx, ty, tz]
# calibrator = MultimediaTsaiCalibrator(n_air=1.0, n_medium=1.333, d=0.0)
# result = calibrator.calibrate(Xw, uv, initial_params)
# print("Estimated parameters:", result.x)