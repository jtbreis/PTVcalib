import numpy as np

from .calibration_method import _CalibrationMethod


class Tsai(_CalibrationMethod):
    def __init__(self, parameters):
        O = parameters['origin']
        theta = parameters['theta']
        f = parameters['focal length']
        xh = parameters['image center correction, x']
        yh = parameters['image center correction, y']
        R = calc_R(parameters['theta'])
        resolution = (1, 1)
        E = np.zeros((3, 5))

    def fit(self, points_world, points_image):
        return super().fit(points_world, points_image)

    def transform_to_pixel(self, points):
        return super().transform(points)

    def transform_to_real_world(self, points):
        return super().inverse_transform(points)


def calc_R(theta):
    tx, ty, tz = theta
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(tx), -np.sin(tx)],
                   [0, np.sin(tx), np.cos(tx)]])  # TODO: look for standard rotation matrix in numpy
    Ry = np.array([[np.cos(ty), 0, np.sin(ty)],
                   [0, 1, 0],
                   [-np.sin(ty), 0, np.cos(ty)]])
    Rz = np.array([[np.cos(tz), -np.sin(tz), 0],
                   [np.sin(tz), np.cos(tz), 0],
                   [0, 0, 1]])
    return np.dot(np.dot(Rx, Ry), Rz)
