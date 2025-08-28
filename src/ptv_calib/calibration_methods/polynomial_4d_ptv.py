from skimage import transform
import numpy as np


def perform_transformation(xy, XY, polynomial_order=3):
    T3rw2px = transform.estimate_transform(
        'polynomial', xy, XY, order=polynomial_order).params
    T3px2rw = transform.estimate_transform(
        'polynomial', XY, xy, order=polynomial_order).params
    return T3rw2px, T3px2rw


def calibrate_4dptv_layer(xy, XY, Z):
    T3rw2px, T3px2rw = perform_transformation(xy, XY)
    calibration = {'posPlane': Z, 'T3rw2px': T3rw2px, 'T3px2rw': T3px2rw}
    return calibration
