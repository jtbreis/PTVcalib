import cv2
import numpy as np


def is_almost_square(facet, tolerance=0.1):
    # facet: Nx2 array of points
    # tolerance: allowed relative difference between width and height
    xs = facet[:, 0]
    ys = facet[:, 1]
    width = xs.max() - xs.min()
    height = ys.max() - ys.min()
    if width == 0 or height == 0:
        return False
    ratio = min(width, height) / max(width, height)
    return ratio >= (1 - tolerance)


def point_in_polygon(point, polygon):
    # point: (x, y), polygon: Nx2 array
    return cv2.pointPolygonTest(np.array(polygon, np.int32), tuple(point), False) >= 0
