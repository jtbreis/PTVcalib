import numpy as np

from src.calibration_methods.soloff import F
from scipy.optimize import least_squares

def perform_soloff(xy, XYZ):
    """
    xy: image coordinates
    XYZ: real world coordinates
    """
    N = 20

    sx = least_squares(lambda a: F(XYZ,a)-xy[:,0], np.zeros(N), method='trf').x
    sy = least_squares(lambda a: F(XYZ,a)-xy[:,1], np.zeros(N), method='trf').x

    return sx, sy