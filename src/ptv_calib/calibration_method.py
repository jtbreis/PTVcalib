from .calibration_methods.soloff import Soloff
from .calibration_methods.tsai import Tsai
from .calibration_methods.polynomial_4d_ptv import Method4DPTV


class CalibrationMethod:
    def __init__(self, method: str, **kwargs):
        if method == "Soloff":
            self._impl = Soloff(**kwargs)
        elif method == "Tsai":
            self._impl = Tsai(**kwargs)
        elif method == "4d-ptv":
            self._impl = Method4DPTV(**kwargs)
        else:
            raise ValueError(f"Unknown calibration method: {method}")

    # Unified API
    def fit(self, XYZ, xy):
        return self._impl.fit(XYZ, xy)

    def transform_to_pixel(self, XYZ):
        return self._impl.transform_to_pixel(XYZ)

    def transform_to_real_world(self, xy):
        return self._impl.transform_to_real_world(xy)

    def __getattr__(self, name):
        # Only called if attribute not found on Calibration itself
        return getattr(self._impl, name)
