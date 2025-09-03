from abc import ABC, abstractmethod

from .soloff import Soloff
from .tsai import Tsai
from .polynomial_4d_ptv import Method4DPTV

# private interface
class _CalibrationMethod(ABC):
    @abstractmethod
    def fit(self, XYZ, xy): ...
    @abstractmethod
    def transform_to_pixel(self, XYZ): ...
    @abstractmethod
    def transform_to_real_world(self, xy): ...


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
        return self._impl.transform(XYZ)

    def transform_to_real_world(self, xy):
        return self._impl.inverse_transform(xy)