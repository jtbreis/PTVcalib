from abc import ABC, abstractmethod


class _CalibrationMethod(ABC):
    @abstractmethod
    def fit(self, XYZ, xy): ...
    @abstractmethod
    def transform_to_pixel(self, XYZ): ...
    @abstractmethod
    def transform_to_real_world(self, xy): ...
