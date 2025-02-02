from dataclasses import dataclass
from typing import Self

from numpy.core.defchararray import equal
from pyquaternion import Quaternion


class PTranslation:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __str__(self):
        return f"PTranslation(x={round(self.x, 6)}, y={round(self.y, 6)}, z={round(self.z, 6)})"

    def __repr__(self):
        return self.__str__()

    def __add__(self, other: Self) -> Self:
        return PTranslation(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: Self) -> Self:
        return PTranslation(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, other: float) -> Self:
        return PTranslation(self.x * other, self.y * other, self.z * other)

    def __truediv__(self, other: float) -> Self:
        return PTranslation(self.x / other, self.y / other, self.z / other)

    def __neg__(self) -> Self:
        return PTranslation(-self.x, -self.y, -self.z)

    def __eq__(self, other: Self) -> bool:
        return self.equals(other)

    def __ne__(self, other: Self) -> bool:
        return not self.__eq__(other)

    def equals(self, other: Self, tolerance: float = 1.0e-13) -> bool:
        return abs(self.x - other.x) <= tolerance and abs(self.y - other.y) <= tolerance and abs(self.z - other.z) <= tolerance

    def __copy__(self):
        return PTranslation(self.x, self.y, self.z)

    def negate(self) -> Self:
        return self.__neg__()

    def inverse(self) -> Self:
        return self.__neg__()

    def to_array(self):
        return [self.x, self.y, self.z]

    @staticmethod
    def from_array(array):
        return PTranslation(array[0], array[1], array[2])

    def rotate_by(self, orientation: Quaternion):
        """
        orientation: A normalized quaternion
        """
        rotated_vector = orientation.rotate(self.to_array())
        return PTranslation.from_array(rotated_vector)
        

