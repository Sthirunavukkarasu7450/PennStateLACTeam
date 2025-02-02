from copy import copy

from pyquaternion import Quaternion

from agents.geometry.translation import PTranslation

from typing import Self


class Pose:
    def __init__(self, position: PTranslation, orientation: Quaternion):
        self.position = position
        self.orientation = orientation

    def __str__(self):
        return f"Pose(position={self.position}, orientation={self.orientation})"

    def __repr__(self):
        return self.__str__()

    def __copy__(self):
        return Pose(copy(self.position), copy(self.orientation))

    def __eq__(self, other: Self):
        return self.position == other.position and self.orientation == other.orientation

    def __ne__(self, other: Self):
        return not self.__eq__(other)