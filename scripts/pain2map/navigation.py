from enum import Enum
from typing import List, Optional, Tuple

import numpy as np


class Direction(Enum):
    Center = (0, 0)

    North = (0, -1)
    East = (1, 0)
    South = (0, 1)
    West = (-1, 0)
    N = North
    E = East
    S = South
    W = West

    Up = North
    Right = East
    Down = South
    Left = West
    U = Up
    R = Right
    D = Down
    L = Left

    def __init__(self, x: int, y: int):
        self.__x = x
        self.__y = y

    @staticmethod
    def from_coordinates(c_from: "Coordinate", c_to: "Coordinate") -> "Direction":
        return direction(c_from, c_to)

    @staticmethod
    def values() -> List["Direction"]:
        return [Direction.North, Direction.East, Direction.South, Direction.West]

    @property
    def x(self) -> int:
        return self.__x

    @property
    def y(self) -> int:
        return self.__y

    def is_horizontal(self) -> bool:
        """

        :return: True if direction is East or West, False otherwise
        """
        return self.x != 0

    def opposite(self) -> "Direction":
        if self == Direction.North:
            return Direction.South
        elif self == Direction.East:
            return Direction.West
        elif self == Direction.South:
            return Direction.North
        elif self == Direction.West:
            return Direction.East
        else:
            return Direction.Center

    def __add__(self, other) -> "Coordinate":
        if isinstance(other, Direction):
            return Coordinate(self.x + other.x, self.y + other.y)
        elif isinstance(other, Coordinate):
            return other + self
        else:
            raise NotImplementedError(f"Adding \"{other}\" to a Coordinate is not supported!")


class Coordinate:
    __PREFIX = "("
    __SEPARATOR = "|"
    __SUFFIX = ")"

    @staticmethod
    def from_string(coordinate: str) -> Optional["Coordinate"]:
        coordinate = coordinate.replace(" ", "")
        if coordinate.startswith(Coordinate.__PREFIX):
            coordinate = coordinate[len(Coordinate.__PREFIX):]
        if coordinate.endswith(Coordinate.__SUFFIX):
            coordinate = coordinate[:-len(Coordinate.__SUFFIX)]
        if Coordinate.__SEPARATOR in coordinate:
            parts = coordinate.split(Coordinate.__SEPARATOR)
            if len(parts) == 2:
                x, y = float(parts[0]), float(parts[1])
                return Coordinate(x, y)
        return None

    @staticmethod
    def distance(a: "Coordinate", b: "Coordinate") -> float:
        return np.sqrt((a.x - b.x)**2 + (a.y - b.y)**2)

    def direction(c_from: "Coordinate", c_to: "Coordinate") -> Direction:
      diff = c_to - c_from
      if diff.x == 0 and diff.y == 0:
          return Direction.Center
      if abs(diff.x) > abs(diff.y):
          if diff.x > 0:
              return Direction.East
          else:
              return Direction.West
      else:
          if diff.y > 0:
              return Direction.South
          else:
              return Direction.North

    def __init__(self, x: float, y: float):
        self.__x = x
        self.__y = y

    @property
    def x(self) -> float:
        return self.__x

    @property
    def y(self) -> float:
        return self.__y

    def resolve(self) -> Tuple[float, float]:
        return self.__x, self.__y

    def __add__(self, other) -> "Coordinate":
        if isinstance(other, Direction):
            return Coordinate(self.x + other.x, self.y + other.y)
        elif isinstance(other, Coordinate):
            return Coordinate(self.x + other.x, self.y + other.y)
        else:
            raise NotImplementedError(f"Adding \"{other}\" to a Coordinate is not supported!")

    def __sub__(self, other) -> "Coordinate":
        if isinstance(other, Direction):
            return Coordinate(self.x - other.x, self.y - other.y)
        elif isinstance(other, Coordinate):
            return Coordinate(self.x - other.x, self.y - other.y)
        else:
            raise NotImplementedError(f"Subtracting \"{other}\" from a Coordinate is not supported!")

    def __mul__(self, other) -> "Coordinate":
        if isinstance(other, (int, float)):
            return Coordinate(self.x * other, self.y * other)
        else:
            raise NotImplementedError(f"Multiplying a Coordinate with \"{other}\" is not supported!")

    def __truediv__(self, other) -> "Coordinate":
        if isinstance(other, (int, float)):
            return Coordinate(self.x / other, self.y / other)
        else:
            raise NotImplementedError(f"Dividing a Coordinate by \"{other}\" is not supported!")

    def __eq__(self, other) -> bool:
        if isinstance(other, Coordinate):
            return self.x == other.x and self.y == other.y
        return False

    def __hash__(self):
        return 61 * hash(self.x) + 51 * hash(self.y)

    def __str__(self):
        return f"{Coordinate.__PREFIX}{self.__x}{Coordinate.__SEPARATOR}{self.__y}{Coordinate.__SUFFIX}"
