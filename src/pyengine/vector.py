from __future__ import annotations

from abc import ABC
from typing import Iterable
from typing import Type
from typing import Union

import numpy as np

NTypes = Union[
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    np.float16,
    np.float32,
    np.float64,
]
DType = Union[int, float, NTypes]


def to_tuple2(data):
    dlen = len(data)
    if dlen == 0:
        return 0, 0
    if dlen == 1:
        if isinstance(data[0], tuple) and len(data[0]) == 2:
            return data[0]
        return data[0], data[0]
    if dlen == 2:
        return data
    raise TypeError("Invalid Arguments Provided")


def to_tuple3(data):
    dlen = len(data)
    if dlen == 0:
        return 0, 0, 0
    if dlen == 1:
        if isinstance(data[0], tuple) and len(data[0]) == 3:
            return data[0]
        return data[0], data[0], data[0]
    if dlen == 2:
        return *to_tuple2(data[0]), data[1]
    if dlen == 3:
        return data
    raise TypeError("Invalid Arguments Provided")


def to_tuple4(data):
    dlen = len(data)
    if dlen == 0:
        return 0, 0, 0, 0
    if dlen == 1:
        if isinstance(data[0], tuple) and len(data[0]) == 4:
            return data[0]
        return data[0], data[0], data[0], data[0]
    if dlen == 2:
        if isinstance(data[0], tuple) and len(data[0]) == 3:
            return *to_tuple3(data[0]), data[1]
        return *to_tuple2(data[0]), *to_tuple2(data[1])
    if dlen == 3:
        return *to_tuple2(data[0]), data[1], data[2]
    if dlen == 4:
        return data
    raise TypeError("Invalid Arguments Provided")


class Vector2c(ABC):
    pass


# noinspection PyUnresolvedReferences
class Vector2(Vector2c, np.ndarray):
    def __new__(cls, *data, dtype: Type[DType] = float):
        return np.array(to_tuple2(data), dtype=dtype).view(cls)

    def __eq__(self, other: Vector2Like) -> bool:
        return np.all(super().__eq__(other))

    def __ne__(self, other: Vector2Like) -> bool:
        return not self.__eq__(other)

    @property
    def x(self) -> DType:
        return self[0]

    @x.setter
    def x(self, value: DType):
        self[0] = value

    @property
    def y(self) -> DType:
        return self[1]

    @y.setter
    def y(self, value: DType):
        self[1] = value

    @property
    def magnitude(self) -> DType:
        return np.linalg.norm(self)

    @magnitude.setter
    def magnitude(self, value: DType):
        self.__imul__(value / self.magnitude)

    # noinspection PyTypeChecker
    @property
    def magnitude_sq(self) -> DType:
        return np.dot(self, self)

    def normalize(self) -> Vector2:
        return self.__truediv__(self.magnitude)

    def normalize_self(self) -> Vector2:
        return self.__itruediv__(self.magnitude)

    def perpendicular(self) -> Vector2:
        return Vector2(self.y, -self.x)

    def perpendicular_self(self) -> Vector2:
        self[:] = self.y, -self.x
        return self

    # noinspection PyTypeChecker
    def dot(self, other: Vector2Like) -> DType:
        return np.dot(self, other)

    # noinspection PyTypeChecker
    def angle(self) -> float:
        return np.angle(self.x + self.y * 1j)

    def angle_between(self, other: Vector2Like) -> float:
        dot: float = self.x * other[0] + self.y * other[1]
        det: float = self.x * other[1] - self.y * other[0]
        return np.arctan2(det, dot)

    def distance(self, other: Vector2Like) -> float:
        return np.linalg.norm(self - other)

    def distance_sq(self, other: Vector2Like) -> float:
        dx: float = self.x - other[0]
        dy: float = self.y - other[1]
        return dx * dx + dy * dy

    def lerp(self, other: Vector2Like, t: float) -> Vector2:
        return (other - self) * t + self

    def smooth_step(self, other: Vector2Like, t: float) -> Vector2:
        t2 = t * t
        t3 = t2 * t
        return (
            (self + self - other - other) * t3 + (3.0 * other - 3.0 * self) * t2 + self * t + self
        )


Vector2Like = Union[Vector2c, np.ndarray, DType, Iterable[DType]]


class Vector3c(ABC):
    pass


# noinspection PyUnresolvedReferences
class Vector3(Vector3c, np.ndarray):
    def __new__(cls, *data, dtype: Type[DType] = float):
        return np.array(to_tuple3(data), dtype=dtype).view(cls)

    def __eq__(self, other: Vector3Like) -> bool:
        return np.all(super().__eq__(other))

    def __ne__(self, other: Vector3Like) -> bool:
        return not self.__eq__(other)

    @property
    def x(self) -> DType:
        return self[0]

    @x.setter
    def x(self, value: DType):
        self[0] = value

    @property
    def y(self) -> DType:
        return self[1]

    @y.setter
    def y(self, value: DType):
        self[1] = value

    @property
    def z(self) -> DType:
        return self[2]

    @z.setter
    def z(self, value: DType):
        self[2] = value

    @property
    def magnitude(self) -> DType:
        return np.linalg.norm(self)

    @magnitude.setter
    def magnitude(self, value: DType):
        self.__imul__(value / self.magnitude)

    # noinspection PyTypeChecker
    @property
    def magnitude_sq(self) -> DType:
        return np.dot(self, self)

    def normalize(self) -> Vector3:
        return self.__truediv__(self.magnitude)

    def normalize_self(self) -> Vector3:
        return self.__itruediv__(self.magnitude)

    # noinspection PyTypeChecker
    def dot(self, other: Vector3Like) -> DType:
        return np.dot(self, other)

    def cross(self, other: Vector3Like):
        return np.cross(self, other).view(Vector3)

    def angle_between(self, other: Vector3Like) -> float:
        x = self.x
        y = self.y
        z = self.z
        length1_sq: float = x * x + (y * y + (z * z))
        v = to_tuple3(other)
        length2_sq: float = v[0] * v[0] + (v[1] * v[1] + (v[2] * v[2]))
        dot: float = x * v[0] + (y * v[1] + (z * v[2]))
        cos: float = dot / np.sqrt(length1_sq * length2_sq)
        # This is because sometimes cos goes above 1 or below -1 because of lost precision
        cos = cos if cos < 1 else 1
        cos = cos if cos > -1 else -1
        return np.acos(cos)

    def distance(self, other: Vector3Like) -> float:
        return np.linalg.norm(self - other)

    def distance_sq(self, other: Vector3Like) -> float:
        dx: float = self.x - other[0]
        dy: float = self.y - other[1]
        return dx * dx + dy * dy

    def lerp(self, other: Vector3Like, t: float) -> Vector3:
        return (other - self) * t + self

    def smooth_step(self, other: Vector2Like, t: float) -> Vector2:
        t2 = t * t
        t3 = t2 * t
        return (
            (self + self - other - other) * t3 + (3.0 * other - 3.0 * self) * t2 + self * t + self
        )


Vector3Like = Union[Vector3c, np.ndarray, DType, Iterable[DType]]


class Vector4c(ABC):
    pass


# noinspection PyUnresolvedReferences
class Vector4(Vector4c, np.ndarray):
    def __new__(cls, *data, dtype: Type[DType] = float):
        return np.array(to_tuple4(data), dtype=dtype).view(cls)

    def __eq__(self, other: Vector4Like) -> bool:
        return np.all(super().__eq__(other))

    def __ne__(self, other: Vector4Like) -> bool:
        return not self.__eq__(other)

    @property
    def x(self) -> DType:
        return self[0]

    @x.setter
    def x(self, value: DType):
        self[0] = value

    @property
    def y(self) -> DType:
        return self[1]

    @y.setter
    def y(self, value: DType):
        self[1] = value

    @property
    def z(self) -> DType:
        return self[2]

    @z.setter
    def z(self, value: DType):
        self[2] = value

    @property
    def w(self) -> DType:
        return self[3]

    @w.setter
    def w(self, value: DType):
        self[3] = value

    @property
    def magnitude(self) -> DType:
        return np.linalg.norm(self)

    @magnitude.setter
    def magnitude(self, value: DType):
        self.__imul__(value / self.magnitude)

    # noinspection PyTypeChecker
    @property
    def magnitude_sq(self) -> DType:
        return np.dot(self, self)

    def normalize(self) -> Vector4:
        return self.__truediv__(self.magnitude)

    def normalize_self(self) -> Vector4:
        return self.__itruediv__(self.magnitude)

    # noinspection PyTypeChecker
    def dot(self, other: Vector4Like) -> DType:
        return np.dot(self, other)

    def cross(self, other: Vector4Like):
        return np.cross(self, other).view(Vector4)

    def angle_between(self, other: Vector4Like) -> float:
        x = self.x
        y = self.y
        z = self.z
        w = self.w
        length1_sq: float = x * x + (y * y + (z * z + (w * w)))
        v = to_tuple4(other)
        length2_sq: float = v[0] * v[0] + (v[1] * v[1] + (v[2] * v[2] + (v[3] * v[3])))
        dot: float = x * v[0] + (y * v[1] + (z * v[2] + (w * v[3])))
        cos: float = dot / np.sqrt(length1_sq * length2_sq)
        # This is because sometimes cos goes above 1 or below -1 because of lost precision
        cos = cos if cos < 1 else 1
        cos = cos if cos > -1 else -1
        return np.acos(cos)

    def distance(self, other: Vector4Like) -> float:
        return np.linalg.norm(self - other)

    def distance_sq(self, other: Vector4Like) -> float:
        dx: float = self.x - other[0]
        dy: float = self.y - other[1]
        return dx * dx + dy * dy

    def lerp(self, other: Vector4Like, t: float) -> Vector4:
        return (other - self) * t + self

    def smooth_step(self, other: Vector2Like, t: float) -> Vector2:
        t2 = t * t
        t3 = t2 * t
        return (
            (self + self - other - other) * t3 + (3.0 * other - 3.0 * self) * t2 + self * t + self
        )


Vector4Like = Union[Vector4c, np.ndarray, DType, Iterable[DType]]
