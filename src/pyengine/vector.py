from __future__ import annotations

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


def vector2(*data: DType, dtype: Type[DType] = float) -> np.ndarray:
    dlen = len(data)
    if dlen == 0:
        return np.array([0, 0], dtype=dtype)
    if dlen == 1:
        if isinstance(data[0], Iterable):
            return vector2(data[0], dtype=dtype)
        return np.array([data[0], data[0]], dtype=dtype)
    if dlen == 2:
        return np.array(data, dtype=dtype)
    raise TypeError("Invalid Arguments Provided")


def vector3(*data: DType, dtype: Type[DType] = float) -> np.ndarray:
    dlen = len(data)
    if dlen == 0:
        return np.array([0, 0, 0], dtype=dtype)
    if dlen == 1:
        if isinstance(data[0], Iterable):
            return vector3(data[0], dtype=dtype)
        return np.array([data[0], data[0], data[0]], dtype=dtype)
    if dlen == 2:
        return np.array([*vector2(data[0], dtype=dtype), data[1]], dtype=dtype)
    if dlen == 3:
        return np.array(data, dtype=dtype)
    raise TypeError("Invalid Arguments Provided")


def vector4(*data: DType, dtype: Type[DType] = float) -> np.ndarray:
    dlen = len(data)
    if dlen == 0:
        return np.array([0, 0, 0, 0], dtype=dtype)
    if dlen == 1:
        if isinstance(data[0], Iterable):
            return vector4(data[0], dtype=dtype)
        return np.array([data[0], data[0], data[0], data[0]], dtype=dtype)
    if dlen == 2:
        if isinstance(data[0], Iterable) and len(data[0]) == 3:
            return np.array([*vector3(data[0], dtype=dtype), data[1]], dtype=dtype)
        return np.array(
            [*vector2(data[0], dtype=dtype), *vector2(data[1], dtype=dtype)], dtype=dtype
        )
    if dlen == 3:
        return np.array([*vector2(data[0], dtype=dtype), data[1], data[2]], dtype=dtype)
    if dlen == 4:
        return np.array(data, dtype=dtype)
    raise TypeError("Invalid Arguments Provided")


def vector_random(order: int, normalize: bool = True) -> np.ndarray:
    vector = 2.0 * np.random.random((order,)) - 1.0
    if normalize:
        return vector_normalize(vector)
    return vector


def vector_magnitude(v: np.ndarray) -> DType:
    return np.linalg.norm(v)


def vector_magnitude_sq(v: np.ndarray) -> DType:
    return vector_dot(v, v)


def vector_normalize(v: np.ndarray, *, out: np.ndarray = None) -> np.ndarray:
    if out is None:
        out = np.array(v)

    out[:] = v.__truediv__(vector_magnitude(v))

    return out


# noinspection PyTypeChecker
def vector_dot(  # TODO - Check Bounds
    v1: np.ndarray, v2: np.ndarray, *, out: np.ndarray = None
) -> DType:
    if out is None:
        out = np.array(v1)

    out[:] = np.dot(v1, v2)
    return out


def vector_cross(  # TODO - Check Bounds
    v1: np.ndarray, v2: np.ndarray, *, out: np.ndarray = None
) -> np.ndarray:
    if out is None:
        out = np.array(v1)

    out[:] = np.cross(v1, v2)
    return out


def vector_angle_between(v1: np.ndarray, v2: np.ndarray) -> DType:  # TODO - Check Bounds
    dot = vector_dot(v1, v2)
    mag1 = vector_magnitude(v1)
    mag2 = vector_magnitude(v2)
    return np.math.acos(dot / (mag1 * mag2))


def vector_distance(v1: np.ndarray, v2: np.ndarray) -> DType:  # TODO - Check Bounds
    return vector_magnitude(v2 - v1)


def vector_distance_sq(v1: np.ndarray, v2: np.ndarray) -> DType:  # TODO - Check Bounds
    return vector_magnitude_sq(v2 - v1)


def vector_lerp(  # TODO - Check Bounds
    v1: np.ndarray, v2: np.ndarray, t: float, *, out: np.ndarray = None
) -> np.ndarray:
    if out is None:
        out = np.array(v1)

    out[:] = (v2 - v1) * t + v1
    return out


def vector2_from_angle2(theta: float) -> np.ndarray:
    return vector2(np.math.cos(theta), np.math.sin(theta), dtype=float)


def vector2_perpendicular(  # TODO - Check Bounds
    v: np.ndarray, *, out: np.ndarray = None
) -> np.ndarray:
    if out is None:
        out = np.array(v)

    out[:] = v[1], -v[0]
    return out


def vector2_angle(v: np.ndarray) -> DType:  # TODO - Check Bounds
    return np.angle(v[0] + v[1] * 1j)


def matrix3(*, dtype: Type[DType] = float) -> np.ndarray:
    return np.identity(3, dtype=dtype)


def matrix4(*, dtype: Type[DType] = float) -> np.ndarray:
    return np.identity(4, dtype=dtype)


def matrix_inverse(m: np.ndarray, *, out: np.ndarray = None) -> np.ndarray:  # TODO - Check Bounds
    if out is None:
        out = np.array(m)

    out[:] = np.linalg.inv(m)
    return out


def matrix_translate(  # TODO - Check Bounds
    m: np.ndarray,
    v: np.ndarray,
    distance: DType = 1.0,
    normalize: bool = False,
    *,
    out: np.ndarray = None,
) -> np.ndarray:
    if normalize:
        v = vector_normalize(v)

    if out is None:
        out = np.array(m)

    if vector_magnitude(v) == 0:
        return out

    out[-1, :-1] = m[-1, :-1] + distance * m[:-1, :-1].dot(v)
    return out


def matrix_translate_abs(  # TODO - Check Bounds
    m: np.ndarray,
    v: np.ndarray,
    distance: DType = 1.0,
    normalize: bool = False,
    *,
    out: np.ndarray = None,
) -> np.ndarray:
    if normalize:
        v = vector_normalize(v)

    if out is None:
        out = np.array(m)

    if vector_magnitude(v) == 0:
        return out

    out[-1, :-1] = m[-1, :-1] + distance * v
    return out


def matrix_rotate(  # TODO - Check Bounds
    m: np.ndarray,
    v: np.ndarray,
    theta: DType = 1.0,
    normalize: bool = False,
    *,
    out: np.ndarray = None,
) -> np.ndarray:
    if normalize:
        v = vector_normalize(v)

    if out is None:
        out = np.array(m)

    if vector_magnitude(v) == 0:
        return out

    s: float = np.sin(theta)
    c: float = np.cos(theta)

    if m.shape[0] == 4:
        tmp = (1.0 - c) * v

        out[:-1, :-1] = (
            m[:-1, :-1].dot(tmp[0] * v + [c, s * v[2], -s * v[1]]),
            m[:-1, :-1].dot(tmp[1] * v + [-s * v[2], c, s * v[0]]),
            m[:-1, :-1].dot(tmp[2] * v + [s * v[1], -s * v[0], c]),
        )
    else:
        out[:-1, :-1] = (out[:-1, :-1].dot([c, s]), out[:-1, :-1].dot([-s, c]))
    return out


def matrix_scale(  # TODO - Check Bounds
    m: np.ndarray,
    v: np.ndarray,
    amount: DType = 1.0,
    normalize: bool = False,
    *,
    out: np.ndarray = None,
) -> np.ndarray:
    if normalize:
        v = vector_normalize(v)

    if out is None:
        out = np.array(m)

    if vector_magnitude(v) == 0:
        return out

    out[:-1, :-1] = m[:-1, :-1] * v * amount
    return out


VECTOR_X: np.ndarray = vector3(1, 0, 0, dtype=float)
VECTOR_Y: np.ndarray = vector3(0, 1, 0, dtype=float)
VECTOR_Z: np.ndarray = vector3(0, 0, 1, dtype=float)

MATRIX_3: np.ndarray = matrix3(dtype=float)
MATRIX_4: np.ndarray = matrix4(dtype=float)
