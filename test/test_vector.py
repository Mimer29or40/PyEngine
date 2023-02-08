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

VectorType = np.ndarray

VectorLike = Union[DType, Iterable[DType]]


def vector2(*data: VectorLike, dtype: Type[DType] = float) -> VectorType:
    """
    0 Args: _      -> (0, 0)
    1 Args: x      -> (x, x)
    1 Args: (x,)   -> (x, x)
    1 Args: (x, y) -> (x, y)
    2 Args: x, y   -> (x, y)

    :param data: The vector data
    :param dtype: The data type
    :return: The vector array
    """
    dlen = len(data)
    if dlen == 0:
        return np.array([0, 0], dtype=dtype)
    if dlen == 1:
        if isinstance(data[0], Iterable):
            return vector2(*data[0], dtype=dtype)
        return np.array([data[0], data[0]], dtype=dtype)
    if dlen == 2:
        return np.array(data, dtype=dtype)
    raise TypeError("Invalid Arguments Provided")


def vector3(*data: VectorLike, dtype: Type[DType] = float) -> VectorType:
    """
    0 Args: _         -> (0, 0, 0)
    1 Args: x         -> (x, x, x)
    1 Args: (x,)      -> (x, x, x)
    1 Args: (x, y)    -> (x, y, 0)
    1 Args: (x, y, z) -> (x, y, z)
    2 Args: x, y      -> (x, y, 0)
    2 Args: (x, y), z -> (x, y, z)
    3 Args: x, y, z   -> (x, y, z)

    :param data: The vector data
    :param dtype: The data type
    :return: The vector array
    """
    dlen = len(data)
    if dlen == 0:
        return np.array([0, 0, 0], dtype=dtype)
    if dlen == 1:
        if isinstance(data[0], Iterable):
            return vector3(*data[0], dtype=dtype)
        return np.array([data[0], data[0], data[0]], dtype=dtype)
    if dlen == 2:
        if isinstance(data[0], Iterable):
            return np.array([*vector2(*data[0], dtype=dtype), data[1]], dtype=dtype)
        return np.array([data[0], data[1], 0], dtype=dtype)
    if dlen == 3:
        return np.array(data, dtype=dtype)
    raise TypeError("Invalid Arguments Provided")


def vector4(*data: VectorLike, dtype: Type[DType] = float) -> VectorType:
    """
    0 Args: _              -> (0, 0, 0, 1)
    1 Args: x              -> (x, x, x, 1)
    1 Args: (x,)           -> (x, x, x, 1)
    1 Args: (x, y)         -> (x, y, 0, 1)
    1 Args: (x, y, z)      -> (x, y, z, 1)
    1 Args: (x, y, z, w)   -> (x, y, z, w)
    2 Args: x, y           -> (x, y, 0, 1)
    2 Args: (x, y), z      -> (x, y, z, 1)
    2 Args: (x, y), (z, w) -> (x, y, z, w)
    2 Args: (x, y, z), w   -> (x, y, z, w)
    3 Args: x, y, z        -> (x, y, z, 1)
    3 Args: (x, y), z, w   -> (x, y, z, w)
    4 Args: x, y, z, w     -> (x, y, z, w)

    :param data: The vector data
    :param dtype: The data type
    :return: The vector array
    """
    dlen = len(data)
    if dlen == 0:
        return np.array([0, 0, 0, 1], dtype=dtype)
    if dlen == 1:
        if isinstance(data[0], Iterable):
            return vector4(*data[0], dtype=dtype)
        return np.array([data[0], data[0], data[0], 1], dtype=dtype)
    if dlen == 2:
        if isinstance(data[0], Iterable):
            if len(data[0]) == 3:
                return np.array([*vector3(*data[0], dtype=dtype), data[1]], dtype=dtype)
            if isinstance(data[1], Iterable):
                return np.array(
                    [*vector2(*data[0], dtype=dtype), *vector2(*data[1], dtype=dtype)], dtype=dtype
                )
            return np.array([*vector2(*data[0], dtype=dtype), data[1], 1], dtype=dtype)
        return np.array([data[0], data[1], 0, 1], dtype=dtype)
    if dlen == 3:
        if isinstance(data[0], Iterable):
            return np.array([*vector2(data[0], dtype=dtype), data[1], data[2]], dtype=dtype)
        return np.array([data[0], data[1], data[2], 1], dtype=dtype)
    if dlen == 4:
        return np.array(data, dtype=dtype)
    raise TypeError("Invalid Arguments Provided")


def main():
    v: VectorType

    # ---------- Vector2 ---------- #
    v = vector2(dtype=int)
    assert all(v == (0, 0))

    v = vector2(1, dtype=int)
    assert all(v == (1, 1))

    v = vector2((1,), dtype=int)
    assert all(v == (1, 1))

    v = vector2((1, 2), dtype=int)
    assert all(v == (1, 2))

    v = vector2(1, 2, dtype=int)
    assert all(v == (1, 2))

    # ---------- Vector3 ---------- #
    v = vector3(dtype=int)
    assert all(v == (0, 0, 0))

    v = vector3(1, dtype=int)
    assert all(v == (1, 1, 1))

    v = vector3((1,), dtype=int)
    assert all(v == (1, 1, 1))

    v = vector3((1, 2), dtype=int)
    assert all(v == (1, 2, 0))

    v = vector3((1, 2, 3), dtype=int)
    assert all(v == (1, 2, 3))

    v = vector3(1, 2, dtype=int)
    assert all(v == (1, 2, 0))

    v = vector3((1, 2), 3, dtype=int)
    assert all(v == (1, 2, 3))

    v = vector3(1, 2, 3, dtype=int)
    assert all(v == (1, 2, 3))

    # ---------- Vector4 ---------- #
    v = vector4(dtype=int)
    assert all(v == (0, 0, 0, 1))

    v = vector4(1, dtype=int)
    assert all(v == (1, 1, 1, 1))

    v = vector4((1,), dtype=int)
    assert all(v == (1, 1, 1, 1))

    v = vector4((1, 2), dtype=int)
    assert all(v == (1, 2, 0, 1))

    v = vector4((1, 2, 3), dtype=int)
    assert all(v == (1, 2, 3, 1))

    v = vector4((1, 2, 3, 4), dtype=int)
    assert all(v == (1, 2, 3, 4))

    v = vector4(1, 2, dtype=int)
    assert all(v == (1, 2, 0, 1))

    v = vector4((1, 2), 3, dtype=int)
    assert all(v == (1, 2, 3, 1))

    v = vector4((1, 2), (3, 4), dtype=int)
    assert all(v == (1, 2, 3, 4))

    v = vector4((1, 2, 3), 4, dtype=int)
    assert all(v == (1, 2, 3, 4))

    v = vector4(1, 2, 3, dtype=int)
    assert all(v == (1, 2, 3, 1))

    v = vector4((1, 2), 3, 4, dtype=int)
    assert all(v == (1, 2, 3, 4))

    v = vector4(1, 2, 3, 4, dtype=int)
    assert all(v == (1, 2, 3, 4))


if __name__ == "__main__":
    main()
