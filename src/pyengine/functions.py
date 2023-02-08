from typing import Collection
from typing import Iterable
from typing import Sized
from typing import TypeVar
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
DType = Union[int, float, NTypes, np.ndarray]

T = TypeVar("T")
TData = TypeVar("TData", bound=DType)

_map = map


def map_number(x: TData, x0: DType, x1: DType, y0: DType, y1: DType) -> TData:
    return ((x - x0) / (x1 - x0) * (y1 - y0)) + y0


def lerp(x: TData, a: DType, b: DType) -> TData:
    return a + x * (b - a)


def clip(x: TData, a: DType, b: DType) -> TData:
    return np.clip(x, a, b)


def radians(x: TData) -> TData:
    return np.radians(x)


def degrees(x: TData) -> TData:
    return np.degrees(x)


def sin(x: TData) -> TData:
    return np.sin(x)


def cos(x: TData) -> TData:
    return np.cos(x)


def tan(x: TData) -> TData:
    return np.tan(x)


def random_seed(seed: int) -> None:
    np.random.seed(int(seed) & 0xFFFFFFFF)


def random(a: TData = None, b: TData = None) -> TData:
    if a is None:
        a = 1.0
    if b is None:
        a, b = 0.0, a
    try:
        a_len = len(a)
    except TypeError:
        a_len = 1
    try:
        b_len = len(b)
    except TypeError:
        b_len = 1
    order = max(a_len, b_len)
    if order == 1:
        order = None
    return (b - a) * np.random.random_sample(order) + a


def random_int(a: int = None, b: int = None) -> int:
    return int(random(a, b))


def random_bool() -> bool:
    return random() < 0.5


def random_from(array: Collection[T]) -> T:
    return list(array)[random_int(len(array))]


def random_normal(loc: float = 0.0, scale: float = 1.0, size: int = None):
    return np.random.normal(loc, scale, size=size)
