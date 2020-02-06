from typing import Union

import numpy as np

from engine.vector import Vector

pymap = map

Number = Union[int, float]


def map(x, x_min, x_max, y_min, y_max):
    return ((x - x_min) / (x_max - x_min) * (y_max - y_min)) + y_min


def lerp(a, b, x):
    return a + x * (b - a)


def constrain(x, a, b):
    return np.clip(x, a, b)


def radians(x):
    return np.radians(x)


def degrees(x):
    return np.degrees(x)


def sin(x):
    return np.sin(x)


def cos(x):
    return np.cos(x)


def tan(x):
    return np.tan(x)


def random_seed(seed):
    np.random.seed(int(seed) & 0xFFFFFFFF)


def random(lower=None, upper=None):
    if lower is None:
        lower = 1.
    if upper is None:
        lower, upper = 0., lower
    try:
        lower_len = len(lower)
    except TypeError:
        lower_len = 0
    try:
        upper_len = len(upper)
    except TypeError:
        upper_len = 0
    order = max(lower_len, upper_len)
    upper = np.array(upper) - lower
    if order > 0:
        return Vector(*(lower + np.random.random((order,)) * upper))
    return lower + np.random.random() * upper


def random_int(lower=None, upper=None):
    return int(random(lower, upper))


def random_bool():
    return random() < 0.5


def random_index(array):
    return list(array)[int(random(len(array)))]


def random_gaussian(size=None):
    return np.random.normal(size=size)


def noise_seed(seed):
    pass


def noise(coord):
    pass


def noise_detail(lod, falloff=0.5):
    pass
