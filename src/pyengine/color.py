from __future__ import annotations

import operator
from dataclasses import dataclass
from enum import Enum
from typing import Callable
from typing import Iterable
from typing import Union

import numpy as np
from OpenGL import GL

DType = Union[int, np.uint8]
ColorType = np.ndarray
ColorLike = Union[DType, Iterable[DType]]

IBlendEqn = Callable[[int, int], int]
IBlendFunc = Callable[[int, int, int, int], int]


class BlendEqn(int, Enum):
    def __new__(cls, gl_ref: int, func: IBlendEqn):
        obj = int.__new__(cls, gl_ref)
        obj._value_ = gl_ref
        obj._func = func
        return obj

    ADD = (GL.GL_FUNC_ADD, operator.add)

    SUBTRACT = (GL.GL_FUNC_SUBTRACT, operator.sub)
    REVERSE_SUBTRACT = (GL.GL_FUNC_REVERSE_SUBTRACT, lambda s, d: d - s)

    MIN = (GL.GL_MIN, min)
    MAX = (GL.GL_MAX, max)

    def __call__(self, s: int, d: int) -> int:
        return self._func(s, d)


class BlendFunc(int, Enum):
    def __new__(cls, gl_ref: int, func: IBlendFunc):
        obj = int.__new__(cls, gl_ref)
        obj._value_ = gl_ref
        obj._func = func
        return obj

    ZERO = (GL.GL_ZERO, lambda c_src, a_src, c_dst, a_dst: 0)
    ONE = (GL.GL_ONE, lambda c_src, a_src, c_dst, a_dst: 255)

    SRC_COLOR = (GL.GL_SRC_COLOR, lambda c_src, a_src, c_dst, a_dst: c_src)
    ONE_MINUS_SRC_COLOR = (
        GL.GL_ONE_MINUS_SRC_COLOR,
        lambda c_src, a_src, c_dst, a_dst: 255 - c_src,
    )
    SRC_ALPHA = (GL.GL_SRC_ALPHA, lambda c_src, a_src, c_dst, a_dst: a_src)
    ONE_MINUS_SRC_ALPHA = (
        GL.GL_ONE_MINUS_SRC_ALPHA,
        lambda c_src, a_src, c_dst, a_dst: 255 - a_src,
    )

    DST_COLOR = (GL.GL_DST_COLOR, lambda c_src, a_src, c_dst, a_dst: c_dst)
    ONE_MINUS_DST_COLOR = (
        GL.GL_ONE_MINUS_DST_COLOR,
        lambda c_src, a_src, c_dst, a_dst: 255 - c_dst,
    )
    DST_ALPHA = (GL.GL_DST_ALPHA, lambda c_src, a_src, c_dst, a_dst: a_dst)
    ONE_MINUS_DST_ALPHA = (
        GL.GL_ONE_MINUS_DST_ALPHA,
        lambda c_src, a_src, c_dst, a_dst: 255 - a_dst,
    )

    def __call__(self, c_src: int, a_src: int, c_dst: int, a_dst: int) -> int:
        return self._func(c_src, a_src, c_dst, a_dst)


@dataclass(frozen=True)
class BlendMode:
    blend_eqn: BlendEqn
    src_func: BlendFunc
    dst_func: BlendFunc


BLEND_NONE = BlendMode(BlendEqn.ADD, BlendFunc.ZERO, BlendFunc.ONE)
BLEND_ALPHA = BlendMode(BlendEqn.ADD, BlendFunc.SRC_ALPHA, BlendFunc.ONE_MINUS_SRC_ALPHA)
BLEND_ADDITIVE = BlendMode(BlendEqn.ADD, BlendFunc.SRC_ALPHA, BlendFunc.ONE)
BLEND_MULTIPLICATIVE = BlendMode(BlendEqn.ADD, BlendFunc.DST_COLOR, BlendFunc.ONE_MINUS_SRC_ALPHA)
BLEND_STENCIL = BlendMode(BlendEqn.ADD, BlendFunc.ZERO, BlendFunc.SRC_ALPHA)
BLEND_ADD_COLORS = BlendMode(BlendEqn.ADD, BlendFunc.ONE, BlendFunc.ONE)
BLEND_SUB_COLORS = BlendMode(BlendEqn.SUBTRACT, BlendFunc.ONE, BlendFunc.ONE)
BLEND_ILLUMINATE = BlendMode(BlendEqn.ADD, BlendFunc.ONE_MINUS_SRC_ALPHA, BlendFunc.SRC_ALPHA)
BLEND_DEFAULT = BLEND_ALPHA

# 0.299R + 0.587G + 0.114B
R_TO_GRAY_F: float = 0.299
G_TO_GRAY_F: float = 0.587
B_TO_GRAY_F: float = 0.114

R_TO_GRAY: int = int(round(R_TO_GRAY_F * 255))
G_TO_GRAY: int = int(round(G_TO_GRAY_F * 255))
B_TO_GRAY: int = int(round(B_TO_GRAY_F * 255))


def to_int(x: Union[int, float]) -> int:
    if isinstance(x, float):
        x = int(round(x * 255))
    if x < 0:
        return 0
    if x > 255:
        return 255
    return x


def to_float(x: Union[int, float]) -> float:
    if isinstance(x, int):
        x = float(x) / 255.0
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def to_gray(r: Union[int, float], g: Union[int, float], b: Union[int, float]) -> int:
    def m(x: Union[int, float], int_value: int, float_value: float):
        if isinstance(x, float):
            return to_int(x * float_value)
        return r * int_value / 255

    return (
        m(r, R_TO_GRAY, R_TO_GRAY_F) + m(g, G_TO_GRAY, G_TO_GRAY_F) + m(b, B_TO_GRAY, B_TO_GRAY_F)
    ) & 0xFF


def color(*data: DType) -> ColorType:
    dlen = len(data)
    if dlen == 0:
        return np.array([0, 0, 0, 255], dtype=np.uint8)
    if dlen == 1:
        if isinstance(data[0], Iterable):
            return color(data[0])
        return np.array([data[0], data[0], data[0], 255], dtype=np.uint8)
    if dlen == 2:
        return np.array([data[0], data[0], data[0], data[1]], dtype=np.uint8)
    if dlen == 3:
        return np.array([data[0], data[1], data[2], 255], dtype=np.uint8)
    if dlen == 4:
        return np.array(data, dtype=np.uint8)
    raise TypeError("Invalid Arguments Provided")


def colorf(*data: DType) -> ColorType:
    return color(*map(to_int, data))


def color_tint(c1: ColorType, c2: ColorType, *, out: ColorType = None) -> ColorType:
    if out is None:
        out = np.array(c1)
    out[:] = (
        to_int(c1[0] * c2[0] / 255),
        to_int(c1[1] * c2[1] / 255),
        to_int(c1[2] * c2[2] / 255),
        to_int(c1[3] * c2[3] / 255),
    )
    return out


def color_grayscale(c: ColorType, *, out: ColorType = None) -> ColorType:
    if out is None:
        out = np.array(c)

    gray = to_gray(c[0], c[1], c[2])

    out[:] = gray, gray, gray, c[3]
    return out


def color_brightness(c: ColorType, brightness: int, *, out: ColorType = None) -> ColorType:
    if out is None:
        out = np.array(c)

    if brightness < -255:
        brightness = -255
    elif brightness > 255:
        brightness = 255

    out[:] = to_int(c[0] + brightness), to_int(c[1] + brightness), to_int(c[2] + brightness), c[3]
    return out


def color_contrast(c: ColorType, contrast: int, *, out: ColorType = None) -> ColorType:
    if out is None:
        out = np.array(c)

    if contrast < -255:
        contrast = -255
    elif contrast > 255:
        contrast = 255

    f: float = (259.0 * (contrast + 255.0)) / (255.0 * (259.0 - contrast))

    out[:] = (
        to_int(f * (c[0] - 128) + 128),
        to_int(f * (c[1] - 128) + 128),
        to_int(f * (c[2] - 128) + 128),
        c[3],
    )
    return out


def color_gamma(c: ColorType, gamma: float, *, out: ColorType = None) -> ColorType:
    if out is None:
        out = np.array(c)

    gamma = 1.0 / gamma

    out[:] = (
        to_int(to_float(c[0]) ** gamma),
        to_int(to_float(c[1]) ** gamma),
        to_int(to_float(c[2]) ** gamma),
        c[3],
    )
    return out


def color_invert(c: ColorType, *, out: ColorType = None) -> ColorType:
    if out is None:
        out = np.array(c)

    out[:] = 255 - c[0], 255 - c[1], 255 - c[2], c[3]
    return out


def color_brighter(c: ColorType, percentage: float, *, out: ColorType = None) -> ColorType:
    if out is None:
        out = np.array(c)

    if percentage < 0.0:
        out[:] = c
        return out
    if percentage > 1.0:
        percentage = 1.0

    # percentage = 1 + percentage * (2 - percentage)  # Quadratic
    percentage = 1 + percentage  # Linear

    out[:] = (
        to_int(c[0] * percentage),
        to_int(c[1] * percentage),
        to_int(c[2] * percentage),
        c[3],
    )
    return out


def color_darker(c: ColorType, percentage: float, *, out: ColorType = None) -> ColorType:
    if out is None:
        out = np.array(c)

    if percentage < 0.0:
        out[:] = c
        return out
    if percentage > 1.0:
        percentage = 1.0

    # percentage = 1 + percentage * (0.5 * percentage - 1)  # Quadratic
    percentage = 0.5 * (2 - percentage)  # Linear

    out[:] = (
        to_int(c[0] * percentage),
        to_int(c[1] * percentage),
        to_int(c[2] * percentage),
        c[3],
    )
    return out


def color_lerp(c1: ColorType, c2: ColorType, t: float, *, out: ColorType = None) -> ColorType:
    if out is None:
        out = np.array(c1)

    if t <= 0:
        out[:] = c1
        return out
    if t >= 1:
        out[:] = c2
        return out

    f: int = int(t * 255)
    f_inv: int = 255 - f

    out[:] = (
        to_int((c1[0] * f_inv + c2[0] * f) / 255),
        to_int((c1[1] * f_inv + c2[1] * f) / 255),
        to_int((c1[2] * f_inv + c2[2] * f) / 255),
        to_int((c1[3] * f_inv + c2[3] * f) / 255),
    )
    return out


def color_blend(
    src: ColorType, dst: ColorType, blend_mode: BlendMode = None, *, out: ColorType = None
) -> ColorType:
    if out is None:
        out = np.array(src)

    if blend_mode is None:
        blend_mode = BLEND_DEFAULT

    rsf: int = blend_mode.src_func(src[0], src[3], dst[0], dst[3])
    gsf: int = blend_mode.src_func(src[1], src[3], dst[1], dst[3])
    bsf: int = blend_mode.src_func(src[2], src[3], dst[2], dst[3])
    asf: int = blend_mode.src_func(src[3], src[3], dst[3], dst[3])

    rdf: int = blend_mode.dst_func(src[0], src[3], dst[0], dst[3])
    gdf: int = blend_mode.dst_func(src[1], src[3], dst[1], dst[3])
    bdf: int = blend_mode.dst_func(src[2], src[3], dst[2], dst[3])
    adf: int = blend_mode.dst_func(src[3], src[3], dst[3], dst[3])

    out[:] = (
        blend_mode.blend_eqn(rsf * src[0], rdf * dst[0]) / 255,
        blend_mode.blend_eqn(gsf * src[1], gdf * dst[1]) / 255,
        blend_mode.blend_eqn(bsf * src[2], bdf * dst[2]) / 255,
        blend_mode.blend_eqn(asf * src[3], adf * dst[3]) / 255,
    )
    return out
