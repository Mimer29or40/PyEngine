from __future__ import annotations

import operator
from abc import ABC
from dataclasses import dataclass
from enum import Enum
from typing import Callable
from typing import ClassVar
from typing import Sequence
from typing import Union
from typing import cast

import numpy as np
from OpenGL import GL

DType = Union[int, np.uint8]

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
    NONE: ClassVar[BlendMode]
    ALPHA: ClassVar[BlendMode]
    ADDITIVE: ClassVar[BlendMode]
    MULTIPLICATIVE: ClassVar[BlendMode]
    STENCIL: ClassVar[BlendMode]
    ADD_COLORS: ClassVar[BlendMode]
    SUB_COLORS: ClassVar[BlendMode]
    ILLUMINATE: ClassVar[BlendMode]

    DEFAULT: ClassVar[BlendMode]

    blend_eqn: BlendEqn
    src_func: BlendFunc
    dst_func: BlendFunc

    def blend(self, src: Colorc, dst: Colorc, out: Color) -> Color:
        r_src: int = src.r
        g_src: int = src.g
        b_src: int = src.b
        a_src: int = src.a

        r_dst: int = dst.r
        g_dst: int = dst.g
        b_dst: int = dst.b
        a_dst: int = dst.a

        rsf: int = self.src_func(r_src, a_src, r_dst, a_dst)
        gsf: int = self.src_func(g_src, a_src, g_dst, a_dst)
        bsf: int = self.src_func(b_src, a_src, b_dst, a_dst)
        asf: int = self.src_func(a_src, a_src, a_dst, a_dst)

        rdf: int = self.dst_func(r_src, a_src, r_dst, a_dst)
        gdf: int = self.dst_func(g_src, a_src, g_dst, a_dst)
        bdf: int = self.dst_func(b_src, a_src, b_dst, a_dst)
        adf: int = self.dst_func(a_src, a_src, a_dst, a_dst)

        out[:] = (
            self.blend_eqn(rsf * r_src, rdf * r_dst) / 255,
            self.blend_eqn(gsf * g_src, gdf * g_dst) / 255,
            self.blend_eqn(bsf * b_src, bdf * b_dst) / 255,
            self.blend_eqn(asf * a_src, adf * a_dst) / 255,
        )
        return out


BlendMode.NONE = BlendMode(BlendEqn.ADD, BlendFunc.ZERO, BlendFunc.ONE)  # noqa
BlendMode.ALPHA = BlendMode(BlendEqn.ADD, BlendFunc.SRC_ALPHA, BlendFunc.ONE_MINUS_SRC_ALPHA)
BlendMode.ADDITIVE = BlendMode(BlendEqn.ADD, BlendFunc.SRC_ALPHA, BlendFunc.ONE)
BlendMode.MULTIPLICATIVE = BlendMode(
    BlendEqn.ADD, BlendFunc.DST_COLOR, BlendFunc.ONE_MINUS_SRC_ALPHA
)
BlendMode.STENCIL = BlendMode(BlendEqn.ADD, BlendFunc.ZERO, BlendFunc.SRC_ALPHA)  # noqa
BlendMode.ADD_COLORS = BlendMode(BlendEqn.ADD, BlendFunc.ONE, BlendFunc.ONE)
BlendMode.SUB_COLORS = BlendMode(BlendEqn.SUBTRACT, BlendFunc.ONE, BlendFunc.ONE)
BlendMode.ILLUMINATE = BlendMode(BlendEqn.ADD, BlendFunc.ONE_MINUS_SRC_ALPHA, BlendFunc.SRC_ALPHA)
BlendMode.DEFAULT = BlendMode.ALPHA

# 0.299R + 0.587G + 0.114B
R_TO_GRAY_F: float = 0.299
G_TO_GRAY_F: float = 0.587
B_TO_GRAY_F: float = 0.114

R_TO_GRAY: int = int(round(R_TO_GRAY_F * 255))
G_TO_GRAY: int = int(round(G_TO_GRAY_F * 255))
B_TO_GRAY: int = int(round(B_TO_GRAY_F * 255))


def to_int(x: Union[int, float]) -> int:
    if isinstance(x, float):
        x = int(x * 255)
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


def to_color_tuple(data):
    dlen = len(data)
    if dlen == 0:
        return 0, 0, 0, 255
    if dlen == 1:
        if isinstance(data[0], (np.ndarray, Sequence)):
            return to_color_tuple(data[0])
        return data[0], data[0], data[0], 255
    if dlen == 2:
        return data[0], data[0], data[0], data[1]
    if dlen == 3:
        return data[0], data[1], data[2], 255
    if dlen == 4:
        return data
    raise TypeError("Invalid Arguments Provided")


class Colorc(ABC, np.ndarray):
    pass


class Color(Colorc, np.ndarray):
    def __new__(cls, *data: ColorLike):
        return np.array(to_color_tuple(data), dtype=np.uint8).view(cls)

    def __eq__(self, other: ColorLike) -> bool:
        return np.all(super().__eq__(to_color_tuple(other)))

    def __ne__(self, other: ColorLike) -> bool:
        return not self.__eq__(other)

    @property
    def r(self) -> int:
        return self[0]

    # noinspection PyUnresolvedReferences
    @r.setter
    def r(self, value: int):
        self[0] = to_int(value)

    @property
    def g(self) -> int:
        return self[1]

    # noinspection PyUnresolvedReferences
    @g.setter
    def g(self, value: int):
        self[1] = to_int(value)

    @property
    def b(self) -> int:
        return self[2]

    # noinspection PyUnresolvedReferences
    @b.setter
    def b(self, value: int):
        self[2] = to_int(value)

    @property
    def a(self) -> int:
        return self[3]

    # noinspection PyUnresolvedReferences
    @a.setter
    def a(self, value: int):
        self[3] = to_int(value)

    @property
    def rf(self) -> float:
        return to_float(self.r)

    @property
    def gf(self) -> float:
        return to_float(self.g)

    @property
    def bf(self) -> float:
        return to_float(self.b)

    @property
    def af(self) -> float:
        return to_float(self.a)

    def tint(self, color: Colorc, out: Color = None) -> Color:
        if out is None:
            out = self
        out[:] = (
            to_int(self.r * color.r / 255),
            to_int(self.g * color.g / 255),
            to_int(self.b * color.b / 255),
            to_int(self.a * color.a / 255),
        )
        return out

    def grayscale(self, out: Color = None) -> Color:
        if out is None:
            out = self
        gray = to_gray(self.r, self.g, self.b)
        out[:] = gray, gray, gray, self.a
        return out

    def brightness(self, brightness: Union[int, float], out: Color = None) -> Color:
        if out is None:
            out = self
        if isinstance(brightness, float):
            brightness = int(brightness * 255)
        if brightness < -255:
            brightness = -255
        elif brightness > 255:
            brightness = 255

        out[:] = (
            to_int(self.r + brightness),
            to_int(self.g + brightness),
            to_int(self.b + brightness),
            self.a,
        )
        return out

    def contrast(self, contrast: Union[int, float], out: Color = None) -> Color:
        if out is None:
            out = self
        if isinstance(contrast, float):
            contrast = int(contrast * 255)
        if contrast < -255:
            contrast = -255
        elif contrast > 255:
            contrast = 255

        f: float = (259.0 * (contrast + 255.0)) / (255.0 * (259.0 - contrast))

        out[:] = (
            to_int(f * (self.r - 128) + 128),
            to_int(f * (self.g - 128) + 128),
            to_int(f * (self.b - 128) + 128),
            self.a,
        )
        return out

    def gamma(self, gamma: float, out: Color = None) -> Color:
        if out is None:
            out = self
        gamma = 1.0 / gamma

        out[:] = (
            to_int(self.rf**gamma),
            to_int(self.gf**gamma),
            to_int(self.bf**gamma),
            self.a,
        )
        return out

    def invert(self, out: Color = None) -> Color:
        if out is None:
            out = self
        out[:] = (
            255 - self.r,
            255 - self.g,
            255 - self.b,
            self.a,
        )
        return out

    def brighter(self, percentage: float, out: Color = None) -> Color:
        if out is None:
            out = self
        if percentage < 0.0:
            out[:] = self
            return out
        if percentage > 1.0:
            percentage = 1.0

        # percentage = 1 + percentage * (2 - percentage)  # Quadratic
        percentage = 1 + percentage  # Linear

        out[:] = (
            to_int(self.r * percentage),
            to_int(self.g * percentage),
            to_int(self.b * percentage),
            self.a,
        )
        return out

    def darker(self, percentage: float, out: Color = None) -> Color:
        if out is None:
            out = self

        if percentage < 0.0:
            out[:] = self
            return out
        if percentage > 1.0:
            percentage = 1.0

        # percentage = 1 + percentage * (0.5 * percentage - 1)  # Quadratic
        percentage = 0.5 * (2 - percentage)  # Linear

        out[:] = (
            to_int(self.r * percentage),
            to_int(self.g * percentage),
            to_int(self.b * percentage),
            self.a,
        )
        return out

    def lerp(self, src: Colorc, t: float, out: Color = None) -> Color:
        if out is None:
            out = self

        if t <= 0:
            out[:] = self
            return out
        if t >= 1:
            out[:] = src
            return out

        f: int = int(t * 255)
        f_inv: int = 255 - f

        out[:] = (
            to_int((self.r * f_inv + src.r * f) / 255),
            to_int((self.g * f_inv + src.g * f) / 255),
            to_int((self.b * f_inv + src.b * f) / 255),
            to_int((self.a * f_inv + src.a * f) / 255),
        )
        return out

    def smooth_step(self, src: ColorLike, t: float, out: Color = None) -> Color:
        if out is None:
            out = self

        t2 = t * t
        t3 = t2 * t

        def calc(a, b):
            return (a + a - b - b) * t3 + (3.0 * b - 3.0 * a) * t2 + a * t + a

        out[:] = (
            to_int(calc(self.rf, src.rf)),
            to_int(calc(self.gf, src.gf)),
            to_int(calc(self.bf, src.bf)),
            to_int(calc(self.af, src.af)),
        )
        return out

    def blend(self, src: Colorc, blend_mode: BlendMode = None, out: Color = None) -> Color:
        if blend_mode is None:
            blend_mode = BlendMode.DEFAULT
        if out is None:
            out = self
        return blend_mode.blend(src, cast(Colorc, self), out)


ColorLike = Union[Colorc, np.ndarray, DType, Sequence[DType]]
