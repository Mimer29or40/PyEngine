from __future__ import annotations

from typing import Union

import numpy as np
import pygame

from .color import ColorLike
from .color import ColorType
from .color import color as _color
from .io import IO
from .vector import VECTOR_X
from .vector import VECTOR_Y
from .vector import VECTOR_Z
from .vector import DType
from .vector import VectorType
from .vector import matrix4
from .vector import matrix_rotate
from .vector import matrix_scale
from .vector import matrix_translate
from .vector import vector3
from .vector import vector4


class Draw:
    view: VectorType = matrix4(dtype=float)
    view_stack: list[VectorType] = []

    color: ColorType = _color(255, 255, 255, 255)
    color_stack: list[ColorType] = []

    thickness: float = 12
    thickness_stack: list[float] = []


def draw_push() -> None:
    Draw.view_stack.append(Draw.view.copy())
    Draw.color_stack.append(Draw.color.copy())
    Draw.thickness_stack.append(Draw.thickness)


def draw_pop() -> None:
    Draw.view = Draw.view_stack.pop(-1)
    Draw.color = Draw.color_stack.pop(-1)
    Draw.thickness = Draw.thickness_stack.pop(-1)


def draw_translate(*v: Union[DType, VectorType]) -> None:
    matrix_translate(Draw.view, vector3(*v), out=Draw.view)


def draw_rotate(theta: float, *axis: Union[DType, VectorType]):
    if len(axis) == 0:
        matrix_rotate(Draw.view, VECTOR_Z, theta, out=Draw.view)
    else:
        matrix_rotate(Draw.view, vector3(*axis), theta, out=Draw.view)


def draw_rotate_x(theta: float):
    matrix_rotate(Draw.view, VECTOR_X, theta, out=Draw.view)


def draw_rotate_y(theta: float):
    matrix_rotate(Draw.view, VECTOR_Y, theta, out=Draw.view)


def draw_rotate_z(theta: float):
    matrix_rotate(Draw.view, VECTOR_Z, theta, out=Draw.view)


def draw_scale(v: VectorType):
    matrix_scale(Draw.view, v, out=Draw.view)


def draw_color(color: ColorType = _color(0, 0, 0, 155)) -> None:
    Draw.color = color


def draw_thickness(thickness: float) -> None:
    Draw.thickness = thickness


def draw_clear(*data: ColorLike) -> None:
    IO.window.fill(_color(*data))


def draw_point(pos: VectorType):
    pos = (vector4(*pos) @ Draw.view).astype(int)[0:2]
    pygame.draw.circle(IO.window, Draw.color, pos, Draw.thickness / 2)


def draw_line(pos0: VectorType, pos1: VectorType):
    pos0 = (vector4(*pos0) @ Draw.view).astype(int)[0:2]
    pos1 = (vector4(*pos1) @ Draw.view).astype(int)[0:2]
    pygame.draw.line(IO.window, Draw.color, pos0, pos1, int(Draw.thickness))


def draw_lines(*points: VectorType):
    points = tuple(map(lambda p: (vector4(*p) @ Draw.view).astype(int)[0:2], points))
    pygame.draw.lines(IO.window, Draw.color, False, points, int(Draw.thickness))


def draw_polygon(*points: VectorType):
    points = tuple(map(lambda p: (vector4(*p) @ Draw.view).astype(int)[0:2], points))
    pygame.draw.polygon(IO.window, Draw.color, points, int(Draw.thickness))


def fill_polygon(*points: VectorType):
    points = tuple(map(lambda p: (vector4(*p) @ Draw.view).astype(int)[0:2], points))
    pygame.draw.polygon(IO.window, Draw.color, points, 0)


def draw_triangle(p1, p2, p3):
    pass


def draw_quad(p1, p2, p3, p4):
    pass


def draw_ellipse(p1, p2):
    pass


def draw_arc(p1, p2, start, stop):
    pass


def draw_text(text, pos):
    pass


def draw_load_pixels(self):
    arr = np.zeros((*IO.window_size, 4), dtype=np.uint8)
    arr[:, :, 3] = 255
    return arr


def draw_update_pixels(self):
    pass
