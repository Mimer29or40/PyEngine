from typing import Tuple

import numpy as np

from .color import color
from .state import State
from .vector import VECTOR_X
from .vector import VECTOR_Y
from .vector import VECTOR_Z
from .vector import matrix_rotate
from .vector import matrix_scale
from .vector import matrix_translate

state: State = State()


class Renderer:
    def __init__(self):
        self.view = np.identity(3, dtype=float)

    def get_flags(self):
        return 0

    def set_background(self, color):
        pass

    def setup(self):
        self.set_background(color())

    def before_draw(self):
        pass

    def after_draw(self):
        pass

    def translate(self, v: np.ndarray):
        matrix_translate(self.view, v, out=self.view)

    def rotate(self, theta: float, axis: np.ndarray = None):
        if axis is None:
            matrix_rotate(self.view, VECTOR_Z, theta, out=self.view)
        else:
            matrix_rotate(self.view, axis, theta, out=self.view)

    def rotate_x(self, theta: float):
        matrix_rotate(self.view, VECTOR_X, theta, out=self.view)

    def rotate_y(self, theta: float):
        matrix_rotate(self.view, VECTOR_Y, theta, out=self.view)

    def rotate_z(self, theta: float):
        matrix_rotate(self.view, VECTOR_Z, theta, out=self.view)

    def scale(self, v: np.ndarray):
        matrix_scale(self.view, v, out=self.view)

    def point(self, p):
        pass

    def line(self, p1, p2):
        pass

    def lines(self, *points):
        pass

    def polygon(self, *points):
        pass

    def triangle(self, p1, p2, p3):
        pass

    def quad(self, p1, p2, p3, p4):
        pass

    def ellipse(self, p1, p2):
        pass

    def arc(self, p1, p2, start, stop):
        pass

    def text(self, text, pos):
        pass

    def load_pixels(self):
        size: Tuple[int, int, int] = (state.viewport[0], state.viewport[1], 4)
        arr = np.zeros(size, dtype=np.uint8)
        arr[:, :, 3] = 255
        return arr

    def update_pixels(self):
        pass
