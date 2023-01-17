import os

import pygame
import numpy as np


_next_id = 1


def _add_constants(constants):
    global _next_id
    out = []
    for c in constants.split(','):
        try:
            value = globals()[c]
        except KeyError:
            value = _Constant(c, _next_id)
            _next_id += 1
        out.append(value)
    return tuple(out)


class _Constant(int):
    def __new__(cls, name, id):
        constant = super().__new__(cls, id)
        constant._name = name
        return constant
    
    def __repr__(self):
        return '{} {}'.format(self._name, super().__repr__())
    
    def __str__(self):
        return self._name


def _from_env(*names):
    for name in names:
        try:
            return os.environ[name] is not None
        except KeyError:
            pass
    return False


MOUSE_BUTTONS = (
    LEFT, CENTER, RIGHT
) = _add_constants('LEFT,CENTER,RIGHT')

MOUSE_WHEEL_BUTTONS = (
    UP, DOWN
) = _add_constants('UP,DOWN')

RECT_MODES = (
    CORNER, CORNERS, RADIUS, CENTER
) = _add_constants('CORNER,CORNERS,RADIUS,CENTER')

ELLIPSE_MODES = (
    RADIUS, CENTER, CORNER, CORNERS
) = _add_constants('RADIUS,CENTER,CORNER,CORNERS')

ARC_MODES = (
    OPEN, CHORD, PIE
) = _add_constants('OPEN,CHORD,PIE')

TEXT_ALIGN_H = (
    LEFT, CENTER, RIGHT
) = _add_constants('LEFT,CENTER,RIGHT')

TEXT_ALIGN_V = (
    TOP, CENTER, BOTTOM
) = _add_constants('TOP,CENTER,BOTTOM')

PROJECTIONS = (
    ORTHOGRAPHIC, PERSPECTIVE
) = _add_constants('ORTHOGRAPHIC,PERSPECTIVE')

RENDERERS = (
    PYGAME1, OPENGL1, PYGAME2, OPENGL2, PYGAME3, OPENGL3
) = _add_constants('PYGAME1,OPENGL1,PYGAME2,OPENGL2,PYGAME3,OPENGL3')
PYGAME, OPENGL = PYGAME3, OPENGL3

MOD_SHIFT, MOD_CTRL, MOD_ALT, MOD_META = 1, 2, 4, 8

MOD_DICT = {
    pygame.K_RSHIFT: MOD_SHIFT,
    pygame.K_LSHIFT: MOD_SHIFT,
    pygame.K_RCTRL: MOD_CTRL,
    pygame.K_LCTRL: MOD_CTRL,
    pygame.K_RALT: MOD_ALT,
    pygame.K_LALT: MOD_ALT,
    pygame.K_RMETA: MOD_META,
    pygame.K_LMETA: MOD_META,
    pygame.K_RSUPER: MOD_META,
    pygame.K_LSUPER: MOD_META
}

PI = np.pi
TWO_PI = np.pi * 2.0

DEBUG_EVENTS = _from_env('ENGINE_DEBUG', 'ENGINE_DEBUG_EVENTS')
DEBUG_RENDER = _from_env('ENGINE_DEBUG', 'ENGINE_DEBUG_RENDER')
