import time

import pygame

from engine2 import util


_keys = {}
_buttons = {}

_pos = util.Vector([0, 0], float)
_pos_prev = util.Vector([0, 0], float)

_scroll = util.Vector([0, 0], float)
_scroll_prev = util.Vector([0, 0], float)


QUIT = 'Quit'
ACTIVE_EVENT = 'ActiveEvent'

KEY_DOWN = 'KeyDown'
KEY_UP = 'KeyUp'
KEY_HOLD = 'KeyHold'
KEY_PRESSED = 'KeyPressed'

MOUSE_BUTTON_DOWN = 'MouseButtonDown'
MOUSE_BUTTON_UP = 'MouseButtonUp'
MOUSE_BUTTON_HOLD = 'MouseButtonHold'
MOUSE_BUTTON_PRESSED = 'MouseButtonPressed'
MOUSE_MOTION = 'MouseMotion'
MOUSE_DRAGGED = 'MouseDragged'
MOUSE_SCROLL = 'MouseScroll'

JOY_AXIS_MOTION = 'JoyAxisMotion'
JOY_BALL_MOTION = 'JoyBallMotion'
JOY_HAT_MOTION = 'JoyHatMotion'
JOY_BUTTON_DOWN = 'JoyButtonDown'
JOY_BUTTON_UP = 'JoyButtonUp'

VIDEO_RESIZE = 'VideoResize'
VIDEO_EXPOSE = 'VideoExpose'

USER_EVENT = 'UserEvent'


def mouse_pos():
    return _pos


def scroll():
    return _scroll


def key_down(key):
    return key in _keys.keys()


def button_down(button):
    return button in _buttons.keys()


def get_events():
    global _keys, _buttons, _pos, _pos_prev, _scroll, _scroll_prev

    events = []

    now = time.perf_counter_ns()

    for e in pygame.event.get():
        if e.type == pygame.KEYDOWN:
            k = e.key
            if k not in _keys:
                _keys[k] = _KeyData(now, e.mod, e.unicode, e.scancode)
        elif e.type == pygame.KEYUP:
            k = e.key
            try:
                _keys[k].up = True
            except KeyError:
                pass
        elif e.type == pygame.MOUSEBUTTONDOWN:
            b = e.button
            if b not in _buttons:
                _buttons[b] = _ButtonData(now, e.pos)
        elif e.type == pygame.MOUSEBUTTONUP:
            b = e.button
            try:
                _buttons[b].up = True
            except KeyError:
                pass
        elif e.type == pygame.MOUSEMOTION:
            _pos = util.Vector(e.pos)
        else:
            events.append(_Event(
                pygame.event.event_name(e.type),
                now,
                **e.__dict__
            ))

    for k, d in _keys.copy().items():
        if d.down:
            events.append(_Event(KEY_DOWN, now, key = k, mod = d.mod))
            _keys[k].down = False
        else:
            events.append(_Event(KEY_HOLD, now, key = k, mod = d.mod))
        if d.up:
            if now - d.time < 1e9 / 5:
                events.append(_Event(KEY_PRESSED, now, key = k, mod = d.mod))
            events.append(_Event(KEY_UP, now, key = k, mod = d.mod))
            del _keys[k]

    for b, d in _buttons.copy().items():
        if _pos != _pos_prev:
            events.append(_Event(
                MOUSE_DRAGGED, now,
                button = b, pos = _pos, rel = _pos - _pos_prev
            ))
        if d.down:
            events.append(_Event(
                MOUSE_BUTTON_DOWN, now,
                button = b, pos = util.Vector(d.pos)
            ))
            _buttons[b].down = False
        else:
            events.append(_Event(
                MOUSE_BUTTON_HOLD, now,
                button = b, pos = util.Vector(d.pos)
            ))
        if d.up:
            if now - d.time < 1e9 / 5:
                events.append(_Event(
                    MOUSE_BUTTON_PRESSED, now,
                    button = b, pos = util.Vector(d.pos)
                ))
            events.append(_Event(
                MOUSE_BUTTON_UP, now,
                button = b, pos = util.Vector(d.pos)
            ))
            del _buttons[b]

    if _pos != _pos_prev:
        events.append(_Event(
            MOUSE_MOTION, now,
            pos = _pos, rel = _pos - _pos_prev
        ))
        _pos_prev = _pos.copy()

    if _scroll != _scroll_prev:
        events.append(_Event(
            MOUSE_SCROLL, now,
            pos = _scroll, rel = _scroll - _scroll_prev
        ))
        _scroll_prev = _scroll.copy()

    return events


class _Event:
    def __init__(self, type, time, **kwargs):
        self.type = type
        self.time = time
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __repr__(self):
        attrs = ['{}={}'.format(k, d) for k, d in self.__dict__.items()]
        return '{}<{}>'.format(self.__class__.__name__, ' '.join(attrs))


class _KeyData:
    def __init__(self, time, mod, unicode, scancode):
        self.time = time
        self.mod = mod
        self.unicode = unicode
        self.scancode = scancode
        self.down = True
        self.up = False

    def __repr__(self):
        attrs = ['{}={}'.format(k, d) for k, d in self.__dict__.items()]
        return '{}<{}>'.format(self.__class__.__name__, ' '.join(attrs))


class _ButtonData:
    def __init__(self, time, pos):
        self.time = time
        self.pos = pos
        self.down = True
        self.up = False

    def __repr__(self):
        attrs = ['{}={}'.format(k, d) for k, d in self.__dict__.items()]
        return '{}<{}>'.format(self.__class__.__name__, ' '.join(attrs))


__all__ = [
    'QUIT',
    'ACTIVE_EVENT',
    'KEY_DOWN',
    'KEY_UP',
    'KEY_HOLD',
    'KEY_PRESSED',
    'MOUSE_BUTTON_DOWN',
    'MOUSE_BUTTON_UP',
    'MOUSE_BUTTON_HOLD',
    'MOUSE_BUTTON_PRESSED',
    'MOUSE_MOTION',
    'MOUSE_DRAGGED',
    'MOUSE_SCROLL',
    'JOY_AXIS_MOTION',
    'JOY_BALL_MOTION',
    'JOY_HAT_MOTION',
    'JOY_BUTTON_DOWN',
    'JOY_BUTTON_UP',
    'VIDEO_RESIZE',
    'VIDEO_EXPOSE',
    'USER_EVENT',
    'mouse_pos',
    'scroll',
    'key_down',
    'button_down',
    'get_events'
]
