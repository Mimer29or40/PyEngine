import time

import numpy as np
import pygame

from engine import *
from engine.event import Event
from engine.render import get_renderer

pygame.init()


class Engine:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    _frame_rate = 0
    _frame_rate_inv = 16_666_666

    @property
    def frame_rate(self):
        return self._frame_rate

    @frame_rate.setter
    def frame_rate(self, value):
        self._frame_rate_inv = 0 if value < 1 else int(1_000_000_000 / value)

    _frame_count = 0

    @property
    def frame_count(self):
        return self._frame_count

    _start_time = 0

    @property
    def time(self):
        if self._start_time > 0:
            return time.perf_counter_ns() - self._start_time
        return -1

    _viewport: Vector = Vector(0, 0).astype(int)

    @property
    def viewport(self):
        return self._viewport.xy

    @property
    def width(self):
        return self._viewport.x

    @property
    def height(self):
        return self._viewport.y

    _mouse = Vector(0, 0)

    @property
    def mouse(self):
        self._mouse.immutable = True
        return self._mouse.xy

    _is_drawing = False
    _pmouse_event, _pmouse_draw = Vector(0, 0), Vector(0, 0)

    @property
    def pmouse(self):
        if self._is_drawing:
            self._pmouse_draw.immutable = True
            return self._pmouse_draw.xy
        self._pmouse_event.immutable = True
        return self._pmouse_event.xy

    _mouse_button = 0

    @property
    def mouse_button(self):
        return self._mouse_button

    _mouse_pressed = False

    @property
    def mouse_pressed(self):
        return self._mouse_pressed

    _key = ""

    @property
    def key(self):
        return self._key

    _key_code = ""

    @property
    def key_code(self):
        return self._key_code

    _key_pressed = False

    @property
    def key_pressed(self):
        return self._key_pressed

    _background, _backgrounds = Color(0), []

    @property
    def background(self):
        self._background.immutable = True
        return self._background

    @background.setter
    def background(self, *value):
        self._background.immutable = False
        self._background.set(*value)

        self._renderer.set_background(self._background)

    _fill, _fills = Color(255), []

    @property
    def fill(self):
        self._fill.immutable = True
        return self._fill

    @fill.setter
    def fill(self, *value):
        self._fill.immutable = False
        self._fill.set(*value)

    _stroke, _strokes = Color(51), []

    @property
    def stroke(self):
        self._stroke.immutable = True
        return self._stroke

    @stroke.setter
    def stroke(self, *value):
        self._stroke.immutable = False
        self._stroke.set(*value)

    _weight, _weights = 5, []

    @property
    def weight(self):
        return self._weight

    @weight.setter
    def weight(self, value):
        self._weight = max(1, int(value))

    _rect_mode, _rect_modes = CORNER, []

    @property
    def rect_mode(self):
        return self._rect_mode

    @rect_mode.setter
    def rect_mode(self, value):
        if value in RECT_MODES:
            self._rect_mode = value

    _ellipse_mode, _ellipse_modes = CENTER, []

    @property
    def ellipse_mode(self):
        return self._ellipse_mode

    @ellipse_mode.setter
    def ellipse_mode(self, value):
        if value in ELLIPSE_MODES:
            self._ellipse_mode = value

    _arc_mode, _arc_modes = OPEN, []

    @property
    def arc_mode(self):
        return self._arc_mode

    @arc_mode.setter
    def arc_mode(self, value):
        if value in ARC_MODES:
            self._arc_mode = value

    _font = pygame.font.SysFont("", 12)
    _text_size = 12

    @property
    def text_size(self):
        return self._text_size

    @text_size.setter
    def text_size(self, value):
        self._text_size = int(np.clip(value, 1, 1000))
        self._text_leading = self._text_size
        self._font = pygame.font.SysFont("", value)

    _text_align = LEFT, TOP

    @property
    def text_align(self):
        return self._text_align

    @text_align.setter
    def text_align(self, h, v=TOP):
        v1 = h if h in TEXT_ALIGN_H else LEFT
        v2 = v if v in TEXT_ALIGN_V else TOP
        self._text_align = v1, v2

    _text_leading = _text_size

    @property
    def text_leading(self):
        return self._text_leading

    @text_leading.setter
    def text_leading(self, value):
        self._text_leading = value

    _renderer = None

    def size(self, width, height, renderer=PYGAME):
        if pygame.display.get_surface() is not None:
            raise Exception("size can only be called once")

        self._renderer = get_renderer(renderer)

        self._viewport.xy = max(100, width), max(100, height)
        self._viewport.immutable = True

        flags = self._renderer.get_flags()

        pygame.display.set_mode(self._viewport.xy.astype(int), flags)
        pygame.display.flip()

    def start(self):
        try:
            self._setup()

            frame_time = 0
            frame = sec = self.time
            frame_count = 0
            while self._start_time > 0:
                self._events()

                t = self.time
                dt = t - frame
                if dt >= self._frame_rate_inv:
                    frame = t
                    self._frame_count += 1

                    self._draw()

                    frame_time = self.time - t

                t = self.time
                dt = t - sec
                if dt >= 1_000_000_000:
                    sec = t
                    self._frame_rate = self._frame_count - frame_count
                    frame_count = self._frame_count

                    frame_time = int(frame_time / 1_000)

                    title = f"FPS({self._frame_rate}) SPF({frame_time} us)"
                    pygame.display.set_caption(title)
        except Exception:
            raise
        finally:
            self._shutdown()

    def stop(self):
        self._start_time = 0

    def _setup(self):
        self._start_time = time.perf_counter_ns()

        self._renderer.setup(self)

    _focused = True
    _modifiers, _keys, _buttons = 0, {}, {}

    def _events(self):
        self._key = ""
        self._key_code = ""
        self._key_pressed = False
        self._mouse_button = 0
        self._mouse_pressed = False

        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                self.stop()

            elif e.type == pygame.ACTIVEEVENT:
                if e.state == 6:
                    self._focused = True
                elif e.state == 2:
                    self._focused = False

            elif e.type == pygame.KEYDOWN and self._focused:
                if e.key in MOD_DICT:
                    self._modifiers |= MOD_DICT[e.key]

                if e.key not in self._keys:
                    self._keys[e.key] = self.time, e.unicode
                    self._key = e.unicode
                    self._key_code = pygame.key.name(e.key)
                    self._key_pressed = True

                    Event(self, "key_pressed", key=e.unicode, key_code=pygame.key.name(e.key))

            elif e.type == pygame.KEYUP and self._focused:
                if e.key in MOD_DICT:
                    mod = MOD_DICT[e.key]
                    self._modifiers -= mod if self._modifiers & mod else 0

                if e.key in self._keys:
                    t, code = self._keys[e.key]
                    if self.time - t < 200_000_000:
                        Event(self, "key_typed", key=code, key_code=pygame.key.name(e.key))
                    Event(self, "key_released", key=code, key_code=pygame.key.name(e.key))
                    del self._keys[e.key]

            elif e.type == pygame.MOUSEBUTTONDOWN and self._focused:
                if e.button not in self._buttons and e.button in MOUSE_BUTTONS:
                    self._buttons[e.button] = time.perf_counter_ns()
                    self._mouse_button = e.button
                    self._mouse_pressed = True

                    Event(self, "mouse_pressed", pos=Vector(*e.pos), button=e.button)
                elif e.button in MOUSE_WHEEL_BUTTONS:
                    Event(self, "mouse_wheel", pos=Vector(*e.pos), dir=1 if e.button == UP else -1)

            elif e.type == pygame.MOUSEBUTTONUP and self._focused:
                if e.button in self._buttons:
                    t = self._buttons[e.button]
                    if time.perf_counter_ns() - t < 200_000_000:
                        Event(self, "mouse_clicked", pos=Vector(*e.pos), button=e.button)
                    Event(self, "mouse_released", pos=Vector(*e.pos), button=e.button)
                    del self._buttons[e.button]

            elif e.type == pygame.MOUSEMOTION and self._focused:
                self._pmouse_event.immutable = False
                self._mouse.immutable = False

                self._pmouse_event.xy = self._mouse.xy
                self._mouse.xy = e.pos

                self._pmouse_event.immutable = True
                self._mouse.immutable = True

                if len(self._buttons) > 0:
                    for b in self._buttons:
                        Event(
                            self,
                            "mouse_dragged",
                            pos=self._mouse.xy,
                            rel=(self._mouse - self._pmouse_event).xy,
                            button=b,
                            count=0,
                        )
                Event(
                    self,
                    "mouse_moved",
                    pos=self._mouse.xy,
                    rel=(self._mouse - self._pmouse_event).xy,
                    button=None,
                    count=0,
                )

        for key, value in self._keys.items():
            self._key = value[1]
            self._key_code = pygame.key.name(key)
            self._key_pressed = True

            Event(self, "key_held", key=value[1], key_code=pygame.key.name(key))

        for button, value in self._buttons.items():
            self._mouse_button = button
            self._mouse_pressed = True

            Event(self, "mouse_held", pos=self._mouse.xy, button=button, count=1)

    def _draw(self):
        self._is_drawing = True

        # self._background.immutable = False
        # self._background.set(0)
        self._backgrounds = []

        self._fill.immutable = False
        self._fill.set(255)
        self._fills = []

        self._stroke.immutable = False
        self._stroke.set(51)
        self._strokes = []

        self._weight = 1
        self._weights = []

        self._rect_mode = CORNER
        self._rect_modes = []

        self._ellipse_mode = CENTER
        self._ellipse_modes = []

        self._arc_mode = OPEN
        self._arc_modes = []

        self.pixels = None

        self._renderer.before_draw(self)

        self._draw_func()

        self._renderer.after_draw(self)

        pygame.display.flip()

        self._pmouse_draw.immutable = False
        self._pmouse_draw.xy = self._mouse.xy

        self._is_drawing = False

    def _shutdown(self):
        pygame.quit()

        self._shutdown_func()

    _event_dict = {
        "mouse_clicked": [],
        "mouse_dragged": [],
        "mouse_moved": [],
        "mouse_pressed": [],
        "mouse_released": [],
        "mouse_held": [],
        "mouse_wheel": [],
        "key_pressed": [],
        "key_released": [],
        "key_held": [],
        "key_typed": [],
    }

    def event(self, func=None, **kwargs):
        def decorator(func):
            self._event_dict[func.__name__].append((func, kwargs))

        return decorator if func is None else decorator(func)

    def _draw_func(self):
        pass

    def draw(self, func):
        self._draw_func = func

    def _shutdown_func(self):
        pass

    def shutdown(self, func):
        self._shutdown_func = func

    _views = []

    def push(self):
        self._backgrounds.append(self._background.copy())
        self._fills.append(self._fill.copy())
        self._strokes.append(self._stroke.copy())
        self._weights.append(self._weight)

        self._views.append(self._renderer.view.copy())

        self._rect_modes.append(self._rect_mode)
        self._ellipse_modes.append(self._ellipse_mode)
        self._arc_modes.append(self._arc_mode)

    def pop(self):
        self._background = self._backgrounds.pop(-1)
        self._fill = self._fills.pop(-1)
        self._stroke = self._strokes.pop(-1)
        self._weight = self._weights.pop(-1)

        self._renderer.view[:] = self._views.pop(-1)

        self._rect_mode = self._rect_modes.pop(-1)
        self._ellipse_mode = self._ellipse_modes.pop(-1)
        self._arc_mode = self._arc_modes.pop(-1)

    def translate(self, *amount):
        if len(amount) == 1:
            self._renderer.translate(amount[0])
        elif len(amount) > 1:
            self._renderer.translate(Vector(*amount))
        else:
            raise TypeError

    def rotate(self, angle, axis=None):
        self._renderer.rotate(angle, axis)

    def rotate_x(self, angle):
        self._renderer.rotate_x(angle)

    def rotate_y(self, angle):
        self._renderer.rotate_y(angle)

    def rotate_z(self, angle):
        self._renderer.rotate_z(angle)

    def scale(self, *amount):
        if len(amount) == 1:
            self._renderer.scale(amount[0])
        elif len(amount) > 1:
            self._renderer.scale(Vector(*amount))
        else:
            raise TypeError

    def point(self, p):
        self._renderer.point(self, p)

    def line(self, p1, p2):
        self._renderer.line(self, p1, p2)

    def lines(self, *points):
        self._renderer.lines(self, *points)

    def polygon(self, *points):
        self._renderer.polygon(self, *points)

    def triangle(self, p1, p2, p3):
        self._renderer.triangle(self, p1, p2, p3)

    def quad(self, p1, p2, p3, p4):
        self._renderer.quad(self, p1, p2, p3, p4)

    def rect(self, p1, p2, *r):
        corner, size = p1.base, p2.base

        if self._rect_mode == CENTER:
            corner.xyz -= size.xyz * 0.5
        elif self._rect_mode == RADIUS:
            corner.xyz -= size.xyz
            size.xyz *= 2
        elif self._rect_mode == CORNER:
            pass
        elif self._rect_mode == CORNERS:
            size.xyz -= corner.xyz

        self.quad(
            corner,
            Vector(corner.x + size.x, corner.y, corner.z),
            Vector(corner.x + size.x, corner.y + size.y, corner.z),
            Vector(corner.x, corner.y + size.y, corner.z),
        )

    def square(self, pos, side):
        self.rect(pos, Vector(side, side))

    def ellipse(self, p1, p2):
        self._renderer.ellipse(self, p1, p2)

    def circle(self, p1, extent):
        self._renderer.ellipse(self, p1, Vector(extent, extent, 0))

    def arc(self, p1, p2, start, stop):
        pass

    def text(self, text, pos):
        self._renderer.text(self, text, pos)

    pixels = None

    def load_pixels(self):
        self.pixels = self._renderer.load_pixels(self)

    def update_pixels(self):
        self._renderer.update_pixels(self)
