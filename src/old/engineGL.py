import time

import pygame
from model import *
from shader import Shader
from util import *
from window import Window

QUIT = "Quit"
ACTIVE_EVENT = "ActiveEvent"

KEY_DOWN = "KeyDown"
KEY_UP = "KeyUp"
KEY_HOLD = "KeyHold"
KEY_PRESSED = "KeyPressed"

MOUSE_BUTTON_DOWN = "MouseButtonDown"
MOUSE_BUTTON_UP = "MouseButtonUp"
MOUSE_BUTTON_HOLD = "MouseButtonHold"
MOUSE_BUTTON_PRESSED = "MouseButtonPressed"
MOUSE_MOTION = "MouseMotion"
MOUSE_DRAGGED = "MouseDragged"
MOUSE_SCROLL = "MouseScroll"

JOY_AXIS_MOTION = "JoyAxisMotion"
JOY_BALL_MOTION = "JoyBallMotion"
JOY_HAT_MOTION = "JoyHatMotion"
JOY_BUTTON_DOWN = "JoyButtonDown"
JOY_BUTTON_UP = "JoyButtonUp"

VIDEO_RESIZE = "VideoResize"
VIDEO_EXPOSE = "VideoExpose"

USER_EVENT = "UserEvent"


class Engine:
    def __init__(self, width, height):
        # pygame Stuff
        pygame.init()

        self.window = Window(width, height)

        self.screen = pygame.display.set_mode(
            self.window.screen_size, pygame.DOUBLEBUF | pygame.OPENGL
        )

        # Engine Stuff
        self._tps_target = None
        self._tps = 0
        self._spt_target = 0

        self._fps_target = None
        self._fps = 0
        self._spf_target = 0

        self._start_time = -1

        self._running = False

        self._event_handlers = {}
        self.game_objects = []

        # Render Stuff
        self.background = Color(0.15, a=1.0)

        self.shader = Shader("vert.glsl", "frag.glsl")

        # Event Stuff
        self._keys = {}
        self._buttons = {}

        self._mouse_pos = Vector([0, 0])
        self._mouse_pos_prev = Vector([0, 0])

        self._scroll_dir = Vector([0, 0])
        self._scroll_dir_prev = Vector([0, 0])

        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_CULL_FACE)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)

    # Engine Stuff
    @property
    def tps(self):
        return self._tps

    @tps.setter
    def tps(self, tps):
        self._tps_target = tps
        self._spt_target = 0 if tps is None or tps < 1 else int(1e9 / tps)

    @property
    def fps(self):
        return self._fps

    @fps.setter
    def fps(self, fps):
        self._fps_target = fps
        self._spf_target = 0 if fps is None or fps < 1 else int(1e9 / fps)

    @property
    def start_time(self):
        return self._start_time

    @property
    def time(self):
        if self._start_time < 0:
            return 0
        return time.perf_counter_ns() - self._start_time

    @property
    def running(self):
        return self._running

    def start(self):
        try:
            self._init()

            self._start_time = time.perf_counter_ns()
            self._running = True

            tick_count, frame_count = 0, 0

            last_tick = self.time
            last_frame = self.time
            last_sec = self.time

            while self._running:
                self._events()

                t = self.time
                dt = t - last_tick
                if dt >= self._spt_target:
                    tick_count += 1
                    last_tick = t

                    t, dt = t / 1e9, dt / 1e9
                    for game_object in self.game_objects:
                        game_object.update(t, dt)

                t = self.time
                dt = t - last_frame
                if dt >= self._spf_target:
                    frame_count += 1
                    last_frame = t

                    self._render(t / 1e9, dt / 1e9)

                t = self.time
                dt = t - last_sec
                if dt >= 1e9:
                    last_sec = t

                    self._tps, self._fps = tick_count, frame_count
                    tick_count, frame_count = 0, 0

        except Exception:
            # print(e)
            raise

        finally:
            self._shutdown()

    def stop(self):
        self._running = False

    def _render(self, t, dt):
        GL.glClearColor(*self.background)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT | GL.GL_STENCIL_BUFFER_BIT)

        # if self.debug:
        # GL.glPolygonMode(
        #     GL.GL_FRONT_AND_BACK,
        #     GL.GL_LINE
        # )

        # self.lights.generateShadows(self.models.draw)

        # Reset Viewpoint
        GL.glViewport(0, 0, *self.window.screen_size)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        self.shader.use()
        self.shader.set_floatm("projection", self.window.projection.data)
        self.shader.set_floatm("view", self.window.view.data)

        # self.shader.setVec3f("viewPos", self.window.position)

        # self.lights.bind(self.shader)

        for game_object in self.game_objects:
            game_object.render(self.shader, t, dt)

        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT | GL.GL_STENCIL_BUFFER_BIT)

        # self.lights.unbind()

        # deltaTimeText = str(self.deltaTime / 1000)
        # self.font.draw(deltaTimeText, 5, 40)

        # for i in range(len(self.drawTextOnScreenMethods)):
        # self.drawTextOnScreenMethods[i](self.font)

        pygame.display.flip()

    def _init(self):
        for handler in self._init_handlers:
            handler()

    def _events(self):
        for e in self._get_events():
            if e.type == QUIT:
                self.stop()
            if e.type == VIDEO_RESIZE:
                self.screen = pygame.display.set_mode((e.w, e.h), self._flags)
            for e_type in (None, e.type):
                if e_type in self._event_handlers.keys():
                    for handler in self._event_handlers[e_type]:
                        handler(e)

    def _update(self, t, dt):
        for handler in self._update_handlers:
            handler(t, dt)

    def _render(self, t, dt):
        for handler in self._render_handlers:
            handler(t, dt)

        pygame.display.flip()

    def _shutdown(self):
        for handler in self._shutdown_handlers:
            handler()

        pygame.quit()

    def on_init(self, func):
        self._init_handlers.add(func)
        return func

    # Event Stuff
    def on_event(self, func=None, *, event=None):
        def decorator(func):
            try:
                self._event_handlers[event].add(func)
            except KeyError:
                self._event_handlers[event] = set()
                self._event_handlers[event].add(func)
            return func

        return decorator if func is None else decorator(func)

    def on_update(self, func):
        self._update_handlers.add(func)
        return func

    def on_render(self, func):
        self._render_handlers.add(func)
        return func

    def on_shutdown(self, func):
        self._shutdown_handlers.add(func)
        return func

    @property
    def mouse_pos(self):
        return self._mouse_pos.copy()

    @property
    def scroll_dir(self):
        return self._scroll_dir.copy()

    def key_down(self, key):
        return key in self._keys.keys()

    def button_down(self, button):
        return button in self._buttons.keys()

    def _get_events(self):
        events = []

        now = time.perf_counter_ns()

        for e in pygame.event.get():
            if e.type == pygame.KEYDOWN:
                k = e.key
                if k not in self._keys:
                    self._keys[k] = _KeyData(now, e.mod, e.unicode, e.scancode)
            elif e.type == pygame.KEYUP:
                k = e.key
                try:
                    self._keys[k].up = True
                except KeyError:
                    pass
            elif e.type == pygame.MOUSEBUTTONDOWN:
                b = e.button
                if b not in self._buttons:
                    self._buttons[b] = _ButtonData(now, e.pos)
            elif e.type == pygame.MOUSEBUTTONUP:
                b = e.button
                try:
                    self._buttons[b].up = True
                except KeyError:
                    pass
            elif e.type == pygame.MOUSEMOTION:
                self._mouse_pos = Vector(e.pos)
            else:
                events.append(_Event(pygame.event.event_name(e.type), now, **e.__dict__))

        for k, d in self._keys.copy().items():
            if d.down:
                events.append(_Event(KEY_DOWN, now, key=k, mod=d.mod))
                self._keys[k].down = False
            else:
                events.append(_Event(KEY_HOLD, now, key=k, mod=d.mod))
            if d.up:
                if now - d.time < 1e9 / 5:
                    events.append(_Event(KEY_PRESSED, now, key=k, mod=d.mod))
                events.append(_Event(KEY_UP, now, key=k, mod=d.mod))
                del self._keys[k]

        for b, d in self._buttons.copy().items():
            if self._mouse_pos != self._mouse_pos_prev:
                events.append(
                    _Event(
                        MOUSE_DRAGGED,
                        now,
                        button=b,
                        pos=self._mouse_pos,
                        rel=self._mouse_pos - self._mouse_pos_prev,
                    )
                )
            if d.down:
                events.append(_Event(MOUSE_BUTTON_DOWN, now, button=b, pos=Vector(d.pos)))
                self._buttons[b].down = False
            else:
                events.append(_Event(MOUSE_BUTTON_HOLD, now, button=b, pos=Vector(d.pos)))
            if d.up:
                if now - d.time < 1e9 / 5:
                    events.append(_Event(MOUSE_BUTTON_PRESSED, now, button=b, pos=Vector(d.pos)))
                events.append(_Event(MOUSE_BUTTON_UP, now, button=b, pos=Vector(d.pos)))
                del self._buttons[b]

        if self._mouse_pos != self._mouse_pos_prev:
            events.append(
                _Event(
                    MOUSE_MOTION,
                    now,
                    pos=self._mouse_pos,
                    rel=self._mouse_pos - self._mouse_pos_prev,
                )
            )
            self._mouse_pos_prev = self._mouse_pos.copy()

        if self._scroll_dir != self._scroll_dir_prev:
            events.append(
                _Event(
                    MOUSE_SCROLL,
                    now,
                    pos=self._scroll_dir,
                    rel=self._scroll_dir - self._scroll_dir_prev,
                )
            )
            self._scroll_dir_prev = self._scroll_dir.copy()

        return events


class GameObject:
    def __init__(self, name):
        self.name = name

        self.position = util.ORIGIN.copy()
        self.rotation = util.IDEN3.copy()
        self.scale = util.ONES.copy()

        self._transformation = util.IDEN4.copy()

        self.model = None

        self.children = []

    def update(self, t, dt):
        self._transformation = util.Matrix(
            [
                [*(self.rotation.data[0] * self.scale.x), 0.0],
                [*(self.rotation.data[1] * self.scale.y), 0.0],
                [*(self.rotation.data[2] * self.scale.z), 0.0],
                [*self.position.data, 1.0],
            ]
        )

        for child in self.children:
            child.update(t, dt)

    def render(self, shader, t, dt):
        if self.model is not None:
            shader.set_floatm("model", self._transformation.data)

            self.model.draw(shader)

        for child in self.children:
            child.render(shader, t, dt)


class _Event:
    def __init__(self, type, time, **kwargs):
        self.type = type
        self.time = time
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __repr__(self):
        a = ["{}={}".format(*k) for k in self.__dict__.items() if k[0] != "type"]
        return "Event.{}<{}>".format(self.type, " ".join(a))


class _KeyData:
    def __init__(self, time, mod, unicode, scancode):
        self.time = time
        self.mod = mod
        self.unicode = unicode
        self.scancode = scancode
        self.down = True
        self.up = False

    def __repr__(self):
        a = ["{}={}".format(*k) for k in self.__dict__.items()]
        return "{}<{}>".format(self.__class__.__name__, " ".join(a))


class _ButtonData:
    def __init__(self, time, pos):
        self.time = time
        self.pos = pos
        self.down = True
        self.up = False

    def __repr__(self):
        a = ["{}={}".format(*k) for k in self.__dict__.items()]
        return "{}<{}>".format(self.__class__.__name__, " ".join(a))
