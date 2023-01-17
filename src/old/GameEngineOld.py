import time
from dataclasses import dataclass

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


class Event:
    def __init__(self, type, time, **kwargs):
        self.type = type
        self.time = time
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __repr__(self):
        attrs = ["{}={}".format(k, d) for k, d in self.__dict__.items()]
        return "{}<{}>".format(self.__class__.__name__, " ".join(attrs))


class Input:
    @dataclass
    class _KeyData:
        time: int
        mod: int
        unicode: str
        scancode: int
        down: bool
        up: bool

    @dataclass
    class _ButtonData:
        time: int
        pos: tuple
        down: bool
        up: bool

    def __init__(self):
        self._other_events = []

        self._keys = {}
        self._buttons = {}

        self._pos = 0, 0
        self._pos_prev = 0, 0

        self._scroll = 0, 0
        self._scroll_prev = 0, 0

    @property
    def mouse_pos(self):
        return self._pos

    @property
    def scroll(self):
        return self._scroll

    def key_down(self, key):
        return key in self._keys

    def button_down(self, button):
        return button in self._buttons

    def get_events(self):
        events = []

        now = time.perf_counter_ns()

        for key, data in self._keys.copy().items():
            if data.down:
                events.append(Event(KEY_DOWN, now, key=key, mod=data.mod))
                self._keys[key].down = False
            else:
                events.append(Event(KEY_HOLD, now, key=key, mod=data.mod))
            if data.up:
                if now - data.time < 1e9 / 5:
                    events.append(Event(KEY_PRESSED, now, key=key, mod=data.mod))
                events.append(Event(KEY_UP, now, key=key, mod=data.mod))
                del self._keys[key]

        for button, data in self._buttons.copy().items():
            if self._pos != self._pos_prev:
                events.append(
                    Event(
                        MOUSE_DRAGGED,
                        now,
                        button=button,
                        pos=self._pos,
                        rel=(self._pos[0] - self._pos_prev[0], self._pos[1] - self._pos_prev[1]),
                    )
                )
            if data.down:
                events.append(Event(MOUSE_BUTTON_DOWN, now, button=button, pos=data.pos))
                self._buttons[button].down = False
            else:
                events.append(Event(MOUSE_BUTTON_HOLD, now, button=button, pos=data.pos))
            if data.up:
                if now - data.time < 1e9 / 5:
                    events.append(Event(MOUSE_BUTTON_PRESSED, now, button=button, pos=data.pos))
                events.append(Event(MOUSE_BUTTON_UP, now, button=button, pos=data.pos))
                del self._buttons[button]

        if self._pos != self._pos_prev:
            events.append(
                Event(
                    MOUSE_MOTION,
                    now,
                    pos=self._pos,
                    rel=(self._pos[0] - self._pos_prev[0], self._pos[1] - self._pos_prev[1]),
                )
            )
            self._pos_prev = self._pos

        if self._scroll != self._scroll_prev:
            events.append(
                Event(
                    MOUSE_SCROLL,
                    now,
                    pos=self._scroll,
                    rel=(
                        self._scroll[0] - self._scroll_prev[0],
                        self._scroll[1] - self._scroll_prev[1],
                    ),
                )
            )
            self._scroll_prev = self._scroll

        events.extend(self._other_events)
        self._other_events = []

        return events


class Game:
    def __init__(self):
        self.name = self.__class__.__name__

    def init(self):
        pass

    def process_events(self, events):
        for event in events:
            if event.type == QUIT:
                stop()

    def update(self, input, t, dt):
        pass

    def render(self, t, dt):
        pass

    def shutdown(self):
        pass


def set_game(game):
    global _game
    if _running:
        raise Exception("Engine is Running")
    _game = game


def set_input(input):
    global _input
    if _running:
        raise Exception("Engine is Running")
    _input = input


def get_tps():
    return _tps_actual


def get_fps():
    return _fps_actual


def set_tps(tps):
    global _tps_target, _tps_actual, _spt_target
    _tps_target = tps
    _tps_actual = tps
    _spt_target = 0 if tps is None or tps < 1 else int(1e9 / tps)


def set_fps(fps):
    global _fps_target, _fps_actual, _spf_target
    _fps_target = fps
    _fps_actual = fps
    _spf_target = 0 if fps is None or fps < 1 else int(1e9 / fps)


def get_start_time():
    return _start_time


def get_time():
    return 0 if _start_time < 0 else time.perf_counter_ns() - _start_time


def run():
    global _start_time, _running, _tps_actual, _fps_actual
    try:
        _game.init()

        _start_time = time.perf_counter_ns()
        _running = True

        tick_count = 0
        frame_count = 0

        last_tick = get_time()
        last_frame = get_time()
        last_sec = get_time()

        while _running:
            _game.process_events(_input.get_events())

            t = get_time()
            dt = t - last_tick
            if dt >= _spt_target:
                tick_count += 1
                last_tick = t

                _game.update(_input, t / 1e9, dt / 1e9)

            t = get_time()
            dt = t - last_frame
            if dt >= _spf_target:
                frame_count += 1
                last_frame = t

                _game.render(t / 1e9, dt / 1e9)

            t = get_time()
            dt = t - last_sec
            if dt >= 1e9:
                last_sec = t

                _tps_actual = tick_count
                _fps_actual = frame_count

                tick_count = 0
                frame_count = 0

    except Exception as e:
        # print(e)
        raise

    _game.shutdown()


def stop():
    global _running
    _running = False


_input = Input()
_game = Game()

_tps_target, _tps_actual, _spt_target = None, 0, 0
_fps_target, _fps_actual, _spf_target = None, 0, 0

set_tps(60)
set_fps(60)

_start_time = -1
_running = False

try:
    import numpy as np
    import pygame

    class InputPygame(Input):
        def get_events(self):
            now = time.perf_counter_ns()

            for e in pygame.event.get():
                if e.type == pygame.KEYDOWN:
                    key = e.key
                    if key not in self._keys:
                        self._keys[key] = self._KeyData(
                            now, e.mod, e.unicode, e.scancode, True, False
                        )

                elif e.type == pygame.KEYUP:
                    key = e.key
                    try:
                        self._keys[key].up = True
                    except KeyError:
                        pass

                elif e.type == pygame.MOUSEBUTTONDOWN:
                    b = e.button
                    if b not in self._buttons:
                        self._buttons[b] = self._ButtonData(now, e.pos, True, False)

                elif e.type == pygame.MOUSEBUTTONUP:
                    b = e.button
                    try:
                        self._buttons[b].up = True
                    except KeyError:
                        pass

                elif e.type == pygame.MOUSEMOTION:
                    self._pos = e.pos

                else:
                    self._other_events.append(
                        Events.new(
                            Events.from_name(pygame.event.event_name(e.type)), now, **e.__dict__
                        )
                    )

            return super().get_events()

    class LineViewer(Game):
        POINTS = 1
        LINES = 2

        def __init__(self, size, data_size=None):
            super().__init__()

            self.size = size

            self.screen = None

            self.render_type = self.POINTS

            data_size = 1 if data_size is None else data_size
            # self.x_data = np.zeros(data_size, dtype = np.float32)
            self.data = np.zeros(data_size, dtype=np.float32)

            self.y_min, self.y_max = -1, 1

        @property
        def screen_shape(self):
            return self.size, self.size

        @property
        def data_size(self):
            return self.data.shape[0]

        def init(self):
            pygame.init()

            pygame.display.set_caption("{} TPS(0) FPS(0)".format(self.name))

            self.screen = pygame.display.set_mode(self.screen_shape)

        def update(self, input, t, dt):
            pygame.display.set_caption("{} TPS({}) FPS({})".format(self.name, get_tps(), get_fps()))

        def render(self, t, dt):
            self.screen.fill((0, 0, 0))

            spacing = self.size / (self.data_size + 1)
            for i, point in enumerate(self.data):
                pos = int(spacing * (i + 1)), self._map(point)

                if i == 0 or self.render_type == self.POINTS:
                    prev_pos = pos
                elif self.render_type == self.LINES:
                    prev_pos = int(spacing * i), self._map(self.data[i - 1])

                pygame.draw.line(self.screen, (255, 255, 255), prev_pos, pos)

            pygame.display.flip()

        def shutdown(self):
            pygame.quit()

        def _map(self, x):
            return self.size / (self.y_min - self.y_max) * (x - self.y_max)

    class ArrayViewer(Game):
        def __init__(self, size, array_size=None):
            super().__init__()

            self.size = size

            self.screen = None

            self.array_size = size if array_size is None else array_size
            self.array = np.zeros(self.array_shape, dtype="uint8")

        @property
        def screen_shape(self):
            return self.size, self.size

        @property
        def scale(self):
            return self.size / self.array_size

        @property
        def array_shape(self):
            return self.array_size, self.array_size, 3

        def init(self):
            pygame.init()

            pygame.display.set_caption("{} TPS(0) FPS(0)".format(self.name))

            self.screen = pygame.display.set_mode(self.screen_shape)

        def update(self, input, t, dt):
            pygame.display.set_caption("{} TPS({}) FPS({})".format(self.name, get_tps(), get_fps()))

        def render(self, t, dt):
            surface = pygame.surfarray.make_surface(np.swapaxes(self.array, 0, 1))
            surface = pygame.transform.scale(surface, self.screen_shape)

            self.screen.blit(surface, (0, 0))

            pygame.display.flip()

        def shutdown(self):
            pygame.quit()

except ImportError:
    pass

try:
    from OpenGL.GL import *

    class Shader:
        def __init__(self, vertex="", fragment="", geometry=""):
            self.program = glCreateProgram()

            if vertex != "":
                self._attach(GL_VERTEX_SHADER, vertex)
            if fragment != "":
                self._attach(GL_FRAGMENT_SHADER, fragment)
            if geometry != "":
                self._attach(GL_GEOMETRY_SHADER, geometry)

            glLinkProgram(self.program)

            self._uniform_locations = {}

        def _attach(self, shader_type, source):
            if "\n" not in source:
                with open(source, "r") as f:
                    source = f.readlines()

            shader = glCreateShader(shader_type)
            glShaderSource(shader, source)
            glCompileShader(shader)

            if glGetShaderiv(shader, GL_COMPILE_STATUS) != 1:
                raise Exception(
                    "Could not compile shader\n"
                    "Shader compilation Log:\n"
                    "" + str(glGetShaderInfoLog(shader))
                )

            glAttachShader(self.program, shader)
            glDeleteShader(shader)

        def _get_uniform_location(self, variable):
            try:
                return self._uniform_locations[variable]
            except KeyError:
                uniform_location = glGetUniformLocation(self.program, variable)
                self._uniform_locations[variable] = uniform_location
            return self._uniform_locations[variable]

        def use(self):
            try:
                glUseProgram(self.program)
            except GLError:
                glGetProgramInfoLog(self.program)
                raise

        def set_int(self, name, *args):
            uniform = self._get_uniform_location(name)
            arg_len = len(args)
            if arg_len == 1:
                glUniform1i(uniform, *args)
            elif arg_len == 2:
                glUniform2i(uniform, *args)
            elif arg_len == 3:
                glUniform3i(uniform, *args)
            elif arg_len == 4:
                glUniform4i(uniform, *args)
            else:
                raise Exception("Wrong Number of Arguments: {}".format(arg_len))

        def set_float(self, name, *args):
            uniform = self._get_uniform_location(name)
            arg_len = len(args)
            if arg_len == 1:
                glUniform1f(uniform, *args)
            elif arg_len == 2:
                glUniform2f(uniform, *args)
            elif arg_len == 3:
                glUniform3f(uniform, *args)
            elif arg_len == 4:
                glUniform4f(uniform, *args)
            else:
                raise Exception("Wrong Number of Arguments: {}".format(arg_len))

        def set_bool(self, name, v1):
            glUniform1i(self._get_uniform_location(name), int(v1))

        def set_int_vec(self, name, vec):
            uniform = self._get_uniform_location(name)
            arg_len = len(vec)
            if arg_len == 1:
                glUniform1iv(uniform, 1, vec)
            elif arg_len == 2:
                glUniform2iv(uniform, 1, vec)
            elif arg_len == 3:
                glUniform3iv(uniform, 1, vec)
            elif arg_len == 4:
                glUniform4iv(uniform, 1, vec)
            else:
                raise Exception("Wrong Number of Arguments: {}".format(arg_len))

        def set_float_vec(self, name, vec):
            uniform = self._get_uniform_location(name)
            arg_len = len(vec)
            if arg_len == 1:
                glUniform1fv(uniform, 1, vec)
            elif arg_len == 2:
                glUniform2fv(uniform, 1, vec)
            elif arg_len == 3:
                glUniform3fv(uniform, 1, vec)
            elif arg_len == 4:
                glUniform4fv(uniform, 1, vec)
            else:
                raise Exception("Wrong Number of Arguments: {}".format(arg_len))

        def set_float_mat(self, name, mat):
            uniform = self._get_uniform_location(name)
            shape = mat.shape
            if shape == (2, 2):
                glUniformMatrix2fv(uniform, 1, GL_FALSE, mat)
            elif shape == (2, 3):
                glUniformMatrix2x3fv(uniform, 1, GL_FALSE, mat)
            elif shape == (2, 4):
                glUniformMatrix2x4fv(uniform, 1, GL_FALSE, mat)
            elif shape == (3, 2):
                glUniformMatrix3x2fv(uniform, 1, GL_FALSE, mat)
            elif shape == (3, 3):
                glUniformMatrix3fv(uniform, 1, GL_FALSE, mat)
            elif shape == (3, 4):
                glUniformMatrix3x4fv(uniform, 1, GL_FALSE, mat)
            elif shape == (4, 2):
                glUniformMatrix4x2fv(uniform, 1, GL_FALSE, mat)
            elif shape == (4, 3):
                glUniformMatrix4x3fv(uniform, 1, GL_FALSE, mat)
            elif shape == (4, 4):
                glUniformMatrix4fv(uniform, 1, GL_FALSE, mat)
            else:
                raise Exception("Wrong Matrix Shape: {}".format(shape))

    class LineViewerGL(LineViewer):
        _vert = (
            "#version 330 core\n"
            "layout (location = 0) in vec2 aPos;\n"
            "void main() {\n"
            "    gl_Position = vec4(vec3(aPos, 0.0), 1.0);\n"
            "}\n"
        )
        _frag = (
            "#version 330 core\n"
            "out vec4 color;\n"
            "void main() {\n"
            "    color = vec4(1.0, 1.0, 1.0, 1.0);\n"
            "}\n"
        )

        def init(self):
            pygame.init()

            pygame.display.set_caption("{} TPS(0) FPS(0)".format(self.name))

            self.screen = pygame.display.set_mode(
                self.screen_shape, pygame.DOUBLEBUF | pygame.OPENGL
            )

            shader = Shader(self._vert, self._frag)
            shader.use()

            glClearColor(0.0, 0.0, 0.0, 1.0)

            glBindVertexArray(glGenVertexArrays(1))
            glBindBuffer(GL_ARRAY_BUFFER, glGenBuffers(1))

            # position
            glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8, ctypes.c_void_p(0))
            glEnableVertexAttribArray(0)

        def render(self, t, dt):
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            vertex = np.zeros((self.data_size, 2), dtype=np.float32)
            space = 2 / (self.data_size + 1)

            vertex[:, 0] = np.linspace(space - 1, 1 - space, num=self.data_size)
            vertex[:, 1] = self.data

            glBufferData(GL_ARRAY_BUFFER, vertex, GL_STATIC_DRAW)

            if self.render_type == self.POINTS:
                glDrawArrays(GL_POINTS, 0, self.data_size)
            elif self.render_type == self.LINES:
                glDrawArrays(GL_LINE_STRIP, 0, self.data_size)

            pygame.display.flip()

    class ArrayViewerGL(ArrayViewer):
        _vert = (
            "#version 330 core\n"
            "layout (location = 0) in vec2 aPos;\n"
            "layout (location = 1) in vec2 aCord;\n"
            "out vec2 cord;\n"
            "void main() {\n"
            "    cord = aCord;\n"
            "    gl_Position = vec4(vec3(aPos, 0.0), 1.0);\n"
            "}\n"
        )
        _frag = (
            "#version 330 core\n"
            "out vec4 color;\n"
            "in vec2 cord;\n"
            "uniform sampler2D map;\n"
            "void main() {\n"
            "    color = vec4(texture(map, cord).rgb, 1.0);\n"
            "}\n"
        )

        def init(self):
            pygame.init()

            pygame.display.set_caption("{} TPS(0) FPS(0)".format(self.name))

            self.screen = pygame.display.set_mode(
                self.screen_shape, pygame.DOUBLEBUF | pygame.OPENGL
            )

            shader = Shader(self._vert, self._frag)
            shader.use()

            glClearColor(0.0, 0.0, 0.0, 1.0)

            glBindBuffer(GL_ARRAY_BUFFER, glGenBuffers(1))
            glBufferData(
                GL_ARRAY_BUFFER,
                np.array(
                    [
                        -1.0,
                        1.0,
                        0.0,
                        0.0,
                        -1.0,
                        -1.0,
                        0.0,
                        1.0,
                        1.0,
                        -1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        0.0,
                    ],
                    dtype=np.float32,
                ),
                GL_STATIC_DRAW,
            )

            # position
            glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(0))
            glEnableVertexAttribArray(0)

            # tex_cords
            glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(8))
            glEnableVertexAttribArray(1)

            glBindTexture(GL_TEXTURE_2D, glGenTextures(1))

        def render(self, t, dt):
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
            glTexImage2D(
                GL_TEXTURE_2D,
                0,
                GL_RGB,
                *self.array_shape[:2],
                0,
                GL_RGB,
                GL_UNSIGNED_BYTE,
                self.array,
            )
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)

            glDrawArrays(GL_QUADS, 0, 4)

            pygame.display.flip()

except ImportError:
    pass
