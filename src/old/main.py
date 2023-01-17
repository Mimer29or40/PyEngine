# System Packages
import math

import noise3
import numpy as np

# Third-Party Packages
import pygame
from fluid import Fluid
from OpenGL.GL import *

# Project Packages
from old import engine as engine

# My Packages
# import Util


class Noise1D(engine.LineViewerGL):
    def __init__(self, size, data_size=None):
        super().__init__(size, data_size)

        self.y_min, self.y_max = 0, 1
        self.sections = 0

        self.noise = noise3.PerlinNoise(seed=69, octaves=1, persistence=0.7)

        self.new_data_size = self.data_size

    def process_events(self, events):
        super().process_events(events)

        for event in events:
            if event.type == engine.KEY_DOWN:
                if event.key == pygame.K_SPACE:
                    self.noise.base_seed = np.random.randint(0, 0xFFFF)
                if event.key == pygame.K_q:
                    self.noise.octaves += 1
                if event.key == pygame.K_a:
                    self.noise.octaves -= 1
                if event.key == pygame.K_l:
                    self.render_type = self.LINES
                if event.key == pygame.K_p:
                    self.render_type = self.POINTS
                self.noise.octaves = max(1, min(self.noise.octaves, 8))

                if event.key == pygame.K_UP:
                    self.new_data_size += 1
                if event.key == pygame.K_DOWN:
                    self.new_data_size -= 1
                if event.key == pygame.K_RIGHT:
                    self.new_data_size += 10
                if event.key == pygame.K_LEFT:
                    self.new_data_size -= 10
                self.new_data_size = max(1, self.new_data_size)

                if event.key == pygame.K_w:
                    self.sections += 1
                if event.key == pygame.K_s:
                    self.sections -= 1
                self.sections = max(0, min(self.sections, 32))
            # elif event.type == GE.KEY_HOLD:
            #     if event.key == pygame.K_RIGHT:
            #         self.start_x += amount
            #     if event.key == pygame.K_LEFT:
            #         self.start_x -= amount

    def update(self, input, t, dt):
        super().update(input, t, dt)

        x = (np.arange(10000) + 1) / 9000
        x = np.linspace(0.0, 10.0, 10000)
        # self.data = np.sin(10 * x) / (1 + x * x)
        near, far = 1.0, 200.0
        self.data = ((1 / (x + 0.000001 + near)) - (1 / near)) / ((1 / far) - (1 / near))

        # rng_x = 0, 1, self.new_data_size
        # noise_arr = self.noise.generate_range(rng_x, t / 4)
        #
        # if self.sections is not None and self.sections > 0:
        #     s = 256 / self.sections
        #     noise_arr = (255 * noise_arr).astype(int) // s * s / 255
        #
        # self.data = 2 * noise_arr[0] - 1


class Noise2D(engine.ArrayViewerGL):
    def __init__(self, size, array_size=None):
        super().__init__(size, array_size)

        self.start_x, self.start_y = 0, 0
        self.sections = 0

        # noise3.fade = lambda t: t
        self.noise = noise3.PerlinNoise(seed=69, octaves=1)
        # self.noise = noise3.WorleyNoise(seed = 69, octaves = 1)
        # self.noise = noise3.ValueNoise(seed = 69, octaves = 1)

    def process_events(self, events):
        super().process_events(events)

        for event in events:
            if event.type == engine.KEY_DOWN:
                if event.key == pygame.K_SPACE:
                    self.noise.octaves += 1
                    if self.noise.octaves > 4:
                        self.noise.octaves = 1
                elif event.key == pygame.K_q:
                    self.sections += 1
                elif event.key == pygame.K_a:
                    self.sections -= 1
                self.sections = max(0, min(self.sections, 32))
            elif event.type == engine.KEY_HOLD:
                amount = 0.05
                if event.key == pygame.K_RIGHT:
                    self.start_x += amount
                if event.key == pygame.K_LEFT:
                    self.start_x -= amount
                if event.key == pygame.K_UP:
                    self.start_y -= amount
                if event.key == pygame.K_DOWN:
                    self.start_y += amount

    def update(self, input, t, dt):
        super().update(input, t, dt)

        rng_x = (self.start_x, self.start_x + 4, self.array_size)
        rng_y = (self.start_y, self.start_y + 4, self.array_size)
        array = self.noise.generate_range(rng_x, rng_y, t)[:, :, 0]

        # array = np.abs(2 * array - 1)
        # array = 4 * (array - array.astype(int))

        self.array = noise3.to_rgb(array, sections=self.sections)


class FluidSim(engine.ArrayViewerGL):
    def __init__(self, size, array_size=None):
        super().__init__(size, array_size=array_size)

        self.fluid = Fluid(self.array_size, 0.00001, 0.000000001)

        self.c = int(0.5 * self.array_size)

        amount = np.random.randint(0, 255)
        for j in [-1, 0, 1]:
            for i in [-1, 0, 1]:
                self.fluid.add_density(self.c + i, self.c + j, amount)

    def process_events(self, events):
        super().process_events(events)

        for event in events:
            if event.type == engine.MOUSE_DRAGGED:
                x = int(event.pos[0] // self.scale)
                y = int(event.pos[1] // self.scale)
                if event.button == 1:
                    dx = event.rel[0] // self.scale
                    dy = event.rel[1] // self.scale

                    for i in [-1, 0, 1]:
                        for j in [-1, 0, 1]:
                            self.fluid.add_velocity(x + i, y + j, dx * 2, dy * 2)

                elif event.button == 3:
                    for i in [-1, 0, 1]:
                        for j in [-1, 0, 1]:
                            self.fluid.add_density(x + i, y + j, 200)
            elif event.type == engine.KEY_DOWN:
                if event.key == pygame.K_SPACE:
                    self.fluid.density *= 0
            elif event.type == engine.KEY_HOLD:
                if event.key == pygame.K_UP:
                    self.fluid.diff += 0.000001
                elif event.key == pygame.K_DOWN:
                    self.fluid.diff -= 0.000001

    def update(self, input, t, dt):
        super().update(input, t, dt)

        # amount = np.random.randint(0, 255)
        amount = 150
        for j in [-1, 0, 1]:
            for i in [-1, 0, 1]:
                self.fluid.add_density(self.c + i, self.c + j, amount)

        # angle = t / 1e9 * np.pi / 180 * 10
        angle = np.random.randint(0, 360) * np.pi / 180

        x = 5 * math.cos(angle)
        y = 5 * math.sin(angle)

        for j in [-1, 0, 1]:
            for i in [-1, 0, 1]:
                self.fluid.add_velocity(self.c + i, self.c + j, x, y)
        self.fluid.step(dt)

        self.array[:, :, 0] = self.fluid.density
        self.array[:, :, 1] = self.fluid.v_x * 255
        self.array[:, :, 2] = self.fluid.v_y * 255


class Heat1D(engine.LineViewerGL):
    def __init__(self, size, data_size=None):
        super().__init__(size, data_size)

        self.y_min, self.y_max = 0, 1

    def process_events(self, events):
        super().process_events(events)

        for event in events:
            if event.type == engine.KEY_DOWN:
                if event.key == pygame.K_l:
                    self.render_type = self.LINES
                elif event.key == pygame.K_p:
                    self.render_type = self.POINTS

    def update(self, input, t, dt):
        super().update(input, t, dt)

        x = np.linspace(-5, 5, num=1000)
        k = 0.1
        pi = np.pi

        self.data = np.exp((-x * x) / (4 * k * t)) / (2 * np.sqrt(pi * k * t))


def rotate(angle):
    return np.array(
        [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
    )


def move(mat, dx, dy):
    mat[2, 0] += dx
    mat[2, 1] += dy


def mag(x):
    return np.linalg.norm(x)


def norm(x):
    _mag = mag(x)
    return x if _mag == 0 else x / _mag


# noinspection PyTupleAssignmentBalance
def dist_to_line(p1, p2, p0):
    x1, y1, x2, y2, x0, y0 = *p1, *p2, *p0
    return np.absolute((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1) / mag(p2 - p1)


# noinspection PyTupleAssignmentBalance
def get_closest_point(p1, p2, p0):
    x1, y1, x2, y2, x0, y0 = *p1, *p2, *p0

    qx, qy = x0 - (y2 - y1), y0 + (x2 - x1)

    return np.array(
        [
            ((x1 * y2 - y1 * x2) * (x0 - qx) - (x1 - x2) * (x0 * qy - y0 * qx))
            / ((x1 - x2) * (y0 - qy) - (y1 - y2) * (x0 - qx)),
            ((x1 * y2 - y1 * x2) * (y0 - qy) - (y1 - y2) * (x0 * qy - y0 * qx))
            / ((x1 - x2) * (y0 - qy) - (y1 - y2) * (x0 - qx)),
        ],
        dtype=float,
    )


class EasingAnimation(engine.Game):
    _vert = (
        "#version 330 core\n"
        "layout (location = 0) in vec2 aPos;\n"
        "uniform vec2 translation;\n"
        # 'uniform mat3 model;\n'
        "void main() {\n"
        "    vec3 pos = vec3(aPos + translation, 1.0);\n"
        # '    vec3 pos = model * vec3(aPos, 1.0);\n'
        "    gl_Position = vec4(pos, 1.0);\n"
        "}\n"
    )
    _frag = (
        "#version 330 core\n"
        "out vec4 color;\n"
        "void main() {\n"
        "    color = vec4(0.45, 0.65, 0.30, 1.0);\n"
        "}\n"
    )

    def __init__(self, size):
        super().__init__()

        self.size = size

        self.screen = None
        self.shader = None

        self.pos = np.zeros(2, dtype=float)
        self.vel = np.zeros(2, dtype=float)
        self.acc = np.zeros(2, dtype=float)

        self.strt = np.zeros(2, dtype=float)
        self.stop = np.zeros(2, dtype=float)
        self.dist = np.zeros(2, dtype=float)

    def _translate(self, pos):
        min = -1
        max = 1
        new_pos = np.zeros(2, dtype=np.float32)
        new_pos[0] = ((max - min) / self.size * pos[0]) + min
        new_pos[1] = ((min - max) / self.size * pos[1]) + max
        return new_pos

    @property
    def screen_shape(self):
        return self.size, self.size

    def init(self):
        pygame.init()

        pygame.display.set_caption("{} TPS(0) FPS(0)".format(self.name))

        self.screen = pygame.display.set_mode(self.screen_shape, pygame.DOUBLEBUF | pygame.OPENGL)

        self.shader = engine.Shader(self._vert, self._frag)
        self.shader.use()

        glClearColor(0.0, 0.0, 0.0, 1.0)

        glBindBuffer(GL_ARRAY_BUFFER, glGenBuffers(1))
        glBufferData(
            GL_ARRAY_BUFFER,
            np.array(
                [
                    -0.01,
                    0.01,
                    -0.01,
                    -0.01,
                    0.01,
                    -0.01,
                    0.01,
                    0.01,
                ],
                dtype=np.float32,
            ),
            GL_STATIC_DRAW,
        )

        # position
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)

    def process_events(self, events):
        for event in events:
            if event.type == engine.QUIT:
                engine.stop()
            elif event.type == engine.MOUSE_BUTTON_DOWN:
                self.strt[:] = self.pos
                self.stop[:] = self._translate(np.array(event.pos))
                self.dist[:] = self.stop - self.strt
                self.acc[:] = norm(self.stop - self.strt)

    def update(self, input, t, dt):
        pygame.display.set_caption(
            "{} TPS({}) FPS({})".format(self.name, engine.get_tps(), engine.get_fps())
        )

        # # if mag(self._to - self.pos) > 0.0001:
        #     # move(self.model, *self.step)
        # new_dist = mag(self.stop - self.pos)
        # if new_dist > 0.001:
        #     # new_acc = norm(self.stop - self.pos)[:]
        #     # if new_dist < 0.5 * self.dist:
        #         # new_acc = -new_acc
        #     # self.acc[:] = new_acc[:]
        #     # self.vel += self.acc * dt
        #     # self.pos += self.vel * dt
        #
        #     new_dist = (self.dist - new_dist) + 0.1
        #     print(new_dist, self.dist, new_dist, 0.5 * self.dist)
        #     if new_dist > 0.5 * self.dist:
        #         self.vel = -self.acc * new_dist + self.dist
        #     else:
        #         self.vel = self.acc * new_dist + 0
        #         # print('There')
        #     # print(mag(self.vel), self.vel, self.acc)
        #     print(self.vel)
        #     self.pos += self.vel * dt
        # else:
        #     self.acc *= 0
        #     self.vel *= 0
        #     self.strt *= 0
        #     self.stop *= 0

        strt_dist_mag = mag(self.strt - self.pos)
        stop_dist_mag = mag(self.stop - self.pos)

        if stop_dist_mag > 0.005:
            # new_acc = norm(self.stop - self.pos)[:] * dt
            new_acc = self.acc * dt
            # new_acc = (self.stop - self.pos)[:] * dt
            # print(dist_to_line(self.strt, self.stop, self.pos))

            at = get_closest_point(self.strt, self.stop, self.pos) - self.pos
            print(mag(at))
            print(self.acc)
            # self.vel += at * dt

            if strt_dist_mag < 0.5 * mag(self.dist):
                self.vel += new_acc * (strt_dist_mag + 0.1)
            else:
                self.vel -= 0.9 * new_acc * (stop_dist_mag + 0.1)

            self.pos += self.vel * dt

            self.pos[0] = max(-1.0, min(self.pos[0], 1.0))
            self.pos[1] = max(-1.0, min(self.pos[1], 1.0))
        else:
            self.acc *= 0
            self.vel *= 0
            self.strt[:] = self.pos
            self.stop[:] = self.pos

    def render(self, t, dt):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        self.shader.set_float_vec("translation", self.pos)

        glDrawArrays(GL_QUADS, 0, 4)

        pygame.display.flip()

    def shutdown(self):
        pygame.quit()


if __name__ == "__main__":
    np.set_printoptions(linewidth=9999, precision=6, edgeitems=10, threshold=4000, suppress=True)

    size = 512
    array_size = 100
    # game = FluidSim(size, array_size)

    data_size = size
    # game = Heat1D(size, data_size)
    game = Noise1D(size, data_size)

    # game = EasingAnimation(size)

    engine.set_tps(60)
    engine.set_fps(60)

    engine.set_game(game)
    engine.set_input(engine.InputPygame())

    engine.run()
    # tps = 60
    # fps = 60
    # engine = GE.Engine(GE.InputPygame(), game, tps, fps)

    # engine.run()
