# System Packages

# Third-Party Packages
import numpy as np

# My Packages
import Util

Util.setup_env(__file__)

# Project Packages


class Fluid:
    def __init__(self, size, diff, viscosity):
        self.size = size
        self.diff = diff
        self.viscosity = viscosity

        self.iterations = 32

        n = self.size

        self.s = np.zeros((n, n))
        self.density = np.zeros((n, n))

        self.v_x = np.zeros((n, n))
        self.v_y = np.zeros((n, n))

        self.v_x0 = np.zeros((n, n))
        self.v_y0 = np.zeros((n, n))

    def step(self, dt):
        self.diffuse(dt, 1, self.v_x0, self.v_x, self.viscosity)
        self.diffuse(dt, 2, self.v_y0, self.v_y, self.viscosity)

        self.project(self.v_x0, self.v_y0, self.v_x, self.v_y)

        self.advect(dt, 1, self.v_x, self.v_x0, self.v_x0, self.v_y0)
        self.advect(dt, 2, self.v_y, self.v_y0, self.v_x0, self.v_y0)

        self.project(self.v_x, self.v_y, self.v_x0, self.v_y0)

        self.diffuse(dt, 0, self.s, self.density, self.diff)
        self.advect(dt, 0, self.density, self.s, self.v_x, self.v_y)

    @staticmethod
    def ix(x, y):
        return y, x
        # return x, y

    def slice(self):
        return slice(1, self.size - 1, None)
        # return slice(0, self.size, None)

    def add_density(self, x, y, amount):
        self.density[self.ix(x, y)] += amount

    def add_velocity(self, x, y, amount_x, amount_y):
        self.v_x[self.ix(x, y)] += amount_x
        self.v_y[self.ix(x, y)] += amount_y

    def set_boundary(self, b, x):
        n = self.size

        i = self.slice()
        j = self.slice()

        x[self.ix(i, 0)] = -x[self.ix(i, 1)] if b == 2 else x[self.ix(i, 1)]
        x[self.ix(i, n - 1)] = -x[self.ix(i, n - 2)] if b == 2 else x[self.ix(i, n - 2)]

        x[self.ix(0, j)] = -x[self.ix(1, j)] if b == 1 else x[self.ix(1, j)]
        x[self.ix(n - 1, j)] = -x[self.ix(n - 2, j)] if b == 1 else x[self.ix(n - 2, j)]

        x[self.ix(0, 0)] = 0.5 * (x[self.ix(1, 0)] + x[self.ix(0, 1)])
        x[self.ix(n - 1, 0)] = 0.5 * (x[self.ix(1, n - 2)] + x[self.ix(1, n - 1)])
        x[self.ix(0, n - 1)] = 0.5 * (x[self.ix(n - 2, 0)] + x[self.ix(n - 1, 1)])
        x[self.ix(n - 1, n - 1)] = 0.5 * (x[self.ix(n - 2, n - 1)] + x[self.ix(n - 1, n - 2)])

    def lin_solve(self, b, x, x0, a, c):
        n = self.size

        i = self.slice()
        j = self.slice()

        c_recip = 1 / c

        for _ in range(self.iterations):
            x[self.ix(i, j)] = (
                x0
                + a
                * (
                    np.roll(x0, 1, axis=1)
                    + np.roll(x0, -1, axis=1)
                    + np.roll(x0, 1, axis=0)
                    + np.roll(x0, -1, axis=0)
                )
            )[self.ix(i, j)] * c_recip

            self.set_boundary(b, x)

    def diffuse(self, dt, b, x, x0, diff):
        n = self.size

        a = dt * diff * (n - 2) * (n - 2)
        self.lin_solve(b, x, x0, a, 1 + 6 * a)

    def project(self, v_x, v_y, p, div):
        n = self.size

        i = self.slice()
        j = self.slice()

        div[self.ix(i, j)] = (
            -0.5
            * (
                np.roll(v_x, 1, axis=1)
                - np.roll(v_x, -1, axis=1)
                + np.roll(v_y, 1, axis=0)
                - np.roll(v_y, -1, axis=0)
            )[self.ix(i, j)]
            / n
        )
        p[self.ix(i, j)] = p[self.ix(i, j)] * 0

        self.set_boundary(0, div)
        self.set_boundary(0, p)
        self.lin_solve(0, p, div, 1, 6)

        v_x[self.ix(i, j)] -= (
            0.5 * (np.roll(p, 1, axis=1) - np.roll(p, -1, axis=1))[self.ix(i, j)] * n
        )

        v_y[self.ix(i, j)] -= (
            0.5 * (np.roll(p, 1, axis=0) - np.roll(p, -1, axis=0))[self.ix(i, j)] * n
        )

        self.set_boundary(1, v_x)
        self.set_boundary(2, v_y)

    def advect(self, dt, b, d, d0, v_x, v_y):
        n = self.size

        i = self.slice()
        j = self.slice()

        dtx = dt * (n - 2)
        dty = dt * (n - 2)

        lin = np.linspace(0, n, n, endpoint=False)
        i_float, j_float = np.meshgrid(lin, lin)

        x = np.clip(i_float - (dtx * v_x), 0.5, n + 0.5)[self.ix(i, j)]
        y = np.clip(j_float - (dty * v_y), 0.5, n + 0.5)[self.ix(i, j)]

        i0 = np.clip(np.int64(x), 0, n - 1)
        i1 = np.clip(i0 + 1, 0, n - 1)
        j0 = np.clip(np.int64(y), 0, n - 1)
        j1 = np.clip(j0 + 1, 0, n - 1)

        s1 = x - i0
        s0 = 1.0 - s1
        t1 = y - j0
        t0 = 1.0 - t1

        d[self.ix(i, j)] = s0 * (t0 * d0[self.ix(i0, j0)] + t1 * d0[self.ix(i0, j1)]) + s1 * (
            t0 * d0[self.ix(i1, j0)] + t1 * d0[self.ix(i1, j1)]
        )

        self.set_boundary(b, d)


class FluidNew:
    def __init__(self, size, diff, viscosity):
        self.size = size
        self.diff = diff
        self.viscosity = viscosity

        self.iterations = 16

        n = self.size

        self.density = np.zeros((n, n))
        self.velocity = np.zeros((2, 2, n, n))

    @property
    def v(self):
        return self.velocity[0, :, :, :]

    @property
    def v_0(self):
        return self.velocity[1, :, :, :]

    def step(self):
        # self.diffuse(self.v, self.v_0)
        # self.project(self.v, self.v_0)
        # self.advect(self.v, self.v_0)
        # self.project(self.v, self.v_0)
        #
        # self.diffuse(self.density)
        # self.advect(self.density)

        pass

    @staticmethod
    def ix(x, y):
        return y, x

    def add_density(self, x, y, amount):
        self.density[self.ix(x, y)] += amount

    def add_velocity(self, x, y, amount=(0, 0)):
        self.v[(slice(None), *self.ix(x, y))] += amount


if __name__ == "__main__":
    np.set_printoptions(linewidth=9999, precision=6, edgeitems=10, threshold=4000, suppress=True)

    size, dt, diff, visc = 16, 0.0001, 0.00001, 0.01

    # fluid = Fluid(size, diff, visc)
    # print(fluid.v_x)
    # fluid.add_density(4, 4, 0.1)
    # fluid.add_velocity(4, 4, 0.2, 0.2)
    # for _ in range(10):
    # fluid.step()
    # print(fluid.density)

    new_fluid = FluidNew(size, diff, visc)
    new_fluid.add_velocity(1, 1, (3, 3))
    new_fluid.add_velocity(1, 1, (3, 3))
    print(new_fluid.velocity)
