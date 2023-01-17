import numpy as np
import util


class Camera:
    def __init__(self):
        self.is_perspective = True

        self._position = util.Z.copy()
        self._focus = util.ORIGIN.copy()

        self._r = util.X.copy()
        self._u = util.Y.copy()
        self._f = util.Z.copy()

        self.fov = 60.0
        self.z_near, self.z_far = 0.0, 200.0

        self._x_min, self._x_max = -10.0, 10.0
        self._y_min, self._y_max = -10.0, 10.0

        self._projection = util.IDEN4.copy()
        self._view = util.IDEN4.copy()

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, position):
        self._position.data = position

    @property
    def focus(self):
        return self._focus

    @focus.setter
    def focus(self, focus):
        self._focus.data = focus

    @property
    def focal_length(self):
        return (self._position - self._focus).magnitude

    def zoom(self, amount):
        if self.is_perspective:
            self._position += self._f * self.focal_length * amount
        else:
            center = (self._x_max + self._x_min) / 2
            dist = (amount + 1) * (self._x_max - self._x_min) / 2
            self._x_min, self._x_max = center - dist, center + dist

    def translate(self, v=None, *, dx=0, dy=0, dz=0):
        if v == None:
            v = util.Vector([dx, dy, dz], float)

        self._position += v
        self._focus += v

    def projection(self, aspect_ratio):
        if self.is_perspective:
            h = 1 / np.math.tan(np.math.radians(self.fov) / 2)

            m00 = h / aspect_ratio
            m11 = h
            m22 = (self.z_near + self.z_far) / (self.z_near - self.z_far)
            m23 = -1.0
            m30 = 0.0
            m31 = 0.0
            m32 = 2 * self.z_near * self.z_far / (self.z_near - self.z_far)
        else:
            screen_height = (self._x_max - self._x_min) / (aspect_ratio * 2)
            self._y_min, self._y_max = -screen_height, screen_height

            m00 = 2.0 / (self._x_max - self._x_min)
            m11 = 2.0 / (self._y_max - self._y_min)
            m22 = 1.0 / (self.z_near - self.z_far)
            m23 = 0.0
            m30 = (self._x_min + self._x_max) / (self._x_min - self._x_max)
            m31 = (self._y_min + self._y_max) / (self._y_min - self._y_max)
            m32 = (self.z_near + self.z_far) / (self.z_near - self.z_far)

        self._projection.data = [
            [m00, 0.0, 0.0, 0.0],
            [0.0, m11, 0.0, 0.0],
            [0.0, 0.0, m22, m23],
            [m30, m31, m32, 1.0],
        ]

        return self._projection

    def view(self):
        r_dot = -self._r.dot(self._position)
        u_dot = -self._u.dot(self._position)
        f_dot = -self._f.dot(self._position)

        self._view.data = [
            [self._r.x, self._u.x, self._f.x, 0],
            [self._r.y, self._u.y, self._f.y, 0],
            [self._r.z, self._u.z, self._f.z, 0],
            [r_dot, u_dot, f_dot, 1],
        ]

        return self._view
