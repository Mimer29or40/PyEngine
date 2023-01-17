import re

import numpy as np


class Vector(np.ndarray):
    @staticmethod
    def from_angle(angle):
        return Vector(np.math.cos(angle), np.math.sin(angle))

    @staticmethod
    def random(order):
        return Vector(*(2 * np.random.random((order,)) - 1)).normalize

    __indices__ = "xyzw"
    __lookup__ = {"x": 0, "y": 1, "z": 2, "w": 3}
    __default__ = {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}
    __match__ = re.compile("^[xyzw]{1,4}$")

    immutable = False

    def __new__(cls, *data, **kwargs):
        return np.array(data, dtype=float).view(cls)

    def __setitem__(self, key, value):
        if self.immutable:
            raise Exception(f"{self.__class__.__name__} is Immutable")
        return super().__setitem__(key, value)

    def __eq__(self, o):
        return np.allclose(self, o)

    def __ne__(self, o):
        return not self.__eq__(o)

    def __getattr__(self, key):
        m = self.__match__.match(key)
        if m:
            if len(m.group(0)) == 1:
                return self[self.__lookup__[m.group(0)]]
            return self[[self.__lookup__[i] for i in m.group(0)]]
        raise AttributeError

    def __setattr__(self, key, value):
        m = self.__match__.match(key)
        if m:
            if len(m.group(0)) == 1:
                self[self.__lookup__[m.group(0)]] = value
                return
            self[[self.__lookup__[i] for i in m.group(0)]] = value
            return
        super().__setattr__(key, value)

    @property
    def x(self):
        return self[:1]

    @x.setter
    def x(self, value):
        self[:1] = value

    @property
    def xy(self):
        return self[:2]

    @xy.setter
    def xy(self, value):
        self[:2] = value

    @property
    def xyz(self):
        return self[:3]

    @xyz.setter
    def xyz(self, value):
        self[:3] = value

    @property
    def xyzw(self):
        return self[:4]

    @xyzw.setter
    def xyzw(self, value):
        self[:4] = value

    @property
    def base(self):
        arr = []
        for i in self.__indices__:
            try:
                arr.append(self[self.__lookup__[i]])
            except IndexError:
                arr.append(self.__default__[i])
        return self.__class__(*arr)

    @property
    def magnitude(self):
        if len(self) < 4:
            return np.linalg.norm(self)
        return np.linalg.norm(self.xyz)

    @magnitude.setter
    def magnitude(self, value):
        if len(self) < 4:
            self.__imul__(value / self.magnitude)
            # self *= value / self.magnitude
            return
        self.xyz *= value / self.magnitude

    @property
    def magnitude_sq(self):
        return self.dot(self)

    @property
    def normalize(self):
        if len(self) < 4:
            return self / self.magnitude
        return self.xyz / self.magnitude

    def dot(self, o, out=None):
        if len(self) < 4:
            return np.dot(self, o)
        return np.dot(self.xyz, o.xyz)

    def cross(self, o):
        return np.cross(self.xyz, o.xyz).view(Vector)

    def angle_between(self, o):
        return np.math.acos(self.dot(o) / (self.magnitude * o.magnitude))

    def dist(self, o):
        if len(self) < 4:
            return np.linalg.norm(self - o)
        return np.linalg.norm(self.xyz - o.xyz)

    def heading(self):
        return np.angle(self.x + self.y * 1j)

    def lerp(self, v, amount):
        if len(self) < 4:
            return self + amount * (v - self)
        return self.xyz + amount * (v.xyz - self.xyz)

    def limit(self, amt):
        if self.magnitude > amt:
            return self.normalize * amt
        return self * 1.0


class Matrix(np.ndarray):
    @staticmethod
    def identity(x):
        return np.identity(x).view(Matrix)

    def __new__(cls, *data):
        return np.array(data).reshape((4, 4)).view(cls)

    def __eq__(self, o):
        return np.allclose(self, o)

    def __ne__(self, o):
        return not self.__eq__(o)

    @property
    def inverse(self):
        return np.linalg.inv(self).view(Matrix)

    def translate(self, v, d=1, out=None, normalize=False):
        a = v.copy().view(Vector).base
        if normalize:
            a = a.normalize

        if out is None:
            out = self

        if a.magnitude == 0:
            return out

        out[-1, :-1] += d * out[:-1, :-1].dot(a.xyz)

        return out

    def rotate(self, v, theta=0, out=None, normalize=False):
        a = v.copy().view(Vector).base
        if normalize:
            a = a.normalize

        if out is None:
            out = self

        if a.magnitude == 0:
            return out

        s, c = np.sin(theta), np.cos(theta)

        if self.shape[0] == 4:
            tmp = (1.0 - c) * a.xyz

            out[:-1, :-1] = (
                out[:-1, :-1].dot(tmp.x * a.xyz + [c, s * a.z, -s * a.y]),
                out[:-1, :-1].dot(tmp.y * a.xyz + [-s * a.z, c, s * a.x]),
                out[:-1, :-1].dot(tmp.z * a.xyz + [s * a.y, -s * a.x, c]),
            )
        elif self.shape[0] == 3:
            out[:-1, :-1] = (out[:-1, :-1].dot([c, s]), out[:-1, :-1].dot([-s, c]))

        return out

    def scale(self, v, d=1, out=None, normalize=False):
        a = v.copy().view(Vector).base
        if normalize:
            a = a.normalize

        if out is None:
            out = self

        if a.magnitude == 0:
            return out

        out[:-1, :-1] *= a.xyz * d

        return out


def format_color(*data):
    if len(data) == 0:
        return 0, 0, 0, 255
    elif len(data) == 1:
        try:
            return format_color(*data[0])
        except TypeError:
            return data[0], data[0], data[0], 255
    elif len(data) == 2:
        return data[0], data[0], data[0], data[1]
    elif len(data) == 3:
        return data[0], data[1], data[2], 255
    elif len(data) >= 4:
        return data[0], data[1], data[2], data[3]


class Color(Vector):
    @staticmethod
    def from_int(x, alpha=True):
        x = int(x)
        r = (x >> 24) & 0xFF
        g = (x >> 16) & 0xFF
        b = (x >> 8) & 0xFF
        a = (x >> 0) & 0xFF
        if alpha:
            return Color(r, g, b, a)
        return Color(g, b, a)

    @staticmethod
    def random(alpha=False):
        return Color(256 * np.random.random((4 if alpha else 3,)))

    __indices__ = "rgba"
    __lookup__ = {"r": 0, "g": 1, "b": 2, "a": 3}
    __default__ = {"r": 0, "g": 0, "b": 0, "a": 255}
    __match__ = re.compile("^[rgba]{1,4}$")

    is_none = False

    def __new__(cls, *data):
        return np.array(format_color(*data), dtype=np.uint8).view(cls)

    def copy(self, order="C"):
        copy = super().copy(order)
        copy.is_none = self.is_none
        return copy

    def set(self, *data):
        if data[0] is None:
            self.is_none = True
        else:
            self.is_none = False
            self[:] = format_color(*data)

    def to_int(self, alpha=True):
        if alpha:
            return int(self.r << 16 | self.g << 8 | self.b)
        return int(self.r << 24 | self.g << 16 | self.b << 8 | self.a)

    def to_float(self):
        return self.astype(float) / 255.0

    def to_gl(self):
        if self.is_none:
            return [-1.0, -1.0, -1.0, -1.0]
        return self.to_float()
