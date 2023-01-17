import numpy as np


def clip(x, x_min, x_max):
    return np.clip(x, x_min, x_max)


def is_int(x):
    _ints = [int] + np.sctypes["int"] + np.sctypes["uint"]
    return type(x) in _ints or (type(x) == np.ndarray and x.dtype in _ints)


def is_float(x):
    _floats = [float] + np.sctypes["float"]
    return type(x) in _floats or (type(x) == np.ndarray and x.dtype in _floats)


class Vector:
    def __init__(self, data, dtype=None, immutable=False):
        self._immutable = immutable
        self._data = np.array(data, dtype=dtype)

    def __str__(self):
        return str(self._data)

    def __repr__(self):
        return "{}([{}], dtype={})".format(
            self.__class__.__name__, ",".join(map(str, [*self._data])), self._data.dtype
        )

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        if self._immutable:
            raise Exception("{} is Immutable".format(self.__class__.__name__))
        self._data[key] = value

    def __eq__(self, o):
        return isinstance(o, self.__class__) and (self._data == o._data).all()

    def __ne__(self, o):
        return not self.__eq__(o)

    def __add__(self, o):
        if isinstance(o, self.__class__):
            data = self._data + o._data
        else:
            data = self._data + o
        return self.__class__(data, self._data.dtype)

    def __sub__(self, o):
        if isinstance(o, self.__class__):
            data = self._data - o._data
        else:
            data = self._data - o
        return self.__class__(data, self._data.dtype)

    def __mul__(self, o):
        if isinstance(o, self.__class__):
            data = self._data * o._data
        else:
            data = self._data * o
        return self.__class__(data, self._data.dtype)

    def __matmul__(self, o):
        if isinstance(o, self.__class__):
            data = np.matmul(self._data, o._data)
        else:
            data = np.matmul(self._data, o)
        return self.__class__(data, self._data.dtype)

    def __truediv__(self, o):
        if isinstance(o, self.__class__):
            data = self._data / o._data
        else:
            data = self._data / o
        return self.__class__(data, self._data.dtype)

    def __radd__(self, o):
        if isinstance(o, self.__class__):
            data = self._data + o._data
        else:
            data = self._data + o
        return self.__class__(data, self._data.dtype)

    def __rsub__(self, o):
        if isinstance(o, self.__class__):
            data = self._data - o._data
        else:
            data = self._data - o
        return self.__class__(data, self._data.dtype)

    def __rmul__(self, o):
        if isinstance(o, self.__class__):
            data = self._data * o._data
        else:
            data = self._data * o
        return self.__class__(data, self._data.dtype)

    def __rmatmul__(self, o):
        if isinstance(o, self.__class__):
            data = np.matmul(self._data, o._data)
        else:
            data = np.matmul(self._data, o)
        return self.__class__(data, self._data.dtype)

    def __rtruediv__(self, o):
        if isinstance(o, self.__class__):
            data = self._data / o._data
        else:
            data = self._data / o
        return self.__class__(data, self._data.dtype)

    def __iadd__(self, o):
        if isinstance(o, self.__class__):
            self._data += o._data.astype(self._data.dtype)
        else:
            self._data += o
        return self

    def __isub__(self, o):
        if isinstance(o, self.__class__):
            self._data -= o._data.astype(self._data.dtype)
        else:
            self._data -= o
        return self

    def __imul__(self, o):
        if isinstance(o, self.__class__):
            self._data *= o._data.astype(self._data.dtype)
        else:
            self._data *= o
        return self

    def __imatmul__(self, o):
        if isinstance(o, self.__class__):
            self._data = np.matmul(self._data, o._data.astype(self._data.dtype))
        else:
            self._data = np.matmul(self._data, o)
        return self

    def __itruediv__(self, o):
        if isinstance(o, self.__class__):
            self._data /= o._data.astype(self._data.dtype)
        else:
            self._data /= o
        return self

    def __neg__(self):
        return self.__class__(-self._data, self._data.dtype)

    def __pos__(self):
        return self.__class__(self._data, self._data.dtype)

    def __floor__(self):
        i = self._data.astype(int)
        return self.__class__(i - (i > self._data), int)

    def __ceil__(self):
        i = self._data.astype(int)
        return self.__class__(i - (i > self._data) + 1, int)

    @property
    def order(self):
        return len(self._data)

    @property
    def data(self):
        return self._data.copy()

    @data.setter
    def data(self, data):
        self._data[:] = np.array(data)

    @property
    def x(self):
        return self[0]

    @x.setter
    def x(self, value):
        self[0] = value

    @property
    def y(self):
        return self[1]

    @y.setter
    def y(self, value):
        self[1] = value

    @property
    def z(self):
        return self[2]

    @z.setter
    def z(self, value):
        self[2] = value

    @property
    def w(self):
        return self[3]

    @w.setter
    def w(self, value):
        self[3] = value

    @property
    def xy(self):
        return self.__class__([self.x, self.y])

    @property
    def xyz(self):
        return self.__class__([self.x, self.y, self.z])

    @property
    def xyzw(self):
        return self.__class__([self.x, self.y, self.z, self.w])

    r = x
    g = y
    b = z
    a = w

    def asint(self):
        return self.__class__(self._data, int)

    def asfloat(self):
        return self.__class__(self._data, float)

    def copy(self):
        return self.__class__(self._data)

    @property
    def transpose(self):
        return self.__class__(self._data.T)

    @property
    def magnitude(self):
        return np.linalg.norm(self._data)

    @property
    def normalize(self):
        mag = self.magnitude
        data = self._data if mag == 0 else self._data / mag
        return self.__class__(data, float)

    def dot(self, o):
        if not isinstance(o, self.__class__):
            raise NotImplementedError
        return np.dot(self._data, o._data)

    def cross(self, o):
        if not isinstance(o, self.__class__):
            raise NotImplementedError
        return np.cross(self._data, o._data)

    def angle(self, o):
        if not isinstance(o, Vector):
            raise NotImplementedError
        return np.math.acos(self.dot(o) / (self.magnitude * o.magnitude))


class Matrix(Vector):
    def __init__(self, data, dtype=None, immutable=False):
        super().__init__(data, dtype, immutable)

        shape = self._data.shape
        if shape[0] != shape[0] or len(shape) > 2:
            raise Exception("Only NxN matrix are supported")

    @property
    def inverse(self):
        return self.__class__(np.linalg.inv(self._data))

    @staticmethod
    def translate4(d, v, out=None):
        if out is None:
            out = IDEN4.copy()

        out[3] = out[0] * d * v.x + out[1] * d * v.y + out[2] * d * v.z + out[3]

        return out

    @staticmethod
    def translate3(d, v, out=None):
        out = Matrix.translate4(d, v, out)

        return Matrix(out.data[3, :-1])

    @staticmethod
    def rotate4(theta, v, out=None):
        c = np.math.cos(theta)
        s = np.math.sin(theta)
        axis = v.normalize
        if axis.magnitude == 0:
            return IDEN4
        temp = (1.0 - c) * axis

        r00 = c + temp.x * axis.x
        r01 = temp.x * axis.y + s * axis.z
        r02 = temp.x * axis.z - s * axis.y

        r10 = temp.y * axis.x - s * axis.z
        r11 = c + temp.y * axis.y
        r12 = temp.y * axis.z + s * axis.x

        r20 = temp.z * axis.x + s * axis.y
        r21 = temp.z * axis.y - s * axis.x
        r22 = c + temp.z * axis.z

        if out is None:
            out = IDEN4.copy()

        out[0] = out[0] * r00 + out[1] * r01 + out[2] * r02
        out[1] = out[0] * r10 + out[1] * r11 + out[2] * r12
        out[2] = out[0] * r20 + out[1] * r21 + out[2] * r22

        return out

    @staticmethod
    def rotate3(theta, v, out=None):
        out = Matrix.rotate4(theta, v, out)

        return Matrix(out.data[:-1, :-1])

    @staticmethod
    def rotate_around4(axis, theta, out=None):
        axis = axis.normalize
        a = np.math.cos(theta / 2.0)
        b, c, d = -axis * np.math.sin(theta / 2.0)
        aa, bb, cc, dd = a * a, b * b, c * c, d * d
        bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d

        if out is None:
            out = IDEN4.copy()

        out[0] = [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac), 0]
        out[1] = [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab), 0]
        out[2] = [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc, 0]
        out[3] = [0.0, 0.0, 0.0, 1.0]

        return out

    @staticmethod
    def rotate_around3(axis, theta, out=None):
        out = Matrix.rotate_around4(axis, theta, out)

        return Matrix(out.data[:-1, :-1])

    @staticmethod
    def scale4(v, out=None):
        if out is None:
            out = IDEN4.copy()

        out[0] *= v.x
        out[1] *= v.y
        out[2] *= v.v

        return out

    @staticmethod
    def scale3(v, out=None):
        out = Matrix.scale4(v, out)

        return Matrix(out.data[:-1, :-1])

    @staticmethod
    def ortho(left, right, bottom, top, z_near=0.0, z_far=1.0, out=None):
        m00 = 2.0 / (right - left)
        m11 = 2.0 / (top - bottom)
        m22 = 1.0 / (z_near - z_far)
        m30 = (left + right) / (left - right)
        m31 = (bottom + top) / (bottom - top)
        m32 = (z_near + z_far) / (z_near - z_far)

        if out is None:
            out = IDEN4.copy()

        out[0] = [m00, 0.0, 0.0, 0.0]
        out[1] = [0.0, m11, 0.0, 0.0]
        out[2] = [0.0, 0.0, m22, 0.0]
        out[3] = [m30, m31, m32, 1.0]

        return out

    @staticmethod
    def perspective(fov, aspect, z_near=0.0, z_far=1.0, out=None):
        h = 1 / np.math.tan(np.math.radians(fov) / 2)
        m00 = h / aspect
        m11 = h
        m22 = (z_near + z_far) / (z_near - z_far)
        m23 = -1
        m32 = 2 * z_near * z_far / (z_near - z_far)

        if out is None:
            out = IDEN4.copy()

        out[0] = [m00, 0.0, 0.0, 0.0]
        out[1] = [0.0, m11, 0.0, 0.0]
        out[2] = [0.0, 0.0, m22, m23]
        out[3] = [0.0, 0.0, m32, 0.0]

        return out

    @staticmethod
    def look_at(eye, center, up, out=None):
        front = (eye - center).normalize
        right = front.cross(up.normalize).normalize
        up = right.cross(front).normalize
        r_dot = -right.dot(eye)
        u_dot = -up.dot(eye)
        f_dot = -front.dot(eye)

        if out is None:
            out = IDEN4.copy()

        out[0] = [right.x, up.x, front.x, 0]
        out[1] = [right.y, up.y, front.y, 0]
        out[2] = [right.z, up.z, front.z, 0]
        out[3] = [r_dot, u_dot, f_dot, 1]

        return out


class Color(Vector):
    def __init__(self, r, g=None, b=None, a=None, *, immutable=False):
        if g is None or b is None:
            g = r
            b = r

        data = np.array([r, g, b])
        if a is None:
            a = 1.0 if is_float(data) else 255

        super().__init__([r, g, b, a], immutable=immutable)

        lower = 0.0 if is_float(self._data) else 0
        upper = 1.0 if is_float(self._data) else 255

        self._data = clip(self._data, lower, upper)

    def __repr__(self):
        return "{}(r={}, g={}, b={}, a={})".format(self.__class__.__name__, *self._data)

    @property
    def luminance(self):
        return (self.r * 299 + self.g * 587 + self.b * 114) / 1000

    def asint(self):
        if self._data.dtype == float:
            return self.__class__(self._data * 255, np.uint8)
        return self.__class__(self._data, int)

    def asfloat(self):
        if self._data.dtype == np.uint8:
            return self.__class__(self._data / 255, float)
        return self.__class__(self._data, float)


ZERO = Vector([0, 0, 0], float, immutable=True)
ONES = Vector([1, 1, 1], float, immutable=True)

ORIGIN = Vector([0, 0, 0], float, immutable=True)

X = Vector([1, 0, 0], float, immutable=True)
Y = Vector([0, 1, 0], float, immutable=True)
Z = Vector([0, 0, 1], float, immutable=True)

X_NEG = Vector([-1, 0, 0], float, immutable=True)
Y_NEG = Vector([0, -1, 0], float, immutable=True)
Z_NEG = Vector([0, 0, -1], float, immutable=True)

IDEN2 = Matrix(np.identity(2), float, immutable=True)
IDEN3 = Matrix(np.identity(3), float, immutable=True)
IDEN4 = Matrix(np.identity(4), float, immutable=True)

WHITE = Color(1.0, 1.0, 1.0, immutable=True)
BLACK = Color(0.0, 0.0, 0.0, immutable=True)
RED = Color(1.0, 0.0, 0.0, immutable=True)
GREEN = Color(0.0, 1.0, 0.0, immutable=True)
BLUE = Color(0.0, 0.0, 1.0, immutable=True)
