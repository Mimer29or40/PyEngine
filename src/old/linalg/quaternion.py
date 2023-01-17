# -*- coding: utf-8 -*-

"""
Quaternion to have nice rotations in 3D.

Copyright (c) 2010, Renaud Blanch <rndblnch at gmail dot com>
Licence: GPLv3 or higher <http://www.gnu.org/licenses/gpl.html>
"""

# imports ####################################################################

from math import asin as _asin
from math import atan2 as _atan2
from math import cos as _cos
from math import sin as _sin
from math import sqrt as _sqrt

from . import vector as _v

# quaternion #################################################################


def quaternion(theta=0, u=(1.0, 0.0, 0.0)):
    w = _cos(theta / 2.0)
    x, y, z = (ui * _sin(theta / 2.0) for ui in u)
    return w, (x, y, z)


def arcball(x, y):
    h2 = x * x + y * y
    if h2 > 1.0:
        h = _sqrt(h2)
        v = x / h, y / h, 0.0
    else:
        v = x, y, _sqrt(1.0 - h2)
    return 0.0, v


def mul(P, Q):
    w1, v1 = P
    w2, v2 = Q
    return (w1 * w2 - _v.dot(v1, v2), _v.sum(_v.mul(w1, v2), _v.mul(w2, v1), _v.cross(v1, v2)))


def product(P=quaternion(), *Qs):
    for Q in Qs:
        P = mul(P, Q)
    return P


def matrix(Q):
    w, (x, y, z) = Q
    return [
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z), 2.0 * (x * z - w * y), 0.0],
        [2.0 * (x * y - w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + w * x), 0.0],
        [2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y), 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def euler_angles(Q):
    w, (x, y, z) = Q
    phi = _atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    theta = _asin(2 * (w * y - x * z))
    psi = _atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return phi, theta, psi


def antipod(Q):
    w, v = Q
    return -w, _v.mul(-1.0, v)


def inverse(Q):
    w, v = Q
    return w, _v.mul(-1.0, v)


def rotate(Q, v):
    _, v = product(Q, (0, v), inverse(Q))
    return v


def theta_u(Q):
    w, v = Q
    norm = _v.norm(v)
    theta = 2.0 * _atan2(norm, w)
    u = _v.mul(1.0 / norm, v)
    return theta, u


def power(Q, a):
    theta, u = theta_u(Q)
    return quaternion(a * theta, u)


def slerp(Q0, Q1, a):
    return mul(power(mul(Q1, inverse(Q0)), a), Q0)
