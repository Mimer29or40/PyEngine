from enum import Enum

import numpy as np

import util


class Window:
    def __init__(self, width, height):
        self._screen_size = util.Vector([width, height])
        
        self.is_perspective = True
        
        self._position = util.Z.copy()
        self._focus = util.ORIGIN.copy()
        
        self._r = util.X.copy()
        self._u = util.Y.copy()
        self._f = util.Z.copy()
        
        self.fov = 60.
        self.z_near, self.z_far = 0., 10.
        
        self._x_min, self._x_max = -5., 5.
        self._y_min, self._y_max = -5., 5.
        
        self._projection = util.IDEN4.copy()
        self._view = util.IDEN4.copy()
    
    @property
    def width(self):
        return self._screen_size.x
    
    @property
    def height(self):
        return self._screen_size.y
    
    @property
    def screen_size(self):
        return self.width, self.height
    
    @property
    def aspect_ratio(self):
        return self.width / self.height
    
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
    
    def screen_to_world(self, screen_pos):
        proj_inv = self._projection.copy().inverse
        view_inv = self._view.copy().inverse
        
        xy = 2 * screen_pos.asfloat() / self._screen_size.asfloat() - 1
        x, y, z, w = xy.x, -xy.y, 0, 1
        
        o = util.Vector([x, y, z, w], float)
        o @= proj_inv @ view_inv
        o = (o / o.w).xyz
        
        z = 0.9
        
        d = util.Vector([x, y, z, w], float)
        d @= proj_inv @ view_inv
        d = (d / d.w).xyz - o
        
        return o + d * (self._f.dot(self.focus - o)) / (self._f.dot(d))
    
    def zoom(self, amount):
        if self.is_perspective:
            self._position = self.focus + self._f * self.focal_length * amount
        else:
            center = (self._x_max + self._x_min) / 2
            dist = amount * (self._x_max - self._x_min) / 2
            self._x_min, self._x_max = center - dist, center + dist
    
    def translate(self, v = None, amount = 1, *, dx = 0, dy = 0, dz = 0):
        if v is None:
            v = util.Vector([dx, dy, dz], float)
        
        self._position += v * amount
        self._focus += v * amount
    
    def rotate(self, v = None, amount = 1, *, alpha = 0, beta = 0, gamma = 0):
        if v is None:
            v = util.Vector([alpha, beta, gamma], float)

        if v.x != 0:
            self._position -= self._focus
            self._position @= util.Matrix.rotate_around3(self._r, v.x * amount)
            self._position += self._focus

            self._f = (self._position - self._focus).normalize
            self._u = self._f.cross(self._r).normalize
        if v.y != 0:
            self._position -= self._focus
            self._position @= util.Matrix.rotate_around3(self._u, v.y * amount)
            self._position += self._focus

            self._f = (self._position - self._focus).normalize
            self._r = self._u.cross(self._f).normalize
        if v.z != 0:
            m = util.Matrix.rotate_around3(self._f, v.z * amount)

            self._r = (self._r @ m).normalize
            self._u = (self._u @ m).normalize
    
    @property
    def projection(self):
        if self.is_perspective:
            h = 1 / np.math.tan(np.math.radians(self.fov) / 2)
            
            m00 = h / self.aspect_ratio
            m11 = h
            m22 = (self.z_near + self.z_far) / (self.z_near - self.z_far)
            m23 = -1.
            m30 = 0.
            m31 = 0.
            m32 = 2 * self.z_near * self.z_far / (self.z_near - self.z_far)
        else:
            screen_height = (self._x_max - self._x_min) / (self.aspect_ratio * 2)
            self._y_min, self._y_max = -screen_height, screen_height
            
            m00 = 2. / (self._x_max - self._x_min)
            m11 = 2. / (self._y_max - self._y_min)
            m22 = 1. / (self.z_near - self.z_far)
            m23 = 0.
            m30 = (self._x_min + self._x_max) / (self._x_min - self._x_max)
            m31 = (self._y_min + self._y_max) / (self._y_min - self._y_max)
            m32 = (self.z_near + self.z_far) / (self.z_near - self.z_far)
        
        self._projection.data = [
            [m00,  0.,  0.,  0.],
            [ 0., m11,  0.,  0.],
            [ 0.,  0., m22, m23],
            [m30, m31, m32,  1.]
        ]
        
        return self._projection
    
    @property
    def view(self):
        r_dot = -self._r.dot(self._position)
        u_dot = -self._u.dot(self._position)
        f_dot = -self._f.dot(self._position)
        
        self._view.data = [
            [self._r.x, self._u.x, self._f.x, 0],
            [self._r.y, self._u.y, self._f.y, 0],
            [self._r.z, self._u.z, self._f.z, 0],
            [    r_dot,     u_dot,     f_dot, 1]
        ]
        
        return self._view
