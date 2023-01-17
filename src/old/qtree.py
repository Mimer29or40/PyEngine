import pygame
import util


class AABB:
    def __init__(self, x_min, y_min, x_max, y_max):
        self.x_min, self.y_min = x_min, y_min
        self.x_max, self.y_max = x_max, y_max

    def __repr__(self):
        return "AABB<x[{},{}) y[{},{})>".format(self.x_min, self.x_max, self.y_min, self.y_max)

    @property
    def x(self):
        return (self.x_max + self.x_min) / 2

    @property
    def y(self):
        return (self.y_max + self.y_min) / 2

    @property
    def center(self):
        return self.x, self.y

    @property
    def width(self):
        return self.x_max - self.x_min

    @property
    def height(self):
        return self.y_max - self.y_min

    def intersects(self, o):
        return isinstance(o, AABB) and not (
            o.x_min > self.x_max
            or o.x_max < self.x_min
            or o.y_min > self.y_max
            or o.y_max < self.y_min
        )

    def contains(self, p):
        return self.x_min <= p[0] < self.x_max and self.y_min <= p[1] < self.y_max


class Point:
    def __init__(self, x, y, meta=None):
        self.x, self.y = x, y
        self.meta = meta

    def __repr__(self):
        return "Point<({},{}) meta[{}]>".format(self.x, self.y, self.meta)

    def __getitem__(self, key):
        return self.x if key == 0 else (self.y if key == 1 else None)


class QuadTree:
    def __init__(self, aabb, capacity):
        self.aabb = AABB(*aabb) if type(aabb) is tuple else aabb

        self.points = []
        self.capacity = capacity

        self.children = None

    def _sub_divide(self):
        self.children = [
            QuadTree((self.x, self.y_min, self.x_max, self.y), capacity),
            QuadTree((self.x_min, self.y_min, self.x, self.y), capacity),
            QuadTree((self.x, self.y, self.x_max, self.y_max), capacity),
            QuadTree((self.x_min, self.y, self.x, self.y_max), capacity),
        ]

        for p in self.points:
            self.insert(p)

        self.points = []

    def insert(self, p):
        if not self.aabb.contains(p):
            return False

        if self.children is None:
            self.points.append(p)

            if len(self.points) > self.capacity:
                self._sub_divide()
        else:
            for c in self.children:
                if c.insert(p):
                    return True
        return False

    def query(self, aabb):
        array = []
        if self.aabb.intersects(aabb):
            if self.children is None:
                array.extend([p for p in points if aabb.contains(p)])
            else:
                for c in self.children:
                    array.extend(c.query(aabb))
        return array
