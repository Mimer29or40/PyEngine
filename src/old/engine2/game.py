import numpy as np

from pyengine import util


class GameObject:
    def __init__(self, name):
        self.name = name

        self.position = util.ORIGIN
        self.rotation = util.IDEN3
        self._transformation = util.IDEN4

        self.model = None

        self.children = []

    def process_events(self, events):
        pass

    def update(self, t, dt):
        self._transformation = util.Matrix(
            [
                [*self.rotation.data[0], 0.0],
                [*self.rotation.data[1], 0.0],
                [*self.rotation.data[2], 0.0],
                [*self.position.data, 1.0],
            ]
        )

        for child in self.children:
            child.update(t, dt)

    def render(self, shader, t, dt):
        if self.model is not None:
            shader.set_float_mat("model", self._transformation)

            self.model.draw(shader)

        for child in self.children:
            child.render(shader, t, dt)
