from .color import Color
from .vector import Vector3
from .vector import Vector3c

X: Vector3c = Vector3(1, 0, 0)
Y: Vector3c = Vector3(0, 1, 0)
Z: Vector3c = Vector3(0, 0, 1)


class Renderer:
    def __init__(self):
        self.view = Matrix.identity(4)

    def get_flags(self):
        return 0

    def set_background(self, color):
        pass

    def setup(self, engine):
        self.set_background(Color())

    def before_draw(self, engine):
        pass

    def after_draw(self, engine):
        pass

    def translate(self, amount):
        self.view.translate(amount, 1)

    def rotate(self, angle, axis=None):
        if axis is None:
            self.view.rotate(Z, angle)
        else:
            self.view.rotate(axis, angle)

    def rotate_x(self, angle):
        self.view.rotate(X, angle)

    def rotate_y(self, angle):
        self.view.rotate(Y, angle)

    def rotate_z(self, angle):
        self.view.rotate(Z, angle)

    def scale(self, amount):
        self.view.scale(amount, 1)

    def point(self, engine, p):
        pass

    def line(self, engine, p1, p2):
        pass

    def lines(self, engine, *points):
        pass

    def polygon(self, engine, *points):
        pass

    def triangle(self, engine, p1, p2, p3):
        pass

    def quad(self, engine, p1, p2, p3, p4):
        pass

    def ellipse(self, engine, p1, p2):
        pass

    def arc(self, engine, p1, p2, start, stop):
        pass

    def text(self, engine, text, pos):
        pass

    def load_pixels(self, engine):
        size = (engine.height, engine.width, 4)
        arr = np.zeros(size, dtype=np.uint8)
        arr[:, :, 3] = 255
        return arr

    def update_pixels(self, engine):
        pass
