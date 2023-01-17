from engine import *


class Box:
    def __init__(self, pos, r):
        self.pos = pos
        self.r = r

    def generate(self):
        boxes = []
        for x in [-1, 0, 1]:
            for y in [-1, 0, 1]:
                for z in [-1, 0, 1]:
                    sum = abs(x) + abs(y) + abs(z)
                    r = self.r / 3
                    if sum > 1:
                        boxes.append(Box(self.pos + Vector(x, y, z) * r, r))
        return boxes

    def show(self):
        engine.push()
        engine.translate(self.pos)
        engine.stroke = None
        engine.fill = 255
        engine.square(self.pos, self.r)
        engine.pop()


if __name__ == "__main__":
    engine.size(400, 300, OPENGL)

    a = 0
    sponge = [Box(Vector(0, 0, 0), 200)]

    @engine.event
    def mouse_pressed():
        global sponge

        next = []
        for b in sponge:
            next.extend(b.generate())
        sponge = next

    @engine.draw
    def draw():
        global a

        engine.background = 51
        engine.stroke = 255
        engine.fill = None
        engine.translate(engine.viewport / 2)
        engine.scale(Vector(0.5, 0.5, 0.5))

        for b in sponge:
            b.show()

        a += 0.01

    engine.start()
