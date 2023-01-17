from engine import *


class Particle:
    def __init__(self):
        self.pos = random(engine.viewport)
        self.prev = self.pos.copy()
        self.vel = Vector(0.0, 0.0)
        self.acc = Vector(0.0, 0.0)

    def update(self):
        self.vel = (self.vel + self.acc).limit(5)
        self.pos += self.vel
        self.acc *= 0

    def show(self):
        engine.stroke = 255, 255
        engine.weight = 4

        engine.line(self.pos, self.prev)
        self.prev.xy = self.pos.xy

    def attracted(self, target):
        force = target - self.pos
        d = constrain(force.magnitude, 1, 25)
        force.magnitude = G / (d * d)
        if d < 20:
            force *= -10
        self.acc += force


if __name__ == "__main__":
    engine.size(400, 400, OPENGL)

    G = 50
    attractors = []
    particles = []

    @engine.event
    def mouse_pressed(e):
        attractors.append(e.pos)

    @engine.draw
    def draw():
        engine.background = 51
        engine.stroke = 255
        engine.weight = 4
        particles.append(Particle())

        if len(particles) > 100:
            particles.pop(0)

        for a in attractors:
            engine.stroke = 0, 255, 0
            engine.fill = 0, 255, 0
            engine.point(a)
        for p in particles:
            for a in attractors:
                p.attracted(a)
            p.update()
            p.show()

    engine.start()
