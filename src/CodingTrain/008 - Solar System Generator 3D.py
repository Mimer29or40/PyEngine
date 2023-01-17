from engine import *


class Planet:
    def __init__(self, r, d, o):
        self.v = Vector.random(3) * d

        self.radius = r
        self.distance = d
        self.angle = random(TWO_PI)
        self.orbit_speed = o
        self.planets = []

    def orbit(self):
        self.angle += self.orbit_speed
        for planet in self.planets:
            planet.orbit()

    def spawn_moons(self, total, level):
        for i in range(total):
            r = self.radius / (level * 2)
            d = random((self.radius + r), (self.radius + r) * 2)
            o = random(-0.1, 0.1)
            planet = Planet(r, d / level, o)
            if level < 3:
                planet.spawn_moons(int(random(0, 3)), level + 1)
            self.planets.append(planet)

    def show(self):
        engine.push()

        p = self.v.cross(Vector(1, 0, 1))
        engine.rotate(self.angle, p)

        engine.stroke = 255

        engine.line(Vector(0), self.v)
        engine.line(Vector(0), p)

        engine.stroke = None
        engine.fill = 255

        engine.translate(self.v)

        engine.circle(Vector(0, 0, 0), self.radius)

        for planet in self.planets:
            planet.show()

        engine.pop()


if __name__ == "__main__":
    engine.size(600, 600, OPENGL)

    sun = Planet(50, 0, 0)
    sun.spawn_moons(5, 1)

    @engine.draw
    def draw():
        engine.background = 0
        engine.scale(0.25, 0.25, 0.25)
        engine.translate(engine.viewport / 2)

        sun.orbit()
        sun.show()

    engine.start()
