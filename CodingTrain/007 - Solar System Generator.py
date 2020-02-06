from engine import *


class Planet:
    def __init__(self, r, d, o):
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
            d = random((self.radius + r),(self.radius + r) * 2)
            o = random(-0.05, 0.05)
            planet = Planet(r, d, o)
            if level < 3:
                planet.spawn_moons(int(random(0, 4)), level + 1)
            self.planets.append(planet)
    
    def show(self):
        engine.push()
        engine.fill = 255, 100
        engine.rotate(self.angle)
        engine.translate(self.distance, 0)
        engine.circle(Vector(0, 0), self.radius * 2)
        for planet in self.planets:
            planet.show()
        engine.pop()


if __name__ == '__main__':
    engine.size(600, 600, OPENGL)
    
    sun = Planet(50, 0, 0)
    sun.spawn_moons(5, 1)
    
    @engine.draw
    def draw():
        engine.background = 0
        engine.translate(engine.viewport / 2)
        sun.orbit()
        sun.show()
    
    engine.start()
