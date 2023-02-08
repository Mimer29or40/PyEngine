from engine import *


class Ship:
    def __init__(self):
        self.pos = engine.viewport / 2
        self.vel = Vector(0.0, 0.0)
        self.r = 10.0
        self.heading = 0.0
        self.boosting = False

    def update(self):
        self.pos += self.vel
        if not self.boosting:
            self.vel *= 0.9

    def boost(self):
        self.vel += Vector.from_angle(self.heading) * 0.0005
        self.boosting = True

    def rotate(self, rotation):
        self.heading += rotation

    def hits(self, asteroid):
        d = self.pos.dist(asteroid.pos)
        return d < self.r + asteroid.r

    def render(self):
        engine.push()

        engine.translate(self.pos)
        engine.rotate(self.heading + np.pi / 2)

        engine.fill = 0
        engine.stroke = 255
        engine.weight = 1

        angle = self.heading + np.pi / 2
        engine.triangle(Vector(-self.r, self.r), Vector(self.r, self.r), Vector(0, -self.r))

        if self.boosting:
            offset = self.r * 0.25
            engine.triangle(
                Vector(-self.r + offset, self.r),
                Vector(self.r - offset, self.r),
                Vector(0, self.r + offset),
            )
            self.boosting = False

        engine.pop()

    def edges(self):
        if self.pos.x > engine.width + self.r:
            self.pos.x = -self.r
        elif self.pos.x < -self.r:
            self.pos.x = engine.width + self.r

        if self.pos.y > engine.height + self.r:
            self.pos.y = -self.r
        elif self.pos.y < -self.r:
            self.pos.y = engine.height + self.r


class Laser:
    def __init__(self, ship):
        self.pos = ship.pos.copy()
        self.vel = ship.vel + Vector.from_angle(ship.heading) * 10.0

    def update(self):
        self.pos += self.vel

    def render(self):
        engine.push()

        engine.fill = 255
        engine.stroke = 255
        engine.weight = 4

        engine.circle(self.pos, 4)

        engine.pop()

    def hits(self, asteroid):
        d = self.pos.dist(asteroid.pos)
        return d < asteroid.r

    def off_screen(self):
        return (
            self.pos.x < 0
            or engine.width < self.pos.x
            or self.pos.y < 0
            or engine.height < self.pos.y
        )


class Asteroid:
    def __init__(self, pos=None, r=None):
        if pos is None:
            self.pos = random(engine.viewport)
        else:
            self.pos = pos.copy()
        self.vel = Vector.random(2)

        if r is None:
            self.r = random(15, 50)
        else:
            self.r = r * 0.5

        self.total = int(random(10, 30))
        self.offset = [random(-self.r * 0.25, self.r * 0.25) for _ in range(self.total)]

    def update(self):
        self.pos += self.vel

    def render(self):
        engine.push()

        engine.fill = None
        engine.stroke = 255
        engine.weight = 1

        engine.translate(self.pos)

        points = []
        for i, offset in enumerate(self.offset):
            angle = map(i, 0, self.total, 0, 2 * np.pi)
            points.append(Vector.from_angle(angle) * (self.r + offset))
        engine.polygon(*points)

        engine.pop()

    def breakup(self):
        return [Asteroid(self.pos, self.r), Asteroid(self.pos, self.r)]

    def edges(self):
        if self.pos.x > engine.width + self.r:
            self.pos.x = -self.r
        elif self.pos.x < -self.r:
            self.pos.x = engine.width + self.r

        if self.pos.y > engine.height + self.r:
            self.pos.y = -self.r
        elif self.pos.y < -self.r:
            self.pos.y = engine.height + self.r


if __name__ == "__main__":
    engine.size(800, 600, OPENGL)

    ship = Ship()
    asteroids = [Asteroid() for _ in range(5)]
    lasers = []
    score = 0

    def gameover():
        global ship, asteroids, lasers, score
        ship = Ship()
        asteroids = [Asteroid() for _ in range(5)]
        lasers = []
        score = 0

    @engine.draw
    def draw():
        global score

        engine.background = 0

        for a in asteroids:
            if ship.hits(a):
                gameover()
                return
            a.update()
            a.edges()
            a.render()

        for l in lasers.copy():
            l.update()
            l.render()
            if l.off_screen():
                lasers.remove(l)
            else:
                for a in asteroids.copy():
                    if l.hits(a):
                        if a.r > 10:
                            score += 1
                            asteroids.extend(a.breakup())
                        asteroids.remove(a)
                        lasers.remove(l)
                        break

        ship.update()
        ship.edges()
        ship.render()

        engine.text(score, Vector(0, 0))

    @engine.event
    def key_pressed(e):
        if e.key == " ":
            lasers.append(Laser(ship))

    @engine.event
    def key_held(e):
        if e.key_code == "right":
            ship.rotate(0.0001)
        if e.key_code == "left":
            ship.rotate(-0.0001)
        if e.key_code == "up":
            ship.boost()

    engine.start()
