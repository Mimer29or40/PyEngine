from engine import *
from engine.event import Event


class Vehicle:
    mutation_chance = 0.01

    def __init__(self, pos, dna=None):
        self.acc = Vector(0.0, 0.0)
        self.vel = Vector(0.0, -2.0)
        self.pos = Vector(*pos.astype(float))

        self.r = 4.0

        self.max_speed, self.max_force = 10.0, 0.5

        self.health = 1.0

        if dna is None:
            # # Food weight
            # # Poison Weight
            # # Food Perception
            # # Poison Perception
            self.dna = random([-2.0, -2.0, 0.0, 0.0], [2.0, 2.0, 100.0, 100.0])
        else:
            self.dna = dna
            if random() < self.mutation_chance:
                self.dna[0] += random(-0.1, 0.1)
            if random() < self.mutation_chance:
                self.dna[1] += random(-0.1, 0.1)
            if random() < self.mutation_chance:
                self.dna[2] += random(-10, 10)
            if random() < self.mutation_chance:
                self.dna[3] += random(-10, 10)

    def clone(self):
        if random() < 0.002:
            return Vehicle(self.pos, self.dna)
        else:
            return None

    def update(self):
        self.health -= 0.005

        self.vel += self.acc
        self.vel.limit(self.max_speed)

        self.pos += self.vel

        self.acc *= 0

    def apply_force(self, force):
        self.acc += force

    def behaviors(self, good, bad):
        steer_g = self.eat(good, 0.2, self.dna[2]) * self.dna[0]
        steer_b = self.eat(bad, -0.5, self.dna[3]) * self.dna[1]
        self.apply_force(steer_g + steer_b)

    def eat(self, values, nutrition, perception):
        closest, record = None, float(1e9)
        for val in reversed(values):
            d = self.pos.dist(val)
            if d < self.max_speed:
                values.remove(val)
                # values.pop(i)
                self.health += nutrition
            elif d < record and d < perception:
                closest, record = val, d
        if closest is not None:
            return self.seek(closest)
        return Vector(0.0, 0.0)

    def seek(self, target):
        desired = target - self.pos
        desired.magnitude = self.max_speed

        return (desired - self.vel).limit(self.max_force)

    def dead(self):
        return self.health < 0

    def show(self):
        engine.push()

        angle = self.vel.heading() + np.pi / -2.0

        engine.translate(self.pos)
        engine.rotate(angle)

        if debug:
            engine.weight = 3
            engine.stroke = 0, 255, 0
            engine.fill = None

            pos = Vector(0, 0)
            point = Vector(0, -self.dna[0] * 25)
            engine.line(pos, point)

            engine.weight = 2

            engine.circle(pos, self.dna[2])

            engine.stroke = 255, 0, 0

            point.xy = (0, self.dna[1] * 25)
            engine.line(pos, point)
            engine.circle(pos, self.dna[3])

        gr = Color(0, 255, 0)
        rd = Color(255, 0, 0)
        col = np.uint8(lerp(np.int16(rd), np.int16(gr), self.health))

        engine.stroke = col
        engine.fill = col
        engine.weight = 1

        pointlist = [
            Vector(0, -self.r * 2),
            Vector(self.r, self.r * 2),
            Vector(-self.r, self.r * 2),
        ]

        engine.polygon(*pointlist)

        engine.pop()

    def boundaries(self):
        d, desired = 25.0, None

        if self.pos.x < d:
            desired = Vector(self.max_speed, self.vel.y)
        elif engine.width - d < self.pos.x:
            desired = Vector(-self.max_speed, self.vel.y)

        if self.pos.y < d:
            desired = Vector(self.vel.x, self.max_speed)
        elif engine.height - d < self.pos.y:
            desired = Vector(self.vel.x, -self.max_speed)

        if desired is not None:
            steer = desired.normalize * self.max_speed - self.vel
            self.apply_force(steer.limit(self.max_force))


if __name__ == "__main__":
    engine.size(800, 600, OPENGL)

    engine.frame_rate = 60

    debug = False

    vehicles = [Vehicle(random(engine.viewport)) for _ in range(50)]
    food = [random(engine.viewport) for _ in range(50)]
    poison = [random(engine.viewport) for _ in range(50)]

    # @engine.event

    def mouse_dragged(e: Event, t="r") -> None:
        vehicles.append(Vehicle(engine.mouse))

    print(mouse_dragged.__annotations__)
    print(mouse_dragged.__code__.co_varnames)
    print(mouse_dragged.__kwdefaults__)

    @engine.event(key=" ")
    def key_pressed():
        global debug
        debug = not debug

    @engine.draw
    def draw():
        engine.background = 51
        engine.font_size = 40

        if random() < 0.1:
            food.append(random(engine.viewport))

        if random() < 0.01:
            poison.append(random(engine.viewport))

        engine.stroke = None

        engine.fill = 0, 255, 0
        for f in food:
            engine.circle(f, 4)

        engine.fill = 255, 0, 0
        for f in poison:
            engine.circle(f, 4)

        for v in list(vehicles):
            v.boundaries()
            v.behaviors(food, poison)
            v.update()
            v.show()

            clone = v.clone()
            if clone is not None:
                vehicles.append(clone)

            if v.dead():
                food.append(v.pos.copy())
                vehicles.remove(v)

        engine.stroke = 255, 255, 255
        engine.text(len(vehicles), Vector(0, 0))

    # engine.start()
