from engine import *


class DNA:
    lifespan = 400
    max_force = 0.2

    def __init__(self, genes=None):
        if genes is None:
            self.genes = []
            for _ in range(DNA.lifespan):
                self.genes.append(Vector.random(2) * DNA.max_force)
        else:
            self.genes = genes.copy()

    def crossover(self, partner):
        mid = int(random(len(self.genes)))
        return DNA(partner.genes[0:mid] + self.genes[mid:-1])

    def mutation(self):
        for i, g in enumerate(self.genes):
            if random(i) < 0.01:
                self.genes[i] = Vector.random(2) * DNA.max_force


class Population:
    def __init__(self):
        self.size = 25
        self.rockets = [Rocket() for _ in range(self.size)]
        self.pool = []

    def evaluate(self, target):
        avg_fit = max_fit = 0.0
        for r in self.rockets:
            r.calc_fitness(target)
            max_fit = max(max_fit, r.fitness)
            avg_fit += r.fitness
        avg_fit /= len(self.rockets)

        for r in self.rockets:
            r.fitness /= max_fit

        self.pool = []

        for r in self.rockets:
            n = r.fitness * 100
            for _ in range(int(n)):
                self.pool.append(r)

        return avg_fit

    def random(self, arr):
        return arr[int(random(len(arr)))]

    def selection(self):
        self.rockets = []
        for _ in range(self.size):
            parent_a = self.random(self.pool).dna
            parent_b = self.random(self.pool).dna
            child = parent_a.crossover(parent_b)
            child.mutation()
            self.rockets.append(Rocket(child))

    def run(self, target):
        for r in self.rockets:
            r.update(target)
            r.show()


class Rocket:
    max_vel = 4

    def __init__(self, dna=None):
        self.pos = engine.viewport / 2
        self.vel = Vector(0.0, 0.0)
        self.acc = Vector(0.0, -0.01)

        self.dna = DNA() if dna is None else dna
        self.fitness = 0.0

        self.hit = self.crashed = False

    def dist_to_target(self, target):
        return self.pos.dist(target)

    def update(self, target):
        d = self.dist_to_target(target)
        if d < 10:
            self.hit = True
            self.pos[:] = target

        if all(barrier_pos < self.pos) and all(self.pos < barrier_pos + barrier_size):
            self.crashed = True

        if not (0 < self.pos.x < engine.width and 0 < self.pos.y < engine.height):
            self.crashed = True

        if not self.hit and not self.crashed:
            self.acc += self.dna.genes[age]

            self.vel += self.acc
            self.pos += self.vel

            self.acc *= 0
            self.vel.limit(Rocket.max_vel)

    def calc_fitness(self, target):
        d = self.dist_to_target(target)
        self.fitness = map(d, 0, engine.width, engine.width, 0)
        if self.hit:
            self.fitness *= 10
        elif self.crashed:
            self.fitness /= 10

    def show(self):
        engine.push()
        engine.stroke = None

        if self.hit:
            engine.fill = 50, 205, 50
        elif self.crashed:
            engine.fill = 128
        else:
            engine.fill = 255, 100

        engine.translate(self.pos)
        engine.rotate(self.vel.heading())
        engine.rect_mode = CENTER

        engine.rect(Vector(0, 0), Vector(25, 5))

        engine.pop()


if __name__ == "__main__":
    engine.size(400, 300)

    target = Vector(engine.width / 2, 50)
    population = Population()

    barrier_size = Vector(engine.width / 8, 10)
    barrier_pos = (engine.viewport - barrier_size) / 2
    barrier_pos.y -= 50

    age = 0
    stat = 0
    generation = 0

    rocket = Rocket()

    @engine.draw
    def draw():
        global age, generation, stat
        engine.background = 5

        # Target
        engine.stroke = 255
        engine.fill = 128
        engine.circle(target, 20)

        engine.fill = 100
        engine.stroke = None
        engine.weight = 2
        engine.circle(target, 10)

        # Barrier
        engine.fill = 255, 0, 0
        engine.stroke = 128
        engine.rect(barrier_pos, barrier_size)
        engine.weight = 1
        engine.stroke = 55, 0, 0

        population.run(target)

        age += 1
        if age >= DNA.lifespan:
            stat = population.evaluate(target)
            population.selection()
            age = 0
            generation += 1

        rocket.update(target)
        rocket.show()

        engine.stroke = 255, 128, 0
        engine.fill = 255, 128, 0
        engine.text(f"Generation: {generation}", Vector(10, 10))
        engine.text(f"Age: {age}", Vector(10, 40))
        engine.text(f"Stat: {stat}", Vector(10, 70))

    engine.start()
