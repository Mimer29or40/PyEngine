# https://github.com/Code-Bullet/SnakeFusion
# https://github.com/Code-Bullet/NEAT-Template-JavaScript

from engine import *

from neural import NeuralNet


class Board:
    def __init__(self, rows, cols, brain = None):
        self.rows, self.cols = rows, cols

        self.board = np.zeros((rows, cols), dtype = np.uint8)
        self.board[:, 0] = self.board[:, -1] = 1
        self.board[0, :] = self.board[-1, :] = 1

        self.board[rows // 2, cols // 2] = 2
        self.snake = [Vector(cols // 2, rows // 2)]

        self.add_food()

        self.dir = Vector(1, 0)
        self.score = 0
        self.age = 0
        self.health = 100

        if brain is None:
            self.brain = NeuralNet(rows * cols + 1, rows * cols // 2, 4)
        else:
            self.brain = brain.copy()

    @property
    def fitness(self):
        fitness = self.age * self.score * self.score
        if len(self.snake) < 10:
            return int(fitness * np.exp2(len(self.snake)))
        return int(fitness * np.exp2(10) * (len(self.snake) - 9))

    @property
    def alive(self):
        return self.health > 0
    
    def copy(self):
        return Board(self.rows, self.cols, brain = self.brain)
    
    def crossover(self, partner):
        brain = self.brain.crossover(partner.brain)
        return Board(self.rows, self.cols, brain = brain)
    
    def mutate(self, mutation_rate):
        self.brain.mutate(mutation_rate)
        return self

    def add_food(self):
        row = int(np.random.random() * self.rows)
        col = int(np.random.random() * self.cols)
        if self.board[row, col] == 0:
            self.board[row, col] = 3
        else:
            self.add_food()

    def move(self, dir):
        self.age += 1
        self.health -= 1

        curr, next = self.snake[-1], self.snake[-1] + dir
        next_val = self.board[int(next.y), int(next.x)]
        # Empty
        if next_val == 0:
            pos = self.snake.pop(0)
            self.board[int(pos.y), int(pos.x)] = 0
        # Wall
        elif next_val == 1:
            self.health = 0
            return
        # Snake
        elif next_val == 2:
            self.health = 0
            return
        # Food
        elif next_val == 3:
            self.score += 1
            self.add_food()
            self.health += self.health // 3

        self.snake.append(next)
        self.board[int(next.y), int(next.x)] = 2

    def update(self):
        inputs = (self.board.astype(float) / 3.).flatten()
        inputs = np.append(inputs, [self.dir.heading()])
        decision = self.brain.output(inputs)
        decision = np.argmax(decision)
        # Up
        if decision == 0:
            self.dir = Vector(0, -1)
        # Down
        elif decision == 1:
            self.dir = Vector(0, 1)
        # Up
        elif decision == 2:
            self.dir = Vector(-1, 0)
        # Down
        elif decision == 3:
            self.dir = Vector(1, 0)

        self.move(self.dir)

    def show(self, size = Vector(500, 500), pos = Vector(300, 0)):
        _size = size // [self.cols, self.rows]

        engine.stroke = 0
        for index, value in np.ndenumerate(self.board):
            # Empty
            if value == 0:
                engine.fill = None
            # Wall
            elif value == 1:
                engine.fill = 128
            # Snake
            elif value == 2:
                engine.fill = 255
            # Food
            elif value == 3:
                engine.fill = 255, 0, 0

            _pos = pos + _size * index[::-1]
            engine.rect(_pos, _size)


class Population:
    def __init__(self, rows, cols, num = 4, size = None, best = None):
        self.generation = 1
        
        size = size if size else 50
        
        if best is None:
            self.boards = [Board(rows, cols) for _ in range(size)]
            self.best = self.boards[0]
        else:
            self.boards = [best.copy().mutate(mutation_rate) for _ in range(size)]
            self.best = best
        
        self.num = num
        self.grid = int(np.ceil(np.sqrt(num)))
    
    def natural_selection(self):
        def select():
            total_fitness = sum(b.fitness for b in self.boards)
            rand = int(random(total_fitness))
            running_total = 0
            for b in self.boards:
                running_total += b.fitness
                if running_total > rand:
                    return b
            return self.boards[0]
        boards = []
        boards.append(self.best.copy())
        for _ in self.boards:
            parent1, parent2 = select(), select()
            child = parent1.crossover(parent2).mutate(mutation_rate)
            boards.append(child)
        
        self.boards = boards
        
        self.generation += 1
    
    def update(self):
        for b in self.boards:
            if b.alive:
                b.update()
        top = sorted(self.boards, key=lambda b: b.fitness, reverse = True)
        if top[0].fitness > self.best.fitness:
            self.best = top[0]
        
        if not any(b.alive for b in self.boards):
            self.natural_selection()
    
    def show(self, _size = Vector(500, 500), _pos = Vector(300, 0)):
        top = sorted(self.boards, key=lambda b: b.fitness, reverse = True)
        
        size = _size // self.grid
        count = 0
        for j in range(self.grid):
            for i in range(self.grid):
                pos = _pos + [size.x * i, size.y * j]
                top[count].show(size, pos)
                if count >= self.num:
                    return
                count += 1


if __name__ == '__main__':
    engine.size(400, 300, OPENGL)
    engine.frame_rate = 60
    engine.text_size = 40
    
    mutation_rate = 0.01
    board = Board(21, 21)
    population = Population(21, 21, num = 1)

    @engine.event
    def key_pressed(e):
        if e.key_code == 'left':
            board.dir = Vector(-1, 0)
        elif e.key_code == 'right':
            board.dir = Vector(1, 0)
        elif e.key_code == 'up':
            board.dir = Vector(0, -1)
        elif e.key_code == 'down':
            board.dir = Vector(0, 1)

    @engine.draw
    def draw():
        engine.background = 51
        
        # rot = map(engine.mouse, 0, engine.viewport, -1, 1) * PI
        # engine.rotate_x(rot.y)
        # engine.rotate_y(rot.x)

        # if board.alive:
            # board.update()
        # board.show()
        
        population.update()
        population.show(Vector(300, 300), Vector(100, 0))
        
        top = sorted(population.boards, key=lambda b: b.fitness, reverse = True)[0]
        
        engine.fill = 255
        engine.text(f'Generation: {population.generation}', Vector(10, 10, -1))
        engine.text(f'Score: {top.score}', Vector(10, 40, -1))
        engine.text(f'Health: {top.health}', Vector(10, 70, -1))
        engine.text(f'Age: {top.age}', Vector(10, 100, -1))
        engine.text(f'Fitness: {top.fitness}', Vector(10, 130, -1))

    engine.start()
