from engine import *


class Cell:
    def __init__(self, i, j, w):
        self.index, self.w = np.int64(Vector(i, j)), w
        self.pos = self.index * w

        self.neighbors = 0
        self.bee, self.revealed = False, False

    def show(self):
        engine.stroke = 0
        engine.fill = None
        engine.square(self.pos, self.w)
        if self.revealed:
            if self.bee:
                engine.fill = 127
                engine.circle(self.pos + self.w * 0.5, self.w * 0.4)
            else:
                engine.fill = 200
                engine.square(self.pos, self.w)
                if self.neighbors > 0:
                    engine.text_align = CENTER
                    engine.fill = 0
                    engine.text(self.neighbors, self.pos + self.w / 2)

    def count_bees(self):
        if self.bee:
            self.neighbors = -1
            return
        self.neighbors = 0
        for x in [-1, 0, 1]:
            cell_i = self.index.x + x
            if cell_i < 0 or cols <= cell_i:
                continue
            for y in [-1, 0, 1]:
                cell_j = self.index.y + y
                if cell_j < 0 or rows <= cell_j:
                    continue
                if grid[cell_i][cell_j].bee:
                    self.neighbors += 1

    def contains(self, point):
        return (self.pos < point).all() and (point < self.pos + self.w).all()

    def reveal(self):
        self.revealed = True
        if self.neighbors == 0:
            self.flood()

    def flood(self):
        for x in [-1, 0, 1]:
            cell_i = self.index.x + x
            if cell_i < 0 or cols <= cell_i:
                continue
            for y in [-1, 0, 1]:
                cell_j = self.index.y + y
                if cell_j < 0 or rows <= cell_j:
                    continue
                neighbor = grid[cell_i][cell_j]
                if not neighbor.revealed:
                    neighbor.reveal()


if __name__ == "__main__":
    engine.size(400, 400, OPENGL)

    engine.frame_rate = 60
    engine.text_size = 20

    w = 20
    total_bees = 30

    cols, rows = np.int64(engine.viewport // w)
    grid = [[Cell(i, j, w) for j in range(rows)] for i in range(cols)]

    options = [[i, j] for i in range(cols) for j in range(rows)]

    for n in range(total_bees):
        index = int(np.random.random() * len(options))
        i, j = options[index]
        options.pop(index)
        grid[i][j].bee = True

    for i in range(cols):
        for j in range(rows):
            grid[i][j].count_bees()

    def game_over():
        for i in range(cols):
            for j in range(rows):
                grid[i][j].revealed = True

    @engine.event
    def mouse_pressed():
        for i in range(cols):
            for j in range(rows):
                if grid[i][j].contains(engine.mouse):
                    grid[i][j].reveal()
                    if grid[i][j].bee:
                        game_over()

    @engine.draw
    def draw():
        engine.background = 255

        for i in range(cols):
            for j in range(rows):
                grid[i][j].show()

    engine.start()
