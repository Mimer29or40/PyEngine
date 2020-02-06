from engine import *


class Cell:
    def __init__(self, pos = None, r = None, c = None):
        self.pos = random(engine.viewport) if pos is None else pos.copy()
        self.r = 60 if r is None else r
        self.c = Color(random(100, 255), 0, random(100, 255), 100) if c is None else c
    
    def clicked(self, pos):
        return self.pos.dist(pos) < self.r
    
    def mitosis(self):
        return Cell(self.pos, self.r * 0.8, self.c)
    
    def move(self):
        self.pos += Vector.random(2) * 2
    
    def show(self):
        engine.stroke = None
        engine.fill = self.c
        engine.circle(self.pos, self.r)


if __name__ == '__main__':
    engine.size(700, 700, OPENGL)
    
    cells = [Cell(), Cell()]
    
    @engine.draw
    def draw():
        engine.background = 200
        
        for c in cells:
            c.move()
            c.show()
        
        engine.text_size = 50
        engine.text(len(cells), Vector(0, 0))
    
    @engine.event
    def mouse_pressed():
        for c in reversed(cells):
            if c.clicked(engine.mouse):
                cells.append(c.mitosis())
                cells.append(c.mitosis())
                cells.remove(c)
    
    engine.start()
