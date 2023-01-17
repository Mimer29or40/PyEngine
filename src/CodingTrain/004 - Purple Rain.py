from engine import *


class Drop:
    def __init__(self):
        x = random(engine.width)
        y = random(-500, -50)
        z = random(0, 20)
        self.pos = Vector(x, y, z)
        self.len = map(z, 0, 20, 10, 20)
        self.vel = map(z, 0, 20, 1, 20)
    
    def fall(self):
        self.pos.y += self.vel
        self.vel += map(self.pos.z, 0, 20, 0, 0.2)
        # self.vel += 0.2
        
        if self.pos.y > engine.height:
            self.pos.y = random(-200, -100)
            self.vel = map(self.pos.z, 0, 20, 4, 10)
    
    def show(self):
        engine.weight = map(self.pos.z, 0, 20, 1, 3)
        engine.stroke = 138, 43, 226
        engine.line(self.pos, self.pos + [0, self.len, 0])


if __name__ == '__main__':
    engine.size(400, 300, OPENGL)
    
    drops = [Drop() for _ in range(500)]
    
    @engine.draw
    def draw():
        engine.background = 230, 230, 250
        for d in drops:
            d.fall()
            d.show()
    
    engine.start()
