from engine import *


class Star:
    def __init__(self):
        x, y = random(-engine.viewport / 2, engine.viewport / 2)
        z = random(engine.width / 2)
        self.pos = Vector(x, y, z)
        self.pz = self.pos.z
    
    def update(self):
        self.pos.z -= speed
        if self.pos.z < 1:
            x, y = random(-engine.viewport / 2, engine.viewport / 2)
            z = random(engine.width / 2)
            self.pos.xyz = [x, y, z]
            self.pz = self.pos.z
    
    def show(self):
        engine.fill = 255
        engine.stroke = None
        
        s = map(self.pos.xy / self.pos.z, 0, 1, 0, engine.viewport / 2)
        r = map(self.pos.z, 0, engine.width / 2, 16, 0)
        engine.circle(s, r)
        
        engine.stroke = 255
        
        p = map(self.pos.xy / self.pz, 0, 1, 0, engine.viewport / 2)
        engine.line(p, s)
        
        self.pz = self.pos.z


if __name__ == '__main__':
    engine.size(400, 400, OPENGL)
    
    speed = 0
    stars = [Star() for _ in range(50)]
    
    @engine.draw
    def draw():
        global speed
        
        speed = map(engine.mouse.x, 0, engine.width, 0, 50)
        
        engine.background = 0
        
        engine.translate(engine.viewport / 2)
        
        for s in stars:
            s.update()
            s.show()
    
    engine.start()
