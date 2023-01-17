from engine import *


class Boundary:
    def __init__(self, x1 = 0, y1 = 0, x2 = 0, y2 = 0, *, a = None, b = None):
        self.a = Vector(x1, y1) if a is None else a
        self.b = Vector(x2, y2) if b is None else b
    
    def show(self):
        engine.stroke = 255
        engine.line(self.a, self.b)


class Particle:
    def __init__(self):
        self.pos = engine.viewport / 2
        self.rays = [Ray(self.pos, radians(a)) for a in range(360)]
    
    def look(self, walls):
        for ray in self.rays:
            closest, record = None, 500_000_000
            for wall in walls:
                pt = ray.cast(wall)
                if pt is not None:
                    d = self.pos.dist(pt)
                    if d < record:
                        closest, record = pt, d
            if closest is not None:
                engine.stroke = 255, 100
                engine.line(self.pos, closest)
    
    def show(self):
        fill = 255
        engine.circle(self.pos, 4)
        for ray in self.rays:
            ray.show()


class Ray:
    def __init__(self, pos, angle):
        self.pos = pos
        self.dir = Vector.from_angle(angle)
    
    def look_at(self, x, y):
        self.dir.x = x - self.dir.x
        self.dir.y = y - self.dir.y
        self.dir.xy = self.dir.normalize
    
    def show(self):
        stroke = 255
        engine.push()
        engine.translate(self.pos)
        engine.line(Vector(0, 0), self.dir * 10)
        engine.pop()
    
    def cast(self, wall):
        x1, y1 = wall.a
        x2, y2 = wall.b
        
        x3, y3 = self.pos
        x4, y4 = self.pos + self.dir
        
        den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if den != 0:
            t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
            u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / den
            if 0 < t < 1 and 0 < u:
                return Vector(x1 + t * (x2 - x1), y1 + t * (y2 - y1))
        return None


if __name__ == '__main__':
    engine.size(400, 400, OPENGL)
    
    engine.frame_rate = 0
    
    walls = [Boundary(a = random(engine.viewport), b = random(engine.viewport)) for i in range(5)]
    walls.append(Boundary(0, 0, engine.width - 1, 0))
    walls.append(Boundary(engine.width - 1, 0, engine.width - 1, engine.height - 1))
    walls.append(Boundary(engine.width - 1, engine.height - 1, 0, engine.height - 1))
    walls.append(Boundary(0, engine.height - 1, 0, 0))
    
    particle = Particle()
    
    @engine.event
    def mouse_moved(e):
        particle.pos.xy = e.pos
    
    @engine.draw
    def draw():
        engine.background = 0
        
        for wall in walls:
            wall.show()
            
        particle.show()
        particle.look(walls)
    
    engine.start()
