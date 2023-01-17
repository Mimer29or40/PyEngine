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
        self._fov = 60
        self._heading = 0
        self._ray_count = 90
        self.pos = Vector(engine.width / 4, engine.height / 2)
        self.rays = []
        
        self._update_rays()
    
    @property
    def fov(self):
        return self._fov
    
    @fov.setter
    def fov(self, value):
        self._fov = value
        self._update_rays()
    
    @property
    def heading(self):
        return self._heading
    
    @heading.setter
    def heading(self, value):
        self._heading = value
        self._update_rays()
    
    def _update_rays(self):
        rng = np.linspace(-self.fov / 2, self.fov / 2, self._ray_count)
        self.rays = [Ray(self.pos, radians(a) + self.heading) for a in rng]
    
    def look(self, walls):
        scene = []
        for ray in self.rays:
            closest, record = None, 500_000_000
            for wall in walls:
                pt = ray.cast(wall)
                if pt is not None:
                    d = self.pos.dist(pt)
                    a = ray.dir.heading() - self.heading
                    if not engine.mouse_pressed:
                        d *= cos(a)
                    if d < record:
                        closest, record = pt, d
            if closest is not None:
                engine.stroke = 255, 100
                engine.line(self.pos, closest)
            scene.append(record)
        return scene
    
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
    engine.size(800, 400, OPENGL)
    engine.frame_rate = 0
    
    view = Vector(engine.width / 2, engine.height)
    
    walls = [Boundary(a = random(view), b = random(view)) for i in range(5)]
    walls.append(Boundary(0, 0, view.x - 1, 0))
    walls.append(Boundary(view.x - 1, 0, view.x - 1, view.y - 1))
    walls.append(Boundary(view.x - 1, view.y - 1, 0, view.y - 1))
    walls.append(Boundary(0, view.y - 1, 0, 0))
    
    particle = Particle()
    
    @engine.event
    def mouse_wheel(e):
        particle.fov += e.count
    
    @engine.event
    def mouse_moved(e):
        particle.heading = (e.pos - particle.pos).heading()
    
    @engine.event
    def key_held(e):
        dir = (engine.mouse - particle.pos).normalize
        norm = Vector(dir.y, -dir.x)
        if e.key == 'w':
            particle.pos += dir
        if e.key == 'a':
            particle.pos += norm
        if e.key == 's':
            particle.pos -= dir
        if e.key == 'd':
            particle.pos -= norm
        particle.pos = np.clip(particle.pos, 1, view - 1)
        particle.heading = dir.heading()
    
    @engine.draw
    def draw():
        engine.background = 0
        
        for wall in walls:
            wall.show()
            
        particle.show()
        scene = particle.look(walls)
        
        engine.translate(view.x, 0)
        engine.stroke = None
        
        w = view.x / len(scene)
        w_sq = view.x * view.x
        for i, dist in enumerate(scene):
            sq = dist * dist
            h = map(dist, 0, view.x, view.y, 0)
            
            engine.fill = int(map(sq, 0, w_sq, 255, 0))
            engine.rect_mode = CENTER
            engine.rect(
                Vector(i * w + w / 2, view.y / 2),
                Vector(w + 1, h)
            )
    
    engine.start()
