from engine import *


if __name__ == '__main__':
    engine.size(400, 300, OPENGL)
    engine.frame_rate = 0
    
    n = 5
    points = []
    percent = 0.5
    curr, prev = Vector(0, 0), Vector(0, 0)
    
    for i in range(n):
        v = Vector.from_angle(i / n * TWO_PI) * engine.width / 2
        v += engine.viewport / 2
        points.append(v)
    
    def reset():
        curr.xy = random(engine.viewport)
        
        engine.background = 0
        engine.stroke = 255
        engine.weight = 8
        
        for p in points:
            engine.point(p)
    
    reset()
    
    @engine.draw
    def draw():
        # if engine.frame_count % 100 == 0:
            # reset()
        
        engine.weight = 1
        engine.stroke = 255
        for _ in range(100):
            next = points[int(random(len(points)))]
            if not next == prev:
                curr.xy = lerp(curr, next, percent)
                engine.point(curr)
            prev.xy = next
    
    engine.start()
