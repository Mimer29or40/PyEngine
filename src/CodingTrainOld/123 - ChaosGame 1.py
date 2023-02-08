from engine import *

if __name__ == "__main__":
    engine.size(400, 300, OPENGL)
    engine.frame_rate = 0

    a = Vector(0, 0)
    b = Vector(engine.width, 0)
    c = Vector(0, engine.height)
    point = random(engine.viewport)

    engine.background = 0
    engine.stroke = 255
    engine.weight = 8

    engine.point(a)
    engine.point(b)
    engine.point(c)

    @engine.draw
    def draw():
        for _ in range(100):
            engine.weight = 2
            engine.point(point)

            r = int(random(3))
            if r == 0:
                engine.stroke = 255, 0, 255
                point.xy = lerp(point, a, 0.5)
            elif r == 1:
                engine.stroke = 0, 255, 255
                point.xy = lerp(point, b, 0.5)
            elif r == 2:
                engine.stroke = 255, 255, 0
                point.xy = lerp(point, c, 0.5)

    engine.start()
