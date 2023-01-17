# http://www.songho.ca/opengl/gl_projectionmatrix.html
# https://www.scratchapixel.com/lessons/3d-basic-rendering/perspective-and-orthographic-projection-matrix/projection-matrix-GPU-rendering-pipeline-clipping
# https://github.com/vicrucann/shader-3dcurve/tree/master/src

from engine import *


class Circle:
    def __init__(self, x, y, r):
        self.pos = Vector(x, y)
        self.r = r

    def dist(self, pos):
        return (pos - self.pos).magnitude - self.r

    def show(self):
        pygame.draw.circle(engine.screen, white, self.pos, self.r, 1)


if __name__ == "__main__":
    engine.size(800, 800, PYGAME)

    engine.frame_rate = 60

    obj = []
    for _ in range(10):
        obj.append(Circle(random(100, 700), random(100, 500), random(10, 100)))

    white = Color(255)

    start_pos = Vector(0, 0)
    end_pos = Vector(0, 0)

    @engine.event
    def mouse_pressed(e):
        if e.button == LEFT:
            start_pos.xy = e.pos

    @engine.event
    def mouse_moved(e):
        end_pos.xy = e.pos

    @engine.draw
    def draw():
        # engine.background = 100
        engine.stroke = 255
        engine.weight = 5

        engine.line(Vector(10, 10), engine.mouse)
        # engine.line(-0.5, -0.5, 0.5, 0.5)

        # for obj in obj:
        #     obj.show()
        #
        # dir = (end_pos - start_pos).normalize
        # if dir.magnitude == 0:
        #     dir = e.Vector([1, 0])
        # pygame.draw.line(engine.screen, white, start_pos, dir * 1000 + start_pos, 1)
        #
        # r = 10
        # pos = start_pos.copy()
        # while 1 < r < 500:
        #     r = int(abs(min([obj.dist(pos) for obj in obj])))
        #     if r > 1:
        #         pygame.draw.circle(engine.screen, white, pos, r, 1)
        #     pos += dir * r
        pass

    # engine.start()
    mat = Matrix.identity(4)
    print(mat)
    mat.translate(Vector(1, 0, 0))
    print(mat)
    mat.translate(Vector(1, 1, 1))
    print(mat)
