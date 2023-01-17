from engine import *

if __name__ == "__main__":
    engine.size(400, 400, OPENGL)

    n1, n2, n3 = 0.9, 0.3, 0.3
    m, a, b = 0.0, 1.0, 1.0
    osc = 0.0

    def super_shape(theta):
        part1 = abs((1.0 / a) * np.cos(theta * m / 4.0)) ** n2
        part2 = abs((1.0 / b) * np.sin(theta * m / 4.0)) ** n3
        part3 = (part1 + part2) ** (1.0 / n1)
        return 0.0 if part3 == 0 else 1.0 / part3

    @engine.draw
    def draw():
        global m, osc

        m = map(np.sin(osc), -1.0, 1.0, 0.0, 10.0)
        osc += 0.005

        engine.background = 51
        engine.translate(engine.viewport / 2)

        engine.stroke = 255
        engine.fill = None

        r = 200
        total = 360

        points = []
        for angle in np.linspace(0, 2 * np.pi, total):
            points.append(Vector.from_angle(angle) * r * super_shape(angle))

        engine.lines(*points)

    engine.start()
