from engine import *

if __name__ == "__main__":
    engine.size(400, 300, OPENGL2)

    engine.frame_rate = 60

    curr = np.zeros((int(engine.height), int(engine.width)), dtype=float)
    prev = np.zeros((int(engine.height), int(engine.width)), dtype=float)
    dampening = 0.95

    index = (slice(1, int(engine.height) - 1, 1), slice(1, int(engine.width) - 1, 1))

    @engine.event
    def mouse_dragged(e):
        prev[int(e.pos.y), int(e.pos.x)] = 500

    @engine.draw
    def draw():
        global curr, prev

        engine.background = 0
        engine.translate(engine.viewport)

        curr[index] = (
            (
                np.roll(prev, -1, axis=1)[index]
                + np.roll(prev, 1, axis=1)[index]
                + np.roll(prev, -1, axis=0)[index]
                + np.roll(prev, 1, axis=0)[index]
            )
            / 2
            - curr[index]
        ) * dampening

        engine.load_pixels()
        engine.pixels[:, :, 0] = curr
        engine.pixels[:, :, 1] = curr
        engine.pixels[:, :, 2] = curr
        engine.update_pixels()

        prev, curr = curr, prev

    engine.start()
