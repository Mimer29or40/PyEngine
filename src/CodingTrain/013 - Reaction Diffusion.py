from engine import *

if __name__ == "__main__":
    np.set_printoptions(threshold=10000, linewidth=9999)

    engine.size(20, 20, OPENGL)

    curr = np.zeros((engine.height, engine.width, 2), dtype=float)
    prev = np.zeros((engine.height, engine.width, 2), dtype=float)
    curr[:, :, 0] = prev[:, :, 0] = 1

    start = random(20, engine.viewport - 20).astype(int)

    curr[start.x : start.x + 10, start.y : start.y + 10, 1] = 1

    da, db = 1.0, 0.5
    feed, k = 0.055, 0.062

    def update():
        global curr, prev

        a = slice(1, engine.height - 1, 1), slice(1, engine.width - 1, 1), 0
        b = slice(1, engine.height - 1, 1), slice(1, engine.width - 1, 1), 1

        laplace_a = 0
        laplace_a += np.roll(prev, +1, axis=1)[a] * 0.2
        laplace_a += np.roll(prev, -1, axis=1)[a] * 0.2
        laplace_a += np.roll(prev, +1, axis=0)[a] * 0.2
        laplace_a += np.roll(prev, -1, axis=0)[a] * 0.2
        laplace_a += np.roll(prev, (-1, -1), axis=(0, 1))[a] * 0.05
        laplace_a += np.roll(prev, (-1, +1), axis=(0, 1))[a] * 0.05
        laplace_a += np.roll(prev, (+1, -1), axis=(0, 1))[a] * 0.05
        laplace_a += np.roll(prev, (+1, +1), axis=(0, 1))[a] * 0.05

        laplace_b = 0
        laplace_b += np.roll(prev, +1, axis=1)[b] * 0.2
        laplace_b += np.roll(prev, -1, axis=1)[b] * 0.2
        laplace_b += np.roll(prev, +1, axis=0)[b] * 0.2
        laplace_b += np.roll(prev, -1, axis=0)[b] * 0.2
        laplace_b += np.roll(prev, (-1, -1), axis=(0, 1))[b] * 0.05
        laplace_b += np.roll(prev, (-1, +1), axis=(0, 1))[b] * 0.05
        laplace_b += np.roll(prev, (+1, -1), axis=(0, 1))[b] * 0.05
        laplace_b += np.roll(prev, (+1, +1), axis=(0, 1))[b] * 0.05

        curr[a] = (
            prev[a] + (da * laplace_a - prev[a] * prev[b] * prev[b] + feed * (1 - prev[a])) * 1
        )
        curr[b] = (
            prev[b] + (db * laplace_b + prev[a] * prev[b] * prev[b] - (k + feed) * prev[b]) * 1
        )

        curr = constrain(curr, 0.0, 1.0)

        curr, prev = prev, curr

    @engine.draw
    def draw():
        for _ in range(1):
            update()

        out = (curr[:, :, 0] - curr[:, :, 1]) * 255
        # curr[start.x:start.x + 10, start.y:start.y + 10, 1] = 1

        # print(out)

        engine.load_pixels()
        # engine.pixels[:, :, 0] = out
        # engine.pixels[:, :, 1] = out
        # engine.pixels[:, :, 2] = out
        engine.pixels[start.x : start.x + 10, start.y : start.y + 10, 0] = 50
        engine.pixels[start.x : start.x + 10, start.y : start.y + 10, 1] = 200
        engine.pixels[start.x : start.x + 10, start.y : start.y + 10, 2] = 100
        engine.update_pixels()

    engine.start()
