from pyengine import *


class Star:
    @classmethod
    def _max_depth(cls) -> float:
        return window_size()[0] * 10

    @classmethod
    def _new_vector(cls) -> VectorType:
        size: VectorType = window_size()
        pos: VectorType = vector_random(2, False) * size * 5
        z = random(cls._max_depth())
        return vector3(pos, z, dtype=float)

    def __init__(self):
        self.pos = self._new_vector()
        self.pz = self.pos[2]

    def update(self, speed: float):
        self.pos[2] -= speed
        if self.pos[2] < 1:
            self.pos = self._new_vector()
            self.pz = self.pos[2]

    def draw(self):
        draw_color(color(255, 255, 255, 255))

        size: VectorType = window_size()

        pos: VectorType = (self.pos[0:2] / self.pos[2]) * size / 2
        pos_prev: VectorType = (self.pos[0:2] / self.pz) * size / 2

        radius: float = map_number(self.pos[2], 0, self._max_depth(), 16, -1)
        draw_thickness(radius)

        draw_point(pos)
        draw_line(pos_prev, pos)

        self.pz = self.pos[2]


@instance
class Engine(AbstractEngine):
    title = "001 - Star Field"
    size = 400, 400
    update_rate = 0
    draw_rate = 60

    stars: list[Star]

    def setup(self) -> None:
        self.stars = [Star() for _ in range(50)]

    def update(self, time: float, delta_time: float) -> None:
        speed: float = map_number(mouse_pos()[0], 0, window_pos()[0], 0, 50)

        for s in self.stars:
            s.update(speed)

    def draw(self, time: float, delta_time: float) -> None:
        draw_clear()

        draw_translate(window_size() / 2)

        for s in self.stars:
            s.draw()

    def destroy(self) -> None:
        pass


if __name__ == "__main__":
    start()
