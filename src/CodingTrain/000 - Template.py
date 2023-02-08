from pyengine import *


@instance
class Engine(AbstractEngine):
    title = "Coding Train"
    size = 400, 400
    update_rate = 0
    draw_rate = 60

    def setup(self) -> None:
        pass

    def update(self, time: float, delta_time: float) -> None:
        pass

    def draw(self, time: float, delta_time: float) -> None:
        draw_clear()

    def destroy(self) -> None:
        pass


if __name__ == "__main__":
    start()
