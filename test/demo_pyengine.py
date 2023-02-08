from pyengine import *


@instance
class EngineDemo(AbstractEngine):
    size = 640, 400
    update_rate = 0
    draw_rate = 60

    def setup(self) -> None:
        pass

    def update(self, time: float, delta_time: float) -> None:
        pass

    def draw(self, time: float, delta_time: float) -> None:
        draw_clear()

        draw_color(color(0, 255, 0, 255))
        draw_lines(vector2(0, 0), mouse_pos(), vector2(window_size()[0], 0))

        draw_color(color(255, 0, 0, 255))
        draw_point(mouse_pos())

    def destroy(self) -> None:
        pass


start()
