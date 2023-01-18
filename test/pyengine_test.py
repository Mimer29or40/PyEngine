from pyengine import *


@instance
class Engine(AbstractEngine):
    def setup(self) -> None:
        pass

    def update(self, time: int, delta_time: int) -> None:
        pass

    def draw(self, time: int, delta_time: int) -> None:
        pass

    def destroy(self) -> None:
        pass


draw_rate(60)

print(viewport())

start()
