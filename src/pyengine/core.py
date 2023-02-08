from __future__ import annotations

import logging
import time as _time
from abc import ABC
from abc import abstractmethod
from typing import Optional
from typing import Tuple
from typing import Type

from .draw import draw_pop
from .draw import draw_push
from .io import destroy as io_destroy
from .io import setup as io_setup
from .io import update as io_update
from .io import window_on_close
from .io import window_swap
from .vector import vector2

logger = logging.getLogger(__name__)


class Engine:
    instance_cls: Optional[Type[AbstractEngine]] = None
    instance: Optional[AbstractEngine] = None

    should_run: bool = False

    start_time: int = 0


def instance(instance_cls: Type[AbstractEngine]) -> Type[AbstractEngine]:
    Engine.instance_cls = instance_cls
    return Engine.instance_cls


def time_nanoseconds() -> int:
    if Engine.start_time > 0:
        return _time.perf_counter_ns() - Engine.start_time
    return 0


def time_microseconds() -> float:
    return time_nanoseconds() / 1_000.0


def time_milliseconds() -> float:
    return time_nanoseconds() / 1_000_000.0


def time_seconds() -> float:
    return time_nanoseconds() / 1_000_000_000.0


class AbstractEngine(ABC):
    title: str = None
    size: Tuple[int, int] = 320, 300
    update_rate: int = 0
    draw_rate: int = 60

    @abstractmethod
    def setup(self) -> None:
        pass

    @abstractmethod
    def update(self, time: float, delta_time: float) -> None:
        pass

    @abstractmethod
    def draw(self, time: float, delta_time: float) -> None:
        pass

    @abstractmethod
    def destroy(self) -> None:
        pass


def start() -> None:
    try:
        setup()

        run()
    except Exception as e:
        logger.critical("An error caused the engine to stop:", exc_info=e)
    finally:
        destroy()


def stop():
    Engine.should_run = False


def setup() -> None:
    if Engine.instance_cls is None:
        raise ValueError("Must provide an AbstractEngine class")
    Engine.instance = Engine.instance_cls()

    Engine.start_time = _time.perf_counter_ns()

    Engine.should_run = True

    size = vector2(Engine.instance.size, dtype=int)
    if Engine.instance.title is None:
        Engine.instance.title = Engine.instance_cls.__name__
    io_setup(size, Engine.instance.title)

    Engine.instance.setup()


def run() -> None:
    current_time = time_nanoseconds()

    update_rate: int = Engine.instance.update_rate
    update_rate_inv: int = 0 if update_rate < 1 else 1_000_000_000 // update_rate
    update_delta: int
    update_last: int = current_time
    update_times: list[int] = list(range(500))

    draw_rate: int = Engine.instance.draw_rate
    draw_rate_inv: int = 0 if draw_rate < 1 else 1_000_000_000 // draw_rate
    draw_delta: int
    draw_last: int = current_time
    draw_times: list[int] = list(range(500))

    times_rate_inv: int = 1_000_000_000
    times_delta: int
    times_last: int = current_time

    while Engine.should_run:
        current_time = time_nanoseconds()
        update_delta = current_time - update_last
        if update_delta >= update_rate_inv:
            update_last = current_time

            update(current_time, update_delta)

            update_times.pop(0)
            update_times.append(time_nanoseconds() - current_time)

        current_time = time_nanoseconds()
        draw_delta = current_time - draw_last
        if draw_delta >= draw_rate_inv:
            draw_last = current_time

            draw(current_time, draw_delta)

            draw_times.pop(0)
            draw_times.append(time_nanoseconds() - current_time)

        current_time = time_nanoseconds()
        times_delta = current_time - times_last
        if times_delta >= times_rate_inv:
            times_last = current_time

            update_times.sort()
            update_len = len(update_times)
            update_avg = sum(update_times) / update_len / 1_000_000.0
            update_5 = update_times[int(update_len * 0.05)] / 1_000_000.0
            update_1 = update_times[int(update_len * 0.01)] / 1_000_000.0

            draw_times.sort(reverse=True)
            draw_len = len(draw_times)
            draw_avg = sum(draw_times) / draw_len / 1_000_000.0
            draw_5 = draw_times[int(draw_len * 0.05)] / 1_000_000.0
            draw_1 = draw_times[int(draw_len * 0.01)] / 1_000_000.0

            print(
                (
                    f"Update: avg={update_avg:0.3f}ms 5%={update_5:0.3f}ms 1%={update_1:0.3f}ms - "
                    f"Draw: avg={draw_avg:0.3f}ms 5%={draw_5:0.3f}ms 1%={draw_1:0.3f}ms"
                )
            )

        _time.sleep(0)


def destroy() -> None:
    Engine.instance.destroy()

    io_destroy()


def update(time: int, delta_time: int):
    io_update(time, delta_time)

    if window_on_close():
        Engine.should_run = False

    time_d: float = time / 1_000_000_000.0
    delta_time_d: float = delta_time / 1_000_000_000.0

    Engine.instance.update(time_d, delta_time_d)


def draw(time: int, delta_time: int):
    time_d: float = time / 1_000_000_000.0
    delta_time_d: float = delta_time / 1_000_000_000.0

    draw_push()
    Engine.instance.draw(time_d, delta_time_d)
    draw_pop()

    window_swap()
