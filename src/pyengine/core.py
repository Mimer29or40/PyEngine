from __future__ import annotations

import logging
import time as _time
from abc import ABC
from abc import abstractmethod
from typing import Optional
from typing import Tuple
from typing import Type

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
    current_time: int = 0

    update_delta: int = 0
    update_last: int = 0

    draw_delta: int = 0
    draw_last: int = 0


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

    io_setup(vector2(Engine.instance.size, dtype=int), "Title")

    Engine.instance.setup()


def run() -> None:
    Engine.update_last = Engine.draw_last = time_nanoseconds()

    update_rate = Engine.instance.update_rate
    update_rate_inv = 0 if update_rate < 1 else 1_000_000_000 // update_rate

    draw_rate = Engine.instance.draw_rate
    draw_rate_inv = 0 if draw_rate < 1 else 1_000_000_000 // draw_rate

    while Engine.should_run:
        Engine.current_time = time_nanoseconds()

        Engine.update_delta = Engine.current_time - Engine.update_last
        if Engine.update_delta >= update_rate_inv:
            Engine.update_last = Engine.current_time

            update(Engine.current_time, Engine.update_delta)

        Engine.draw_delta = Engine.current_time - Engine.draw_last
        if Engine.draw_delta >= draw_rate_inv:
            Engine.draw_last = Engine.current_time

            draw(Engine.current_time, Engine.draw_delta)

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

    Engine.instance.draw(time_d, delta_time_d)

    window_swap()
