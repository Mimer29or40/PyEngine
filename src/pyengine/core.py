import logging
import time as _time

import pygame

from .io import destroy as io_destroy
from .io import setup as io_setup
from .io import update as io_update
from .io import window_on_close
from .state import State
from .state import time
from .vector import vector2

logger = logging.getLogger(__name__)

state: State = State()


def start(width: int = 320, height: int = 200, renderer=None) -> None:
    try:
        state.viewport = vector2(width, height, dtype=int)

        setup()

        run()
    except Exception as e:
        logger.critical("An error caused the engine to stop:", exc_info=e)
    finally:
        destroy()


def stop():
    state.should_run = False


def setup() -> None:
    if state.instance_cls is None:
        raise ValueError("Must provide an AbstractEngine class")
    state.instance = state.instance_cls()

    state.start_time = _time.perf_counter_ns()

    state.should_run = True

    io_setup(state.viewport, "Title")

    state.instance.setup()


def run() -> None:
    state.update_last = state.draw_last = time()

    while state.should_run:
        state.current_time = time()

        state.update_delta = state.current_time - state.update_last
        if state.update_delta >= state.update_rate_inv:
            state.update_last = state.current_time

            update(state.current_time, state.update_delta)

        state.draw_delta = state.current_time - state.draw_last
        if state.draw_delta >= state.draw_rate_inv:
            state.draw_last = state.current_time

            draw(state.current_time, state.draw_delta)

        _time.sleep(0)


def destroy() -> None:
    state.instance.destroy()

    io_destroy()


def update(time: int, delta_time: int):
    io_update(time, delta_time)

    if window_on_close():
        state.should_run = False

    time_d: float = time / 1_000_000_000.0
    delta_time_d: float = delta_time / 1_000_000_000.0

    state.instance.update(time_d, delta_time_d)


def draw(time: int, delta_time: int):
    time_d: float = time / 1_000_000_000.0
    delta_time_d: float = delta_time / 1_000_000_000.0

    state.instance.draw(time_d, delta_time_d)

    pygame.display.flip()
