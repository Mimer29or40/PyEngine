import logging
import time as _time

from .state import State
from .state import time
from .vector import Vector2

logger = logging.getLogger(__name__)

state: State = State()


def start(width: int = 320, height: int = 200, renderer=None) -> None:
    try:
        state.viewport = Vector2(width, height)

        setup()

        run()
    except Exception as e:
        logger.critical("An error caused the engine to stop:", exc_info=e)
    finally:
        destroy()


def setup() -> None:
    if state.instance_cls is None:
        raise ValueError("Must provide an AbstractEngine class")
    state.instance = state.instance_cls()

    state.start_time = _time.perf_counter_ns()

    state.should_run = True

    state.instance.setup()


def run() -> None:
    state.update_last = state.draw_last = time()

    while state.should_run:
        state.current_time = time()

        state.update_delta = state.current_time - state.update_last
        if state.update_delta >= state.update_rate_inv:
            state.update_last = state.current_time

            state.instance.update(state.current_time, state.update_delta)

        state.draw_delta = state.current_time - state.draw_last
        if state.draw_delta >= state.draw_rate_inv:
            state.draw_last = state.current_time

            state.instance.draw(state.current_time, state.draw_delta)


def destroy() -> None:
    state.instance.destroy()
