from __future__ import annotations

import time as _time
from typing import Optional
from typing import Type

from .types import AbstractEngine


def instance(instance_cls: Type[AbstractEngine]) -> Type[AbstractEngine]:
    state.instance_cls = instance_cls
    return state.instance_cls


def time() -> int:
    if state.start_time > 0:
        return _time.perf_counter_ns() - state.start_time
    return 0


def time_s() -> float:
    return time() / 1_000_000_000.0


def update_rate(update_rate: int = None, /) -> int:
    if update_rate is not None:
        state.update_rate = update_rate
        state.update_rate_inv = 0 if update_rate < 1 else 1_000_000_000 // update_rate
    return state.update_rate


def draw_rate(draw_rate: int = None, /) -> int:
    if draw_rate is not None:
        state.draw_rate = draw_rate
        state.draw_rate_inv = 0 if draw_rate < 1 else 1_000_000_000 // draw_rate
    return state.draw_rate


class State:
    __instance__: State = None

    def __new__(cls) -> State:
        if cls.__instance__ is None:
            cls.__instance__ = super().__new__(cls)
        return cls.__instance__

    def __init__(self):
        self.instance_cls: Optional[Type[AbstractEngine]] = None
        self.instance: Optional[AbstractEngine] = None

        self.should_run: bool = False

        self.start_time: int = 0
        self.current_time: int = 0

        self.update_rate: int = 0
        self.update_rate_inv: int = 0
        self.update_delta: int = 0
        self.update_last: int = 0

        self.draw_delta: int = 0
        self.draw_last: int = 0
        self.draw_rate: int = 0
        self.draw_rate_inv: int = 0


state: State = State()
