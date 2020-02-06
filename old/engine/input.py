import time
from dataclasses import dataclass
from . import events


@dataclass
class KeyData:
    time: int
    mod: int
    unicode: str
    scancode: int
    down: bool
    up: bool


@dataclass
class ButtonData:
    time: int
    pos: tuple
    down: bool
    up: bool


class Input:
    def __init__(self):
        self._other_events = []

        self._keys = {}
        self._buttons = {}

        self._pos = 0, 0
        self._pos_prev = 0, 0

        self._scroll = 0, 0
        self._scroll_prev = 0, 0

    @property
    def mouse_pos(self):
        return self._pos

    @property
    def scroll(self):
        return self._scroll
    
    def key_down(self, key):
        return key in self._keys
    
    def button_down(self, button):
        return button in self._buttons

    def get_events(self):
        event_list = []

        now = time.perf_counter_ns()

        for key, data in self._keys.copy().items():
            if data.down:
                event_list.append(events.Event(
                    events.KEY_DOWN,
                    now,
                    key = key,
                    mod = data.mod
                ))
                self._keys[key].down = False
            else:
                event_list.append(events.Event(
                    events.KEY_HOLD,
                    now,
                    key = key,
                    mod = data.mod
                ))
            if data.up:
                if now - data.time < 1e9 / 5:
                    event_list.append(events.Event(
                        events.KEY_PRESSED,
                        now,
                        key = key,
                        mod = data.mod
                    ))
                event_list.append(events.Event(
                    events.KEY_UP,
                    now,
                    key = key,
                    mod = data.mod
                ))
                del self._keys[key]

        for button, data in self._buttons.copy().items():
            if self._pos != self._pos_prev:
                event_list.append(events.Event(
                    events.MOUSE_DRAGGED,
                    now,
                    button = button,
                    pos = self._pos,
                    rel = (
                        self._pos[0] - self._pos_prev[0],
                        self._pos[1] - self._pos_prev[1]
                    )
                ))
            if data.down:
                event_list.append(events.Event(
                    events.MOUSE_BUTTON_DOWN,
                    now,
                    button = button,
                    pos = data.pos
                ))
                self._buttons[button].down = False
            else:
                event_list.append(events.Event(
                    events.MOUSE_BUTTON_HOLD,
                    now,
                    button = button,
                    pos = data.pos
                ))
            if data.up:
                if now - data.time < 1e9 / 5:
                    event_list.append(events.Event(
                        events.MOUSE_BUTTON_PRESSED,
                        now,
                        button = button,
                        pos = data.pos
                    ))
                event_list.append(events.Event(
                    events.MOUSE_BUTTON_UP,
                    now,
                    button = button,
                    pos = data.pos
                ))
                del self._buttons[button]

        if self._pos != self._pos_prev:
            event_list.append(events.Event(
                events.MOUSE_MOTION,
                now,
                pos = self._pos,
                rel = (
                    self._pos[0] - self._pos_prev[0],
                    self._pos[1] - self._pos_prev[1]
                )
            ))
            self._pos_prev = self._pos

        if self._scroll != self._scroll_prev:
            event_list.append(events.Event(
                events.MOUSE_SCROLL,
                now,
                pos = self._scroll,
                rel = (
                    self._scroll[0] - self._scroll_prev[0],
                    self._scroll[1] - self._scroll_prev[1]
                )
            ))
            self._scroll_prev = self._scroll

        event_list.extend(self._other_events)
        self._other_events = []

        return event_list
