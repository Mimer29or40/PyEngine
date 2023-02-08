from __future__ import annotations

import sys
from dataclasses import dataclass
from enum import Enum
from enum import Flag
from typing import ClassVar
from typing import Dict
from typing import Final
from typing import List
from typing import Optional
from typing import Sequence

import numpy as np
import pygame

from .vector import VectorType
from .vector import vector2


@dataclass
class Input:
    PRESS: ClassVar[int] = 0
    RELEASE: ClassVar[int] = 1
    REPEAT: ClassVar[int] = 2

    state: int = -1
    state_change: int = -1

    held: bool = False
    held_time: int = sys.maxsize

    down_time: int = 0
    down_count: int = 0


@dataclass
class ButtonInput(Input):
    dragging: bool = False
    down_pos: Final[VectorType] = vector2(0, 0, dtype=float)


class Button(Enum):
    @classmethod
    def _missing_(cls, value) -> Button:
        return Button.UNKNOWN

    def __repr__(self) -> str:
        return f"Button.{self.name}"

    UNKNOWN = 0

    ONE = 1
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8

    LEFT = ONE
    MIDDLE = TWO
    RIGHT = THREE


class Key(Enum):
    @classmethod
    def _missing_(cls, value) -> Key:
        return Key.UNKNOWN

    def __repr__(self) -> str:
        return f"Key.{self.name}"

    UNKNOWN = -1

    A = pygame.K_a
    B = pygame.K_b
    C = pygame.K_c
    D = pygame.K_d
    E = pygame.K_e
    F = pygame.K_f
    G = pygame.K_g
    H = pygame.K_h
    I = pygame.K_i  # noqa
    J = pygame.K_j
    K = pygame.K_k
    L = pygame.K_l
    M = pygame.K_m
    N = pygame.K_n
    O = pygame.K_o  # noqa
    P = pygame.K_p
    Q = pygame.K_q
    R = pygame.K_r
    S = pygame.K_s
    T = pygame.K_t
    U = pygame.K_u
    V = pygame.K_v
    W = pygame.K_w
    X = pygame.K_x
    Y = pygame.K_y
    Z = pygame.K_z

    K1 = pygame.K_1
    K2 = pygame.K_2
    K3 = pygame.K_3
    K4 = pygame.K_4
    K5 = pygame.K_5
    K6 = pygame.K_6
    K7 = pygame.K_7
    K8 = pygame.K_8
    K9 = pygame.K_9
    K0 = pygame.K_0

    GRAVE = pygame.K_BACKQUOTE
    MINUS = pygame.K_MINUS
    EQUAL = pygame.K_EQUALS
    L_BRACKET = pygame.K_LEFTBRACKET
    R_BRACKET = pygame.K_RIGHTBRACKET
    BACKSLASH = pygame.K_BACKSLASH
    SEMICOLON = pygame.K_SEMICOLON
    APOSTROPHE = pygame.K_QUOTE
    COMMA = pygame.K_COMMA
    PERIOD = pygame.K_PERIOD
    SLASH = pygame.K_SLASH

    F1 = pygame.K_F1
    F2 = pygame.K_F2
    F3 = pygame.K_F3
    F4 = pygame.K_F4
    F5 = pygame.K_F5
    F6 = pygame.K_F6
    F7 = pygame.K_F7
    F8 = pygame.K_F8
    F9 = pygame.K_F9
    F10 = pygame.K_F10
    F11 = pygame.K_F11
    F12 = pygame.K_F12
    F13 = pygame.K_F13
    F14 = pygame.K_F14
    F15 = pygame.K_F15
    # F16 = pygame.K_F16
    # F17 = pygame.K_F17
    # F18 = pygame.K_F18
    # F19 = pygame.K_F19
    # F20 = pygame.K_F20
    # F21 = pygame.K_F21
    # F22 = pygame.K_F22
    # F23 = pygame.K_F23
    # F24 = pygame.K_F24
    # F25 = pygame.K_F25

    UP = pygame.K_UP
    DOWN = pygame.K_DOWN
    LEFT = pygame.K_LEFT
    RIGHT = pygame.K_RIGHT

    TAB = pygame.K_TAB
    CAPS_LOCK = pygame.K_CAPSLOCK
    ENTER = pygame.K_RETURN
    BACKSPACE = pygame.K_BACKSPACE
    SPACE = pygame.K_SPACE

    L_SHIFT = pygame.K_LSHIFT
    R_SHIFT = pygame.K_RSHIFT
    L_CONTROL = pygame.K_LCTRL
    R_CONTROL = pygame.K_RCTRL
    L_ALT = pygame.K_LALT
    R_ALT = pygame.K_RALT
    L_SUPER = pygame.K_LSUPER
    R_SUPER = pygame.K_RSUPER

    MENU = pygame.K_MENU
    ESCAPE = pygame.K_ESCAPE
    PRINT_SCREEN = pygame.K_PRINTSCREEN
    SCROLL_LOCK = pygame.K_SCROLLLOCK
    PAUSE = pygame.K_PAUSE
    INSERT = pygame.K_INSERT
    DELETE = pygame.K_DELETE
    HOME = pygame.K_HOME
    END = pygame.K_END
    PAGE_UP = pygame.K_PAGEUP
    PAGE_DOWN = pygame.K_PAGEDOWN

    KP_0 = pygame.K_KP_0
    KP_1 = pygame.K_KP_1
    KP_2 = pygame.K_KP_2
    KP_3 = pygame.K_KP_3
    KP_4 = pygame.K_KP_4
    KP_5 = pygame.K_KP_5
    KP_6 = pygame.K_KP_6
    KP_7 = pygame.K_KP_7
    KP_8 = pygame.K_KP_8
    KP_9 = pygame.K_KP_9

    NUM_LOCK = pygame.K_NUMLOCK
    KP_DIVIDE = pygame.K_KP_DIVIDE
    KP_MULTIPLY = pygame.K_KP_MULTIPLY
    KP_SUBTRACT = pygame.K_KP_MINUS
    KP_ADD = pygame.K_KP_PLUS
    KP_DECIMAL = pygame.K_KP_PERIOD
    KP_EQUAL = pygame.K_KP_EQUALS
    KP_ENTER = pygame.K_KP_ENTER


class Modifier(Flag):
    NONE = pygame.KMOD_NONE

    L_SHIFT = pygame.KMOD_LSHIFT  # BIT 1
    R_SHIFT = pygame.KMOD_RSHIFT  # BIT 2
    SHIFT = L_SHIFT | R_SHIFT  # BIT 1|2

    # _04 = 4  # BIT 3
    # _08 = 8  # BIT 4
    # _16 = 16  # BIT 5
    # _32 = 32  # BIT 6

    L_CONTROL = pygame.KMOD_LCTRL  # BIT 7
    R_CONTROL = pygame.KMOD_RCTRL  # BIT 8
    CTRL = L_CONTROL | R_CONTROL  # BIT 7|8

    L_ALT = pygame.KMOD_LALT  # BIT 9
    R_ALT = pygame.KMOD_RALT  # BIT 10
    ALT = L_ALT | R_ALT  # BIT 9|10

    L_META = pygame.KMOD_LMETA  # BIT 11
    R_META = pygame.KMOD_RMETA  # BIT 12
    META = L_META | R_META  # BIT 11|12

    NUM = pygame.KMOD_NUM  # BIT 13
    CAPS = pygame.KMOD_CAPS  # BIT 14
    MODE = pygame.KMOD_MODE  # BIT 15
    SCROLL = 32768  # BIT 16


MOUSE_SHOWN: Final[int] = 0
MOUSE_HIDDEN: Final[int] = 1
MOUSE_CAPTURED: Final[int] = 2


class IO:
    input_hold_frequency: int = 1_000_000
    input_double_press_delay: int = 200_000_000

    # ---------- Window ---------- #
    window: pygame.Surface

    window_close_requested: bool = False
    window_on_close: bool = False

    window_focused: bool = False
    window_focused_change: Optional[bool] = None
    window_on_focused: bool = False

    window_minimized: bool = False
    window_minimized_change: Optional[bool] = None
    window_on_minimized: bool = False

    window_maximized: bool = False
    window_maximized_change: Optional[bool] = None
    window_on_maximized: bool = False

    window_pos: Final[VectorType] = vector2(0, 0, dtype=int)
    window_pos_change: Optional[VectorType] = None
    window_on_pos: bool = False

    window_size: Final[VectorType] = vector2(0, 0, dtype=int)
    window_size_change: Optional[VectorType] = None
    window_on_size: bool = False

    window_content_scale: Final[VectorType] = vector2(0, 0, dtype=float)
    window_content_scale_change: Optional[VectorType] = None
    window_on_content_scale: bool = False

    window_framebuffer_size: Final[VectorType] = vector2(0, 0, dtype=int)
    window_framebuffer_size_change: Optional[VectorType] = None
    window_on_framebuffer_size: bool = False

    window_refresh_requested: bool = False
    window_on_refresh: bool = False

    window_dropped: Final[List[str]] = []
    window_dropped_change: Final[List[str]] = []
    window_on_dropped: bool = False

    # ---------- Mouse ---------- #
    mouse_state: int = MOUSE_SHOWN
    mouse_state_change: Optional[int] = None
    mouse_on_state: bool = False

    mouse_entered: bool = False
    mouse_entered_change: Optional[bool] = None
    mouse_on_entered: bool = False

    mouse_pos: Final[VectorType] = vector2(0, 0, dtype=int)
    mouse_pos_change: Optional[VectorType] = None
    mouse_on_pos: bool = False
    mouse_pos_do_event: bool = True

    mouse_pos_delta: Final[VectorType] = vector2(0, 0, dtype=int)

    mouse_scroll: Final[VectorType] = vector2(0, 0, dtype=int)
    mouse_scroll_change: Optional[VectorType] = None
    mouse_on_scroll: bool = False

    mouse_button_states: Dict[Button, ButtonInput] = {}

    mouse_button_down: Final[List[Button]] = []
    mouse_button_up: Final[List[Button]] = []
    mouse_button_repeated: Final[List[Button]] = []
    mouse_button_held: Final[List[Button]] = []
    mouse_button_dragged: Final[List[Button]] = []

    # ---------- Keyboard ---------- #
    keyboard_typed: str = ""
    keyboard_typed_changes: str = ""
    keyboard_on_typed: bool = False

    keyboard_key_states: Dict[Key, Input] = {}

    keyboard_key_down: List[Key] = []
    keyboard_key_up: List[Key] = []
    keyboard_key_repeated: List[Key] = []
    keyboard_key_held: List[Key] = []

    # ---------- Modifier ---------- #
    modifier_active: Modifier = Modifier.NONE

    modifier_include_lock_mods: bool = False


def setup(size: VectorType, title: str) -> None:
    from pygame._sdl2 import Window

    pygame.init()
    print("PYGAME INIT")

    # ---------- Window ---------- #
    IO.window = pygame.display.set_mode(size, pygame.RESIZABLE)
    pygame.display.flip()

    IO.window_focused = pygame.mouse.get_focused()
    IO.window_minimized = False
    IO.window_maximized = False

    _window = Window.from_display_module()
    IO.window_pos[:] = _window.position
    IO.window_size[:] = _window.size

    window_title(title)

    # ---------- Mouse ---------- #
    IO.mouse_pos[:] = pygame.mouse.get_pos()

    for button in Button:
        IO.mouse_button_states[button] = ButtonInput()

    mouse_show()
    mouse_raw_input(True)
    mouse_sticky(False)

    # ---------- Keyboard ---------- #
    for key in Key:
        IO.keyboard_key_states[key] = Input()

    keyboard_sticky(False)

    # ---------- Modifier ---------- #
    modifier_include_lock_mods(False)


def update(time: int, delta_time: int) -> None:
    for e in pygame.event.get():
        # ---------- Window ---------- #
        if e.type == pygame.QUIT:
            IO.window_close_requested = True
        elif e.type == pygame.WINDOWCLOSE:
            IO.window_close_requested = True
        elif e.type == pygame.WINDOWFOCUSGAINED:
            IO.window_focused_change = True
        elif e.type == pygame.WINDOWFOCUSLOST:
            IO.window_focused_change = False
        elif e.type == pygame.WINDOWMINIMIZED:
            IO.window_minimized_change = True
        elif e.type == pygame.WINDOWMAXIMIZED:
            IO.window_maximized_change = True
        elif e.type == pygame.WINDOWRESTORED:
            IO.window_minimized_change = False
            IO.window_maximized_change = False
        elif e.type == pygame.WINDOWMOVED:
            IO.window_pos_change = vector2(e.x, e.y)
        elif e.type == pygame.WINDOWRESIZED:
            IO.window_size_change = vector2(e.x, e.y)
        elif e.type == pygame.DROPFILE:
            IO.window_dropped_change.append(e.file)
        elif e.type == pygame.WINDOWENTER:
            IO.mouse_entered_change = True
        elif e.type == pygame.WINDOWLEAVE:
            IO.mouse_entered_change = False
        elif e.type == pygame.MOUSEMOTION:
            IO.mouse_pos_change = vector2(*e.pos)
        elif e.type == pygame.MOUSEWHEEL:
            IO.mouse_scroll_change = vector2(e.x, e.y)
        elif e.type == pygame.MOUSEBUTTONDOWN:
            IO.mouse_button_states[Button(e.button)].state_change = Input.PRESS
        elif e.type == pygame.MOUSEBUTTONUP:
            IO.mouse_button_states[Button(e.button)].state_change = Input.RELEASE
        elif e.type == pygame.KEYDOWN:
            IO.keyboard_key_states[Key(e.key)].state_change = Input.PRESS
            IO.modifier_active = Modifier(e.mod)
            if not IO.modifier_include_lock_mods:
                IO.modifier_active &= ~(Modifier.NUM | Modifier.CAPS | Modifier.SCROLL)
        elif e.type == pygame.KEYUP:
            IO.keyboard_key_states[Key(e.key)].state_change = Input.RELEASE
            IO.modifier_active = Modifier(e.mod)
            if not IO.modifier_include_lock_mods:
                IO.modifier_active &= ~(Modifier.NUM | Modifier.CAPS | Modifier.SCROLL)
        elif e.type == pygame.TEXTINPUT:
            IO.keyboard_typed_changes += e.text
        else:
            print(f"Unhandled Pygame Event: {e}")

        # ACTIVEEVENT       gain, state
        # JOYAXISMOTION joy(deprecated), instance_id, axis, value
        # JOYBALLMOTION joy(deprecated), instance_id, ball, rel
        # JOYHATMOTION joy(deprecated), instance_id, hat, value
        # JOYBUTTONUP joy(deprecated), instance_id, button
        # JOYBUTTONDOWN joy(deprecated), instance_id, button
        # USEREVENT code

        # AUDIODEVICEADDED   which, iscapture (SDL backend >= 2.0.4)
        # AUDIODEVICEREMOVED which, iscapture (SDL backend >= 2.0.4)
        # FINGERMOTION       touch_id, finger_id, x, y, dx, dy
        # FINGERDOWN         touch_id, finger_id, x, y, dx, dy
        # FINGERUP           touch_id, finger_id, x, y, dx, dy
        # MULTIGESTURE       touch_id, x, y, pinched, rotated, num_fingers
        # TEXTEDITING        text, start, length

        # DROPTEXT                 text (SDL backend >= 2.0.5)
        # MIDIIN
        # MIDIOUT
        # CONTROLLERDEVICEADDED    device_index
        # JOYDEVICEADDED           device_index
        # CONTROLLERDEVICEREMOVED  instance_id
        # JOYDEVICEREMOVED         instance_id
        # CONTROLLERDEVICEREMAPPED instance_id
        # KEYMAPCHANGED            (SDL backend >= 2.0.4)
        # CLIPBOARDUPDATE
        # RENDER_TARGETS_RESET     (SDL backend >= 2.0.2)
        # RENDER_DEVICE_RESET      (SDL backend >= 2.0.4)
        # LOCALECHANGED            (SDL backend >= 2.0.14)

        # WINDOWSHOWN            Window became shown
        # WINDOWHIDDEN           Window became hidden
        # WINDOWEXPOSED          Window got updated by some external event
        # WINDOWSIZECHANGED      Window changed its size
        # WINDOWTAKEFOCUS        Window was offered focus (SDL backend >= 2.0.5)
        # WINDOWHITTEST          Window has a special hit test (SDL backend >= 2.0.5)
        # WINDOWICCPROFCHANGED   Window ICC profile changed (SDL backend >= 2.0.18)
        # WINDOWDISPLAYCHANGED   Window moved on a new display (SDL backend >= 2.0.18)

    # ---------- Window ---------- #
    IO.window_on_close = False
    if IO.window_close_requested:
        IO.window_on_close = True

        IO.window_close_requested = False

    IO.window_on_focused = False
    if IO.window_focused_change is not None:
        IO.window_focused = IO.window_focused_change
        IO.window_on_focused = True

        IO.window_focused_change = None

    IO.window_on_minimized = False
    if IO.window_minimized_change is not None and IO.window_minimized != IO.window_minimized_change:
        IO.window_minimized = IO.window_minimized_change
        IO.window_on_minimized = True

        IO.window_minimized_change = None

    IO.window_on_maximized = False
    if IO.window_maximized_change is not None and IO.window_maximized != IO.window_maximized_change:
        IO.window_maximized = IO.window_maximized_change
        IO.window_on_maximized = True

        IO.WINDOW_MAXIMIZED_CHANGE = None

    IO.window_on_pos = False
    if IO.window_pos_change is not None:
        IO.window_pos[:] = IO.window_pos_change
        IO.window_on_pos = True

        IO.window_pos_change = None

    IO.window_on_size = False
    if IO.window_size_change is not None:
        IO.window_size[:] = IO.window_size_change
        IO.window_on_size = True

        IO.window_size_change = None

    IO.window_on_content_scale = False
    if IO.window_content_scale_change is not None:
        IO.window_content_scale[:] = IO.window_content_scale_change
        IO.window_on_content_scale = True

        IO.window_content_scale_change = None

    IO.window_on_framebuffer_size = False
    if IO.window_framebuffer_size_change is not None:
        IO.window_framebuffer_size[:] = IO.window_framebuffer_size_change
        IO.window_on_framebuffer_size = True

        IO.window_framebuffer_size_change = None

    IO.window_on_refresh = False
    if IO.window_refresh_requested:
        IO.window_on_refresh = True

        IO.window_refresh_requested = False

    IO.window_on_dropped = False
    if len(IO.window_dropped_change) > 0:
        IO.window_dropped.clear()
        IO.window_dropped.extend(IO.window_dropped_change)

        IO.window_dropped_change.clear()

    # ---------- Mouse ---------- #
    IO.mouse_on_state = False
    if IO.mouse_state_change is not None:
        IO.mouse_state = IO.mouse_state_change
        IO.mouse_on_state = True

        if IO.mouse_state == MOUSE_SHOWN:
            pygame.event.set_grab(False)
            pygame.mouse.set_visible(True)
            IO.mouse_pos_do_event = True
        elif IO.mouse_state == MOUSE_HIDDEN:
            pygame.event.set_grab(False)
            pygame.mouse.set_visible(False)
            IO.mouse_pos_do_event = True
        elif IO.mouse_state == MOUSE_CAPTURED:
            pygame.event.set_grab(True)
            # pygame.mouse.set_visible(False)
            size = IO.window.get_width() / 2, IO.window.get_height() / 2
            pygame.mouse.set_pos(size)
            IO.mouse_pos_do_event = False

        IO.mouse_state_change = None

    IO.mouse_on_entered = False
    if IO.mouse_entered_change is not None:
        IO.mouse_entered = IO.mouse_entered_change
        IO.mouse_on_entered = True

        if IO.mouse_entered:
            IO.mouse_pos_change = IO.mouse_pos.copy()

        IO.mouse_entered_change = None

    IO.mouse_on_pos = False
    IO.mouse_pos_delta[:] = 0
    if IO.mouse_pos_change is not None:
        IO.mouse_pos_delta[:] = IO.mouse_pos_change - IO.mouse_pos
        IO.mouse_pos[:] = IO.mouse_pos_change
        if IO.mouse_pos_do_event:
            IO.mouse_on_pos = True
        IO.mouse_pos_do_event = True
        if IO.mouse_state == MOUSE_CAPTURED:
            size = IO.window.get_width() / 2, IO.window.get_height() / 2
            if any(IO.mouse_pos != size):
                pygame.mouse.set_pos(size)
                IO.mouse_pos_do_event = False

        IO.mouse_pos_change = None

    IO.mouse_on_scroll = False
    IO.mouse_scroll[:] = 0
    if IO.mouse_scroll_change is not None:
        IO.mouse_scroll[:] = IO.mouse_scroll_change
        IO.mouse_on_scroll = True

        IO.mouse_scroll_change = None

    IO.mouse_button_down.clear()
    IO.mouse_button_up.clear()
    IO.mouse_button_repeated.clear()
    IO.mouse_button_held.clear()
    IO.mouse_button_dragged.clear()
    for button, input in IO.mouse_button_states.items():
        input.state = input.state_change
        input.state_change = -1
        if input.state == Input.PRESS:
            tolerance: int = 2

            inc: bool = (
                all((IO.mouse_pos - input.down_pos) < tolerance)
                and time - input.down_time < IO.input_double_press_delay
            )

            input.held = True
            input.held_time = time + IO.input_hold_frequency
            input.down_time = time
            input.down_count = input.down_count + 1 if inc else 1
            input.down_pos[:] = IO.mouse_pos

            IO.mouse_button_down.append(button)
        elif input.state == Input.RELEASE:
            input.held = False
            input.held_time = sys.maxsize

            IO.mouse_button_up.append(button)
        elif input.state == Input.REPEAT:
            IO.mouse_button_repeated.append(button)
        input.dragging = False
        if input.held:
            if time - input.held_time >= IO.input_hold_frequency:
                IO.mouse_button_held.append(button)
                input.held_time += IO.input_hold_frequency
            if any(IO.mouse_pos_delta != 0):
                input.dragging = True
                IO.mouse_button_dragged.append(button)

    # ---------- Keyboard ---------- #
    IO.keyboard_typed = ""
    IO.keyboard_on_typed = False
    if IO.keyboard_typed_changes != "":
        IO.keyboard_typed = IO.keyboard_typed_changes
        IO.keyboard_on_typed = True

        IO.keyboard_typed_changes = ""

    IO.keyboard_key_down.clear()
    IO.keyboard_key_up.clear()
    IO.keyboard_key_repeated.clear()
    IO.keyboard_key_held.clear()
    for key, input in IO.keyboard_key_states.items():
        input.state = input.state_change
        input.state_change = -1
        if input.state == Input.PRESS:
            inc: bool = time - input.down_time < IO.input_double_press_delay

            input.held = True
            input.held_time = time + IO.input_hold_frequency
            input.down_time = time
            input.down_count = input.down_count + 1 if inc else 1

            IO.keyboard_key_down.append(key)
        elif input.state == Input.RELEASE:
            input.held = False
            input.held_time = sys.maxsize

            IO.keyboard_key_up.append(key)
        elif input.state == Input.REPEAT:
            IO.keyboard_key_repeated.append(key)
        if input.held and time - input.held_time >= IO.input_hold_frequency:
            input.held_time += IO.input_hold_frequency
            IO.keyboard_key_held.append(key)

    # ---------- Modifier ---------- #


def destroy() -> None:
    pygame.quit()


# -------------------- Window State -------------------- #


def window_on_close() -> bool:
    return IO.window_on_close


def window_focused() -> bool:
    return IO.window_focused


def window_on_focused() -> bool:
    return IO.window_on_focused


def window_minimized() -> bool:
    return IO.window_minimized


def window_on_minimized() -> bool:
    return IO.window_on_minimized


def window_maximized() -> bool:
    return IO.window_maximized


def window_on_maximized() -> bool:
    return IO.window_on_maximized


def window_pos() -> VectorType:
    return IO.window_pos.copy()


def window_on_pos_change() -> bool:
    return IO.window_on_pos


def window_size() -> VectorType:
    return IO.window_size.copy()


def window_on_size_change() -> bool:
    return IO.window_on_size


def window_content_scale() -> VectorType:
    return IO.window_content_scale


def window_on_content_scale_change() -> bool:
    return IO.window_on_content_scale


def window_framebuffer_size() -> VectorType:
    return IO.window_framebuffer_size


def window_on_framebuffer_size_change() -> bool:
    return IO.window_on_framebuffer_size


def window_on_refresh() -> bool:
    return IO.window_on_refresh


def window_dropped() -> Sequence[str]:
    return IO.window_dropped


def window_on_dropped() -> bool:
    return IO.window_on_dropped


# -------------------- Window Functions -------------------- #

# def window_make_current() -> None:
#     glfwMakeContextCurrent(IO.window)
#     org.lwjgl.opengl.GL.createCapabilities()


# def window_unmake_current() -> None:
#     org.lwjgl.opengl.GL.setCapabilities(None)
#     glfwMakeContextCurrent(0)


def window_swap() -> None:
    pygame.display.flip()


def window_title(title: str) -> None:
    pygame.display.set_caption(title)


def clipboard(data: bytes = None) -> Optional[bytes]:
    if data is None:
        return pygame.scrap.get(pygame.SCRAP_TEXT)
    pygame.scrap.put(pygame.SCRAP_TEXT, data)


# -------------------- Mouse State -------------------- #


def mouse_entered() -> bool:
    return IO.mouse_entered


def mouse_on_entered() -> bool:
    return IO.mouse_on_entered


def mouse_pos() -> VectorType:
    return IO.mouse_pos.copy()


def mouse_pos_delta() -> VectorType:
    return IO.mouse_pos_delta.copy()


def mouse_on_pos_change() -> bool:
    return IO.mouse_on_pos


def mouse_scroll() -> VectorType:
    return IO.mouse_scroll


def mouse_on_scroll_change() -> bool:
    return IO.mouse_on_scroll


def mouse_button_down(button: Button) -> bool:
    return IO.mouse_button_states[button].state == Input.PRESS


def mouse_buttons_down() -> Sequence[Button]:
    return IO.mouse_button_down


def mouse_button_down_count(button: Button) -> int:
    return IO.mouse_button_states[button].down_count


def mouse_on_button_down() -> bool:
    return len(IO.mouse_button_down) > 0


def mouse_button_up(button: Button) -> bool:
    return IO.mouse_button_states[button].state == Input.RELEASE


def mouse_buttons_up() -> Sequence[Button]:
    return IO.mouse_button_up


def mouse_on_button_up() -> bool:
    return len(IO.mouse_button_up) > 0


def mouse_button_repeated(button: Button) -> bool:
    return IO.mouse_button_states[button].state == Input.REPEAT


def mouse_buttons_repeated() -> Sequence[Button]:
    return IO.mouse_button_repeated


def mouse_on_button_repeated() -> bool:
    return len(IO.mouse_button_repeated) > 0


def mouse_button_held(button: Button) -> bool:
    return IO.mouse_button_states[button].held


def mouse_buttons_held() -> Sequence[Button]:
    return IO.mouse_button_held


def mouse_on_button_held() -> bool:
    return len(IO.mouse_button_held) > 0


def mouse_button_dragged(button: Button) -> bool:
    return IO.mouse_button_states[button].dragging


def mouse_buttons_dragged() -> Sequence[Button]:
    return IO.mouse_button_dragged


def mouse_on_button_dragged() -> bool:
    return len(IO.mouse_button_dragged) > 0


# -------------------- Mouse Functions -------------------- #


def mouse_show() -> None:
    IO.mouse_state_change = MOUSE_SHOWN


def mouse_shown() -> bool:
    return IO.mouse_state == MOUSE_SHOWN


def mouse_hide() -> None:
    IO.mouse_state_change = MOUSE_HIDDEN


def mouse_hidden() -> bool:
    return IO.mouse_state == MOUSE_HIDDEN


def mouse_capture() -> None:
    IO.mouse_state_change = MOUSE_CAPTURED


def mouse_captured() -> bool:
    return IO.mouse_state == MOUSE_CAPTURED


def mouse_raw_input(raw_input: bool = None) -> bool:
    return raw_input or False


def mouse_sticky(sticky: bool = None) -> bool:
    return sticky or False


# -------------------- Keyboard State -------------------- #


def keyboard_typed() -> str:
    return IO.keyboard_typed


def keyboard_on_typed() -> bool:
    return IO.keyboard_on_typed


def keyboard_key_down(key: Key) -> bool:
    return IO.keyboard_key_states[key].state == Input.PRESS


def keyboard_keys_down() -> Sequence[Key]:
    return IO.keyboard_key_down


def keyboard_key_down_count(key: Key) -> int:
    return IO.keyboard_key_states[key].down_count


def keyboard_on_key_down() -> bool:
    return len(IO.keyboard_key_down) > 0


def keyboard_key_up(key: Key) -> bool:
    return IO.keyboard_key_states[key].state == Input.RELEASE


def keyboard_keys_up() -> Sequence[Key]:
    return IO.keyboard_key_up


def keyboard_on_key_up() -> bool:
    return len(IO.keyboard_key_up) > 0


def keyboard_key_repeated(key: Key) -> bool:
    return IO.keyboard_key_states[key].state == Input.REPEAT


def keyboard_keys_repeated() -> Sequence[Key]:
    return IO.keyboard_key_repeated


def keyboard_on_key_repeated() -> bool:
    return len(IO.keyboard_key_repeated) > 0


def keyboard_key_held(key: Key) -> bool:
    return IO.keyboard_key_states[key].held


def keyboard_keys_held() -> Sequence[Key]:
    return IO.keyboard_key_held


def keyboard_on_key_held() -> bool:
    return len(IO.keyboard_key_held) > 0


# -------------------- Keyboard Functions -------------------- #


def keyboard_sticky(sticky: bool = None) -> bool:
    pass


# -------------------- Modifier State -------------------- #


def modifier_get() -> Modifier:
    return IO.modifier_active


def modifier_any(modifiers: Modifier) -> bool:
    return (IO.modifier_active & modifiers) != Modifier.NONE


def modifier_all(modifiers: Modifier) -> bool:
    return (IO.modifier_active & modifiers) == modifiers


def modifier_only(modifiers: Modifier) -> bool:
    return IO.modifier_active == modifiers


# -------------------- Modifier Functions -------------------- #


def modifier_include_lock_mods(include_lock_mods: bool = None) -> bool:
    if include_lock_mods is not None:
        IO.modifier_include_lock_mods = include_lock_mods
    return IO.modifier_include_lock_mods
