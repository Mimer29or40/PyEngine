QUIT = "Quit"
ACTIVE_EVENT = "ActiveEvent"

KEY_DOWN = "KeyDown"
KEY_UP = "KeyUp"
KEY_HOLD = "KeyHold"
KEY_PRESSED = "KeyPressed"

MOUSE_BUTTON_DOWN = "MouseButtonDown"
MOUSE_BUTTON_UP = "MouseButtonUp"
MOUSE_BUTTON_HOLD = "MouseButtonHold"
MOUSE_BUTTON_PRESSED = "MouseButtonPressed"
MOUSE_MOTION = "MouseMotion"
MOUSE_DRAGGED = "MouseDragged"
MOUSE_SCROLL = "MouseScroll"

JOY_AXIS_MOTION = "JoyAxisMotion"
JOY_BALL_MOTION = "JoyBallMotion"
JOY_HAT_MOTION = "JoyHatMotion"
JOY_BUTTON_DOWN = "JoyButtonDown"
JOY_BUTTON_UP = "JoyButtonUp"

VIDEO_RESIZE = "VideoResize"
VIDEO_EXPOSE = "VideoExpose"

USER_EVENT = "UserEvent"


class Event:
    def __init__(self, type, time, **kwargs):
        self.type = type
        self.time = time
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __repr__(self):
        attrs = ["{}={}".format(k, d) for k, d in self.__dict__.items()]
        return "{}<{}>".format(self.__class__.__name__, " ".join(attrs))
