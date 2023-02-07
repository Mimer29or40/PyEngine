from engine import *


class Event:
    type: int
    time: int
    modifiers: int

    def __init__(self, engine, _type, **kwargs):
        object.__setattr__(self, "type", _type)
        object.__setattr__(self, "time", engine.time)
        object.__setattr__(self, "modifiers", engine._modifiers)
        for k in kwargs.items():
            object.__setattr__(self, *k)

        if DEBUG_EVENTS:
            print(self)

        for func, kwargs in engine._event_dict[_type]:
            for k, v in kwargs.items():
                if getattr(self, k, None) != v:
                    break
            else:
                try:
                    func(self)
                except TypeError:
                    func()

    def __repr__(self):
        a = " ".join("{}={}".format(*k) for k in self.__dict__.items())
        return f"Event[{a}]"

    def __setattr__(self, name, value):
        raise AttributeError("attribute is readonly")

    def is_shift_down(self):
        return bool(self.modifiers & MOD_SHIFT)

    def is_control_down(self):
        return bool(self.modifiers & MOD_CTRL)

    def is_alt_down(self):
        return bool(self.modifiers & MOD_ALT)

    def is_meta_down(self):
        return bool(self.modifiers & MOD_META)
