from .events import QUIT


class Game:
    def __init__(self):
        self.name = self.__class__.__name__

    def init(self):
        pass

    def process_events(self, events):
        for event in events:
            if event.type == QUIT:
                stop()

    def update(self, input, t, dt):
        pass

    def render(self, t, dt):
        pass

    def shutdown(self):
        pass
