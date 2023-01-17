from engine import engine


class Slider:
    def __init__(self, pos, min, max, start, increment):
        self.pos = pos.copy()
        self.min, self.max = min, max
        self.value = start
        self.increment = increment

        self.w, self.h = 200, 20
        self.size = 7
        self.dragging = False

        self._pos = Vector(0, 0, 0)
        self._size = Vector(0, 0, 0)

    def update(self):
        if self.dragging:
            if engine.mouse_pressed:
                x1 = self.x_position(self.value)
                d = engine.mouse.x - x1
                change = d * (self.max - self.min) / self.w
                self.value += change
                self.value -= self.value % self.increment
                self.value = constrain(self.value, self.min, self.max)
            else:
                self.dragging = False
                print("End Drag")
        else:
            x1 = self.x_position(self.value)
            x, y = engine.mouse
            if (
                engine.mouse_pressed
                and self.pos.y <= y <= self.pos.y + self.h
                and x1 - self.size / 2 <= x <= x1 + self.size / 2
            ):
                print("Start Drag")
                self.dragging = True

    def x_position(self, val):
        val = map(val, self.min, self.max, self.pos.x, self.pos.x + self.w)
        return constrain(val, self.pos.x, self.pos.x + self.w)

    def show(self):
        self.update()

        engine.push()

        engine.stroke = 255
        engine.fill = 0

        self._pos.xy = self.pos.xy + [0, self.h / 3]
        self._size.xy = [self.w, self.h / 3]

        engine.rect(self._pos, self._size)

        x1 = self.x_position(self.value)

        engine.stroke = 200
        engine.fill = 128

        self._pos.xyz = [x1 - self.size / 2, self.pos.y, 10]
        self._size.xy = [self.size, self.h]

        engine.rect(self._pos, self._size)

        engine.pop()
