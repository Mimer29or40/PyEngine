from engine import *


class Snake:
    def __init__(self):
        self.pos = Vector(0, 0)
        self.dir = Vector(1, 0)
        self.total = 0
        self.tail = []
    
    def eat(self, pos):
        d = self.pos.dist(pos)
        if d < 1:
            self.total += 1
            return True
        return False
    
    def death(self):
        for pos in self.tail:
            d = self.pos.dist(pos)
            if d < 1:
                print('Starting Over')
                self.total = 0
                self.tail.clear()
                break
    
    def update(self):
        if self.total > 0:
            if self.total == len(self.tail) and len(self.tail) > 0:
                self.tail.pop(0)
            self.tail.append(self.pos.copy())
        
        self.pos += self.dir * scale
        self.pos = np.clip(self.pos, 0, engine.viewport - scale)
    
    def show(self):
        engine.fill = 255
        for t in self.tail:
            engine.square(t, scale)
        engine.square(self.pos, scale)


if __name__ == '__main__':
    engine.size(400, 400, OPENGL)
    
    engine.frame_rate = 10
    
    scale = 20
    snake = Snake()
    food = Vector(0, 0)
    
    def pick_location():
        size = (engine.viewport / scale).astype(int)
        food[:] = random(size).astype(int) * scale
    
    pick_location()
    
    @engine.event
    def mouse_pressed():
        snake.total += 1
    
    @engine.event
    def key_pressed(e):
        if engine.key_code == 'up':
            snake.dir[:] = 0, -1
        elif engine.key_code == 'down':
            snake.dir[:] = 0, 1
        elif engine.key_code == 'left':
            snake.dir[:] = -1, 0
        elif engine.key_code == 'right':
            snake.dir[:] = 1, 0
    
    @engine.draw
    def draw():
        engine.background = 51
        
        if snake.eat(food):
            pick_location()
        
        snake.death()
        snake.update()
        snake.show()
        
        engine.fill = 255, 0, 100
        engine.square(food, scale)
    
    engine.start()
