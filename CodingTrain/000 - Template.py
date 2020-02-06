from engine import *


if __name__ == '__main__':
    engine.size(400, 400)
    
    @engine.draw
    def draw():
        pass
    
    engine.start()
