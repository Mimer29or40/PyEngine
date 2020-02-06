from engine import *


if __name__ == '__main__':
    engine.size(400, 400, OPENGL)
    
    pos = Vector(20, 20)
    slider = Slider(pos, 0.1, 10, 2, 0.1)
    
    def sgn(val):
        return 0 if val == 0 else val / np.abs(val)
    
    @engine.draw
    def draw():
        engine.background = 51
        engine.translate(engine.viewport / 2)
        
        a, b = 100., 100.
        n = slider.value
        
        engine.stroke = 255
        engine.fill = None
        
        points = []
        for angle in np.linspace(0, 2 * np.pi, 180):
            na = 2 / n
            pos = Vector.from_angle(angle)
            points.append(np.abs(pos) ** na * pos * [a, b])
        
        engine.lines(*points)
        
        engine.translate(-engine.viewport / 2)
        
        slider.show()
    
    engine.start()
