import pygame

from old import engine as e

engine = e.Engine(800, 600)

background = e.Color(0)


@engine.on_render
def render(t, dt):
    engine.screen.fill(background)

    pygame.draw.line(engine.screen, (255, 255, 255), (0, 0), engine.mouse_pos)

    pygame.draw.circle(engine.screen, (255, 0, 0), engine.mouse_pos, 50, 50)


engine.start()
