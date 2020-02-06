# System Packages

# Third-Party Packages
import OpenGL.GL as GL
import OpenGL.GLU as GLU
import pygame

# My Packages
# import Util

# Project Packages
from engine import *

np.set_printoptions(
    linewidth = 9999,
    precision = 6,
    edgeitems = 10,
    threshold = 4000,
    suppress = True
)

engine = Engine(400, 300)


@engine.event_handler(event = KEY_DOWN)
def handler(event):
    if event.key == pygame.K_SPACE:
        engine.camera.is_perspective = not engine.camera.is_perspective


@engine.event_handler(event = KEY_HOLD)
def handler(event):
    if event.key == pygame.K_UP:
        game_object.scale.x += 0.001
    if event.key == pygame.K_DOWN:
        game_object.scale.x -= 0.001


@engine.event_handler(event = MOUSE_DRAGGED)
def handler(event):
    if event.button == 1:
        rel = event.rel.asfloat() * 0.005
        engine.camera.translate(dx = -rel.x, dy = rel.y)
    # if event.button == 3:
        # game_object.scale.x -= 0.001


@engine.event_handler(event = MOUSE_BUTTON_DOWN)
def handler(event):
    if event.button == 1:
        # aspect = engine.width / engine.height
        # m = engine.camera.focal_length * np.math.tan(np.math.radians(engine.camera.fov / 2))
        # # # print(engine.camera.focal_length, engine.camera.fov / 2)
        # x = ((event.pos.x * 2 / engine.width) - 1)
        # y = ((event.pos.y * 2 / engine.height) - 1) * -1
        # # x = event.pos.x / engine.width
        # # y = event.pos.y / engine.height * -1
        # z = (engine.camera.focal_length * 2 / (engine.camera.z_far - engine.camera.z_near)) - 1
        # # z = engine.camera.focal_length / (engine.camera.z_far - engine.camera.z_near)
        # # z = 1
        # # v = engine.camera.focus + (x * engine.camera._r) + (y * engine.camera._u)
        # # print(event.pos, m, x, y, v)
        # # # q = engine.camera.focus + (m * engine.camera._r)
        # # # print(q)
        # pv = (engine.camera.projection(aspect) @ engine.camera.view()).inverse
        # print(engine.camera.projection(aspect))
        # print(engine.camera.view())
        # # pv = (engine.camera.view() @ engine.camera.projection(aspect)).inverse
        # v = util.Vector([x, y, z, 1], float)
        # v @= pv
        # v /= v.w
        # print(x, y, z, v.xyz)
        
        # print(event.pos)
        
        aspect = engine.width / engine.height
        
        x = (2 * event.pos.x / engine.width - 1)
        y = -(2 * event.pos.y / engine.height - 1)
        z = 0.1
        w = 1
        
        vec = util.Vector([x, y, z, w], float)
        vec @= engine.camera.projection(aspect).inverse
        vec @= engine.camera.view().inverse
        # print(vec.w)
        vec /= vec.w
        
        # print(vec)
        
        x = (2 * event.pos.x / engine.width - 1)
        y = -(2 * event.pos.y / engine.height - 1)
        z = 0.5
        w = 1
        
        vec = util.Vector([x, y, z, w], float)
        vec @= engine.camera.projection(aspect).inverse
        vec @= engine.camera.view().inverse
        # print(vec.w)
        vec /= vec.w
        
        # print(vec)
        
        # v1 = util.Vector([*engine.camera.focus, .1]) @ engine.camera.view() @ engine.camera.projection(aspect)
        v1 = util.Vector([1, 0, 0, 1], float) @ engine.camera.view() @ engine.camera.projection(aspect)
        # v1 = util.Vector([0, 1, 0, 1], float) @ engine.camera.view() @ engine.camera.projection(aspect)
        # v1 = util.Vector([0, 0, 1, 1], float) @ engine.camera.view() @ engine.camera.projection(aspect)
        print(v1, (engine.camera.focal_length - engine.camera.z_near) / 10)
        v1 /= v1.w
        
        v1.x = (v1.x + 1) / 2 * engine.width
        v1.y = (-v1.y + 1) / 2 * engine.height
        
        # z1 = 1 - (1 / engine.camera.focal_length)
        # z2 = (1 / engine.camera.focal_length)
        
        print(v1.z, (engine.camera.focal_length - engine.camera.z_near) / 10)
    if event.button == 4:
        engine.camera.zoom(-0.1)
    if event.button == 5:
        engine.camera.zoom(0.1)


mesh = Mesh('Square')
mesh.bind_vertices([
    0, 0, 0, 0, 1,
    1, 0, 0, 1, 1,
    1, 1, 0, 1, 0,
    0, 1, 0, 0, 0
], [0, 1, 2, 0, 2, 3])

game_object = GameObject('Test')
game_object.model = mesh
game_object.position.x = 0.5

engine.game_objects.append(game_object)


engine.start()
