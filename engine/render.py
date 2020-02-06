from functools import wraps

from engine import *
from engine.gl import *

from OpenGL.GL import *


X = Vector(1, 0, 0)
Y = Vector(0, 1, 0)
Z = Vector(0, 0, 1)


def get_renderer(renderer):
    if renderer == PYGAME1:
        return RenderPygame1()
    elif renderer == PYGAME3:
        return RenderPygame3()
    elif renderer == OPENGL2:
        return RenderOpenGL2()
    elif renderer == OPENGL3:
        return RenderOpenGL3()
    else:
        raise Exception(f'incorrect renderer {renderer}')


def not_supported(func):
    func.run = True
    @wraps(func)
    def wrapper(*args, **kwargs):
        if func.run:
            print(f'{func.__name__} is not supported for this renderer')
            func.run = False
    return wrapper


class Renderer:
    def __init__(self):
        self.view = Matrix.identity(4)

    def get_flags(self):
        return 0

    def set_background(self, color):
        pass

    def setup(self, engine):
        self.set_background(Color())

    def before_draw(self, engine):
        pass

    def after_draw(self, engine):
        pass

    def translate(self, amount):
        self.view.translate(amount, 1)

    def rotate(self, angle, axis = None):
        if axis is None:
            self.view.rotate(Z, angle)
        else:
            self.view.rotate(axis, angle)

    def rotate_x(self, angle):
        self.view.rotate(X, angle)

    def rotate_y(self, angle):
        self.view.rotate(Y, angle)

    def rotate_z(self, angle):
        self.view.rotate(Z, angle)

    def scale(self, amount):
        self.view.scale(amount, 1)

    def point(self, engine, p):
        pass

    def line(self, engine, p1, p2):
        pass
    
    def lines(self, engine, *points):
        pass
    
    def polygon(self, engine, *points):
        pass

    def triangle(self, engine, p1, p2, p3):
        pass

    def quad(self, engine, p1, p2, p3, p4):
        pass

    def ellipse(self, engine, p1, p2):
        pass

    def arc(self, engine, p1, p2, start, stop):
        pass

    def text(self, engine, text, pos):
        pass
    
    def load_pixels(self, engine):
        size = (engine.height, engine.width, 4)
        arr = np.zeros(size, dtype = np.uint8)
        arr[:, :, 3] = 255
        return arr
    
    def update_pixels(self, engine):
        pass


class RenderPygame(Renderer):
    def __init__(self):
        super().__init__()

    def get_flags(self):
        return 0

    def set_background(self, color):
        pygame.display.get_surface().fill(color.rgb)

    def before_draw(self, engine):
        self.view[:] = np.identity(4)
    
    def load_pixels(self, engine):
        size = (engine.height, engine.width, 4)
        buf = pygame.display.get_surface().get_buffer().raw
        return np.frombuffer(buf, dtype = np.uint8).reshape(size).copy()
    
    def update_pixels(self, engine):
        pygame.display.get_surface().blit(
            pygame.surfarray.make_surface(
                np.swapaxes(engine.pixels[:, :, :3], 0, 1)
            ),
            (0, 0)
        )


class RenderPygame1(RenderPygame):
    translate = not_supported(RenderPygame.translate)
    rotate = not_supported(RenderPygame.rotate)
    rotate_x = not_supported(RenderPygame.rotate_x)
    rotate_y = not_supported(RenderPygame.rotate_y)
    rotate_z = not_supported(RenderPygame.rotate_z)
    scale = not_supported(RenderPygame.scale)
    
    def point(self, engine, p):
        if not engine.stroke.is_none:
            pygame.draw.circle(
                pygame.display.get_surface(),
                engine.stroke,
                p.base.astype(int).xy,
                int(engine.weight)
            )

    def line(self, engine, p1, p2):
        if not engine.stroke.is_none:
            pygame.draw.line(
                pygame.display.get_surface(),
                engine.stroke,
                p1.base.astype(int).xy,
                p2.base.astype(int).xy,
                int(engine.weight)
            )
    
    def lines(self, engine, *points):
        if not engine.stroke.is_none:
            pygame.draw.lines(
                pygame.display.get_surface(),
                engine.stroke,
                False,
                [p.base.astype(int).xy for p in points],
                int(engine.weight)
            )
    
    def polygon(self, engine, *points):
        if not engine.fill.is_none or not engine.stroke.is_none:
            points = [p.base.astype(int).xy for p in points]
            
            if not engine.fill.is_none:
                pygame.draw.polygon(
                    pygame.display.get_surface(),
                    engine.fill,
                    points
                )
            if not engine.stroke.is_none:
                pygame.draw.polygon(
                    pygame.display.get_surface(),
                    engine.stroke,
                    points,
                    int(engine.weight)
                )

    def triangle(self, engine, p1, p2, p3):
        self.polygon(engine, p1, p2, p3)

    def quad(self, engine, p1, p2, p3, p4):
        self.polygon(engine, p1, p2, p3, p4)

    def ellipse(self, engine, p1, p2):
        if not engine.fill.is_none or not engine.stroke.is_none:
            top_left, size = p1.base, p2.base

            if engine.ellipse_mode == CENTER:
                top_left.xyz -= size.xyz * 0.5
            elif engine.ellipse_mode == RADIUS:
                top_left.xyz -= size.xyz
                size.xyz *= 2.0
            elif engine.ellipse_mode == CORNER:
                pass
            elif engine.ellipse_mode == CORNERS:
                size.xyz -= top_left.xyz
            
            rect = [*top_left.astype(int).xy, *size.astype(int).xy]
            
            if not engine.fill.is_none:
                pygame.draw.ellipse(
                    pygame.display.get_surface(),
                    engine.fill,
                    rect
                )
            if not engine.stroke.is_none:
                pygame.draw.ellipse(
                    pygame.display.get_surface(),
                    engine.stroke,
                    rect,
                    int(engine.weight)
                )

    def arc(self, engine, p1, p2, start, stop):
        if not engine.fill.is_none or not engine.stroke.is_none:
            top_left, size = p1.base, p2.base

            if engine.ellipse_mode == CENTER:
                top_left.xyz -= size.xyz * 0.5
            elif engine.ellipse_mode == RADIUS:
                top_left.xyz -= size.xyz
                size.xyz *= 2.0
            elif engine.ellipse_mode == CORNER:
                pass
            elif engine.ellipse_mode == CORNERS:
                size.xyz -= top_left.xyz
            
            rect = [*top_left.astype(int).xy, *size.astype(int).xy]
            
            if not engine.fill.is_none:
                pygame.draw.arc(
                    pygame.display.get_surface(),
                    engine.fill,
                    rect,
                    start,
                    stop
                )
            if not engine.stroke.is_none:
                pygame.draw.arc(
                    pygame.display.get_surface(),
                    engine.stroke,
                    rect,
                    start,
                    stop,
                    int(engine.weight)
                )

    def text(self, engine, text, pos):
        surf = engine._font.render(str(text), True, engine.fill)
        
        w, h = surf.get_size()
        top_left = pos.base
        
        if engine.text_align[0] == LEFT:
            pass
        elif engine.text_align[0] == CENTER:
            top_left.x -= w * 0.5
        elif engine.text_align[0] == RIGHT:
            top_left.x -= w
        
        if engine.text_align[1] == TOP:
            pass
        elif engine.text_align[1] == CENTER:
            top_left.y -= h * 0.5
        elif engine.text_align[1] == BOTTOM:
            top_left.y -= h
        
        pygame.display.get_surface().blit(
            surf,
            top_left.astype(int).xy
        )


class RenderPygame3(RenderPygame):
    def point(self, engine, p):
        if not engine.stroke.is_none:
            pygame.draw.circle(
                pygame.display.get_surface(),
                engine.stroke,
                (p.base @ self.view).astype(int).xy,
                int(engine.weight)
            )

    def line(self, engine, p1, p2):
        if not engine.stroke.is_none:
            pygame.draw.line(
                pygame.display.get_surface(),
                engine.stroke,
                (p1.base @ self.view).astype(int).xy,
                (p2.base @ self.view).astype(int).xy,
                int(engine.weight)
            )
    
    def lines(self, engine, *points):
        if not engine.stroke.is_none:
            pygame.draw.lines(
                pygame.display.get_surface(),
                engine.stroke,
                False,
                [(p.base @ self.view).astype(int).xy for p in points],
                int(engine.weight)
            )
    
    def polygon(self, engine, *points):
        if not engine.fill.is_none or not engine.stroke.is_none:
            points = [(p.base @ self.view).astype(int).xy for p in points]
            
            if not engine.fill.is_none:
                pygame.draw.polygon(
                    pygame.display.get_surface(),
                    engine.fill,
                    points
                )
            if not engine.stroke.is_none:
                pygame.draw.polygon(
                    pygame.display.get_surface(),
                    engine.stroke,
                    points,
                    int(engine.weight)
                )

    def triangle(self, engine, p1, p2, p3):
        self.polygon(engine, p1, p2, p3)

    def quad(self, engine, p1, p2, p3, p4):
        self.polygon(engine, p1, p2, p3, p4)

    def ellipse(self, engine, p1, p2):
        if not engine.fill.is_none or not engine.stroke.is_none:
            top_left, size = p1.base, p2.base

            if engine.ellipse_mode == CENTER:
                top_left.xyz -= size.xyz * 0.5
            elif engine.ellipse_mode == RADIUS:
                top_left.xyz -= size.xyz
                size.xyz *= 2.0
            elif engine.ellipse_mode == CORNER:
                pass
            elif engine.ellipse_mode == CORNERS:
                size.xyz -= top_left.xyz
            
            rect = [
                *(top_left @ self.view).astype(int).xy,
                *(size @ self.view).astype(int).xy
            ]
            
            if not engine.fill.is_none:
                pygame.draw.ellipse(
                    pygame.display.get_surface(),
                    engine.fill,
                    rect
                )
            if not engine.stroke.is_none:
                pygame.draw.ellipse(
                    pygame.display.get_surface(),
                    engine.stroke,
                    rect,
                    int(engine.weight)
                )

    def arc(self, engine, p1, p2, start, stop):
        if not engine.fill.is_none or not engine.stroke.is_none:
            top_left, size = p1.base, p2.base

            if engine.ellipse_mode == CENTER:
                top_left.xyz -= size.xyz * 0.5
            elif engine.ellipse_mode == RADIUS:
                top_left.xyz -= size.xyz
                size.xyz *= 2.0
            elif engine.ellipse_mode == CORNER:
                pass
            elif engine.ellipse_mode == CORNERS:
                size.xyz -= top_left.xyz
            
            rect = [
                *(top_left @ self.view).astype(int).xy,
                *(size @ self.view).astype(int).xy
            ]
            
            if not engine.fill.is_none:
                pygame.draw.arc(
                    pygame.display.get_surface(),
                    engine.fill,
                    rect,
                    start,
                    stop
                )
            if not engine.stroke.is_none:
                pygame.draw.arc(
                    pygame.display.get_surface(),
                    engine.stroke,
                    rect,
                    start,
                    stop,
                    int(engine.weight)
                )

    def text(self, engine, text, pos):
        surf = engine._font.render(str(text), True, engine.fill)
        
        w, h = surf.get_size()
        top_left = pos.base
        
        if engine.text_align[0] == LEFT:
            pass
        elif engine.text_align[0] == CENTER:
            top_left.x -= w * 0.5
        elif engine.text_align[0] == RIGHT:
            top_left.x -= w
        
        if engine.text_align[1] == TOP:
            pass
        elif engine.text_align[1] == CENTER:
            top_left.y -= h * 0.5
        elif engine.text_align[1] == BOTTOM:
            top_left.y -= h
        
        pygame.display.get_surface().blit(
            surf,
            (top_left @ self.view).astype(int).xy
        )


class RenderOpenGL(Renderer):
    def __init__(self):
        super().__init__()
        
        self.proj = Matrix.identity(4)
        
        self.pixel_shader, self.pixel_vertex = None, None
        self.pixel_texture = None

    def get_flags(self):
        return pygame.OPENGL | pygame.DOUBLEBUF

    def set_background(self, color):
        glClearColor(*color.to_float())
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    def setup(self, engine):
        super().setup(engine)
        
        self.proj[0, 0] = 2. / engine.width
        self.proj[1, 1] = 2. / engine.height
        self.proj[2, 2] = 2. / max(engine.viewport * engine.viewport)

        if DEBUG_RENDER:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        self.pixel_shader, self.pixel_vertex = Shader(
            (GL_VERTEX_SHADER, PIXEL_VERT3),
            (GL_FRAGMENT_SHADER, PIXEL_FRAG3)
        ), VertexArray('float32', 2)
        self.pixel_vertex.set(
            np.array([-1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0]),
            GL_DYNAMIC_DRAW
        )
        self.pixel_texture = glGenTextures(1)
    
    def before_draw(self, engine):
        x, y, z = Vector(1, 0, 0), Vector(0, -1, 0), Vector(0, 0, -1)
        cam_z = (engine.height / 2.) / np.tan(np.pi / 3. / 2.)
        pos = Vector(engine.width / 2, engine.height / 2, cam_z)

        self.view[:] = [
            [x.x, y.x, z.x, 0],
            [x.y, y.y, z.y, 0],
            [x.z, y.z, z.z, 0],
            [-x.dot(pos), -y.dot(pos), -z.dot(pos), 1]
        ]
    
    def load_pixels(self, engine):
        size = (engine.height, engine.width, 4)
        buf = glReadPixels(0, 0, *engine.viewport, GL_RGBA, GL_UNSIGNED_BYTE)
        return np.flipud(np.frombuffer(buf, dtype = np.uint8).copy().reshape(size))
    
    def update_pixels(self, engine):
        glBindTexture(GL_TEXTURE_2D, self.pixel_texture)
        glTexImage2D(
            GL_TEXTURE_2D, 0, GL_RGBA,
            *engine.viewport, 0, GL_RGBA,
            GL_UNSIGNED_BYTE, engine.pixels
        )
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glActiveTexture(GL_TEXTURE0)
        
        self.pixel_vertex.bind()

        self.pixel_shader.use()
        self.pixel_shader.set_int('TextMap', 0)

        glDrawArrays(GL_QUADS, 0, 4)

        self.pixel_vertex.unbind()


class RenderOpenGL2(RenderOpenGL):
    def __init__(self):
        super().__init__()
        
        self.vertex = None
        
        self.point_shader = None
        self.line_shader = None
        self.lines_shader = None
        self.quad_shader = None
        self.ellipse_shader = None
        self.ellipse_outline_shader = None
        self.arc_shader = None
        self.arc_outline_shader = None
        self.text_shader = None
        self.text_texture = 0
        self.pixel_shader = None
        self.pixel_texture = None
    
    def setup(self, engine):
        super().setup(engine)
        
        self.vertex = VertexArray('float32', 3)
        self.vertex.bind()
        
        self.point_shader = Shader(
            (GL_VERTEX_SHADER, VERT2),
            (GL_GEOMETRY_SHADER, POINT_GEOM2),
            (GL_FRAGMENT_SHADER, FRAG2)
        )
        self.line_shader = Shader(
            (GL_VERTEX_SHADER, VERT2),
            (GL_GEOMETRY_SHADER, LINE_GEOM2),
            (GL_FRAGMENT_SHADER, FRAG2)
        )
        self.lines_shader = Shader(
            (GL_VERTEX_SHADER, VERT2),
            (GL_GEOMETRY_SHADER, LINES_GEOM2),
            (GL_FRAGMENT_SHADER, FRAG2)
        )
        self.quad_shader = Shader(
            (GL_VERTEX_SHADER, VERT2),
            (GL_FRAGMENT_SHADER, FRAG2)
        )
        self.ellipse_shader = Shader(
            (GL_VERTEX_SHADER, VERT2),
            (GL_GEOMETRY_SHADER, ELLIPSE_GEOM2),
            (GL_FRAGMENT_SHADER, FRAG2)
        )
        self.ellipse_outline_shader = Shader(
            (GL_VERTEX_SHADER, VERT2),
            (GL_GEOMETRY_SHADER, ELLIPSE_OUTLINE_GEOM2),
            (GL_FRAGMENT_SHADER, FRAG2)
        )
        self.arc_shader = Shader(
            (GL_VERTEX_SHADER, VERT2),
            (GL_GEOMETRY_SHADER, ARC_GEOM2),
            (GL_FRAGMENT_SHADER, FRAG2)
        )
        self.arc_outline_shader = Shader(
            (GL_VERTEX_SHADER, VERT2),
            (GL_GEOMETRY_SHADER, ARC_OUTLINE_GEOM2),
            (GL_FRAGMENT_SHADER, FRAG2)
        )
        self.text_shader = Shader(
            (GL_VERTEX_SHADER, VERT2),
            (GL_GEOMETRY_SHADER, TEXT_GEOM2),
            (GL_FRAGMENT_SHADER, TEXT_FRAG2)
        )
        self.texture_id = glGenTextures(1)

    def point(self, engine, p):
        if not engine.stroke.is_none:
            self.vertex.set(p.base.xyz, GL_DYNAMIC_DRAW)

            self.point_shader.use()
            self.point_shader.set_floatm('pv', self.view @ self.proj)
            self.point_shader.set_floatv('color', engine.stroke.to_gl())
            self.point_shader.set_floatv('viewport', engine.viewport)
            self.point_shader.set_float('thickness', engine.weight)

            glDrawArrays(GL_POINTS, 0, 1)

    def line(self, engine, p1, p2):
        if not engine.stroke.is_none:
            self.vertex.set(np.array([
                *p1.base.xyz, *p2.base.xyz
            ]), GL_DYNAMIC_DRAW)

            self.line_shader.use()
            self.line_shader.set_floatm('pv', self.view @ self.proj)
            self.line_shader.set_floatv('color', engine.stroke.to_gl())
            self.line_shader.set_float('thickness', engine.weight)
            self.line_shader.set_floatv('viewport', engine.viewport)

            glDrawArrays(GL_LINES, 0, 2)

    def lines(self, engine, *points):
        if not engine.stroke.is_none:
            data = [points[0].base.xyz, points[0].base.xyz, points[1].base.xyz]
            for i in range(len(points) - 2):
                data.append(points[i + 0].base.xyz)
                data.append(points[i + 1].base.xyz)
                data.append(points[i + 2].base.xyz)
            self.vertex.set(np.array(data), GL_DYNAMIC_DRAW)

            self.line_shader.use()
            self.line_shader.set_floatm('pv', self.view @ self.proj)
            self.line_shader.set_floatv('color', engine.stroke.to_gl())
            self.line_shader.set_float('thickness', engine.weight)
            self.line_shader.set_floatv('viewport', engine.viewport)

            glDrawArrays(GL_TRIANGLES, 0, len(points) * 3)
    
    def triangle(self, engine, p1, p2, p3):
        if not engine.fill.is_none:
            self.vertex.set(np.array([
                *p1.base.xyz, *p2.base.xyz, *p3.base.xyz
            ]), GL_DYNAMIC_DRAW)

            self.quad_shader.use()
            self.quad_shader.set_floatm('pv', self.view @ self.proj)
            self.quad_shader.set_floatv('color', engine.fill.to_gl())

            glDrawArrays(GL_TRIANGLES, 0, 3)

        if not engine.stroke.is_none:
            self.vertex.set(np.array([
                *p1.base.xyz, *p2.base.xyz, *p3.base.xyz,
                *p2.base.xyz, *p3.base.xyz, *p1.base.xyz,
                *p3.base.xyz, *p1.base.xyz, *p2.base.xyz
            ]), GL_DYNAMIC_DRAW)

            self.lines_shader.use()
            self.lines_shader.set_floatm('pv', self.view @ self.proj)
            self.lines_shader.set_floatv('color', engine.stroke.to_gl())
            self.lines_shader.set_floatv('viewport', engine.viewport)
            self.lines_shader.set_float('thickness', engine.weight)

            glDrawArrays(GL_TRIANGLES, 0, 9)

    def quad(self, engine, p1, p2, p3, p4):
        if not engine.fill.is_none:
            self.vertex.set(np.array([
                *p1.base.xyz, *p2.base.xyz, *p3.base.xyz, *p4.base.xyz
            ]), GL_DYNAMIC_DRAW)

            self.quad_shader.use()
            self.quad_shader.set_floatm('pv', self.view @ self.proj)
            self.quad_shader.set_floatv('color', engine.fill.to_gl())

            glDrawArrays(GL_QUADS, 0, 4)

        if not engine.stroke.is_none:
            self.vertex.set(np.array([
                *p4.base.xyz, *p1.base.xyz, *p2.base.xyz,
                *p1.base.xyz, *p2.base.xyz, *p3.base.xyz,
                *p2.base.xyz, *p3.base.xyz, *p4.base.xyz,
                *p3.base.xyz, *p4.base.xyz, *p1.base.xyz
            ]), GL_DYNAMIC_DRAW)

            self.lines_shader.use()
            self.lines_shader.set_floatm('pv', self.view @ self.proj)
            self.lines_shader.set_floatv('color', engine.stroke.to_gl())
            self.lines_shader.set_floatv('viewport', engine.viewport)
            self.lines_shader.set_float('thickness', engine.weight)

            glDrawArrays(GL_LINES_ADJACENCY, 0, 12)
    
    def ellipse(self, engine, p1, p2):
        center, radius = p1.base, p2.base

        if engine.ellipse_mode == CENTER:
            radius.xyz *= 0.5
        elif engine.ellipse_mode == RADIUS:
            pass
        elif engine.ellipse_mode == CORNER:
            center.xyz += radius.xyz * 0.5
        elif engine.ellipse_mode == CORNERS:
            radius.xyz = (radius - center).xyz * 0.5
            center.xyz += radius.xyz
        
        self.vertex.set(center.base.xyz, GL_DYNAMIC_DRAW)

        if not engine.fill.is_none:
            self.ellipse_shader.use()
            self.ellipse_shader.set_floatm('pv', self.view @ self.proj)
            self.ellipse_shader.set_floatv('color', engine.fill.to_gl())
            self.ellipse_shader.set_floatv('radius', radius.xy)

            glDrawArrays(GL_POINTS, 0, 1)

        if not engine.stroke.is_none:
            self.ellipse_outline_shader.use()
            self.ellipse_outline_shader.set_floatm('pv', self.view @ self.proj)
            self.ellipse_outline_shader.set_floatv('color', engine.stroke.to_gl())
            self.ellipse_outline_shader.set_floatv('radius', radius.xy)
            self.ellipse_outline_shader.set_floatv('viewport', engine.viewport)
            self.ellipse_outline_shader.set_float('thickness', engine.weight)

            glDrawArrays(GL_POINTS, 0, 1)
    
    def arc(self, engine, p1, p2, start, stop):
        center, radius = p1.base, p2.base

        if engine.ellipse_mode == CENTER:
            radius.xyz *= 0.5
        elif engine.ellipse_mode == RADIUS:
            pass
        elif engine.ellipse_mode == CORNER:
            center.xyz += radius.xyz * 0.5
        elif engine.ellipse_mode == CORNERS:
            radius.xyz = (radius - center).xyz * 0.5
            center.xyz += radius.xyz
        
        self.vertex.set(center.base.xyz, GL_DYNAMIC_DRAW)

        if not engine.fill.is_none:
            self.arc_shader.use()
            self.arc_shader.set_floatm('pv', pv)
            self.arc_shader.set_floatv('color', engine.fill.to_gl())
            self.arc_shader.set_floatv('radius', radius.xy)
            self.arc_shader.set_float('start', start)
            self.arc_shader.set_float('stop', stop)
            if engine.arc_mode == OPEN:
                self.arc_shader.set_int('mode', 0)
            elif engine.arc_mode == CHORD:
                self.arc_shader.set_int('mode', 1)
            elif engine.arc_mode == PIE:
                self.arc_shader.set_int('mode', 2)

            glDrawArrays(GL_POINTS, 0, 1)

        if not engine.stroke.is_none:
            self.arc_outline_shader.use()
            self.arc_outline_shader.set_floatm('pv', pv)
            self.arc_outline_shader.set_floatv('color', engine.stroke.to_gl())
            self.arc_outline_shader.set_floatv('radius', radius.xy)
            self.arc_outline_shader.set_floatv('viewport', engine.viewport)
            self.arc_outline_shader.set_float('thickness', engine.weight)
            self.arc_outline_shader.set_float('start', start)
            self.arc_outline_shader.set_float('stop', stop)
            if engine.arc_mode == OPEN:
                self.arc_outline_shader.set_int('mode', 0)
            elif engine.arc_mode == CHORD:
                self.arc_outline_shader.set_int('mode', 1)
            elif engine.arc_mode == PIE:
                self.arc_outline_shader.set_int('mode', 2)

            glDrawArrays(GL_POINTS, 0, 1)

    def text(self, engine, text, pos):
        arr, size = engine._font.render_raw(str(text), size = engine._text_size)
        
        w, h = size
        top_left = pos.base
        
        if engine.text_align[0] == LEFT:
            pass
        elif engine.text_align[0] == CENTER:
            top_left.x -= w * 0.5
        elif engine.text_align[0] == RIGHT:
            top_left.x -= w

        if engine.text_align[1] == TOP:
            pass
        elif engine.text_align[1] == CENTER:
            top_left.y -= h * 0.5
        elif engine.text_align[1] == BOTTOM:
            top_left.y -= h
        
        glBindTexture(GL_TEXTURE_2D, self.text_texture)
        glTexImage2D(
            GL_TEXTURE_2D,
            0, GL_RED,
            w, h,
            0, GL_RED,
            GL_UNSIGNED_BYTE,
            np.frombuffer(arr, dtype = np.uint8)
        )
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glActiveTexture(GL_TEXTURE0)
        
        self.vertex.set(top_left.base.xyz, GL_DYNAMIC_DRAW)

        self.text_shader.use()
        self.text_shader.set_floatm('pv', self.view @ self.proj)
        self.text_shader.set_floatv('color', engine.fill.to_gl())
        self.text_shader.set_floatv('viewport', engine.viewport)
        self.text_shader.set_float('textSize', w, h)
        self.text_shader.set_int('text', 0)

        glDrawArrays(GL_POINTS, 0, 1)


class RenderOpenGL3(RenderOpenGL):
    def __init__(self):
        super().__init__()
        
        self.point_shader, self.point_vertex = None, None
        self.point_group = []
        
        self.line_shader, self.line_vertex = None, None
        self.line_group = []
        
        self.lines_shader, self.lines_vertex = None, None
        
        self.triangle_shader, self.triangle_vertex = None, None
        self.triangle_group = []
        
        self.quad_shader, self.quad_vertex = None, None
        self.quad_group = []
        
        self.poly_shader, self.poly_vertex = None, None
        self.poly_buffer = None
        
        self.ellipse_shader, self.ellipse_vertex = None, None
        self.ellipse_group = []
        
        self.arc_shader = None
        self.arc_outline_shader = None
        
        self.text_shader, self.text_vertex = None, None
        self.text_texture = 0

    def setup(self, engine):
        super().setup(engine)
        
        self.point_shader, self.point_vertex = Shader(
            (GL_VERTEX_SHADER, POINT_VERT3),
            (GL_GEOMETRY_SHADER, POINT_GEOM3),
            (GL_FRAGMENT_SHADER, FRAG3)
        ), VertexArray('float32', 3, 4, 1, 4, 4, 4, 4)
        self.line_shader, self.line_vertex = Shader(
            (GL_VERTEX_SHADER, LINE_VERT3),
            (GL_GEOMETRY_SHADER, LINE_GEOM3),
            (GL_FRAGMENT_SHADER, FRAG3)
        ), VertexArray('float32', 3, 3, 4, 1, 4, 4, 4, 4)
        self.lines_shader, self.lines_vertex = Shader(
            (GL_VERTEX_SHADER, LINES_VERT3),
            (GL_GEOMETRY_SHADER, LINES_GEOM3),
            (GL_FRAGMENT_SHADER, FRAG3)
        ), VertexArray('float32', 3)
        self.triangle_shader, self.triangle_vertex = Shader(
            (GL_VERTEX_SHADER, TRIANGLE_VERT3),
            (GL_GEOMETRY_SHADER, TRIANGLE_GEOM3),
            (GL_FRAGMENT_SHADER, FRAG3)
        ), VertexArray('float32', 3, 3, 3, 4, 4, 1, 4, 4, 4, 4)
        self.quad_shader, self.quad_vertex = Shader(
            (GL_VERTEX_SHADER, QUAD_VERT3),
            (GL_GEOMETRY_SHADER, QUAD_GEOM3),
            (GL_FRAGMENT_SHADER, FRAG3)
        ), VertexArray('float32', 3, 3, 3, 3, 4, 4, 1, 4, 4, 4, 4)
        self.poly_shader, self.poly_vertex = Shader(
            (GL_VERTEX_SHADER, POLY_VERT3),
            (GL_GEOMETRY_SHADER, POLY_GEOM3),
            (GL_FRAGMENT_SHADER, FRAG3)
        ), VertexArray('float32', 3)
        self.poly_buffer = Buffer(GL_SHADER_STORAGE_BUFFER, 'float32')
        self.poly_buffer.bind_base(1)
        self.ellipse_shader, self.ellipse_vertex = Shader(
            (GL_VERTEX_SHADER, ELLIPSE_VERT3),
            (GL_GEOMETRY_SHADER, ELLIPSE_GEOM3),
            (GL_FRAGMENT_SHADER, FRAG3)
        ), VertexArray('float32', 3, 2, 4, 4, 1, 4, 4, 4, 4)
        # self.arc_shader = Shader(
        #     (GL_VERTEX_SHADER, _VERT),
        #     (GL_GEOMETRY_SHADER, _ARC_GEOM),
        #     (GL_FRAGMENT_SHADER, _FRAG)
        # )
        # self.arc_outline_shader = Shader(
        #     (GL_VERTEX_SHADER, _VERT),
        #     (GL_GEOMETRY_SHADER, _ARC_OUTLINE_GEOM),
        #     (GL_FRAGMENT_SHADER, _FRAG)
        # )
        self.text_shader, self.text_vertex = Shader(
            (GL_VERTEX_SHADER, TEXT_VERT3),
            (GL_GEOMETRY_SHADER, TEXT_GEOM3),
            (GL_FRAGMENT_SHADER, TEXT_FRAG3)
        ), VertexArray('float32', 3, 2, 4, 4, 4, 4, 4)
        self.texture_id = glGenTextures(1)

    def after_draw(self, engine):
        if len(self.point_group) > 0:
            self.point_vertex.bind()

            data = np.array(self.point_group)

            self.point_vertex.set(data, GL_DYNAMIC_DRAW)

            self.point_shader.use()
            self.point_shader.set_floatm('proj', self.proj)
            self.point_shader.set_floatv('viewport', engine.viewport)

            glDrawArrays(GL_POINTS, 0, len(self.point_group))

            self.point_vertex.unbind()
            self.point_group = []
        
        if len(self.line_group) > 0:
            self.line_vertex.bind()

            data = np.array(self.line_group)

            self.line_vertex.set(data, GL_DYNAMIC_DRAW)

            self.line_shader.use()
            self.line_shader.set_floatm('proj', self.proj)
            self.line_shader.set_floatv('viewport', engine.viewport)

            glDrawArrays(GL_POINTS, 0, len(self.line_group))

            self.line_vertex.unbind()
            self.line_group = []
        
        if len(self.triangle_group) > 0:
            self.triangle_vertex.bind()

            data = np.array(self.triangle_group)

            self.triangle_vertex.set(data, GL_DYNAMIC_DRAW)

            self.triangle_shader.use()
            self.triangle_shader.set_floatm('proj', self.proj)
            self.triangle_shader.set_floatv('viewport', engine.viewport)

            glDrawArrays(GL_POINTS, 0, len(self.triangle_group))

            self.triangle_vertex.unbind()
            self.triangle_group = []
        
        if len(self.quad_group) > 0:
            self.quad_vertex.bind()

            data = np.array(self.quad_group)

            self.quad_vertex.set(data, GL_DYNAMIC_DRAW)

            self.quad_shader.use()
            self.quad_shader.set_floatm('proj', self.proj)
            self.quad_shader.set_floatv('viewport', engine.viewport)

            glDrawArrays(GL_POINTS, 0, len(self.quad_group))

            self.quad_vertex.unbind()
            self.quad_group = []
        
        if len(self.ellipse_group) > 0:
            self.ellipse_vertex.bind()

            data = np.array(self.ellipse_group)

            self.ellipse_vertex.set(data, GL_DYNAMIC_DRAW)

            self.ellipse_shader.use()
            self.ellipse_shader.set_floatm('proj', self.proj)
            self.ellipse_shader.set_floatv('viewport', engine.viewport)

            glDrawArrays(GL_POINTS, 0, len(self.ellipse_group))

            self.ellipse_vertex.unbind()
            self.ellipse_group = []

    def point(self, engine, p):
        self.point_group.append([
            *p.base.xyz,
            *engine.stroke.to_gl(),
            engine.weight,
            *self.view.flatten()
        ])

    def line(self, engine, p1, p2):
        self.line_group.append([
            *p1.base.xyz,
            *p2.base.xyz,
            *engine.stroke.to_gl(),
            engine.weight,
            *self.view.flatten()
        ])
    
    def lines(self, engine, *points):
        if not engine.stroke.is_none:
            self.lines_vertex.bind()

            data = [points[0].base.xyz, points[0].base.xyz, points[1].base.xyz]
            for i in range(len(points) - 2):
                data.append(points[i + 0].base.xyz)
                data.append(points[i + 1].base.xyz)
                data.append(points[i + 2].base.xyz)

            self.lines_vertex.set(np.array(data), GL_DYNAMIC_DRAW)

            self.lines_shader.use()
            self.lines_shader.set_floatm('proj', self.proj)
            self.lines_shader.set_floatm('view', self.view)
            self.lines_shader.set_floatv('viewport', engine.viewport)
            self.lines_shader.set_floatv('stroke', engine.stroke.to_gl())
            self.lines_shader.set_float('weight', engine.weight)

            glDrawArrays(GL_TRIANGLES, 0, len(points) * 3)

            self.lines_vertex.unbind()
    
    def triangle(self, engine, p1, p2, p3):
        self.triangle_group.append([
            *p1.base.xyz,
            *p2.base.xyz,
            *p3.base.xyz,
            *engine.fill.to_gl(),
            *engine.stroke.to_gl(),
            engine.weight,
            *self.view.flatten()
        ])

    def quad(self, engine, p1, p2, p3, p4):
        self.quad_group.append([
            *p1.base.xyz,
            *p2.base.xyz,
            *p3.base.xyz,
            *p4.base.xyz,
            *engine.fill.to_gl(),
            *engine.stroke.to_gl(),
            engine.weight,
            *self.view.flatten()
        ])

    def polygon(self, engine, *points):
        if not engine.fill.is_none:
            self.poly_vertex.bind()
            self.poly_vertex.set(points[0].base.xyz, GL_DYNAMIC_DRAW)
            
            self.poly_buffer.bind()
            
            data = np.array([p.base for p in points])
            self.poly_buffer.set(data, GL_DYNAMIC_DRAW)

            self.poly_shader.use()
            self.poly_shader.set_floatm('proj', self.proj)
            self.poly_shader.set_floatm('view', self.view)
            self.poly_shader.set_floatv('fill', engine.fill.to_gl())

            glDrawArrays(GL_POINTS, 0, 1)

            self.poly_vertex.unbind()
        
        if not engine.stroke.is_none:
            self.lines_vertex.bind()

            n = len(points)
            data = []
            for i in range(n):
                data.append(points[(i + 0) % n].base.xyz)
                data.append(points[(i + 1) % n].base.xyz)
                data.append(points[(i + 2) % n].base.xyz)
            data = np.array(data)

            self.lines_vertex.set(data, GL_DYNAMIC_DRAW)

            self.lines_shader.use()
            self.lines_shader.set_floatm('proj', self.proj)
            self.lines_shader.set_floatm('view', self.view)
            self.lines_shader.set_floatv('viewport', engine.viewport)
            self.lines_shader.set_floatv('stroke', engine.stroke.to_gl())
            self.lines_shader.set_float('weight', engine.weight)

            glDrawArrays(GL_TRIANGLES, 0, len(points) * 3)

            self.lines_vertex.unbind()

    def ellipse(self, engine, p1, p2):
        center, radius = p1.base, p2.base

        if engine.ellipse_mode == CENTER:
            radius.xyz *= 0.5
        elif engine.ellipse_mode == RADIUS:
            pass
        elif engine.ellipse_mode == CORNER:
            center.xyz += radius.xyz * 0.5
        elif engine.ellipse_mode == CORNERS:
            radius.xyz = (radius - center).xyz * 0.5
            center.xyz += radius.xyz
        
        self.ellipse_group.append([
            *center.xyz,
            *radius.xy,
            *engine.fill.to_gl(),
            *engine.stroke.to_gl(),
            engine.weight,
            *self.view.flatten()
        ])

    def arc(self, engine, p1, p2, start, stop):
        pass

    def text(self, engine, text, pos):
        arr, size = engine._font.render_raw(str(text), size = engine._text_size)
        
        w, h = size
        top_left = pos.base
        
        if engine.text_align[0] == LEFT:
            pass
        elif engine.text_align[0] == CENTER:
            top_left.x -= w * 0.5
        elif engine.text_align[0] == RIGHT:
            top_left.x -= w

        if engine.text_align[1] == TOP:
            pass
        elif engine.text_align[1] == CENTER:
            top_left.y -= h * 0.5
        elif engine.text_align[1] == BOTTOM:
            top_left.y -= h
        
        glBindTexture(GL_TEXTURE_2D, self.text_texture)
        glTexImage2D(
            GL_TEXTURE_2D,
            0, GL_RED,
            w, h,
            0, GL_RED,
            GL_UNSIGNED_BYTE,
            np.frombuffer(arr, dtype = np.uint8)
        )
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glActiveTexture(GL_TEXTURE0)
        
        self.text_vertex.bind()

        data = np.array([
            *top_left.xyz,
            w, h,
            *engine.fill.to_gl(),
            *self.view.flatten()
        ])

        self.text_vertex.set(data, GL_DYNAMIC_DRAW)

        self.text_shader.use()
        self.text_shader.set_floatm('proj', self.proj)
        self.text_shader.set_floatv('viewport', engine.viewport)
        self.text_shader.set_int('text', 0)

        glDrawArrays(GL_POINTS, 0, 1)

        self.text_vertex.unbind()


VERT2 = '''
#version 330

uniform mat4 pv;

layout(location = 0) in vec3 aPosition;

out vec3 position;

void main(void)
{
    position = aPosition;
    gl_Position = pv * vec4(aPosition, 1.0);
}
'''

FRAG2 = '''
#version 330

out vec4 FragColor;

uniform vec4 color;

void main(void)
{
    FragColor = color;
}
'''

POINT_GEOM2 = '''
#version 330

uniform float thickness;
uniform vec2 viewport;

const float PI = 3.141592653;

layout(points) in;
layout(triangle_strip, max_vertices = 16) out;

vec3 toScreenSpace(vec4 v)
{
    return vec3(v.xy / v.w * viewport, v.z / v.w - 0.000);
}

void main(void)
{
    const int segments = 16;
    
    vec3 point = toScreenSpace(gl_in[0].gl_Position);
    
    for (int i = 0; i < segments; i++) {
        float angle = ceil(i / 2.0) * 2.0 * PI / segments;
        angle *= (i % 2 == 0 ? 1. : -1.);
        
        vec2 off = thickness * vec2(cos(angle), sin(angle));
        gl_Position = vec4((point.xy + off) / viewport, point.z, 1.0);
        EmitVertex();
    }
    EndPrimitive();
}
'''

LINE_GEOM2 = '''
#version 330 core

layout(lines) in;
layout(triangle_strip, max_vertices = 4) out;

uniform float thickness;
uniform vec2 viewport;

vec3 toScreenSpace(vec4 v)
{
    return vec3(v.xy / v.w * viewport, v.z / v.w - 0.000);
}

void main(void)
{
    vec3 p0 = toScreenSpace(gl_in[0].gl_Position);
    vec3 p1 = toScreenSpace(gl_in[1].gl_Position);
    
    vec2 dir = normalize(p1.xy - p0.xy);
    vec2 norm = thickness * vec2(-dir.y, dir.x);
    
    gl_Position = vec4((p0.xy + norm) / viewport, p0.z, 1.0);
    EmitVertex();
    
    gl_Position = vec4((p0.xy - norm) / viewport, p0.z, 1.0);
    EmitVertex();
    
    gl_Position = vec4((p1.xy + norm) / viewport, p1.z, 1.0);
    EmitVertex();
    
    gl_Position = vec4((p1.xy - norm) / viewport, p1.z, 1.0);
    EmitVertex();
    
    EndPrimitive();
}
'''

LINES_GEOM2 = '''
#version 330

uniform mat4 pv;
uniform vec2 viewport;
uniform float thickness;

layout(triangles) in;
layout(triangle_strip, max_vertices = 7) out;

vec3 toScreenSpace(vec4 v)
{
    return vec3(v.xy / v.w * viewport, (v.z - 0.001) / v.w);
}

void main(void)
{
    vec3 p[3];
    p[0] = toScreenSpace(gl_in[0].gl_Position);
    p[1] = toScreenSpace(gl_in[1].gl_Position);
    p[2] = toScreenSpace(gl_in[2].gl_Position);
    
    // Perform Naive Culling
    vec2 area = viewport * 4;
    if(p[1].x < -area.x || p[1].x > area.x) return;
    if(p[1].y < -area.y || p[1].y > area.y) return;
    if(p[2].x < -area.x || p[2].x > area.x) return;
    if(p[2].y < -area.y || p[2].y > area.y) return;
    
    // Determines the normals for the first two line segments
    vec2 v0 = normalize(p[1].xy - p[0].xy);
    vec2 v1 = normalize(p[2].xy - p[1].xy);
    
    vec2 n0 = thickness * vec2(-v0.y, v0.x);
    vec2 n1 = thickness * vec2(-v1.y, v1.x);
    
    // Determines location of bevel
    vec2 p1, p2;
    if(dot(v0, n1) > 0) {
        p1 = p[1].xy + n0;
        p2 = p[1].xy + n1;
    }
    else {
        p1 = p[1].xy - n1;
        p2 = p[1].xy - n0;
    }
    // Generates Bevel at Joint
    gl_Position = vec4(p1 / viewport, p[1].z, 1.0);
    EmitVertex();
    
    gl_Position = vec4(p2 / viewport, p[1].z, 1.0);
    EmitVertex();
    
    gl_Position = vec4(p[1].xy / viewport, p[1].z, 1.0);
    EmitVertex();
    
    EndPrimitive();
    
    // Generates Line Strip
    gl_Position = vec4((p[1].xy + n1) / viewport, p[1].z, 1.0);
    EmitVertex();
    
    gl_Position = vec4((p[1].xy - n1) / viewport, p[1].z, 1.0);
    EmitVertex();
    
    gl_Position = vec4((p[2].xy + n1) / viewport, p[2].z, 1.0);
    EmitVertex();
    
    gl_Position = vec4((p[2].xy - n1) / viewport, p[2].z, 1.0);
    EmitVertex();
    
    EndPrimitive();
}
'''

ELLIPSE_GEOM2 = '''
#version 330

in vec3 position[1];

uniform mat4 pv;
uniform vec2 radius;

const float PI = 3.141592653;

layout(points) in;
layout(triangle_strip, max_vertices = 32) out;

void main(void)
{
    const int segments = 32;
    
    for (int i = 0; i < segments; i++) {
        float angle = ceil(i / 2.0) * 2.0 * PI / float(segments);
        angle *= (i % 2 == 0 ? 1. : -1.);
        
        vec2 off = radius * vec2(cos(angle), sin(angle));
        gl_Position = pv * vec4(position[0].xy + off, 0.0, 1.0);
        EmitVertex();
    }
    EndPrimitive();
}
'''

ELLIPSE_OUTLINE_GEOM2 = '''
#version 330

in vec3 position[1];

uniform mat4 pv;
uniform vec2 radius;
uniform vec2 viewport;
uniform float thickness;

const float PI = 3.141592653;

layout(points) in;
layout(triangle_strip, max_vertices = 224) out;

vec3 toScreenSpace(vec4 v)
{
    return vec3(v.xy / v.w * viewport, (v.z - 0.001) / v.w);
}

void main(void)
{
    const int segments = 32;  // (SegmentsMax+1)*4
    
    vec4 Points[segments];
    
    // Generates Vertices of ellipse from point and radius
    for (int i = 0; i < segments; i++) {
        float angle = i * 2.0 * PI / float(segments);
        
        vec2 off = radius * vec2(cos(angle), sin(angle));
        Points[i] = pv * vec4(position[0].xy + off, 0.0, 1.0);
    }
    
    // Generates Line List with Adjacency minus the 4th point
    // Because bevel joints are only between segments 0 and 1
    vec3 p[3];
    for (int i = 0; i < segments; i++) {
        if (i == 0) {
            p[0] = toScreenSpace(Points[segments - 1]);
            p[1] = toScreenSpace(Points[i + 0]);
            p[2] = toScreenSpace(Points[i + 1]);
        }
        else if (i < segments - 1) {
            p[0] = toScreenSpace(Points[i - 1]);
            p[1] = toScreenSpace(Points[i + 0]);
            p[2] = toScreenSpace(Points[i + 1]);
        }
        else {
            p[0] = toScreenSpace(Points[i - 1]);
            p[1] = toScreenSpace(Points[i + 0]);
            p[2] = toScreenSpace(Points[0]);
        }
        
        // Perform Naive Culling
        vec2 area = viewport * 4;
        if(p[1].x < -area.x || p[1].x > area.x) return;
        if(p[1].y < -area.y || p[1].y > area.y) return;
        if(p[2].x < -area.x || p[2].x > area.x) return;
        if(p[2].y < -area.y || p[2].y > area.y) return;
        
        // Determines the normals for the first two line segments
        vec2 v0 = normalize(p[1].xy - p[0].xy);
        vec2 v1 = normalize(p[2].xy - p[1].xy);
        
        vec2 n0 = thickness * vec2(-v0.y, v0.x);
        vec2 n1 = thickness * vec2(-v1.y, v1.x);
        
        // Determines location of bevel
        vec2 p1, p2;
        if(dot(v0, n1) > 0) {
            p1 = p[1].xy + n0;
            p2 = p[1].xy + n1;
        }
        else {
            p1 = p[1].xy - n1;
            p2 = p[1].xy - n0;
        }
        // Generates Bevel at Joint
        gl_Position = vec4(p1 / viewport, p[1].z, 1.0);
        EmitVertex();
        
        gl_Position = vec4(p2 / viewport, p[1].z, 1.0);
        EmitVertex();
        
        gl_Position = vec4(p[1].xy / viewport, p[1].z, 1.0);
        EmitVertex();
        
        EndPrimitive();
        
        // Generates Line Strip
        gl_Position = vec4((p[1].xy + n1) / viewport, p[1].z, 1.0);
        EmitVertex();
        
        gl_Position = vec4((p[1].xy - n1) / viewport, p[1].z, 1.0);
        EmitVertex();
        
        gl_Position = vec4((p[2].xy + n1) / viewport, p[2].z, 1.0);
        EmitVertex();
        
        gl_Position = vec4((p[2].xy - n1) / viewport, p[2].z, 1.0);
        EmitVertex();
        
        EndPrimitive();
    }
}
'''

ARC_GEOM2 = '''
#version 330

in vec3 position[1];

uniform mat4 pv;
uniform vec2 radius;
uniform float start;
uniform float stop;
uniform int mode;

const float PI = 3.141592653;

layout(points) in;
layout(triangle_strip, max_vertices = 64) out;

vec2 intersect(vec2 l1, vec2 l2, vec2 l3, vec2 l4)
{
    float d = (l1.x - l2.x) * (l3.y - l4.y) - (l1.y - l2.y) * (l3.x - l4.x);
    float t = (l1.x - l3.x) * (l3.y - l4.y) - (l1.y - l3.y) * (l3.x - l4.x);
    return l1 + t / d * (l2 - l1);
}

void main(void)
{
    const int segments = 8;
    
    float angle = (stop + start) / 2.0 - PI;
    vec2 anchor = position[0].xy + radius * vec2(cos(angle), sin(angle));
    vec2 startPos = position[0].xy + radius * vec2(cos(start), sin(start));
    vec2 stopPos = position[0].xy + radius * vec2(cos(stop), sin(stop));
    
    bool flag = false;
    for (int i = 0; i < segments + 1; i++) {
        angle = start + i / float(segments) * (stop - start);
        
        vec2 off = radius * vec2(cos(angle), sin(angle));
        gl_Position = pv * vec4(position[0].xy + off, 0.0, 1.0);
        EmitVertex();
        
        if (0 < i && i < segments) {
            // OPEN, CHORD
            if (mode == 0 || mode == 1) {
                vec2 pos = intersect(startPos, stopPos, anchor, position[0].xy + off);
                gl_Position = pv * vec4(pos, 0.0, 1.0);
            }
            // PIE
            else {
                gl_Position = gl_in[0].gl_Position;
            }
            EmitVertex();
        }
        flag = !flag;
    }
    EndPrimitive();
}
'''

ARC_OUTLINE_GEOM2 = '''
#version 330

in vec3 position[1];

uniform mat4 pv;
uniform vec2 radius;
uniform vec2 viewport;
uniform float thickness;
uniform float start;
uniform float stop;
uniform int mode;

const float PI = 3.141592653;

layout(points) in;
layout(triangle_strip, max_vertices = 224) out;

vec3 toScreenSpace(vec4 v)
{
    return vec3(v.xy / v.w * viewport, (v.z - 0.001) / v.w);
}

void main(void)
{
    const int segments = 8;  // (SegmentsMax+1)*4
    
    vec4 Points[segments + 2];
    
    // Generates Vertices of ellipse from point and radius
    for (int i = 0; i < segments + 1; i++) {
        float angle = start + i / float(segments) * (stop - start);
        
        vec2 off = radius * vec2(cos(angle), sin(angle));
        Points[i] = pv * vec4(position[0].xy + off, 0.0, 1.0);
    }
    // PIE
    if (mode == 2) {
        Points[segments + 1] = pv * vec4(position[0].xy, 0.0, 1.0);
    }
    
    // Generates Line List with Adjacency minus the 4th point
    // Because bevel joints are only between segments 0 and 1
    vec3 p[3];
    int count = Points.length() - (mode == 0 || mode == 1 ? 1 : 0);
    for (int i = 0; i < count; i++) {
        if (i == 0) {
            if (mode == 0) {
                p[0] = toScreenSpace(Points[i + 0]);
                p[1] = toScreenSpace(Points[i + 0]);
                p[2] = toScreenSpace(Points[i + 1]);
            }
            else {
                p[0] = toScreenSpace(Points[count - 1]);
                p[1] = toScreenSpace(Points[i + 0]);
                p[2] = toScreenSpace(Points[i + 1]);
            }
        }
        else if (i < count - 1) {
            p[0] = toScreenSpace(Points[i - 1]);
            p[1] = toScreenSpace(Points[i + 0]);
            p[2] = toScreenSpace(Points[i + 1]);
        }
        else if (mode == 1 || mode == 2) {
            p[0] = toScreenSpace(Points[i - 1]);
            p[1] = toScreenSpace(Points[i + 0]);
            p[2] = toScreenSpace(Points[0]);
        }
        
        // Perform Naive Culling
        vec2 area = viewport * 4;
        if(p[1].x < -area.x || p[1].x > area.x) return;
        if(p[1].y < -area.y || p[1].y > area.y) return;
        if(p[2].x < -area.x || p[2].x > area.x) return;
        if(p[2].y < -area.y || p[2].y > area.y) return;
        
        // Determines the normals for the first two line segments
        vec2 v0 = normalize(p[1].xy - p[0].xy);
        vec2 v1 = normalize(p[2].xy - p[1].xy);
        
        vec2 n0 = thickness * vec2(-v0.y, v0.x);
        vec2 n1 = thickness * vec2(-v1.y, v1.x);
        
        // Determines location of bevel
        vec2 p1, p2;
        if(dot(v0, n1) > 0) {
            p1 = p[1].xy + n0;
            p2 = p[1].xy + n1;
        }
        else {
            p1 = p[1].xy - n1;
            p2 = p[1].xy - n0;
        }
        // Generates Bevel at Joint
        gl_Position = vec4(p1 / viewport, p[1].z, 1.0);
        EmitVertex();
        
        gl_Position = vec4(p2 / viewport, p[1].z, 1.0);
        EmitVertex();
        
        gl_Position = vec4(p[1].xy / viewport, p[1].z, 1.0);
        EmitVertex();
        
        EndPrimitive();
        
        // Generates Line Strip
        gl_Position = vec4((p[1].xy + n1) / viewport, p[1].z, 1.0);
        EmitVertex();
        
        gl_Position = vec4((p[1].xy - n1) / viewport, p[1].z, 1.0);
        EmitVertex();
        
        gl_Position = vec4((p[2].xy + n1) / viewport, p[2].z, 1.0);
        EmitVertex();
        
        gl_Position = vec4((p[2].xy - n1) / viewport, p[2].z, 1.0);
        EmitVertex();
        
        EndPrimitive();
    }
}
'''

TEXT_GEOM2 = '''
#version 330

out vec2 TextCoord;

uniform vec2 viewport;
uniform vec2 textSize;

layout(points) in;
layout(triangle_strip, max_vertices = 4) out;

vec3 toScreenSpace(vec4 v)
{
    return vec3(v.xy / v.w * viewport, (v.z - 0.001) / v.w);
}

void main(void)
{
    vec3 point = toScreenSpace(gl_in[0].gl_Position);
    vec2 offset = vec2(0.0);
    float scale = 2.0;
    
    TextCoord = vec2(0.0, 0.0);
    gl_Position = vec4((point.xy + offset) / viewport, point.z, 1.0);
    EmitVertex();
    
    TextCoord = vec2(1.0, 0.0);
    offset = vec2(textSize.x - 1.0, 0.0) * scale;
    gl_Position = vec4((point.xy + offset) / viewport, point.z, 1.0);
    EmitVertex();
    
    TextCoord = vec2(0.0, 1.0);
    offset = vec2(0.0, -textSize.y + 1.0) * scale;
    gl_Position = vec4((point.xy + offset) / viewport, point.z, 1.0);
    EmitVertex();
    
    TextCoord = vec2(1.0, 1.0);
    offset = vec2(textSize.x - 1.0, -textSize.y + 1.0) * scale;
    gl_Position = vec4((point.xy + offset) / viewport, point.z, 1.0);
    EmitVertex();
    
    EndPrimitive();
}
'''

TEXT_FRAG2 = '''
#version 330

out vec4 FragColor;

in vec2 TextCoord;

uniform vec4 color;
uniform sampler2D text;

void main(void)
{
    FragColor = color;
    FragColor.a *= texture(text, TextCoord).r;
    //FragColor = vec4(TextCoord, 0.0, 1.0);
}
'''

FRAG3 = '''
#version 430

in vec4 color;

out vec4 FragColor;

void main(void)
{
    FragColor = color;
}
'''

POINT_VERT3 = '''
#version 430

layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec4 aStroke;
layout(location = 2) in float aWeight;
layout(location = 3) in mat4 aView;

uniform mat4 proj;

out vec4 stroke;
out float weight;

void main(void)
{
    stroke = aStroke;
    weight = aWeight;
    gl_Position = proj * aView * vec4(aPosition, 1.0);
}
'''

POINT_GEOM3 = '''
#version 430

layout(points) in;
layout(triangle_strip, max_vertices = 16) out;

const float PI = 3.141592653;

uniform vec2 viewport;

in vec4 stroke[1];
in float weight[1];

out vec4 color;

vec3 toScreenSpace(vec4 v)
{
    return vec3(v.xy / v.w * viewport, v.z / v.w - 0.000);
}

void main(void)
{
    if (stroke[0].x >= 0.0) {
        color = stroke[0];
        
        const int segments = 16;
        
        vec3 point = toScreenSpace(gl_in[0].gl_Position);
        
        for (int i = 0; i < segments; i++) {
            float angle = ceil(i / 2.0) * 2.0 * PI / segments;
            angle *= (i % 2 == 0 ? 1. : -1.);
            
            vec2 off = weight[0] * vec2(cos(angle), sin(angle));
            gl_Position = vec4((point.xy + off) / viewport, point.z, 1.0);
            EmitVertex();
        }
        EndPrimitive();
    }
}
'''

LINE_VERT3 = '''
#version 430

layout(location = 0) in vec3 aPosition0;
layout(location = 1) in vec3 aPosition1;
layout(location = 2) in vec4 aStroke;
layout(location = 3) in float aWeight;
layout(location = 4) in mat4 aView;

uniform mat4 proj;

out vec4 position0;
out vec4 position1;
out vec4 stroke;
out float weight;

void main(void)
{
    position0 = proj * aView * vec4(aPosition0, 1.0);
    position1 = proj * aView * vec4(aPosition1, 1.0);
    stroke = aStroke;
    weight = aWeight;
}
'''

LINE_GEOM3 = '''
#version 430

layout(points) in;
layout(triangle_strip, max_vertices = 4) out;

uniform vec2 viewport;

in vec4 position0[1];
in vec4 position1[1];
in vec4 stroke[1];
in float weight[1];

out vec4 color;

vec3 toScreenSpace(vec4 v)
{
    return vec3(v.xy / v.w * viewport, v.z / v.w - 0.000);
}

void main(void)
{
    if (stroke[0].x >= 0.0) {
        color = stroke[0];
        
        vec3 p0 = toScreenSpace(position0[0]);
        vec3 p1 = toScreenSpace(position1[0]);
        
        vec2 dir = normalize(p1.xy - p0.xy);
        vec2 norm = weight[0] * vec2(-dir.y, dir.x);
        
        gl_Position = vec4((p0.xy + norm) / viewport, p0.z, 1.0);
        EmitVertex();
        
        gl_Position = vec4((p0.xy - norm) / viewport, p0.z, 1.0);
        EmitVertex();
        
        gl_Position = vec4((p1.xy + norm) / viewport, p1.z, 1.0);
        EmitVertex();
        
        gl_Position = vec4((p1.xy - norm) / viewport, p1.z, 1.0);
        EmitVertex();
        
        EndPrimitive();
    }
}
'''

LINES_VERT3 = '''
#version 430

layout(location = 0) in vec3 aPosition;

uniform mat4 proj;
uniform mat4 view;

void main(void)
{
    gl_Position = proj * view * vec4(aPosition, 1.0);
}
'''

LINES_GEOM3 = '''
#version 430

layout(triangles) in;
layout(triangle_strip, max_vertices = 16) out;

const float PI = 3.141592653;

uniform vec2 viewport;
uniform vec4 stroke;
uniform float weight;

out vec4 color;

vec3 toScreenSpace(vec4 v)
{
    return vec3(v.xy / v.w * viewport, v.z / v.w - 0.000);
}

void drawLine(vec3 p0, vec3 p1, vec3 p2)
{
    // Perform Naive Culling
    vec2 area = viewport * 4;
    if(p1.x < -area.x || p1.x > area.x) return;
    if(p1.y < -area.y || p1.y > area.y) return;
    if(p2.x < -area.x || p2.x > area.x) return;
    if(p2.y < -area.y || p2.y > area.y) return;
    
    vec2 v0 = normalize(p1.xy - p0.xy);
    vec2 v1 = normalize(p2.xy - p1.xy);
    
    vec2 n0 = weight * vec2(-v0.y, v0.x);
    vec2 n1 = weight * vec2(-v1.y, v1.x);
    
    // Determines location of bevel
    vec2 _p1, _p2;
    if(dot(v0, n1) > 0) {
        _p1 = p1.xy + n0;
        _p2 = p1.xy + n1;
    }
    else {
        _p1 = p1.xy - n1;
        _p2 = p1.xy - n0;
    }
    
    // Generates Bevel at Joint
    gl_Position = vec4(_p1 / viewport, p1.z, 1.0);
    EmitVertex();
    
    gl_Position = vec4(_p2 / viewport, p1.z, 1.0);
    EmitVertex();
    
    gl_Position = vec4(p1.xy / viewport, p1.z, 1.0);
    EmitVertex();
    
    EndPrimitive();
    
    // Generates Line Strip
    gl_Position = vec4((p1.xy + n1) / viewport, p1.z, 1.0);
    EmitVertex();
    
    gl_Position = vec4((p1.xy - n1) / viewport, p1.z, 1.0);
    EmitVertex();
    
    gl_Position = vec4((p2.xy + n1) / viewport, p2.z, 1.0);
    EmitVertex();
    
    gl_Position = vec4((p2.xy - n1) / viewport, p2.z, 1.0);
    EmitVertex();
    
    EndPrimitive();
}

void main(void)
{
    color = stroke;
    
    vec3 p0 = toScreenSpace(gl_in[0].gl_Position);
    vec3 p1 = toScreenSpace(gl_in[1].gl_Position);
    vec3 p2 = toScreenSpace(gl_in[2].gl_Position);
        
    drawLine(p0, p1, p2);
}
'''

TRIANGLE_VERT3 = '''
#version 430

layout(location = 0) in vec3 aPosition0;
layout(location = 1) in vec3 aPosition1;
layout(location = 2) in vec3 aPosition2;
layout(location = 3) in vec4 aFill;
layout(location = 4) in vec4 aStroke;
layout(location = 5) in float aWeight;
layout(location = 6) in mat4 aView;

uniform mat4 proj;

out vec4 position0;
out vec4 position1;
out vec4 position2;
out vec4 fill;
out vec4 stroke;
out float weight;

void main(void)
{
    position0 = proj * aView * vec4(aPosition0, 1.0);
    position1 = proj * aView * vec4(aPosition1, 1.0);
    position2 = proj * aView * vec4(aPosition2, 1.0);
    fill = aFill;
    stroke = aStroke;
    weight = aWeight;
}
'''

TRIANGLE_GEOM3 = '''
#version 430

layout(points) in;
layout(triangle_strip, max_vertices = 24) out;

uniform vec2 viewport;

in vec4 position0[1];
in vec4 position1[1];
in vec4 position2[1];
in vec4 fill[1];
in vec4 stroke[1];
in float weight[1];

out vec4 color;

vec3 toScreenSpace(vec4 v)
{
    return vec3(v.xy / v.w * viewport, v.z / v.w - 0.001);
}

void drawLine(vec3 p0, vec3 p1, vec3 p2)
{
    // Perform Naive Culling
    vec2 area = viewport * 4;
    if(p1.x < -area.x || p1.x > area.x) return;
    if(p1.y < -area.y || p1.y > area.y) return;
    if(p2.x < -area.x || p2.x > area.x) return;
    if(p2.y < -area.y || p2.y > area.y) return;
    
    vec2 v0 = normalize(p1.xy - p0.xy);
    vec2 v1 = normalize(p2.xy - p1.xy);
    
    vec2 n0 = weight[0] * vec2(-v0.y, v0.x);
    vec2 n1 = weight[0] * vec2(-v1.y, v1.x);
    
    // Determines location of bevel
    vec2 _p1, _p2;
    if(dot(v0, n1) > 0) {
        _p1 = p1.xy + n0;
        _p2 = p1.xy + n1;
    }
    else {
        _p1 = p1.xy - n1;
        _p2 = p1.xy - n0;
    }
    
    // Generates Bevel at Joint
    gl_Position = vec4(_p1 / viewport, p1.z, 1.0);
    EmitVertex();
    
    gl_Position = vec4(_p2 / viewport, p1.z, 1.0);
    EmitVertex();
    
    gl_Position = vec4(p1.xy / viewport, p1.z, 1.0);
    EmitVertex();
    
    EndPrimitive();
    
    // Generates Line Strip
    gl_Position = vec4((p1.xy + n1) / viewport, p1.z, 1.0);
    EmitVertex();
    
    gl_Position = vec4((p1.xy - n1) / viewport, p1.z, 1.0);
    EmitVertex();
    
    gl_Position = vec4((p2.xy + n1) / viewport, p2.z, 1.0);
    EmitVertex();
    
    gl_Position = vec4((p2.xy - n1) / viewport, p2.z, 1.0);
    EmitVertex();
    
    EndPrimitive();
}

void main(void)
{
    if (fill[0].x >= 0.0) {
        color = fill[0];
        
        gl_Position = position0[0];
        EmitVertex();
        
        gl_Position = position1[0];
        EmitVertex();
        
        gl_Position = position2[0];
        EmitVertex();
        
        EndPrimitive();
    }
    
    if (stroke[0].x >= 0.0) {
        color = stroke[0];
        
        vec3 p0 = toScreenSpace(position0[0]);
        vec3 p1 = toScreenSpace(position1[0]);
        vec3 p2 = toScreenSpace(position2[0]);
        
        drawLine(p0, p1, p2);
        drawLine(p1, p2, p0);
        drawLine(p2, p0, p1);
    }
}
'''

QUAD_VERT3 = '''
#version 430

layout(location = 0) in vec3 aPosition0;
layout(location = 1) in vec3 aPosition1;
layout(location = 2) in vec3 aPosition2;
layout(location = 3) in vec3 aPosition3;
layout(location = 4) in vec4 aFill;
layout(location = 5) in vec4 aStroke;
layout(location = 6) in float aWeight;
layout(location = 7) in mat4 aView;

uniform mat4 proj;

out vec4 position0;
out vec4 position1;
out vec4 position2;
out vec4 position3;
out vec4 fill;
out vec4 stroke;
out float weight;

void main(void)
{
    position0 = proj * aView * vec4(aPosition0, 1.0);
    position1 = proj * aView * vec4(aPosition1, 1.0);
    position2 = proj * aView * vec4(aPosition2, 1.0);
    position3 = proj * aView * vec4(aPosition3, 1.0);
    fill = aFill;
    stroke = aStroke;
    weight = aWeight;
}
'''

QUAD_GEOM3 = '''
#version 430

layout(points) in;
layout(triangle_strip, max_vertices = 32) out;

uniform vec2 viewport;

in vec4 position0[1];
in vec4 position1[1];
in vec4 position2[1];
in vec4 position3[1];
in vec4 fill[1];
in vec4 stroke[1];
in float weight[1];

out vec4 color;

vec3 toScreenSpace(vec4 v)
{
    return vec3(v.xy / v.w * viewport, v.z / v.w - 0.001);
}

void drawLine(vec3 p0, vec3 p1, vec3 p2)
{
    // Perform Naive Culling
    vec2 area = viewport * 4;
    if(p1.x < -area.x || p1.x > area.x) return;
    if(p1.y < -area.y || p1.y > area.y) return;
    if(p2.x < -area.x || p2.x > area.x) return;
    if(p2.y < -area.y || p2.y > area.y) return;
    
    vec2 v0 = normalize(p1.xy - p0.xy);
    vec2 v1 = normalize(p2.xy - p1.xy);
    
    vec2 n0 = weight[0] * vec2(-v0.y, v0.x);
    vec2 n1 = weight[0] * vec2(-v1.y, v1.x);
    
    // Determines location of bevel
    vec2 _p1, _p2;
    if(dot(v0, n1) > 0) {
        _p1 = p1.xy + n0;
        _p2 = p1.xy + n1;
    }
    else {
        _p1 = p1.xy - n1;
        _p2 = p1.xy - n0;
    }
    
    // Generates Bevel at Joint
    gl_Position = vec4(_p1 / viewport, p1.z, 1.0);
    EmitVertex();
    
    gl_Position = vec4(_p2 / viewport, p1.z, 1.0);
    EmitVertex();
    
    gl_Position = vec4(p1.xy / viewport, p1.z, 1.0);
    EmitVertex();
    
    EndPrimitive();
    
    // Generates Line Strip
    gl_Position = vec4((p1.xy + n1) / viewport, p1.z, 1.0);
    EmitVertex();
    
    gl_Position = vec4((p1.xy - n1) / viewport, p1.z, 1.0);
    EmitVertex();
    
    gl_Position = vec4((p2.xy + n1) / viewport, p2.z, 1.0);
    EmitVertex();
    
    gl_Position = vec4((p2.xy - n1) / viewport, p2.z, 1.0);
    EmitVertex();
    
    EndPrimitive();
}

void main(void)
{
    if (fill[0].x >= 0.0) {
        color = fill[0];
        
        vec3 p02 = position0[0].xyz - position2[0].xyz;
        vec3 p13 = position1[0].xyz - position3[0].xyz;
        
        if (dot(p02, p02) < dot(p13, p13)) {
            gl_Position = position1[0];
            EmitVertex();
            
            gl_Position = position0[0];
            EmitVertex();
            
            gl_Position = position2[0];
            EmitVertex();
            
            gl_Position = position3[0];
            EmitVertex();
        }
        else {
            gl_Position = position0[0];
            EmitVertex();
            
            gl_Position = position1[0];
            EmitVertex();
            
            gl_Position = position3[0];
            EmitVertex();
            
            gl_Position = position2[0];
            EmitVertex();
        }
        
        EndPrimitive();
    }
    
    if (stroke[0].x >= 0.0) {
        color = stroke[0];
        
        vec3 p0 = toScreenSpace(position0[0]);
        vec3 p1 = toScreenSpace(position1[0]);
        vec3 p2 = toScreenSpace(position2[0]);
        vec3 p3 = toScreenSpace(position3[0]);
        
        drawLine(p3, p0, p1);
        drawLine(p0, p1, p2);
        drawLine(p1, p2, p3);
        drawLine(p2, p3, p0);
    }
}
'''

POLY_VERT3 = '''
#version 430

layout(location = 0) in vec3 aPosition;

void main(void)
{
    
}
'''

POLY_GEOM3 = '''
#version 430

layout(points) in;
layout(triangle_strip, max_vertices = 256) out;
    
layout(std140, binding = 1) buffer Vertices { vec4 vertices[]; };

const float EPSILON = 0.0000000001f;

int indices[256];

uniform mat4 proj;
uniform mat4 view;
uniform vec4 fill;

out vec4 color;

float wedge(vec4 a, vec4 b)
{
    return a.x * b.y - a.y * b.x;
}

bool valid_triangle(int n, int prev_i, int curr_i, int next_i)
{
    vec4 prev = vertices[prev_i];
    vec4 curr = vertices[curr_i];
    vec4 next = vertices[next_i];
    
    if (wedge(next - curr, prev - curr) < EPSILON) return false;
    for (int p = 0; p < n; p++) {
        if (p == prev_i || p == curr_i || p == next_i) continue;
        if (wedge(curr - prev, vertices[p] - prev) >= EPSILON &&
           wedge(next - curr, vertices[p] - curr) >= EPSILON && 
           wedge(prev - next, vertices[p] - next) >= EPSILON) return false;
    }
    return true;
}

void main(void)
{
    color = fill;
    
    int n = vertices.length();
    
    float a = 0.0;
    for (int p = n - 1, q = 0; q < n; p = q++) {
        a += wedge(vertices[p], vertices[q]);
    }
    if (a * 0.5 > 0.0) {
        for (int i = 0; i < n; i++) indices[i] = i;
    }
    else {
        for (int i = 0; i < n; i++) indices[i] = n - 1 - i;
    }
    
    int i = 0, count = 2 * n;
    while (n >= 3 && count > 0) {
        count--;
        int prev_i = indices[(i + n - 1) % n];
        int curr_i = indices[(i + n + 0) % n];
        int next_i = indices[(i + n + 1) % n];
        if (valid_triangle(n, prev_i, curr_i, next_i)) {
            gl_Position = proj * view * vertices[prev_i];
            EmitVertex();
            
            gl_Position = proj * view * vertices[curr_i];
            EmitVertex();
            
            gl_Position = proj * view * vertices[next_i];
            EmitVertex();
            
            EndPrimitive();
            
            for (int s = i % n, t = i % n + 1; t < n; s = t++) {
                indices[s] = indices[t];
            }
            n--; count = 2 * n;
        }
        else i++;
    }
}
'''

ELLIPSE_VERT3 = '''
#version 430

layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec2 aRadius;
layout(location = 2) in vec4 aFill;
layout(location = 3) in vec4 aStroke;
layout(location = 4) in float aWeight;
layout(location = 5) in mat4 aView;

uniform mat4 proj;

out vec3 position;
out vec2 radius;
out vec4 fill;
out vec4 stroke;
out float weight;
out mat4 pv;

void main(void)
{
    position = aPosition;
    radius = aRadius;
    fill = aFill;
    stroke = aStroke;
    weight = aWeight;
    pv = proj * aView;
}
'''

ELLIPSE_GEOM3 = '''
#version 430

layout(points) in;
layout(triangle_strip, max_vertices = 128) out;

const float PI = 3.141592653;

uniform vec2 viewport;

in vec3 position[1];
in vec2 radius[1];
in vec4 fill[1];
in vec4 stroke[1];
in float weight[1];
in mat4 pv[1];

out vec4 color;

vec3 toScreenSpace(vec4 v)
{
    return vec3(v.xy / v.w * viewport, v.z / v.w - 0.001);
}

void drawLine(vec3 p0, vec3 p1, vec3 p2)
{
    // Perform Naive Culling
    vec2 area = viewport * 4;
    if(p1.x < -area.x || p1.x > area.x) return;
    if(p1.y < -area.y || p1.y > area.y) return;
    if(p2.x < -area.x || p2.x > area.x) return;
    if(p2.y < -area.y || p2.y > area.y) return;
    
    vec2 v0 = normalize(p1.xy - p0.xy);
    vec2 v1 = normalize(p2.xy - p1.xy);
    
    vec2 n0 = weight[0] * vec2(-v0.y, v0.x);
    vec2 n1 = weight[0] * vec2(-v1.y, v1.x);
    
    // Determines location of bevel
    vec2 _p1, _p2;
    if(dot(v0, n1) > 0) {
        _p1 = p1.xy + n0;
        _p2 = p1.xy + n1;
    }
    else {
        _p1 = p1.xy - n1;
        _p2 = p1.xy - n0;
    }
    
    // Generates Bevel at Joint
    gl_Position = vec4(_p1 / viewport, p1.z, 1.0);
    EmitVertex();
    
    gl_Position = vec4(_p2 / viewport, p1.z, 1.0);
    EmitVertex();
    
    gl_Position = vec4(p1.xy / viewport, p1.z, 1.0);
    EmitVertex();
    
    EndPrimitive();
    
    // Generates Line Strip
    gl_Position = vec4((p1.xy + n1) / viewport, p1.z, 1.0);
    EmitVertex();
    
    gl_Position = vec4((p1.xy - n1) / viewport, p1.z, 1.0);
    EmitVertex();
    
    gl_Position = vec4((p2.xy + n1) / viewport, p2.z, 1.0);
    EmitVertex();
    
    gl_Position = vec4((p2.xy - n1) / viewport, p2.z, 1.0);
    EmitVertex();
    
    EndPrimitive();
}

void main(void)
{
    const int segments = 16;
    
    bool shouldFill = fill[0].x >= 0.0;
    
    if (shouldFill) color = fill[0];
    
    vec4 points[segments];
    for (int i = 0; i < segments; i++) {
        float angle = ceil(i / 2.0) * 2.0 * PI / float(segments);
        angle *= (i % 2 == 0 ? 1. : -1.);
        
        vec2 off = radius[0] * vec2(cos(angle), sin(angle));
        int pointIndex = i % 2 == 0 ? int(0.5 * i) : int(segments - 0.5 * (i + 1));
        points[pointIndex] = pv[0] * vec4(position[0].xy + off, 0.0, 1.0);
        
        if (shouldFill) {
            gl_Position = points[pointIndex];
            EmitVertex();
        }
    }
    if (shouldFill) EndPrimitive();
    
    if (stroke[0].x >= 0.0) {
        color = stroke[0];
        
        vec3 p0, p1, p2;
        for (int i = 0; i < segments; i++) {
            if (i == 0) {
                p0 = toScreenSpace(points[segments - 1]);
                p1 = toScreenSpace(points[i + 0]);
                p2 = toScreenSpace(points[i + 1]);
            }
            else if (i < segments - 1) {
                p0 = toScreenSpace(points[i - 1]);
                p1 = toScreenSpace(points[i + 0]);
                p2 = toScreenSpace(points[i + 1]);
            }
            else {
                p0 = toScreenSpace(points[i - 1]);
                p1 = toScreenSpace(points[i + 0]);
                p2 = toScreenSpace(points[0]);
            }
            drawLine(p0, p1, p2);
        }
    }
}
'''

TEXT_VERT3 = '''
#version 430

layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec2 aSize;
layout(location = 2) in vec4 aFill;
layout(location = 3) in mat4 aView;

uniform mat4 proj;

out vec2 size;
out vec4 fill;

void main(void)
{
    size = aSize;
    fill = aFill;
    gl_Position = proj * aView * vec4(aPosition, 1.0);
}
'''

TEXT_GEOM3 = '''
#version 430

layout(points) in;
layout(triangle_strip, max_vertices = 4) out;

const float PI = 3.141592653;

uniform vec2 viewport;

in vec2 size[1];
in vec4 fill[1];

out vec4 color;
out vec2 coord;

vec3 toScreenSpace(vec4 v)
{
    return vec3(v.xy / v.w * viewport, (v.z) / v.w - 0.01);
}

void main(void)
{
    const float scale = 2.0;
    
    vec3 point = toScreenSpace(gl_in[0].gl_Position);
    vec2 offset;
    
    color = fill[0];
    
    coord = vec2(0.0, 0.0);
    offset = vec2(0.0);
    gl_Position = vec4((point.xy + offset) / viewport, point.z, 1.0);
    EmitVertex();
    
    coord = vec2(1.0, 0.0);
    offset = vec2(size[0].x - 0.0, 0.0) * scale;
    gl_Position = vec4((point.xy + offset) / viewport, point.z, 1.0);
    EmitVertex();
    
    coord = vec2(0.0, 1.0);
    offset = vec2(0.0, 0.0 - size[0].y) * scale;
    gl_Position = vec4((point.xy + offset) / viewport, point.z, 1.0);
    EmitVertex();
    
    coord = vec2(1.0, 1.0);
    offset = vec2(size[0].x - 0.0, 0.0 - size[0].y) * scale;
    gl_Position = vec4((point.xy + offset) / viewport, point.z, 1.0);
    EmitVertex();
    
    EndPrimitive();
}
'''

TEXT_FRAG3 = '''
#version 430

uniform sampler2D text;

in vec4 color;
in vec2 coord;

out vec4 FragColor;

void main(void)
{
    FragColor = color;
    FragColor.a *= texture(text, coord).r;
}
'''

PIXEL_VERT3 = '''
#version 430

layout(location = 0) in vec2 aPosition;

out vec2 TextCoord;

void main(void)
{
    gl_Position = vec4(aPosition, 0.0, 1.0);
    TextCoord = vec2(aPosition.x < 0 ? 0.0 : 1.0, aPosition.y < 0 ? 1.0 : 0.0);
}
'''

PIXEL_FRAG3 = '''
#version 430

uniform sampler2D TextMap;

in vec2 TextCoord;

out vec4 FragColor;

void main(void)
{
    FragColor = texture(TextMap, TextCoord);
}
'''
