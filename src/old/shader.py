import functools

from OpenGL.GL import *


def _do_nothing_on_gl_error(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except GLError:
            return None
    return wrapper


class Shader:
    def __init__(self, vertex = '', fragment = '', geometry = ''):
        self.program = glCreateProgram()
        
        if vertex != '':
            self._attach(GL_VERTEX_SHADER, vertex)
        if fragment != '':
            self._attach(GL_FRAGMENT_SHADER, fragment)
        if geometry != '':
            self._attach(GL_GEOMETRY_SHADER, geometry)
        
        glLinkProgram(self.program)
        
        self._uniforms = {}
    
    def _attach(self, shader_type, source):
        if '\n' not in source:
            with open(source, 'r') as f:
                source = f.readlines()
        
        shader = glCreateShader(shader_type)
        glShaderSource(shader, source)
        glCompileShader(shader)
        
        if glGetShaderiv(shader, GL_COMPILE_STATUS) != 1:
            raise Exception(
                'Could not compile shader\n'
                'Shader compilation Log:\n'
                '' + str(glGetShaderInfoLog(shader))
            )
        
        glAttachShader(self.program, shader)
        glDeleteShader(shader)
    
    def _get_uniform_location(self, var):
        try:
            return self._uniforms[var]
        except KeyError:
            pass
        self._uniforms[var] = glGetUniformLocation(self.program, var)
        return self._uniforms[var]
    
    def use(self):
        try:
            glUseProgram(self.program)
        except GLError:
            print(glGetProgramInfoLog(self.program))
            raise
    
    @_do_nothing_on_gl_error
    def set_int(self, name, *args):
        uniform = self._get_uniform_location(name)
        arg_len = len(args)
        if arg_len == 1:
            glUniform1i(uniform, *args)
        elif arg_len == 2:
            glUniform2i(uniform, *args)
        elif arg_len == 3:
            glUniform3i(uniform, *args)
        elif arg_len == 4:
            glUniform4i(uniform, *args)
        else:
            raise Exception('Wrong Number of Arguments: {}'.format(arg_len))
    
    @_do_nothing_on_gl_error
    def set_float(self, name, *args):
        uniform = self._get_uniform_location(name)
        arg_len = len(args)
        if arg_len == 1:
            glUniform1f(uniform, *args)
        elif arg_len == 2:
            glUniform2f(uniform, *args)
        elif arg_len == 3:
            glUniform3f(uniform, *args)
        elif arg_len == 4:
            glUniform4f(uniform, *args)
        else:
            raise Exception('Wrong Number of Arguments: {}'.format(arg_len))
    
    @_do_nothing_on_gl_error
    def set_bool(self, name, v1):
        glUniform1i(self._get_uniform_location(name), int(v1))
    
    @_do_nothing_on_gl_error
    def set_intv(self, name, vec):
        uniform = self._get_uniform_location(name)
        arg_len = len(vec)
        if arg_len == 1:
            glUniform1iv(uniform, 1, vec)
        elif arg_len == 2:
            glUniform2iv(uniform, 1, vec)
        elif arg_len == 3:
            glUniform3iv(uniform, 1, vec)
        elif arg_len == 4:
            glUniform4iv(uniform, 1, vec)
        else:
            raise Exception('Wrong Number of Arguments: {}'.format(arg_len))
    
    @_do_nothing_on_gl_error
    def set_floatv(self, name, vec):
        uniform = self._get_uniform_location(name)
        arg_len = len(vec)
        if arg_len == 1:
            glUniform1fv(uniform, 1, vec)
        elif arg_len == 2:
            glUniform2fv(uniform, 1, vec)
        elif arg_len == 3:
            glUniform3fv(uniform, 1, vec)
        elif arg_len == 4:
            glUniform4fv(uniform, 1, vec)
        else:
            raise Exception('Wrong Number of Arguments: {}'.format(arg_len))
    
    @_do_nothing_on_gl_error
    def set_floatm(self, name, mat):
        uniform = self._get_uniform_location(name)
        shape = mat.shape
        if shape == (2, 2):
            glUniformMatrix2fv(uniform, 1, GL_FALSE, mat)
        elif shape == (2, 3):
            glUniformMatrix2x3fv(uniform, 1, GL_FALSE, mat)
        elif shape == (2, 4):
            glUniformMatrix2x4fv(uniform, 1, GL_FALSE, mat)
        elif shape == (3, 2):
            glUniformMatrix3x2fv(uniform, 1, GL_FALSE, mat)
        elif shape == (3, 3):
            glUniformMatrix3fv(uniform, 1, GL_FALSE, mat)
        elif shape == (3, 4):
            glUniformMatrix3x4fv(uniform, 1, GL_FALSE, mat)
        elif shape == (4, 2):
            glUniformMatrix4x2fv(uniform, 1, GL_FALSE, mat)
        elif shape == (4, 3):
            glUniformMatrix4x3fv(uniform, 1, GL_FALSE, mat)
        elif shape == (4, 4):
            glUniformMatrix4fv(uniform, 1, GL_FALSE, mat)
        else:
            raise Exception('Wrong Matrix Shape: {}'.format(shape))
