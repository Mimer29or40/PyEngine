import numpy as np

from OpenGL.GL import *


_types = {
    np.dtype('int8'): GL_BYTE,
    np.dtype('uint8'): GL_UNSIGNED_BYTE,
    np.dtype('int16'): GL_SHORT,
    np.dtype('uint16'): GL_UNSIGNED_SHORT,
    np.dtype('int32'): GL_INT,
    np.dtype('uint32'): GL_UNSIGNED_INT,
    np.dtype('float32'): GL_FLOAT,
    np.dtype('float64'): GL_DOUBLE
}


def _set(var, dtype):
    def wrapper(self, name, *args):
        dim = len(args)
        try:
            func = globals()[f'glUniform{dim}{var}']
            func(self._get_uniform(name), *map(dtype, args))
        except KeyError as e:
            raise TypeError from e

    def wrapper_v(self, name, vec):
        dim = len(vec)
        try:
            func = globals()[f'glUniform{dim}{var}v']
            func(self._get_uniform(name), 1, list(map(dtype, vec)))
        except KeyError as e:
            raise TypeError from e

    def wrapper_m(self, name, mat):
        dim1, dim2 = mat.shape
        dim = f'{dim1}x{dim2}' if dim1 != dim2 else f'{dim1}'
        try:
            func = globals()[f'glUniformMatrix{dim}{var}v']
            func(self._get_uniform(name), 1, GL_FALSE, mat)
        except KeyError as e:
            raise TypeError from e

    return wrapper, wrapper_v, wrapper_m


class Shader(int):
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls, glCreateProgram())

    def __init__(self, *shaders):
        super().__init__()
        self._uniforms = {}
        self._shaders = []

        for stype, source in shaders:
            if '\n' not in source:
                with open(source, 'r') as f:
                    source = f.readlines()
            shader = glCreateShader(stype)
            glShaderSource(shader, source)
            glCompileShader(shader)

            result = glGetShaderiv(shader, GL_COMPILE_STATUS)
            if not result:
                log = glGetShaderInfoLog(shader).decode().split('\n\n')
                log = '\n'.join(log)
                raise RuntimeError(
                    f'Shader compile failure ({result}): {log}',
                    stype
                )
            glAttachShader(self, shader)
            glDeleteShader(shader)
            self._shaders.append(shader)

        glLinkProgram(self)
        link_status = glGetProgramiv(self, GL_LINK_STATUS)
        if link_status == GL_FALSE:
            log = glGetProgramInfoLog(self).decode().split('\n\n')
            log = '\n'.join(log)
            raise RuntimeError(
                f'Link failure ({link_status}): {log}',
                self
            )
        glValidateProgram(self)
        validation = glGetProgramiv(self, GL_VALIDATE_STATUS)
        if validation == GL_FALSE:
            log = glGetProgramInfoLog(self).decode().split('\n\n')
            log = '\n'.join(log)
            raise RuntimeError(
                f'Validation failure ({validation}): {log}',
                self
            )

    def _get_uniform(self, var):
        try:
            return self._uniforms[var]
        except KeyError:
            self._uniforms[var] = glGetUniformLocation(self, var)
        return self._uniforms[var]

    def use(self):
        glUseProgram(self)

    def delete(self):
        for s in self._shaders:
            glDetachShader(self, s)
        glDeleteProgram(self)

    set_int, set_intv, set_intm = _set('i', int)
    set_bool, set_boolv, set_boolm = _set('i', int)
    set_float, set_floatv, set_floatm = _set('f', float)


class Buffer(int):
    def __new__(cls, btype, dtype):
        return super().__new__(cls, glGenBuffers(1))

    def __init__(self, btype, dtype):
        super().__init__()
        self._btype = btype
        self._dtype = np.dtype(dtype)
        self._shape = None

    @property
    def size(self):
        return np.prod(self._shape) * self._dtype.itemsize
    
    def bind_base(self, base):
        glBindBufferBase(self._btype, base, self)

    def bind(self):
        glBindBuffer(self._btype, self)

    def unbind(self):
        glBindBuffer(self._btype, 0)

    def get(self):
        d = glGetBufferSubData(self._btype, 0, self.size)
        return np.ndarray(shape = self._shape, dtype = self._dtype, buffer = d)

    def set(self, data, usage = GL_STATIC_DRAW):
        self._shape = data.shape
        glBufferData(self._btype, self.size, data.astype(self._dtype), usage)


class VertexArray:
    def __init__(self, dtype, *sizes):
        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)
        
        self.dtype = np.dtype(dtype)
        self.attributes = []
        
        self.stride = 0
        for i, size in enumerate(sizes):
            self.attributes.append(i)
            self.stride += size * self.dtype.itemsize
        
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        
        offset = 0
        for i, size in enumerate(sizes):
            glVertexAttribPointer(
                i, size,
                _types[self.dtype], GL_FALSE,
                self.stride, ctypes.c_void_p(offset)
            )
            offset += size * self.dtype.itemsize

    def bind(self):
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        for i in self.attributes:
            glEnableVertexAttribArray(i)

    def unbind(self):
        glBindVertexArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        for i in self.attributes:
            glDisableVertexAttribArray(i)
    
    def set(self, data, usage = GL_STATIC_DRAW):
        glBufferData(GL_ARRAY_BUFFER, data.astype(self.dtype), usage)
