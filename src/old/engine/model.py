import ctypes

import numpy as np
import OpenGL.GL as GL

_attributes = dict(size=2, tex=2)
_locations, _offsets, _vertex_len = {}, {}, 0
for i, k in enumerate(_attributes.keys()):
    _locations[k] = i
    _offsets[k] = ctypes.c_void_p(_vertex_len * GL.sizeof(ctypes.c_float))
    _vertex_len += _attributes[k]
_vertex_len *= GL.sizeof(ctypes.c_float)


class Model:
    def __init__(self):
        self.meshes = {}
        self.materials = {}
        self.textures = {}


class Mesh:
    def __init__(self, name):
        self.name = name

        self.material = None

        self._index_len = 0

        self._vao = GL.glGenVertexArrays(1)
        self._vbo = GL.glGenBuffers(1)
        self._ebo = GL.glGenBuffers(1)

    def bind_vertices(self, vertices, indices):
        GL.glBindVertexArray(self._vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, np.array(vertices, np.float32), GL.GL_STATIC_DRAW)

        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self._ebo)
        GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, np.array(indices, np.int32), GL.GL_STATIC_DRAW)

        self._index_len = len(indices)

        for name in _attributes:
            GL.glVertexAttribPointer(
                _locations[name],
                _attributes[name],
                GL.GL_FLOAT,
                GL.GL_FALSE,
                _vertex_len,
                _offsets[name],
            )
            GL.glEnableVertexAttribArray(_locations[name])

        GL.glBindVertexArray(0)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, 0)

    def draw(self, shader):
        if self.material is not None:
            self.material.bind(shader)

        GL.glBindVertexArray(self._vao)
        GL.glDrawElements(GL.GL_TRIANGLES, self._index_len, GL.GL_UNSIGNED_INT, None)

        GL.glBindVertexArray(0)

        if self.material is not None:
            self.material.unbind(shader)
