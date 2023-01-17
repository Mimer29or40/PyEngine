import ctypes

import numpy as np
import OpenGL.GL as GL
from PIL import Image

import util


_attrs = dict(pos = 3, tex = 2)
_locs, _offs, _v_len = {}, {}, 0
for i, k in enumerate(_attrs.keys()):
    _locs[k] = i
    _offs[k] = ctypes.c_void_p(_v_len * GL.sizeof(ctypes.c_float))
    _v_len += _attrs[k]
_v_len *= GL.sizeof(ctypes.c_float)


MATERIAL_REGISTRY = {}
TEXTURE_REGISTRY = {}


class Model:
    def __init__(self, name):
        self.name = name
        
        self.meshes = []
    
    def draw(self, shader):
        for mesh in self.meshes:
            mesh.draw(shader)


class Mesh:
    def __init__(self, name):
        self.name = name
        
        self.material_name = None
        
        self._index_len = 0
        
        self._vao = GL.glGenVertexArrays(1)
        self._vbo = GL.glGenBuffers(1)
        self._ebo = GL.glGenBuffers(1)
    
    def bind_vertices(self, vertices, indices):
        GL.glBindVertexArray(self._vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._vbo)
        GL.glBufferData(
            GL.GL_ARRAY_BUFFER,
            np.array(vertices, np.float32),
            GL.GL_STATIC_DRAW
        )
        
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self._ebo)
        GL.glBufferData(
            GL.GL_ELEMENT_ARRAY_BUFFER,
            np.array(indices, np.int32),
            GL.GL_STATIC_DRAW
        )
        
        self._index_len = len(indices)
        
        for name in _attrs.keys():
            GL.glVertexAttribPointer(
                _locs[name], _attrs[name],
                GL.GL_FLOAT, GL.GL_FALSE,
                _v_len, _offs[name]
            )
            GL.glEnableVertexAttribArray(_locs[name])
        
        GL.glBindVertexArray(0)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, 0)
    
    def draw(self, shader):
        if self.material_name is not None:
            MATERIAL_REGISTRY[self.material_name].bind(shader)
        
        GL.glBindVertexArray(self._vao)
        GL.glDrawElements(
            GL.GL_TRIANGLES, self._index_len, GL.GL_UNSIGNED_INT, None
        )
        
        GL.glBindVertexArray(0)
        
        if self.material_name is not None:
            MATERIAL_REGISTRY[self.material_name].unbind(shader)


class Material:
    def __init__(self, name):
        global MATERIAL_REGISTRY
        
        self.name = name
        
        self.texture_ids = set()
        
        self.k_a = util.Color(0.2, 0.2, 0.2)
        self.k_d = util.Color(0.8, 0.8, 0.8)
        self.k_s = util.Color(0.1, 0.1, 0.1)
        self.k_e = util.Color(0., 0., 0.)
        self.t_f = util.Color(1., 1., 1.)
        self.illum = 0
        self.d = 1.
        self.n_s = 100
        self.sharpness = 60
        self.n_i = 1.
        
        MATERIAL_REGISTRY[name] = self
    
    def __del__(self):
        global MATERIAL_REGISTRY
        
        del MATERIAL_REGISTRY[self.name]
    
    def __eq__(self, other):
        try:
            return self.name == self.other
        except AttributeError:
            return False
    
    def __ne__(self, other):
        return not self.__eq__(other)
    
    def bind(self, shader):
        shader.set_floatv('material.Ka', self.k_a)
        shader.set_floatv('material.Kd', self.k_d)
        shader.set_floatv('material.Ks', self.k_s)
        shader.set_floatv('material.Ke', self.k_e)
        shader.set_floatv('material.Tf', self.t_f)
        shader.set_int('material.illum', self.illum)
        shader.set_float('material.d', self.d)
        shader.set_int('material.Ns', self.n_s)
        shader.set_int('material.sharpness', self.sharpness)
        shader.set_float('material.Ni', self.n_i)
        
        for i, tex_id in enumerate(self.texture_ids):
            texture = TEXTURE_REGISTRY[tex_id]
            
            shader.set_bool('material.use_' + texture.image_type, True)
            shader.set_int('material.' + texture.image_type, i)
            
            GL.glActiveTexture(GL.GL_TEXTURE0 + i)
            GL.glBindTexture(GL.GL_TEXTURE_2D, texture.id)
    
    def unbind(self, shader):
        for i, tex_id in enumerate(self.texture_ids):
            texture = TEXTURE_REGISTRY[tex_id]
            
            shader.set_bool('material.use_' + texture.image_type, False)
            
            GL.glActiveTexture(GL.GL_TEXTURE0 + i)
            GL.glBindTexture(GL.GL_TEXTURE_2D, 0)


class Texture:
    def __init__(self, name):
        global TEXTURE_REGISTRY
        
        self.name = name
        
        self.id = glGenTextures(1)
        
        self.data = np.zeros((0, 0, 4), dtype = np.uint8)
        
        TEXTURE_REGISTRY[name] = self
    
    def __del__(self):
        global TEXTURE_REGISTRY
        
        del TEXTURE_REGISTRY[self.name]
    
    def gen(self,
        wrapS = GL.GL_REPEAT, wrapT = GL.GL_REPEAT,
        minFil = GL.GL_LINEAR, magFil = GL.GL_LINEAR
    ):
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.id)
        
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, wrapS)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, wrapT)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, minFil)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, magFil)
        
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D,
            0, GL.GL_RGBA,
            *self.data.shape[:2],
            0, GL.GL_RGBA,
            GL.GL_UNSIGNED_BYTE,
            self.data
        )
        GL.glGenerateMipmap(GL.GL_TEXTURE_2D)
        
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
    
    @classmethod
    def from_file(cls, name, path):
        texture = cls(name)
        
        image_array = np.asarray(Image.open(path))
        
        filler = np.ones((*image_array.shape[:2], 1), dtype = np.uint8) * 255
        while image_array.shape[2] < 4:
            image_array = np.concatenate((image_array, filler), axis = -1)
        
        texture.data = image_array
        
        return texture
    
    @classmethod
    def from_array(cls, name, array):
        texture = cls(name)
        
        if len(array.shape) == 2:
            arr = np.uint8((array * 255) if util.is_float(array) else array)
            color = np.zeros((*arr.shape, 3), dtype = np.uint8)
            color[:,:,0] = color[:,:,1] = color[:,:,2] = arr
        elif len(array.shape) == 3:
            arr = np.uint8((array * 255) if util.is_float(array) else array)
            color = arr
        else:
            raise Exception('Array is wrong shape', array.shape)
        
        filler = np.ones((*color.shape[:2], 1), dtype = np.uint8) * 255
        while color.shape[2] < 4:
            color = np.concatenate((color, filler), axis = -1)
        
        texture.data = color
        
        return texture
