import os

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = 'True'
os.environ['PYGAME_FREETYPE'] = 'True'

import numpy as np

from .constants import *
from .functions import *
from .vector import Vector, Matrix, Color
from .engine import Engine

engine = Engine()

from .gui import *

del Engine
del os
