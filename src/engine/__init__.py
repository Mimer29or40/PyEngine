import os

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "True"
os.environ["PYGAME_FREETYPE"] = "True"

import numpy as np

from .constants import *
from .engine import Engine
from .functions import *
from .vector import Color
from .vector import Matrix
from .vector import Vector

engine = Engine()

from .gui import *

del Engine
del os
