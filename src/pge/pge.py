import random


class PixelGameEngine:
    def __init__(self, app_name, seed=None):
        self.app_name = app_name
        if seed is None:
            random.seed(seed)
