import numpy as np


class Ray:
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction / np.linalg.norm(direction)

    def eval(self, t):
        return self.origin + (t * self.direction)


