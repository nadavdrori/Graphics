import math

import numpy as np


class Sphere:
    def __init__(self, position, radius, material_index):
        self.position = position
        self.radius = radius
        self.material_index = material_index

    def find_intersections(self, ray):
        L = self.position - ray.origin
        t_ca = np.dot(L, ray.direction)
        if t_ca < 0:
            return None
        d_2 = np.dot(L, L) - t_ca ** 2
        if d_2 > self.radius ** 2:
            return None
        t_hc = np.sqrt(self.radius ** 2 - d_2)
        t_0 = t_ca - t_hc
        t_1 = t_ca + t_hc

        if t_0 > 0:
            return ray.eval(t_0)
        elif t_1 > 0:
            return ray.eval(t_1)
        else:
            return None

    def get_normal(self, point):
        norm = (point - self.position)
        return norm / np.linalg.norm(norm)


