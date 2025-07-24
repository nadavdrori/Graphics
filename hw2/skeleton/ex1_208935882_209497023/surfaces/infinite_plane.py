import numpy as np

from surfaces.ray import Ray

class InfinitePlane:
    def __init__(self, normal, offset, material_index):
        self.normal = np.array(normal)
        self.normal = normal / np.linalg.norm(self.normal)
        self.offset = offset
        self.material_index = material_index

    def find_intersections(self, ray: Ray):
        eps = 1e-6
        denominator = np.dot(self.normal, ray.direction)  # dot product of the normal and ray
        if np.abs(denominator) < eps:  # Check if the ray is parallel to the plane
            return None  # No intersection if parallel
        t = (self.offset - np.dot(self.normal, ray.origin)) / denominator
        if t > 0:
            return ray.eval(t)
        else:
            return None


    def get_normal(self, point):
        return self.normal


