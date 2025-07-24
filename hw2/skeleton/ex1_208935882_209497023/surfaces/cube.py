import numpy as np
from surfaces.ray import Ray


class Cube:
    def __init__(self, position, scale, material_index):
        self.position = np.array(position)
        self.scale = scale
        self.material_index = material_index


    def find_intersections(self, ray: Ray):
        # Compute bounds of cube
        half_scale = self.scale / 2
        max_bound = self.position + half_scale
        min_bound = self.position - half_scale
        
        
        t_max = np.inf
        t_min = -np.inf

        for i in range(3):  # Check axis
            ray_dir = 1 / ray.direction[i]
            t0 = (min_bound[i] - ray.origin[i]) * ray_dir
            t1 = (max_bound[i] - ray.origin[i]) * ray_dir
            if ray_dir < 0:
                t0, t1 = t1, t0
            t_min = max(t_min, t0)
            t_max = min(t_max, t1)
            if t_min > t_max:
                return None  # No intersection

        if t_min < 0: # Check if the cube is behind the ray
            return None  
        
        return ray.eval(t_min)
    
    
    def get_normal(self, point):
        # Compute normal based on which face of the cube the point is on
        half_scale = self.scale / 2
        max_bound = self.position + half_scale
        min_bound = self.position - half_scale

        # Normal for each face of the cube
        if np.abs(point[0] - min_bound[0]) < 1e-6:
            return np.array([-1, 0, 0])
        elif np.abs(point[0] - max_bound[0]) < 1e-6:
            return np.array([1, 0, 0])
        elif np.abs(point[1] - min_bound[1]) < 1e-6:
            return np.array([0, -1, 0])
        elif np.abs(point[1] - max_bound[1]) < 1e-6:
            return np.array([0, 1, 0])
        elif np.abs(point[2] - min_bound[2]) < 1e-6:
            return np.array([0, 0, -1])
        elif np.abs(point[2] - max_bound[2]) < 1e-6:
            return np.array([0, 0, 1])
        else:
            raise ValueError("Point is not on the surface of the cube")
