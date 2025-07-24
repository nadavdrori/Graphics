import numpy as np

class Camera:
    def __init__(self, position, look_at, up_vector, screen_distance, screen_width):
        self.position = np.array(position)
        self.look_at = np.array(look_at) / np.linalg.norm(look_at)
        self.up_vector = np.array(up_vector)
        self.screen_distance = screen_distance
        self.screen_width = screen_width
        self.direction = np.subtract(look_at, position) / np.linalg.norm(np.subtract(look_at, position))  # Normalized direction for camera.
        right_vector = np.cross(self.direction, self.up_vector)
        right_vector = right_vector / np.linalg.norm(right_vector)
        self.right_vector = np.array(right_vector)
        self.fixed_up_vector = np.cross(right_vector, self.direction)
        self.fixed_up_vector = self.fixed_up_vector / np.linalg.norm(self.fixed_up_vector)
