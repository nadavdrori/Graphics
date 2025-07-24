# Ray tracer by: Yotam Zvieli & Nadav Drori.

import argparse
from PIL import Image
import numpy as np
from camera import Camera
from surfaces.ray import Ray
from light import Light
from material import Material
from scene_settings import SceneSettings
from surfaces.cube import Cube
from surfaces.infinite_plane import InfinitePlane
from surfaces.sphere import Sphere


def parse_scene_file(file_path):
    objects = []
    camera = None
    scene_settings = None
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            obj_type = parts[0]
            params = [float(p) for p in parts[1:]]
            if obj_type == "cam":
                camera = Camera(params[:3], params[3:6], params[6:9], params[9], params[10])
            elif obj_type == "set":
                scene_settings = SceneSettings(params[:3], params[3], params[4])
            elif obj_type == "mtl":
                material = Material(params[:3], params[3:6], params[6:9], params[9], params[10])
                objects.append(material)
            elif obj_type == "sph":
                sphere = Sphere(params[:3], params[3], int(params[4]))
                objects.append(sphere)
            elif obj_type == "pln":
                plane = InfinitePlane(params[:3], params[3], int(params[4]))
                objects.append(plane)
            elif obj_type == "box":
                cube = Cube(params[:3], params[3], int(params[4]))
                objects.append(cube)
            elif obj_type == "lgt":
                light = Light(params[:3], params[3:6], params[6], params[7], params[8])
                objects.append(light)
            else:
                raise ValueError("Unknown object type: {}".format(obj_type))
    return camera, scene_settings, objects


def save_image(image_array, name):
    image = Image.fromarray(np.uint8(image_array))

    # Save the image to a file
    image.save(f"output/{name}.png")


def main():
    parser = argparse.ArgumentParser(description='Python Ray Tracer')
    parser.add_argument('scene_file', type=str, help='Path to the scene file')
    parser.add_argument('output_image', type=str, help='Name of the output image file')
    parser.add_argument('--width', type=int, default=500, help='Image width')
    parser.add_argument('--height', type=int, default=500, help='Image height')
    args = parser.parse_args()

    # Parse the scene file
    camera, scene_settings, objects = parse_scene_file(args.scene_file)

    width = args.width
    height = args.height

    image_array = np.zeros((height, width, 3))
    screen_width = camera.screen_width
    size_of_pixel = screen_width / width
    screen_height = size_of_pixel * height

    lights = [obj for obj in objects if isinstance(obj, Light)]
    materials = [obj for obj in objects if isinstance(obj, Material)]
    scene_objects = [obj for obj in objects if not (isinstance(obj, Light) or isinstance(obj, Material))]

    for i in range(width):
        for j in range(height):
            ray = generate_ray(camera, i, j, width, height, screen_height)
            nearest_hit = find_intersections(ray, scene_objects)
            picxel_color = find_pixel_color(nearest_hit, ray, scene_objects, materials, lights, scene_settings,
                                            scene_settings.background_color, 0, scene_settings.max_recursions)
            image_array[j][width - 1 - i] = picxel_color

    # Save the output image
    save_image(image_array, args.output_image)


def generate_ray(camera, x, y, image_width, image_height, screen_height):
    # section 1 in the project general flow section
    pixel_position = calculate_pixel_position(camera, x, y, image_width, image_height, screen_height)
    direction = pixel_position - camera.position
    return Ray(camera.position, direction / np.linalg.norm(direction))


def calculate_pixel_position(camera, x, y, image_width, image_height, screen_height):
    # We want to find the middle of the pixel on the screen.
    screen_x = (x + 0.5) / image_width
    screen_y = (y + 0.5) / image_height

    # Find the position of uppermost pixel on screen.
    upper_most_left = (camera.position + camera.screen_distance * camera.direction
                       + camera.fixed_up_vector * (screen_height / 2) - camera.right_vector * (camera.screen_width / 2))

    # add pixel offset to the uppermost pixel.
    pixel_position = upper_most_left + camera.right_vector * camera.screen_width * screen_x - camera.fixed_up_vector * screen_height * screen_y

    return pixel_position


def find_intersections(ray, objects):
    # Steps 2 + 3 - find the closest intersection.
    all_intersections = []
    for obj in objects:
        maybe_hit_point = obj.find_intersections(ray)
        if maybe_hit_point is not None:
            all_intersections.append((maybe_hit_point, obj))
    if len(all_intersections) == 0:
        return None  # if none => hit the background.
    else:
        all_intersections = \
            sorted(all_intersections, key=lambda point_and_index: np.linalg.norm(ray.origin - point_and_index[0]))
        return all_intersections[0]


def find_pixel_color(nearest_hit, ray, objects, materials, lights, scene_settings, background_color, recursion_depth,
                     max_recursion_depth):
    # Steps 4 + 5 in general process
    if nearest_hit is None:
        return background_color

    intersection_point, hit_obj = nearest_hit
    material = materials[hit_obj.material_index - 1]

    color = np.zeros(3)

    normal = hit_obj.get_normal(intersection_point)  
    view_direction = -ray.direction

    for light in lights:
        # Soft shadows
        hit_ratio = generate_soft_shadows(hit_obj, intersection_point, light, objects, scene_settings)

        # Diffuse light
        light_direction = light.position - intersection_point  
        light_direction = light_direction / np.linalg.norm(light_direction)  
        diffuse_intensity = max(np.dot(normal, light_direction), 0)  
        diffuse_color = material.diffuse_color * light.color * diffuse_intensity  

        # Specular color
        reflection_direction = 2 * np.dot(normal, light_direction) * normal - light_direction 
        specular_intensity = np.power(max(np.dot(view_direction, reflection_direction), 0), material.shininess)  
        specular_color = material.specular_color * light.color * specular_intensity * light.specular_intensity

        color += ((background_color * material.transparency) +
                  (diffuse_color + specular_color) * (1 - material.transparency) + material.reflection_color) * hit_ratio * 255 * 0.75

    # Reflection and recursion
    if recursion_depth < max_recursion_depth:
        reflection_direction = ray.direction - 2 * np.dot(ray.direction, normal) * normal
        reflection_ray = Ray(intersection_point + reflection_direction * 1e-4, reflection_direction)
        reflection_hit = find_intersections(reflection_ray, objects)
        if reflection_hit is not None:
            reflection_color = find_pixel_color(reflection_hit, reflection_ray, objects, materials, lights, scene_settings,
                                                background_color, recursion_depth + 1, max_recursion_depth)
        else:
            reflection_color = background_color  # If there's no hit, return background color
        color += material.reflection_color * reflection_color * (1 - material.transparency)

    return np.clip(color, 0, 255)


def generate_soft_shadows(hit_obj, intersection_point, light, objects, scene_settings):
    N = int(scene_settings.root_number_shadow_rays)
    total_rays = N * N
    rays_hit = 0

    light_to_point = intersection_point - light.position
    light_to_point_normalized = light_to_point / np.linalg.norm(light_to_point)

    # choose axis the rectangle - arbitrary.
    if np.abs(light_to_point_normalized[0]) < np.abs(light_to_point_normalized[1]):
        rect_x = np.cross(light_to_point_normalized, [1, 0, 0])
    else:
        rect_x = np.cross(light_to_point_normalized, [0, 1, 0])
    rect_x = rect_x / np.linalg.norm(rect_x)
    rect_y = np.cross(light_to_point_normalized, rect_x)

    for i in range(N):
        for j in range(N):
            # Generate a random point within each cell of the grid
            u = (i + np.random.uniform()) / N
            v = (j + np.random.uniform()) / N

            # Calculate the position of the random point on the light's surface inside the i, j cell
            random_point = (light.position +
                            (u - 0.5) * 2 * light.radius * rect_x +
                            (v - 0.5) * 2 * light.radius * rect_y)

            shadow_dir = intersection_point - random_point
            shadow_dir_normalized = shadow_dir / np.linalg.norm(shadow_dir)
            shadow_ray = Ray(random_point, shadow_dir_normalized)

            shadow_hit = find_intersections(shadow_ray, objects)

            if shadow_hit is not None and shadow_hit[1] == hit_obj:
                rays_hit += 1

    shadow_hit_rate = rays_hit / total_rays
    light_intensity = (1 - light.shadow_intensity) + light.shadow_intensity * shadow_hit_rate

    return light_intensity


if __name__ == '__main__':
    main()
