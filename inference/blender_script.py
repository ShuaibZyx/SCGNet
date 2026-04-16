import argparse
import math
import os
from os.path import basename, dirname, join, splitext, isfile
import random
import sys
import time
from mathutils import Vector
import numpy as np
import bpy
import bmesh
import shutil


parser = argparse.ArgumentParser()
parser.add_argument(
    "--object_path",
    type=str,
    required=True,
    help="Path to the object file",
)
parser.add_argument("--output_dir", type=str, default="./rendering_random_1views")
parser.add_argument(
    "--engine", type=str, default="CYCLES", choices=["CYCLES", "BLENDER_EEVEE"]
)
parser.add_argument("--resolution", type=int, default=256)
parser.add_argument(
    "--use_material", type=str, default="false", help="Whether use material."
)
parser.add_argument(
    "--transpose",
    type=str,
    default="true",
    help="Whether to transpose mesh.",
)

if "--" in sys.argv:
    argv = sys.argv[sys.argv.index("--") + 1 :]
else:
    argv = sys.argv[1:]

args = parser.parse_args(argv)


context = bpy.context
scene = context.scene
render = scene.render

render.engine = args.engine
render.image_settings.file_format = "PNG"  # ('PNG', 'OPEN_EXR', 'JPEG, ...)
render.image_settings.color_mode = "RGBA"  # ('RGB', 'RGBA', ...)
render.image_settings.color_depth = "8"  # ('8', '16')
render.resolution_x = args.resolution
render.resolution_y = args.resolution
render.resolution_percentage = 100

scene.cycles.device = "GPU"
scene.cycles.samples = 128
scene.cycles.diffuse_bounces = 1
scene.cycles.glossy_bounces = 1
scene.cycles.transparent_max_bounces = 3
scene.cycles.transmission_bounces = 3
scene.cycles.filter_width = 0.01
scene.cycles.use_denoising = True
scene.render.film_transparent = False

# Set the device_type
cycles_preferences = bpy.context.preferences.addons["cycles"].preferences
cycles_preferences.compute_device_type = "CUDA"
cuda_devices = cycles_preferences.get_devices_for_type("CUDA")
for device in cuda_devices:
    device.use = True


def setup_compositor_nodes():
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    links = tree.links

    for n in tree.nodes:
        tree.nodes.remove(n)

    rl = tree.nodes.new(type="CompositorNodeRLayers")
    composite = tree.nodes.new(type="CompositorNodeComposite")
    links.new(rl.outputs["Image"], composite.inputs["Image"])
    image_save = tree.nodes.new(type="CompositorNodeOutputFile")
    links.new(rl.outputs["Image"], image_save.inputs["Image"])

    return image_save


def compose_RT(R, T):
    return np.hstack((R, T.reshape(-1, 1)))


def sample_spherical(radius_min=1.5, radius_max=2.0, maxz=1.6, minz=-0.75):
    correct = False
    while not correct:
        vec = np.random.uniform(-1, 1, 3)
        radius = np.random.uniform(radius_min, radius_max, 1)
        vec = vec / np.linalg.norm(vec, axis=0) * radius[0]
        if maxz > vec[2] > minz:
            correct = True
    return vec


def set_camera_location(camera, option: str):
    assert option in ["fixed", "random", "front"]

    if option == "fixed":
        x, y, z = 0, -2.25, 0
    elif option == "random":
        x, y, z = sample_spherical(radius_min=2.0, radius_max=2.5, maxz=0.75, minz=0.25)
    elif option == "front":
        x, y, z = 0, -np.random.uniform(2.0, 2.5, 1)[0], 0

    camera.location = x, y, z
    direction = -camera.location
    rot_quat = direction.to_track_quat("-Z", "Y")
    camera.rotation_euler = rot_quat.to_euler()
    camera.data.lens = 40
    return camera


def add_lighting(option: str) -> None:
    assert option in ["fixed", "random"]

    if "Light" in bpy.data.objects:
        bpy.data.objects["Light"].select_set(True)
        bpy.ops.object.delete()

    bpy.ops.object.light_add(type="AREA")
    light = bpy.data.lights["Area"]
    light_object = bpy.data.objects["Area"]

    if option == "fixed":
        light.energy = 30000
        light_object.location = (0, 1, 0.5)

    elif option == "random":
        light.energy = random.uniform(80000, 120000)
        light_object.location = (
            random.uniform(-2.0, 2.0),
            random.uniform(-2.0, 2.0),
            random.uniform(1.0, 3.0),
        )

    light_object.scale = (200, 200, 200)


def reset_scene() -> None:
    for obj in bpy.data.objects:
        if obj.type not in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(obj, do_unlink=True)
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)
    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)


def load_object(object_path: str) -> None:
    if object_path.endswith(".glb"):
        bpy.ops.import_scene.gltf(filepath=object_path)
    elif object_path.endswith(".fbx"):
        bpy.ops.import_scene.fbx(filepath=object_path)
    elif object_path.endswith(".obj"):
        bpy.ops.wm.obj_import(filepath=object_path)
    else:
        raise ValueError(f"Unsupported file type: {object_path}")


def rotate_mesh(mesh_name: str, degrees: float, axis: tuple) -> None:
    # 确保场景中有指定的mesh
    obj = bpy.data.objects.get(mesh_name)
    if obj is None:
        raise ValueError(f"Object with name '{mesh_name}' not found in the scene.")

    # 将对象设置为活动对象
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    # 开始编辑模式
    bpy.ops.object.mode_set(mode="EDIT")

    # 获取mesh的编辑网格数据
    mesh = obj.data
    bm = bmesh.from_edit_mesh(mesh)

    # 使用numpy创建旋转矩阵
    theta = np.radians(degrees)
    axis = np.array(axis)
    axis = axis / np.linalg.norm(axis)
    a = np.cos(theta / 2)
    b, c, d = -axis * np.sin(theta / 2)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    rotation_matrix = np.array(
        [
            [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
            [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
            [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc],
        ]
    )

    # 对网格中的每个顶点应用旋转矩阵
    for v in bm.verts:
        v.co = np.dot(rotation_matrix, np.array(v.co))

    # 确保修改被应用
    bmesh.update_edit_mesh(mesh)

    # 退出编辑模式
    bpy.ops.object.mode_set(mode="OBJECT")


def scene_bbox(single_obj=None, ignore_matrix=False):
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    for obj in scene_meshes() if single_obj is None else [single_obj]:
        found = True
        for coord in obj.bound_box:
            coord = Vector(coord)
            if not ignore_matrix:
                coord = obj.matrix_world @ coord
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
    if not found:
        raise RuntimeError("no objects in scene to compute bounding box for")
    return Vector(bbox_min), Vector(bbox_max)


def scene_root_objects():
    for obj in bpy.context.scene.objects.values():
        if not obj.parent:
            yield obj


def scene_meshes():
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, (bpy.types.Mesh)):
            yield obj


def normalize_scene(box_scale: float):
    bbox_min, bbox_max = scene_bbox()
    scale = box_scale / max(bbox_max - bbox_min)
    for obj in scene_root_objects():
        obj.scale = obj.scale * scale
    bpy.context.view_layer.update()
    bbox_min, bbox_max = scene_bbox()
    offset = -(bbox_min + bbox_max) / 2
    for obj in scene_root_objects():
        obj.matrix_world.translation += offset
    bpy.ops.object.select_all(action="DESELECT")


def setup_camera():
    cam = scene.objects["Camera"]
    cam.location = (0, 1.2, 0)
    cam.data.lens = 24
    cam.data.sensor_width = 32
    cam.data.sensor_height = 32
    cam_constraint = cam.constraints.new(type="TRACK_TO")
    cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
    cam_constraint.up_axis = "UP_Y"
    return cam, cam_constraint

def align_model_to_camera(model, camera):
    # 计算相机到模型的方向向量
    direction = model.location - camera.location
    direction = direction.normalized()

    # 计算旋转四元数，使模型的正面（假设为'-Z'轴）朝向相机
    # 注意：'-Z'表示模型的局部Z轴指向相反方向，即模型的正面
    rot_quat = direction.to_track_quat('-Z', 'Y')

    # 应用旋转到模型
    model.rotation_mode = 'QUATERNION'
    model.rotation_quaternion = rot_quat


def save_images(object_file: str) -> None:
    reset_scene()
    load_object(object_file)
    
    object_uid, ext = splitext(basename(object_file))
    
    normalize_scene(box_scale=1)
    add_lighting(option="fixed")

    camera, cam_constraint = setup_camera()
    
    base_object = bpy.data.objects[object_uid]
    if args.transpose == "true":
        rotate_mesh(object_uid, -90, (1, 0, 0))
    align_model_to_camera(base_object, camera)
    # base_object.rotation_euler[2] = camera.rotation_euler[2]
    rotate_mesh(object_uid, 45, (0, 1, 0))
    
    empty = bpy.data.objects.new("Empty", None)
    scene.collection.objects.link(empty)
    cam_constraint.target = empty
    
    if args.use_material == "true":
        add_building_material(base_object)
    
    # img_dir = join(args.output_dir, object_uid)
    img_dir = args.output_dir
    os.makedirs(img_dir, exist_ok=True)

    # save the handled object
    # bpy.ops.wm.obj_export(
    #     filepath=join(img_dir, "model.obj"),
    #     export_materials=True if args.use_material else False,
    # )

    # Prepare to save camera parameters
    # cam_params = {
    #     "intrinsics": get_calibration_matrix_K_from_blender(
    #         camera.data, return_principles=True
    #     ),
    #     "poses": [],
    # }

    image_save = setup_compositor_nodes()

    image_save.base_path = img_dir
    image_save.file_slots[0].path = object_uid

    # Set the camera position
    camera = set_camera_location(camera, option="fixed")

    # Save camera RT matrix (C2W)
    # location, rotation = camera.matrix_world.decompose()[0:2]
    # RT = compose_RT(rotation.to_matrix(), np.array(location))
    # cam_params["poses"].append(RT)

    bpy.ops.render.render(write_still=True)

    file_save_path = join(img_dir, f"{object_uid}0001.png")
    # 检查是否为文件
    if isfile(file_save_path):
        
        name, extension = splitext(basename(file_save_path))

        # 获取新文件名，去掉文件名部分的最后三个字符
        new_name = name.rsplit("0001", 1)[0]

        # 构建完整的新文件名
        new_filename = new_name + extension

        # 构建完整的旧文件路径和新文件路径
        old_filepath = join(img_dir, name + extension)
        new_filepath = join(img_dir, new_filename)

        # 重命名文件
        shutil.move(old_filepath, new_filepath) 

    # Save camera intrinsics and poses
    # np.savez(join(img_dir, "camera.npz"), **cam_params)


def get_calibration_matrix_K_from_blender(camera, return_principles=False):
    render = bpy.context.scene.render
    width = render.resolution_x * render.pixel_aspect_x
    height = render.resolution_y * render.pixel_aspect_y
    focal_length = camera.lens
    sensor_width = camera.sensor_width
    sensor_height = camera.sensor_height
    focal_length_x = width * (focal_length / sensor_width)
    focal_length_y = height * (focal_length / sensor_height)
    optical_center_x = width / 2
    optical_center_y = height / 2
    K = np.array(
        [
            [focal_length_x, 0, optical_center_x],
            [0, focal_length_y, optical_center_y],
            [0, 0, 1],
        ]
    )
    if return_principles:
        return np.array(
            [
                [focal_length_x, focal_length_y],
                [optical_center_x, optical_center_y],
                [width, height],
            ]
        )
    else:
        return K


def add_building_material(base_object):
    # Create a new material
    material = bpy.data.materials.new(name="Building Material")
    material.use_nodes = True
    material.node_tree.nodes.clear()

    nodes = material.node_tree.nodes
    links = material.node_tree.links

    # Create a principled BSDF node
    principled_bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")

    # Set some default values for the principled BSDF
    principled_bsdf.inputs["Roughness"].default_value = 0.5
    principled_bsdf.inputs["Metallic"].default_value = 0.0
    # principled_bsdf.inputs["Specular"].default_value = 0.5

    # Create a noise texture node for detail
    noise_tex = nodes.new(type="ShaderNodeTexNoise")
    noise_tex.inputs["Scale"].default_value = 10.0  # Adjust the scale for the noise
    noise_tex.inputs["Detail"].default_value = 2.0  # Adjust the detail level
    noise_tex.inputs["Roughness"].default_value = (
        1.0  # Adjust the roughness of the noise
    )

    # Create a color ramp to modify the noise texture
    color_ramp = nodes.new(type="ShaderNodeValToRGB")
    color_ramp.color_ramp.interpolation = "LINEAR"
    color_ramp.color_ramp.elements.new(0.0).color = (0.8, 0.8, 0.8, 1.0)  # Light color
    color_ramp.color_ramp.elements.new(1.0).color = (0.2, 0.2, 0.2, 1.0)  # Dark color

    # Link the noise texture to the color ramp
    links.new(noise_tex.outputs["Color"], color_ramp.inputs["Fac"])

    # Create a mix RGB node to blend the color ramp with a base color
    mix_rgb = nodes.new(type="ShaderNodeMixRGB")
    mix_rgb.inputs["Fac"].default_value = 0.5  # Adjust the blend factor
    mix_rgb.inputs["Color1"].default_value = (
        random.uniform(0.5, 0.8),
        random.uniform(0.3, 0.6),
        random.uniform(0.2, 0.5),
        1.0,
    )  # Random base color

    # Link the color ramp to the mix RGB
    links.new(color_ramp.outputs["Color"], mix_rgb.inputs["Color2"])

    # Link the mix RGB to the principled BSDF base color
    links.new(mix_rgb.outputs["Color"], principled_bsdf.inputs["Base Color"])

    # Create a material output node
    material_output = nodes.new(type="ShaderNodeOutputMaterial")

    # Link the principled BSDF to the material output
    links.new(principled_bsdf.outputs["BSDF"], material_output.inputs["Surface"])

    # Assign the material to the base object
    base_object.data.materials.append(material)


def main(args):
    try:
        start_i = time.time()
        local_path = args.object_path
        save_images(local_path)
        end_i = time.time()
        print("Finished", local_path, "in", end_i - start_i, "seconds")
    except Exception as e:
        print("Failed to render", args.object_path)
        print(e)


if __name__ == "__main__":
    main(args)
