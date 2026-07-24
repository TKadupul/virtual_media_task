import argparse
import json
import math
from pathlib import Path
import sys

import bpy
from mathutils import Vector


SUPPORTED_PRIMITIVES = {
    "cube",
    "sphere",
    "cylinder",
    "cone",
    "torus",
    "plane",
    "text",
}
MAX_OBJECTS = 100
MAX_LIGHTS = 12


def parse_arguments():
    arguments = sys.argv
    arguments = arguments[arguments.index("--") + 1 :] if "--" in arguments else []
    parser = argparse.ArgumentParser(
        description="Interpret a safe JSON scene plan and render it in Blender."
    )
    parser.add_argument("--spec", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args(arguments)


def vector3(value, field_name):
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{field_name} must be a list of three numbers.")
    try:
        return tuple(float(number) for number in value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field_name} must contain only numbers.") from error


def color4(value, field_name):
    if not isinstance(value, list) or len(value) not in {3, 4}:
        raise ValueError(f"{field_name} must contain three or four numbers.")
    values = [max(0.0, min(1.0, float(number))) for number in value]
    if len(values) == 3:
        values.append(1.0)
    return tuple(values)


def load_spec(path):
    with open(path, "r", encoding="utf-8") as handle:
        spec = json.load(handle)

    required = {"render", "materials", "objects", "camera", "lights"}
    missing = sorted(required.difference(spec))
    if missing:
        raise ValueError(f"Scene JSON is missing: {', '.join(missing)}")
    if not isinstance(spec["materials"], list):
        raise ValueError("materials must be a list.")
    if not isinstance(spec["objects"], list):
        raise ValueError("objects must be a list.")
    if not isinstance(spec["lights"], list):
        raise ValueError("lights must be a list.")
    if len(spec["objects"]) > MAX_OBJECTS:
        raise ValueError(f"A scene may contain at most {MAX_OBJECTS} objects.")
    if len(spec["lights"]) > MAX_LIGHTS:
        raise ValueError(f"A scene may contain at most {MAX_LIGHTS} lights.")

    object_ids = set()
    for item in spec["objects"]:
        if not isinstance(item, dict):
            raise ValueError("Every object must be a JSON object.")
        object_id = str(item.get("id", ""))
        primitive = item.get("primitive")
        if not object_id or object_id in object_ids:
            raise ValueError("Every object needs a unique non-empty id.")
        if primitive not in SUPPORTED_PRIMITIVES:
            raise ValueError(f"Unsupported primitive {primitive!r} for {object_id}.")
        object_ids.add(object_id)

    camera_mode = spec["camera"].get("mode")
    if camera_mode not in {"orbit", "static", "keyframes"}:
        raise ValueError("camera.mode must be orbit, static, or keyframes.")
    return spec


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for datablocks in (
        bpy.data.meshes,
        bpy.data.curves,
        bpy.data.materials,
        bpy.data.cameras,
        bpy.data.lights,
    ):
        for datablock in list(datablocks):
            if datablock.users == 0:
                datablocks.remove(datablock)


def create_materials(material_specs):
    materials = {}
    for item in material_specs:
        material_id = str(item.get("id", "")).strip()
        if not material_id or material_id in materials:
            raise ValueError("Every material needs a unique non-empty id.")

        material = bpy.data.materials.new(name=material_id)
        material.use_nodes = True
        base_color = color4(item.get("color", [0.5, 0.5, 0.5, 1]), "color")
        material.diffuse_color = base_color
        principled = material.node_tree.nodes.get("Principled BSDF")
        principled.inputs["Base Color"].default_value = base_color
        principled.inputs["Metallic"].default_value = max(
            0.0, min(1.0, float(item.get("metallic", 0.0)))
        )
        principled.inputs["Roughness"].default_value = max(
            0.0, min(1.0, float(item.get("roughness", 0.6)))
        )
        if "emission_color" in item:
            emission = color4(item["emission_color"], "emission_color")
            if "Emission Color" in principled.inputs:
                principled.inputs["Emission Color"].default_value = emission
                principled.inputs["Emission Strength"].default_value = float(
                    item.get("emission_strength", 1.0)
                )
        materials[material_id] = material
    return materials


def add_primitive(item):
    primitive = item["primitive"]
    location = vector3(item.get("location", [0, 0, 0]), "location")

    if primitive == "cube":
        bpy.ops.mesh.primitive_cube_add(size=2.0, location=location)
    elif primitive == "sphere":
        bpy.ops.mesh.primitive_uv_sphere_add(
            segments=48,
            ring_count=24,
            radius=1.0,
            location=location,
        )
    elif primitive == "cylinder":
        bpy.ops.mesh.primitive_cylinder_add(
            vertices=48,
            radius=1.0,
            depth=2.0,
            location=location,
        )
    elif primitive == "cone":
        bpy.ops.mesh.primitive_cone_add(
            vertices=48,
            radius1=1.0,
            radius2=float(item.get("top_radius", 0.0)),
            depth=2.0,
            location=location,
        )
    elif primitive == "torus":
        bpy.ops.mesh.primitive_torus_add(
            major_radius=1.0,
            minor_radius=float(item.get("minor_radius", 0.25)),
            major_segments=64,
            minor_segments=20,
            location=location,
        )
    elif primitive == "plane":
        bpy.ops.mesh.primitive_plane_add(size=2.0, location=location)
    elif primitive == "text":
        bpy.ops.object.text_add(location=location)
        bpy.context.object.data.body = str(item.get("text", "Text"))[:100]
        bpy.context.object.data.align_x = "CENTER"
        bpy.context.object.data.align_y = "CENTER"
        bpy.context.object.data.extrude = float(item.get("extrude", 0.03))
        bpy.context.object.data.bevel_depth = float(item.get("text_bevel", 0.01))

    obj = bpy.context.object
    obj.name = str(item["id"])
    obj.rotation_mode = "XYZ"
    obj.rotation_euler = tuple(
        math.radians(value)
        for value in vector3(item.get("rotation_deg", [0, 0, 0]), "rotation_deg")
    )
    obj.scale = vector3(item.get("scale", [1, 1, 1]), "scale")
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    if bool(item.get("smooth", primitive in {"sphere", "cylinder", "cone", "torus"})):
        if hasattr(obj.data, "polygons"):
            for polygon in obj.data.polygons:
                polygon.use_smooth = True

    bevel = max(0.0, min(0.5, float(item.get("bevel", 0.0))))
    if bevel and primitive not in {"plane", "text"}:
        modifier = obj.modifiers.new(name="Rounded edges", type="BEVEL")
        modifier.width = bevel
        modifier.segments = 3
    return obj


def apply_object_animation(obj, item, frame_end):
    keyframes = item.get("keyframes", [])
    if not isinstance(keyframes, list):
        raise ValueError(f"keyframes for {obj.name} must be a list.")
    for keyframe in keyframes:
        frame = max(1, min(frame_end, int(keyframe["frame"])))
        if "location" in keyframe:
            obj.location = vector3(keyframe["location"], "keyframe.location")
            obj.keyframe_insert(data_path="location", frame=frame)
        if "rotation_deg" in keyframe:
            obj.rotation_euler = tuple(
                math.radians(value)
                for value in vector3(
                    keyframe["rotation_deg"],
                    "keyframe.rotation_deg",
                )
            )
            obj.keyframe_insert(data_path="rotation_euler", frame=frame)
        if "scale" in keyframe:
            obj.scale = vector3(keyframe["scale"], "keyframe.scale")
            obj.keyframe_insert(data_path="scale", frame=frame)


def build_objects(spec, materials):
    frame_end = int(spec["render"]["num_frames"])
    objects = {}
    for item in spec["objects"]:
        obj = add_primitive(item)
        material_id = item.get("material")
        if material_id:
            if material_id not in materials:
                raise ValueError(
                    f"Object {item['id']} refers to unknown material {material_id}."
                )
            if hasattr(obj.data, "materials"):
                obj.data.materials.append(materials[material_id])
        objects[item["id"]] = obj

    # Parent composite-object parts after all objects exist. Preserve each
    # part's world transform while allowing a torso/body animation to carry
    # its attached head, limbs, equipment, or other sub-parts.
    for item in spec["objects"]:
        parent_id = item.get("parent")
        if not parent_id:
            continue
        if parent_id not in objects:
            raise ValueError(
                f"Object {item['id']} has unknown parent {parent_id}."
            )
        child = objects[item["id"]]
        parent = objects[parent_id]
        child.parent = parent
        child.matrix_parent_inverse = parent.matrix_world.inverted()

    for item in spec["objects"]:
        apply_object_animation(objects[item["id"]], item, frame_end)
    return objects


def build_lights(light_specs):
    for index, item in enumerate(light_specs):
        light_type = str(item.get("type", "AREA")).upper()
        if light_type not in {"AREA", "SUN", "POINT", "SPOT"}:
            raise ValueError(f"Unsupported light type: {light_type}")
        data = bpy.data.lights.new(
            name=str(item.get("id", f"light_{index}")),
            type=light_type,
        )
        data.energy = max(0.0, float(item.get("energy", 1000.0)))
        data.color = color4(item.get("color", [1, 1, 1]), "light.color")[:3]
        if light_type == "AREA":
            data.shape = "DISK"
            data.size = max(0.1, float(item.get("size", 5.0)))
        if light_type == "SPOT":
            data.spot_size = math.radians(float(item.get("spot_size_deg", 45.0)))

        obj = bpy.data.objects.new(data.name, data)
        bpy.context.collection.objects.link(obj)
        obj.location = vector3(item.get("location", [4, -4, 8]), "light.location")
        obj.rotation_euler = tuple(
            math.radians(value)
            for value in vector3(
                item.get("rotation_deg", [25, 0, 35]),
                "light.rotation_deg",
            )
        )


def add_tracking_constraint(camera, target_location):
    bpy.ops.object.empty_add(
        type="PLAIN_AXES",
        location=vector3(target_location, "camera.target"),
    )
    target = bpy.context.object
    target.name = "Camera tracking target"
    constraint = camera.constraints.new(type="TRACK_TO")
    constraint.target = target
    constraint.track_axis = "TRACK_NEGATIVE_Z"
    constraint.up_axis = "UP_Y"


def build_camera(camera_spec, frame_end):
    bpy.ops.object.camera_add()
    camera = bpy.context.object
    camera.name = "Render camera"
    camera.data.lens = float(camera_spec.get("lens_mm", 48.0))
    camera.data.sensor_width = 36.0
    camera.rotation_mode = "XYZ"
    target = camera_spec.get("target", [0, 0, 1.5])
    mode = camera_spec["mode"]

    if mode == "orbit":
        bpy.ops.object.empty_add(type="PLAIN_AXES", location=(0, 0, 0))
        pivot = bpy.context.object
        pivot.name = "Camera orbit pivot"
        camera.parent = pivot

        radius = float(camera_spec.get("radius", 7.0))
        height = float(camera_spec.get("height", 3.0))
        start_angle = math.radians(float(camera_spec.get("start_angle_deg", -90.0)))
        camera.location = (
            radius * math.cos(start_angle),
            radius * math.sin(start_angle),
            height,
        )
        add_tracking_constraint(camera, target)

        direction = camera_spec.get("direction", "clockwise")
        sign = -1.0 if direction == "clockwise" else 1.0
        rotation = sign * math.radians(
            float(camera_spec.get("rotation_degrees", 360.0))
        )
        pivot.rotation_mode = "XYZ"
        pivot.rotation_euler = (0, 0, 0)
        pivot.keyframe_insert(data_path="rotation_euler", frame=1)
        pivot.rotation_euler = (0, 0, rotation)
        pivot.keyframe_insert(data_path="rotation_euler", frame=frame_end)

    else:
        camera.location = vector3(
            camera_spec.get("location", [0, -7, 3]),
            "camera.location",
        )
        add_tracking_constraint(camera, target)
        if mode == "keyframes":
            keyframes = camera_spec.get("keyframes", [])
            if not keyframes:
                raise ValueError("A keyframed camera needs camera.keyframes.")
            for keyframe in keyframes:
                frame = max(1, min(frame_end, int(keyframe["frame"])))
                camera.location = vector3(
                    keyframe["location"],
                    "camera.keyframe.location",
                )
                camera.keyframe_insert(data_path="location", frame=frame)

    bpy.context.scene.camera = camera


def configure_world(spec):
    world_spec = spec.get("world", {})
    world = bpy.context.scene.world
    world.use_nodes = True
    background = world.node_tree.nodes.get("Background")
    background.inputs["Color"].default_value = color4(
        world_spec.get("background_color", [0.05, 0.07, 0.10, 1]),
        "world.background_color",
    )
    background.inputs["Strength"].default_value = float(
        world_spec.get("strength", 0.35)
    )


def configure_render(render_spec, output_path):
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = int(render_spec["num_frames"])
    scene.render.fps = int(render_spec["fps"])
    scene.render.fps_base = 1.0
    scene.render.resolution_x = int(render_spec["width"])
    scene.render.resolution_y = int(render_spec["height"])
    scene.render.resolution_percentage = 100

    engine = str(render_spec.get("engine", "EEVEE")).upper()
    if engine == "CYCLES":
        scene.render.engine = "BLENDER_EEVEE_NEXT"
        try:
            scene.render.engine = "CYCLES"
            scene.cycles.samples = max(
                8, min(256, int(render_spec.get("samples", 32)))
            )
            scene.cycles.use_denoising = True
            scene.cycles.device = "GPU"
        except Exception as error:
            print(f"Cycles unavailable; using Eevee: {error}")
            scene.render.engine = "BLENDER_EEVEE_NEXT"
    else:
        scene.render.engine = "BLENDER_EEVEE_NEXT"

    scene.render.image_settings.file_format = "FFMPEG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    scene.render.ffmpeg.format = "MPEG4"
    scene.render.ffmpeg.codec = "H264"
    scene.render.ffmpeg.constant_rate_factor = "MEDIUM"
    scene.render.filepath = str(output_path)
    scene.render.use_file_extension = False
    if hasattr(scene.view_settings, "look"):
        try:
            scene.view_settings.look = "AgX - Medium High Contrast"
        except TypeError:
            pass


def main():
    args = parse_arguments()
    spec_path = Path(args.spec).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    spec = load_spec(spec_path)
    clear_scene()
    edit_preferences = bpy.context.preferences.edit
    if hasattr(edit_preferences, "keyframe_new_interpolation_type"):
        edit_preferences.keyframe_new_interpolation_type = "LINEAR"

    materials = create_materials(spec["materials"])
    build_objects(spec, materials)
    build_lights(spec["lights"])
    build_camera(spec["camera"], int(spec["render"]["num_frames"]))
    configure_world(spec)
    configure_render(spec["render"], output_path)

    print(
        "Rendering generic prompt scene:",
        output_path,
        f"({spec['render']['num_frames']} frames at {spec['render']['fps']} fps)",
    )
    bpy.ops.render.render(animation=True)
    print("Completed:", output_path)


if __name__ == "__main__":
    main()
