#!/usr/bin/env python3
"""Deterministic reference renderer: camera orbits clockwise around a stationary
stone statue exactly two complete times.

Outputs an H.264 (yuv420p) MP4 built from explicit mathematical camera
trajectories and chronological keyframes. No external assets/network/Blender.
"""
import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import imageio_ffmpeg


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--frames", type=int, default=49)
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--width", type=int, default=720)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--orbits", type=float, default=2.0,
                    help="Number of complete clockwise camera revolutions.")
    p.add_argument("--out", type=str, default="statue_orbit_original.mp4")
    return p.parse_args()


# ----------------------------------------------------------------------------
# Scene definition: a stationary stone statue standing at the world origin on
# a ground plane. The statue is built from simple primitives (base pedestal,
# torso, head, one raised arm) so that different facing sides are visually
# distinguishable:
#   - Front: face + two "eyes" visible, arm raised on viewer's right.
#   - Back: flat featureless slab (no face), a vertical seam line.
#   - Left/Right profiles: visible arm silhouette differs (arm only on one side).
# The camera orbits around this stationary subject; the subject itself never
# rotates.
# ----------------------------------------------------------------------------

STATUE_HEIGHT = 2.6      # world units, base to top of head
STATUE_RADIUS = 0.55     # torso radius (cylinder-like)
HEAD_RADIUS = 0.32
BASE_HEIGHT = 0.35
BASE_RADIUS = 0.9
ARM_LENGTH = 0.9
ARM_AZIMUTH_DEG = 90.0   # arm sticks out toward world +X (statue's own "right")

CAMERA_RADIUS = 5.0
CAMERA_HEIGHT = 1.6
FOV_DEG = 50.0


def project(point_world, cam_pos, cam_target, cam_up, fov_deg, width, height):
    """Simple pinhole projection of a 3D world point to 2D pixel coordinates.

    Returns (x_px, y_px, depth) where depth > 0 means in front of camera.
    """
    forward = cam_target - cam_pos
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, cam_up)
    right = right / np.linalg.norm(right)
    true_up = np.cross(right, forward)

    rel = point_world - cam_pos
    x_cam = np.dot(rel, right)
    y_cam = np.dot(rel, true_up)
    z_cam = np.dot(rel, forward)  # depth along view direction

    if z_cam <= 1e-6:
        return None

    f = 1.0 / math.tan(math.radians(fov_deg) / 2.0)
    aspect = width / height
    x_ndc = (x_cam * f / aspect) / z_cam
    y_ndc = (y_cam * f) / z_cam

    x_px = (x_ndc * 0.5 + 0.5) * width
    y_px = (1.0 - (y_ndc * 0.5 + 0.5)) * height
    return x_px, y_px, z_cam


def camera_state(t_frac, orbits, radius, height):
    """Return camera position for normalized time t_frac in [0, 1].

    Clockwise orbit as viewed from above (looking down +Y... here Y is up,
    so "from above" means looking down the -Y axis): angle decreases over
    time, which corresponds to clockwise motion when viewed from above with
    a standard right-handed (X-right, Z-toward-viewer) coordinate frame.
    """
    total_angle = 2.0 * math.pi * orbits
    angle = -total_angle * t_frac  # negative => clockwise from top-down view
    cam_x = radius * math.sin(angle)
    cam_z = radius * math.cos(angle)
    cam_y = height
    return np.array([cam_x, cam_y, cam_z], dtype=np.float64), angle


def draw_ground_grid(draw, cam_pos, cam_target, cam_up, width, height):
    grid_extent = 8
    step = 1.0
    color = (90, 90, 95)
    i = -grid_extent
    while i <= grid_extent:
        p1 = np.array([i, 0.0, -grid_extent])
        p2 = np.array([i, 0.0, grid_extent])
        p3 = np.array([-grid_extent, 0.0, i])
        p4 = np.array([grid_extent, 0.0, i])
        for a, b in ((p1, p2), (p3, p4)):
            pa = project(a, cam_pos, cam_target, cam_up, FOV_DEG, width, height)
            pb = project(b, cam_pos, cam_target, cam_up, FOV_DEG, width, height)
            if pa and pb:
                draw.line([(pa[0], pa[1]), (pb[0], pb[1])], fill=color, width=1)
        i += step


def statue_polygons():
    """Return a list of (name, list_of_world_points, base_color) describing
    the statue as a set of vertical quad faces around a polygon cross-section,
    plus a head and an arm, so that faces are distinguishable by azimuth.
    """
    n_sides = 8
    faces = []

    def ring(radius, y):
        return [
            np.array([radius * math.cos(2 * math.pi * k / n_sides),
                       y,
                       radius * math.sin(2 * math.pi * k / n_sides)])
            for k in range(n_sides)
        ]

    base_bottom = ring(BASE_RADIUS, 0.0)
    base_top = ring(BASE_RADIUS, BASE_HEIGHT)
    for k in range(n_sides):
        k2 = (k + 1) % n_sides
        quad = [base_bottom[k], base_bottom[k2], base_top[k2], base_top[k]]
        faces.append(("base", quad, (120, 112, 100)))

    torso_bottom_y = BASE_HEIGHT
    torso_top_y = STATUE_HEIGHT - 2 * HEAD_RADIUS
    torso_bottom = ring(STATUE_RADIUS, torso_bottom_y)
    torso_top = ring(STATUE_RADIUS, torso_top_y)
    for k in range(n_sides):
        k2 = (k + 1) % n_sides
        mid_angle = (2 * math.pi * k / n_sides + 2 * math.pi * k2 / n_sides) / 2.0
        deg = math.degrees(mid_angle) % 360.0
        if deg < 45 or deg >= 315:
            shade = (196, 188, 172)  # front-ish, brightest: has the "face" side
        elif 135 <= deg < 225:
            shade = (110, 104, 96)   # back, darkest, featureless
        else:
            shade = (150, 143, 130)  # side profiles
        quad = [torso_bottom[k], torso_bottom[k2], torso_top[k2], torso_top[k]]
        faces.append(("torso", quad, shade))

    head_bottom = ring(HEAD_RADIUS, torso_top_y)
    head_top = ring(HEAD_RADIUS * 0.55, STATUE_HEIGHT)
    for k in range(n_sides):
        k2 = (k + 1) % n_sides
        mid_angle = (2 * math.pi * k / n_sides + 2 * math.pi * k2 / n_sides) / 2.0
        deg = math.degrees(mid_angle) % 360.0
        if deg < 45 or deg >= 315:
            shade = (210, 200, 182)
        elif 135 <= deg < 225:
            shade = (120, 114, 104)
        else:
            shade = (165, 157, 142)
        quad = [head_bottom[k], head_bottom[k2], head_top[k2], head_top[k]]
        faces.append(("head", quad, shade))

    return faces


def statue_face_features():
    """Small distinguishing markers: two eyes on the front (angle 0), a
    vertical seam line on the back (angle 180), placed as short 3D segments.
    """
    features = []
    face_y = STATUE_HEIGHT - HEAD_RADIUS * 1.3
    eye_offset = HEAD_RADIUS * 0.35
    eye_z = HEAD_RADIUS * 0.95
    left_eye = np.array([-eye_offset, face_y, eye_z])
    right_eye = np.array([eye_offset, face_y, eye_z])
    features.append(("eye", left_eye, (30, 30, 30)))
    features.append(("eye", right_eye, (30, 30, 30)))

    seam_top = np.array([0.0, STATUE_HEIGHT - HEAD_RADIUS * 2, -STATUE_RADIUS * 1.0])
    seam_bottom = np.array([0.0, BASE_HEIGHT, -STATUE_RADIUS * 1.0])
    features.append(("seam", (seam_top, seam_bottom), (60, 56, 52)))

    return features


def arm_segment():
    arm_angle = math.radians(ARM_AZIMUTH_DEG)
    shoulder_y = STATUE_HEIGHT - 2 * HEAD_RADIUS - 0.15
    shoulder = np.array([STATUE_RADIUS * math.cos(arm_angle), shoulder_y,
                          STATUE_RADIUS * math.sin(arm_angle)])
    hand = shoulder + np.array([
        ARM_LENGTH * math.cos(arm_angle),
        0.25,
        ARM_LENGTH * math.sin(arm_angle),
    ])
    return shoulder, hand


def face_visibility_score(face_points, cam_pos):
    center = np.mean(face_points, axis=0)
    v1 = face_points[1] - face_points[0]
    v2 = face_points[3] - face_points[0]
    normal = np.cross(v1, v2)
    n = np.linalg.norm(normal)
    if n < 1e-9:
        return -1.0, center
    normal = normal / n
    to_cam = cam_pos - center
    to_cam = to_cam / (np.linalg.norm(to_cam) + 1e-9)
    return float(np.dot(normal, to_cam)), center


def render_frame(width, height, cam_pos, cam_target, cam_up, faces, features, arm_pts):
    img = Image.new("RGB", (width, height), (18, 20, 26))
    draw = ImageDraw.Draw(img)

    draw_ground_grid(draw, cam_pos, cam_target, cam_up, width, height)

    visible = []
    for name, pts, color in faces:
        score, center = face_visibility_score(pts, cam_pos)
        if score <= 0.02:
            continue
        depth = np.linalg.norm(center - cam_pos)
        visible.append((depth, name, pts, color))

    visible.sort(key=lambda x: -x[0])  # painter's algorithm: far first

    for depth, name, pts, color in visible:
        projected = [project(p, cam_pos, cam_target, cam_up, FOV_DEG, width, height)
                     for p in pts]
        if any(p is None for p in projected):
            continue
        poly = [(p[0], p[1]) for p in projected]
        draw.polygon(poly, fill=color, outline=(40, 38, 34))

    shoulder, hand = arm_pts
    p_shoulder = project(shoulder, cam_pos, cam_target, cam_up, FOV_DEG, width, height)
    p_hand = project(hand, cam_pos, cam_target, cam_up, FOV_DEG, width, height)
    if p_shoulder and p_hand:
        draw.line([(p_shoulder[0], p_shoulder[1]), (p_hand[0], p_hand[1])],
                   fill=(130, 122, 108), width=8)
        draw.ellipse([p_hand[0] - 6, p_hand[1] - 6, p_hand[0] + 6, p_hand[1] + 6],
                     fill=(130, 122, 108))

    for name, data, color in features:
        if name == "eye":
            p = project(data, cam_pos, cam_target, cam_up, FOV_DEG, width, height)
            if p and p[2] > 0:
                r = 5
                draw.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=color)
        elif name == "seam":
            top, bottom = data
            pt = project(top, cam_pos, cam_target, cam_up, FOV_DEG, width, height)
            pb = project(bottom, cam_pos, cam_target, cam_up, FOV_DEG, width, height)
            if pt and pb and pt[2] > 0 and pb[2] > 0:
                draw.line([(pt[0], pt[1]), (pb[0], pb[1])], fill=color, width=3)

    return img


def add_hud(img, frame_idx, total_frames, angle_deg):
    draw = ImageDraw.Draw(img)
    text = f"frame {frame_idx + 1}/{total_frames}  cam_angle={angle_deg:6.1f} deg"
    draw.rectangle([0, 0, img.width, 22], fill=(0, 0, 0))
    draw.text((6, 5), text, fill=(255, 255, 255))
    return img


def main():
    args = parse_args()
    width, height, n_frames, fps = args.width, args.height, args.frames, args.fps

    faces = statue_polygons()
    features = statue_face_features()
    arm_pts = arm_segment()

    cam_up = np.array([0.0, 1.0, 0.0])
    cam_target = np.array([0.0, STATUE_HEIGHT * 0.4, 0.0])

    out_path = Path(args.out)

    writer = imageio_ffmpeg.write_frames(
        str(out_path),
        (width, height),
        fps=fps,
        codec="libx264",
        pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p",
        macro_block_size=None,
    )
    writer.send(None)

    try:
        for i in range(n_frames):
            t_frac = i / (n_frames - 1) if n_frames > 1 else 0.0
            cam_pos, angle = camera_state(t_frac, args.orbits, CAMERA_RADIUS, CAMERA_HEIGHT)
            img = render_frame(width, height, cam_pos, cam_target, cam_up, faces, features, arm_pts)
            add_hud(img, i, n_frames, math.degrees(angle) % 360.0)
            frame = np.asarray(img, dtype=np.uint8)
            frame = np.ascontiguousarray(frame)
            writer.send(frame)
    finally:
        writer.close()

    print(f"Wrote {n_frames} frames at {fps} fps, {width}x{height} -> {out_path}")
    print(f"Camera orbits: {args.orbits} (clockwise, viewed from above)")

    # Sanity check: first and last camera states should match exactly when
    # orbits is an integer (full revolutions return to the start pose).
    cam0, ang0 = camera_state(0.0, args.orbits, CAMERA_RADIUS, CAMERA_HEIGHT)
    cam1, ang1 = camera_state(1.0, args.orbits, CAMERA_RADIUS, CAMERA_HEIGHT)
    print(f"start cam pos: {cam0}, end cam pos: {cam1}")
    print(f"start angle deg: {math.degrees(ang0)%360:.4f}, "
          f"end angle deg: {math.degrees(ang1)%360:.4f}")


if __name__ == "__main__":
    main()
