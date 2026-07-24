#!/usr/bin/env python3
"""Deterministic reference renderer: continuous FIRST-PERSON driving view from
inside a red sedan, looking forward through the windshield along a straight
two-lane road. A blue sedan starts ahead in the same lane. The red sedan
(the camera itself, ego-centric) steers into the adjacent lane, completely
overtakes the blue sedan, returns to its original lane, slows down, performs
one continuous 180-degree U-turn, and drives back in the opposite direction.
The blue sedan keeps driving forward the whole time, so once the red sedan
has turned around it drives back toward the blue sedan, which becomes
visible through the windshield again.

Outputs an H.264 (yuv420p) MP4 built from explicit mathematical trajectories
and chronological keyframes. No external assets/network/Blender.
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
    p.add_argument("--out", type=str, default="test3_original.mp4")
    return p.parse_args()


# ----------------------------------------------------------------------------
# World: X = along-road travel direction, Z = across-road (lane) offset,
# Y = up. The camera is mounted INSIDE the red sedan (ego vehicle) near the
# windshield and moves/yaws together with it -- this is a first-person POV,
# not an external orbit. The red sedan's own body is never drawn (we are
# inside it); a fixed dashboard/windshield-frame overlay is drawn in screen
# space every frame to keep the "inside a car" read consistent. The blue
# sedan is a separate, persistent 3D object with its own trajectory.
# ----------------------------------------------------------------------------

ROAD_HALF_LEN = 90.0
LANE_OFFSET = 2.5
PAVED_HALF_WIDTH = 4.3
GRASS_HALF_WIDTH = 11.0
GROUND_SEG_LEN = 2.0        # chunk length for large ground quads (culling)

LANE_NEAR_Z = -LANE_OFFSET  # lane both cars start in
LANE_FAR_Z = LANE_OFFSET    # passing / final lane

SKY_TOP = (128, 168, 214)
SKY_BOTTOM = (206, 222, 232)
GRASS_COLOR = (64, 112, 58)
ROAD_COLOR = (58, 58, 64)
EDGE_LINE_COLOR = (235, 235, 225)
DASH_COLOR = (232, 208, 78)

CAR_LENGTH = 4.4
CAR_WIDTH = 1.8
BODY_HEIGHT = 0.9
CABIN_HEIGHT = 0.55
CABIN_X0 = -0.85
CABIN_X1 = 0.55
CABIN_HALF_W = 0.68

# Ego (red) camera mounting, relative to the red car's own reference frame.
EYE_HEIGHT = 1.15           # world Y of the driver's eyes
CAM_FORWARD_OFFSET = 0.9    # eyes sit forward of the car's center, near the wheel
LOOK_AHEAD_DIST = 20.0      # distance ahead used to build the look target
TARGET_DROP = 0.35          # look slightly down at the road, not level with horizon
FOV_DEG = 72.0

TREE_X_RANGE = (-80.0, 86.0)
TREE_X_STEP = 12.0
TREE_Z_OFFSETS = (8.0, -8.0)

# Timeline breakpoints (fractions of the clip, t in [0, 1]).
T_FOLLOW_END = 0.10          # red still trailing blue, same lane
T_LANE_OUT_START = 0.10
T_LANE_OUT_END = 0.18        # lane change into passing lane complete
T_PASS_END = 0.40            # red now well ahead of blue, still in passing lane
T_LANE_IN_START = 0.40
T_LANE_IN_END = 0.48         # lane change back to original lane complete
T_CRUISE_END = 0.58          # brief cruise ahead of blue before slowing
T_TURN_START = 0.60          # fully slowed, U-turn begins
T_TURN_END = 0.82            # U-turn complete (heading reversed, lane swapped)

RED_X_KEYFRAMES = [
    (0.00, -20.0),
    (T_FOLLOW_END, -14.0),
    (T_LANE_OUT_END, -4.0),
    (T_PASS_END, 30.0),
    (T_LANE_IN_END, 38.0),
    (T_CRUISE_END, 43.0),
    (T_TURN_START, 45.0),
]
RED_X_RETURN_START = 45.0    # == X at T_TURN_START (continuity)
# Slower, gradual closing speed after the turn so the blue sedan is clearly
# visible ahead (within the camera FOV) for many frames before the two cars
# pass each other, rather than snapping from "barely in frame" to "gone".
RED_X_KEYFRAMES_RETURN = [
    (T_TURN_END, RED_X_RETURN_START),
    (0.94, 12.0),
    (1.00, -6.0),
]

BLUE_X_START = -4.0
BLUE_SPEED = 30.0             # world units per unit of normalized time t


def smootherstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * x * (x * (x * 6.0 - 15.0) + 10.0)


def piecewise_linear(t, keyframes):
    ts = [k[0] for k in keyframes]
    xs = [k[1] for k in keyframes]
    if t <= ts[0]:
        return xs[0]
    if t >= ts[-1]:
        return xs[-1]
    for i in range(len(ts) - 1):
        if ts[i] <= t <= ts[i + 1]:
            span = ts[i + 1] - ts[i]
            frac = 0.0 if span <= 0 else (t - ts[i]) / span
            return xs[i] + (xs[i + 1] - xs[i]) * frac
    return xs[-1]


def lane_z(t):
    """Red car's lane offset (Z) for the pre-turn phase only."""
    if t < T_LANE_OUT_START:
        return LANE_NEAR_Z
    if t < T_LANE_OUT_END:
        f = smootherstep((t - T_LANE_OUT_START) / (T_LANE_OUT_END - T_LANE_OUT_START))
        return LANE_NEAR_Z + (LANE_FAR_Z - LANE_NEAR_Z) * f
    if t < T_LANE_IN_START:
        return LANE_FAR_Z
    if t < T_LANE_IN_END:
        f = smootherstep((t - T_LANE_IN_START) / (T_LANE_IN_END - T_LANE_IN_START))
        return LANE_FAR_Z + (LANE_NEAR_Z - LANE_FAR_Z) * f
    return LANE_NEAR_Z


def red_state(t):
    """Return (x, z, heading_rad) for the red (ego) sedan at time t."""
    if t <= T_TURN_START:
        x = piecewise_linear(t, RED_X_KEYFRAMES)
        z = lane_z(t)
        heading = 0.0
        return x, z, heading

    if t <= T_TURN_END:
        # Continuous semicircular U-turn: heading sweeps 0 -> pi while
        # position traces a true semicircle of radius LANE_OFFSET, pivoting
        # from the near lane to the far lane.
        f = smootherstep((t - T_TURN_START) / (T_TURN_END - T_TURN_START))
        theta = math.pi * f
        r = LANE_OFFSET
        x = RED_X_RETURN_START + r * math.sin(theta)
        z = -r * math.cos(theta)
        heading = theta
        return x, z, heading

    x = piecewise_linear(t, RED_X_KEYFRAMES_RETURN)
    z = LANE_FAR_Z
    heading = math.pi
    return x, z, heading


def blue_state(t):
    x = BLUE_X_START + BLUE_SPEED * t
    return x, LANE_NEAR_Z, 0.0


def camera_for_ego(x, z, heading):
    forward = np.array([math.cos(heading), 0.0, math.sin(heading)])
    cam_pos = np.array([x, EYE_HEIGHT, z]) + forward * CAM_FORWARD_OFFSET
    cam_target = cam_pos + forward * LOOK_AHEAD_DIST
    cam_target[1] = EYE_HEIGHT - TARGET_DROP
    cam_up = np.array([0.0, 1.0, 0.0])
    return cam_pos, cam_target, cam_up


NEAR_PLANE = 0.15


def camera_basis(cam_pos, cam_target, cam_up):
    forward = cam_target - cam_pos
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, cam_up)
    right = right / np.linalg.norm(right)
    true_up = np.cross(right, forward)
    return forward, right, true_up


def to_camera_space(point_world, cam_pos, right, true_up, forward):
    rel = point_world - cam_pos
    return np.array([np.dot(rel, right), np.dot(rel, true_up), np.dot(rel, forward)])


def _clip_plane(points_cam, test_fn):
    """Sutherland-Hodgman clip of a convex polygon (camera-space points)
    against a linear half-space `test_fn(p) > 0` (inside)."""
    if not points_cam:
        return []
    out = []
    n = len(points_cam)
    for i in range(n):
        cur = points_cam[i]
        nxt = points_cam[(i + 1) % n]
        v_cur = test_fn(cur)
        v_nxt = test_fn(nxt)
        cur_in = v_cur > 0
        nxt_in = v_nxt > 0
        if cur_in:
            out.append(cur)
        if cur_in != nxt_in:
            f = v_cur / (v_cur - v_nxt)
            out.append(cur + (nxt - cur) * f)
    return out


def clip_near(points_cam, near=NEAR_PLANE, fov_deg=FOV_DEG, aspect=1.0):
    """Clip a convex polygon (camera-space points) against the near plane
    and the four side frustum planes. Side-plane clipping is essential: a
    face at a grazing angle to the camera can have a vertex just barely in
    front of the near plane but with huge lateral offset, which the
    perspective divide would otherwise blow up into a screen-filling
    polygon. Clipping to the actual view frustum keeps every projected
    point within a sane, bounded range."""
    k_y = math.tan(math.radians(fov_deg) / 2.0)
    k_x = k_y * aspect
    pts = points_cam
    pts = _clip_plane(pts, lambda p: p[2] - near)
    pts = _clip_plane(pts, lambda p: k_x * p[2] - p[0])
    pts = _clip_plane(pts, lambda p: k_x * p[2] + p[0])
    pts = _clip_plane(pts, lambda p: k_y * p[2] - p[1])
    pts = _clip_plane(pts, lambda p: k_y * p[2] + p[1])
    return pts


def project_cam_point(pt_cam, fov_deg, width, height):
    x_cam, y_cam, z_cam = pt_cam
    f = 1.0 / math.tan(math.radians(fov_deg) / 2.0)
    aspect = width / height
    x_ndc = (x_cam * f / aspect) / z_cam
    y_ndc = (y_cam * f) / z_cam
    x_px = (x_ndc * 0.5 + 0.5) * width
    y_px = (1.0 - (y_ndc * 0.5 + 0.5)) * height
    return x_px, y_px, z_cam


def sky_gradient(width, height):
    top = np.array(SKY_TOP, dtype=np.float64)
    bottom = np.array(SKY_BOTTOM, dtype=np.float64)
    grad = np.linspace(0.0, 1.0, height)[:, None]
    column = top[None, :] * (1.0 - grad) + bottom[None, :] * grad
    arr = np.tile(column[:, None, :], (1, width, 1)).astype(np.uint8)
    return Image.fromarray(arr)


def ground_quad(x0, x1, z0, z1, y=0.0):
    return [
        np.array([x0, y, z0]),
        np.array([x1, y, z0]),
        np.array([x1, y, z1]),
        np.array([x0, y, z1]),
    ]


def chunked_ground_quads(x0, x1, z0, z1, seg_len, color, name="ground"):
    """Split a long ground strip into short segments so each can be
    independently culled once the camera sits inside the scene (rather than
    far outside it looking in)."""
    faces = []
    x = x0
    while x < x1:
        xe = min(x + seg_len, x1)
        faces.append((name, ground_quad(x, xe, z0, z1), color))
        x += seg_len
    return faces


def dash_lane_markings():
    faces = []
    dash_len, gap_len, dash_half_w = 2.0, 1.6, 0.10
    x = -ROAD_HALF_LEN
    while x < ROAD_HALF_LEN:
        x_end = min(x + dash_len, ROAD_HALF_LEN)
        faces.append(("dash", ground_quad(x, x_end, -dash_half_w, dash_half_w), DASH_COLOR))
        x += dash_len + gap_len
    return faces


def static_ground_faces():
    faces = []
    faces.extend(chunked_ground_quads(
        -ROAD_HALF_LEN, ROAD_HALF_LEN, -GRASS_HALF_WIDTH, -PAVED_HALF_WIDTH,
        GROUND_SEG_LEN, GRASS_COLOR, "grass"))
    faces.extend(chunked_ground_quads(
        -ROAD_HALF_LEN, ROAD_HALF_LEN, PAVED_HALF_WIDTH, GRASS_HALF_WIDTH,
        GROUND_SEG_LEN, GRASS_COLOR, "grass"))
    faces.extend(chunked_ground_quads(
        -ROAD_HALF_LEN, ROAD_HALF_LEN, -PAVED_HALF_WIDTH, PAVED_HALF_WIDTH,
        GROUND_SEG_LEN, ROAD_COLOR, "road"))
    edge_w = 0.12
    faces.extend(chunked_ground_quads(
        -ROAD_HALF_LEN, ROAD_HALF_LEN, -PAVED_HALF_WIDTH, -PAVED_HALF_WIDTH + edge_w,
        GROUND_SEG_LEN, EDGE_LINE_COLOR, "edge"))
    faces.extend(chunked_ground_quads(
        -ROAD_HALF_LEN, ROAD_HALF_LEN, PAVED_HALF_WIDTH - edge_w, PAVED_HALF_WIDTH,
        GROUND_SEG_LEN, EDGE_LINE_COLOR, "edge"))
    faces.extend(dash_lane_markings())
    return faces


def box_faces(cx, cz, heading, length, width, height, palette, y0=0.0):
    """Visible (non-bottom) faces of a box centered at (cx, cz), yawed by
    `heading` radians, base at y0."""
    hl, hw = length / 2.0, width / 2.0
    corners_local = {
        "FL": (hl, -hw), "FR": (hl, hw), "BR": (-hl, hw), "BL": (-hl, -hw),
    }
    c, s = math.cos(heading), math.sin(heading)
    world = {}
    for k, (lx, lz) in corners_local.items():
        wx = lx * c - lz * s + cx
        wz = lx * s + lz * c + cz
        world[k] = (wx, wz)

    def pt(key, y):
        wx, wz = world[key]
        return np.array([wx, y, wz])

    y_top = y0 + height
    faces = [
        ("front", [pt("FL", y0), pt("FR", y0), pt("FR", y_top), pt("FL", y_top)], palette["front"]),
        ("back", [pt("BR", y0), pt("BL", y0), pt("BL", y_top), pt("BR", y_top)], palette["back"]),
        ("right", [pt("FR", y0), pt("BR", y0), pt("BR", y_top), pt("FR", y_top)], palette["right"]),
        ("left", [pt("BL", y0), pt("FL", y0), pt("FL", y_top), pt("BL", y_top)], palette["left"]),
        ("top", [pt("FL", y_top), pt("FR", y_top), pt("BR", y_top), pt("BL", y_top)], palette["top"]),
    ]
    return faces


def tree_faces(cx, cz):
    trunk_color = (92, 62, 38)
    canopy_color = (46, 96, 46)
    faces = []
    faces.extend(box_faces(cx, cz, 0.0, 0.3, 0.3, 1.2,
                            {"front": trunk_color, "back": trunk_color,
                             "left": trunk_color, "right": trunk_color,
                             "top": trunk_color}, y0=0.0))
    faces.extend(box_faces(cx, cz, 0.0, 1.3, 1.3, 1.4,
                            {"front": canopy_color, "back": canopy_color,
                             "left": canopy_color, "right": canopy_color,
                             "top": (56, 110, 56)}, y0=1.2))
    return faces


def car_palette(kind):
    if kind == "red":
        base = (196, 40, 38)
        return {
            "body": {"front": (232, 104, 64), "back": (92, 16, 16),
                     "left": base, "right": base, "top": (150, 26, 24)},
            "cabin": {"front": (208, 224, 236), "back": (30, 30, 36),
                      "left": (168, 32, 30), "right": (168, 32, 30),
                      "top": (140, 20, 20)},
        }
    base = (40, 90, 196)
    return {
        "body": {"front": (110, 170, 232), "back": (14, 34, 92),
                 "left": base, "right": base, "top": (26, 66, 150)},
        "cabin": {"front": (208, 224, 236), "back": (26, 28, 36),
                  "left": (30, 78, 168), "right": (30, 78, 168),
                  "top": (20, 50, 140)},
    }


def car_polygons(cx, cz, heading, kind):
    palette = car_palette(kind)
    faces = box_faces(cx, cz, heading, CAR_LENGTH, CAR_WIDTH, BODY_HEIGHT,
                       palette["body"], y0=0.0)
    cab_len = CABIN_X1 - CABIN_X0
    cab_local_cx = (CABIN_X0 + CABIN_X1) / 2.0
    c, s = math.cos(heading), math.sin(heading)
    cab_cx = cx + cab_local_cx * c
    cab_cz = cz + cab_local_cx * s
    faces.extend(box_faces(cab_cx, cab_cz, heading, cab_len, 2 * CABIN_HALF_W,
                            CABIN_HEIGHT, palette["cabin"], y0=BODY_HEIGHT))
    return faces


def build_tree_positions():
    xs = np.arange(TREE_X_RANGE[0], TREE_X_RANGE[1], TREE_X_STEP)
    positions = []
    for i, x in enumerate(xs):
        z = TREE_Z_OFFSETS[i % 2]
        positions.append((float(x), z))
    return positions


TREE_POSITIONS = build_tree_positions()
STATIC_GROUND_FACES = static_ground_faces()


def project_poly(points, cam_pos, right, true_up, forward, fov_deg, width, height):
    """Convert a world-space convex polygon to 2D screen points, clipping
    against the camera near plane first so faces straddling or behind the
    camera are handled correctly instead of producing blown-up artifacts."""
    cam_pts = [to_camera_space(p, cam_pos, right, true_up, forward) for p in points]
    clipped = clip_near(cam_pts, fov_deg=fov_deg, aspect=width / height)
    if len(clipped) < 3:
        return None
    screen = [project_cam_point(p, fov_deg, width, height) for p in clipped]
    return screen


def render_frame(t, width, height):
    rx, rz, rh = red_state(t)
    bx, bz, bh = blue_state(t)
    cam_pos, cam_target, cam_up = camera_for_ego(rx, rz, rh)
    forward, right, true_up = camera_basis(cam_pos, cam_target, cam_up)

    img = sky_gradient(width, height)
    draw = ImageDraw.Draw(img)

    all_faces = list(STATIC_GROUND_FACES)
    for tx, tz in TREE_POSITIONS:
        all_faces.extend(tree_faces(tx, tz))
    all_faces.extend(car_polygons(bx, bz, bh, "blue"))

    depth_faces = []
    for name, pts, color in all_faces:
        proj = project_poly(pts, cam_pos, right, true_up, forward, FOV_DEG, width, height)
        if proj is None:
            continue
        avg_depth = sum(p[2] for p in proj) / len(proj)
        poly2d = [(p[0], p[1]) for p in proj]
        depth_faces.append((avg_depth, poly2d, color))

    depth_faces.sort(key=lambda f: f[0], reverse=True)
    for _, poly2d, color in depth_faces:
        draw.polygon(poly2d, fill=color)

    draw_dashboard_overlay(draw, width, height)
    return img


def draw_dashboard_overlay(draw, width, height):
    """Fixed screen-space overlay: dark A-pillars framing the windshield, a
    red-tinted hood/dashboard band along the bottom, a steering wheel, and a
    rear-view mirror -- keeps the "looking out of a red sedan" read
    consistent every frame regardless of camera motion outside."""
    pillar_w = width * 0.05
    pillar_color = (24, 22, 22)
    draw.polygon([(0, 0), (pillar_w, 0), (pillar_w * 0.55, height * 0.42), (0, height * 0.42)],
                 fill=pillar_color)
    draw.polygon([(width, 0), (width - pillar_w, 0),
                  (width - pillar_w * 0.55, height * 0.42), (width, height * 0.42)],
                 fill=pillar_color)

    hood_top = height * 0.82
    draw.polygon([(0, height), (width, height), (width, hood_top),
                  (width * 0.5, hood_top - height * 0.03), (0, hood_top)],
                 fill=(46, 44, 46))
    hood_edge_top = hood_top - height * 0.03
    draw.polygon([(0, hood_edge_top), (width, hood_edge_top),
                  (width * 0.62, hood_edge_top - height * 0.05),
                  (width * 0.38, hood_edge_top - height * 0.05)],
                 fill=(150, 30, 28))

    wheel_cx, wheel_cy = width * 0.30, height * 0.99
    wheel_r = width * 0.13
    draw.ellipse([wheel_cx - wheel_r, wheel_cy - wheel_r,
                  wheel_cx + wheel_r, wheel_cy + wheel_r],
                 outline=(18, 18, 18), width=int(width * 0.018))
    draw.ellipse([wheel_cx - wheel_r * 0.22, wheel_cy - wheel_r * 0.22,
                  wheel_cx + wheel_r * 0.22, wheel_cy + wheel_r * 0.22],
                 fill=(18, 18, 18))
    draw.line([(wheel_cx, wheel_cy), (wheel_cx, wheel_cy - wheel_r)],
               fill=(18, 18, 18), width=max(2, int(width * 0.01)))

    mirror_w, mirror_h = width * 0.10, height * 0.045
    mx0 = width * 0.5 - mirror_w / 2.0
    draw.polygon([(mx0, 0), (mx0 + mirror_w, 0),
                  (mx0 + mirror_w * 0.9, mirror_h), (mx0 + mirror_w * 0.1, mirror_h)],
                 fill=(22, 22, 24))


def main():
    args = parse_args()
    width, height = args.width, args.height
    n_frames = args.frames
    out_path = Path(args.out)

    frames = []
    for i in range(n_frames):
        t = i / (n_frames - 1) if n_frames > 1 else 0.0
        img = render_frame(t, width, height)
        frames.append(np.ascontiguousarray(np.array(img.convert("RGB"), dtype=np.uint8)))

    writer = imageio_ffmpeg.write_frames(
        str(out_path), size=(width, height), fps=args.fps,
        codec="libx264", pix_fmt_in="rgb24", pix_fmt_out="yuv420p",
        macro_block_size=16,
    )
    writer.send(None)
    try:
        for frame in frames:
            writer.send(frame)
    finally:
        writer.close()

    print(f"Wrote {n_frames} frames to {out_path} ({width}x{height} @ {args.fps}fps)")

    # Sanity prints: verify overtake/turn timeline is well-formed.
    r0 = red_state(0.0)
    r_turn0 = red_state(T_TURN_START)
    r_turn1 = red_state(T_TURN_END)
    r1 = red_state(1.0)
    print(f"red start: {r0}, turn start: {r_turn0}, turn end: {r_turn1}, end: {r1}")
    print(f"heading start deg: {math.degrees(r0[2]):.1f}, "
          f"heading end deg: {math.degrees(r1[2]):.1f}")


if __name__ == "__main__":
    main()
