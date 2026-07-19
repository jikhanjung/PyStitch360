"""Dual GoPro HERO5 Black fixed sideline rig generator.

Generates 3D-printable STL files for a one-piece rig holding two HERO5 Black
bodies (bare, no Frame) at a fixed relative yaw/pitch matching the field
calibration measured by PyStitch360.

Two cradle variants are produced:
  - toploader : camera drops in from the top, snaps under front-post nubs
  - rearload  : camera slides in from the back under two top rails until it
                hits the front stops; a floor snap tab latches behind it.
                The back is fully open (touchscreen unobstructed).

Outputs (written next to this script):
  - dual_gopro_rig.stl / test_cradle.stl / rig_preview.png            (toploader)
  - dual_gopro_rig_rearload.stl / test_cradle_rearload.stl /
    rig_preview_rearload.png                                          (rearload)

Run:  python hardware/rig_generator.py

Requires: trimesh, manifold3d, scipy, numpy, matplotlib (~/venv/PyStitch360).
"""

import numpy as np
import trimesh
from pathlib import Path

OUT_DIR = Path(__file__).parent

# ---------------------------------------------------------------- parameters
# Camera body (HERO5/6/7 Black shared body, bare without The Frame)
CAM_W = 62.3        # width  (mm)
CAM_H = 44.9        # height (mm)
CAM_D = 24.6        # body depth excluding lens bump (lens adds ~8.4 mm)
FIT_CLR = 0.4       # snug-fit clearance added to each cavity dimension

# Cradle structure (shared)
WALL = 3.0          # wall thickness
FLOOR = 3.0         # bottom plate thickness
FRONT_LIP_H = 12.0  # solid lip height on the (otherwise open) front face
POST_FRONT_W = 3.5  # front post/column width over the camera front face
                    # (kept small: lens bump sits ~4 mm from the body edge)
POST_SIDE_D = 4.5   # front post depth along the side face

# Toploader specifics
POST_RISE = 2.0     # posts/back wall rise above camera top for the snap nubs
NUB = 1.0           # snap nub inward protrusion over the camera top edge
BACK_BORDER = 6.0   # back-wall frame border around the touchscreen window

# Rearload specifics
RAIL_OVH = 2.5      # top rail inward overhang over the camera top edges
RAIL_H = 4.0        # top rail height
RAIL_CLR = 0.3      # sliding clearance between camera top and rail underside
REAR_STRIP = 5.0    # solid side-wall strip depth at the rear end
REAR_EXT = 6.0      # floor tail behind the camera (carries the snap tab)
TAB_W = 14.0        # floor snap tab width
TAB_HINGE = 14.0    # tab hinge position forward of the camera rear face
TAB_BUMP_H = 2.0    # snap bump height above the floor
ROOF_HOLE_W = 36.0  # roof finger-hole width (shutter button access)
ROOF_HOLE_D = 20.0  # roof finger-hole depth

# Lens position within the body (HERO5 Black; lens sits in a top corner).
# Confirmed: seen from behind, the lens is on the body's LEFT side
# (front view: right). In cradle coords (+y forward, z up) that is -x.
LENS_EDGE = 14.5    # lens center distance from the nearer side edge
LENS_TOP = 14.5     # lens center distance from the top edge
LENS_X = -(CAM_W / 2 - LENS_EDGE)  # lens x in cradle coords (upright camera)

# Rig geometry (from PyStitch360 calibration data)
YAW_SPLIT_DEG = 68.8   # total divergence between optical axes
PITCH_DOWN_DEG = 18.0  # downward tilt of both cameras
CAM_GAP = 3.0          # minimum gap between the two cradles

# Base plate
BASE_T = 8.0        # plate thickness
BASE_MARGIN = 6.0   # outline margin around the wedge footprints
NUT_AF = 11.4       # 1/4"-20 hex nut across-flats + clearance (nominal 11.11)
NUT_T = 5.7         # nut pocket depth (nut is 5.45 thick)
SCREW_D = 6.6       # 1/4" screw clearance hole


def box(x0, x1, y0, y1, z0, z1):
    (x0, x1), (y0, y1), (z0, z1) = sorted((x0, x1)), sorted((y0, y1)), sorted((z0, z1))
    b = trimesh.creation.box((x1 - x0, y1 - y0, z1 - z0))
    b.apply_translation(((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2))
    return b


def hull(points):
    return trimesh.convex.convex_hull(np.asarray(points, dtype=float))


def union(meshes):
    return trimesh.boolean.union(meshes, engine="manifold")


def difference(a, cutters):
    return trimesh.boolean.difference([a] + cutters, engine="manifold")


def cavity_dims():
    return CAM_W + FIT_CLR, CAM_H + FIT_CLR, CAM_D + FIT_CLR


def build_cradle_toploader():
    """Camera faces +y, z up, origin at floor center; drops in from the top.

    Open top (shutter), open sides between the back wall and the front corner
    posts (side buttons / USB door), open front above a 12 mm lip (lens bump
    protrudes freely). The camera slides its rear top edge under the rear lip,
    then the front presses down past ramped nubs on the front posts.
    """
    iw, ih, idp = cavity_dims()
    ow, od = iw + 2 * WALL, idp + 2 * WALL
    z_top = FLOOR + ih + POST_RISE
    y_back_in, y_front_in = -idp / 2, idp / 2

    parts = [box(-ow / 2, ow / 2, -od / 2, od / 2, 0, FLOOR)]  # floor

    # back wall with touchscreen window
    back = box(-ow / 2, ow / 2, y_back_in - WALL, y_back_in, 0, z_top)
    win = box(-(iw / 2 - BACK_BORDER), iw / 2 - BACK_BORDER,
              y_back_in - WALL - 1, y_back_in + 1,
              FLOOR + BACK_BORDER, FLOOR + ih - BACK_BORDER)
    parts.append(difference(back, [win]))

    # rear top lip (overhangs the camera top rear edge)
    parts.append(box(-iw / 2 + 8, iw / 2 - 8, y_back_in, y_back_in + 1.2,
                     FLOOR + ih + 0.2, z_top))

    # front lip (below the lens / front LCD area)
    parts.append(box(-ow / 2, ow / 2, y_front_in, y_front_in + WALL,
                     0, FLOOR + FRONT_LIP_H))

    # front corner posts (L-shaped) + ramped snap nubs
    for sx in (-1, 1):
        x_out, x_in = sx * ow / 2, sx * iw / 2
        fx0, fx1 = sorted((x_out, x_out - sx * (POST_FRONT_W + WALL)))
        parts.append(box(fx0, fx1, y_front_in, y_front_in + WALL, 0, z_top))
        sx0, sx1 = sorted((x_out, x_in))
        parts.append(box(sx0, sx1, y_front_in - POST_SIDE_D, y_front_in + WALL,
                         0, z_top))
        z1, z2 = FLOOR + ih + 0.2, z_top
        y0, y1 = y_front_in - POST_SIDE_D, y_front_in
        tip_x = x_in - sx * NUB
        parts.append(hull([(x_in, y0, z1), (x_in, y1, z1),
                           (x_in, y0, z2), (x_in, y1, z2),
                           (tip_x, y0, z2), (tip_x, y1, z2)]))

    return union(parts), []


def build_cradle_rearload(roof=False):
    """Camera faces +y, z up, origin at floor center; slides in from the back.

    Fully open back (touchscreen unobstructed), open top between two side
    rails (shutter), open sides between the front posts and short rear strips
    (side buttons / USB door). The camera slides forward under the chamfered
    top rails until it hits the front stops; a floor snap tab clicks up behind
    its rear face. Press the tab bump down to release.

    Returns (mesh, local_cutters) — the cutters carve flex clearance under the
    snap tab out of the wedge when the cradle is placed on the rig base.
    """
    iw, ih, idp = cavity_dims()
    ow = iw + 2 * WALL
    z_rail_bot = FLOOR + ih + RAIL_CLR
    z_top = z_rail_bot + RAIL_H
    y_rear, y_front_in = -idp / 2, idp / 2
    y_floor_rear = y_rear - REAR_EXT

    parts = [box(-ow / 2, ow / 2, y_floor_rear, y_front_in + WALL, 0, FLOOR)]

    # front stop: bottom lip + two narrow full-height columns at the edges
    parts.append(box(-ow / 2, ow / 2, y_front_in, y_front_in + WALL,
                     0, FLOOR + FRONT_LIP_H))
    for sx in (-1, 1):
        x_out, x_in = sx * ow / 2, sx * iw / 2
        fx0, fx1 = sorted((x_out, x_out - sx * (POST_FRONT_W + WALL)))
        parts.append(box(fx0, fx1, y_front_in, y_front_in + WALL, 0, z_top))
        # side wall: solid strips front and rear, window between
        sx0, sx1 = sorted((x_out, x_in))
        parts.append(box(sx0, sx1, y_front_in - POST_SIDE_D, y_front_in + WALL,
                         0, z_top))
        parts.append(box(sx0, sx1, y_rear, y_rear + REAR_STRIP, 0, z_top))
        # top rail along the full side, chamfered underside at the overhang
        tip_x = x_in - sx * RAIL_OVH
        profile = [(x_out, z_rail_bot), (x_out, z_top),
                   (tip_x, z_top), (tip_x, z_top - 1.2), (x_in, z_rail_bot)]
        parts.append(hull([(px, y, pz) for px, pz in profile
                           for y in (y_rear, y_front_in + WALL)]))

    # optional roof plate over the top opening, with a finger hole above the
    # shutter button (sun/rain shade + extra stiffness)
    if roof:
        plate = box(-ow / 2, ow / 2, y_rear, y_front_in + WALL,
                    z_top, z_top + WALL)
        hole = box(-ROOF_HOLE_W / 2, ROOF_HOLE_W / 2,
                   -ROOF_HOLE_D / 2, ROOF_HOLE_D / 2,
                   z_top - 1, z_top + WALL + 1)
        parts.append(difference(plate, [hole]))

    # floor snap tab: bump with an insertion ramp, stop face behind the camera
    y_stop = y_rear - 0.1
    parts.append(hull([(x, y, z) for x in (-TAB_W / 2 + 1, TAB_W / 2 - 1)
                       for y, z in [(y_stop, FLOOR), (y_stop, FLOOR + TAB_BUMP_H),
                                    (y_stop - 3.5, FLOOR)]]))
    cradle = union(parts)

    # U-slot around the tab so it can flex down (hinge toward the camera)
    y_hinge = y_rear + TAB_HINGE
    slots = [box(sx * (TAB_W / 2), sx * (TAB_W / 2 + 2.0),
                 y_floor_rear - 1, y_hinge, -1, FLOOR + 0.6)
             for sx in (-1, 1)]
    cradle = difference(cradle, slots)

    # clearance pocket below the tab (cut out of the wedge on the full rig)
    cutters = [box(-TAB_W / 2 - 2, TAB_W / 2 + 2, y_floor_rear - 0.1,
                   y_hinge - 1, -5, -0.01)]
    return cradle, cutters


def build_rig(cradle, local_cutters):
    yaw = np.radians(YAW_SPLIT_DEG / 2)
    pitch = np.radians(PITCH_DOWN_DEG)
    lo, hi = cradle.bounds
    hw, hd = (hi[0] - lo[0]) / 2, (hi[1] - lo[1]) / 2

    placed, wedge_pts, lens_axes, opt_centers, cutters = [], [], [], [], []
    # nominal optical center in cradle-local coords (camera front-face center)
    oc_local = np.array([[0.0, CAM_D / 2, FLOOR + CAM_H / 2]])
    dx0 = hw * np.cos(yaw) + hd * np.sin(yaw) + CAM_GAP / 2
    for side in (-1, 1):  # left, right camera
        dx = side * dx0
        T = (trimesh.transformations.translation_matrix((dx, 0, 0)) @
             trimesh.transformations.rotation_matrix(side * -yaw, (0, 0, 1)) @
             trimesh.transformations.rotation_matrix(-pitch, (1, 0, 0)))
        lift = -trimesh.transform_points(cradle.vertices, T)[:, 2].min()
        L = trimesh.transformations.translation_matrix((0, 0, lift)) @ T
        m = cradle.copy()
        m.apply_transform(L)
        placed.append(m)
        lens_axes.append(T[:3, :3] @ np.array([0.0, 1.0, 0.0]))
        opt_centers.append(trimesh.transform_points(oc_local, L)[0])
        for c in local_cutters:
            cutters.append(c.copy().apply_transform(L))

        # wedge: hull between the cradle floor slab and its shadow on z=0
        corners = np.array([(x, y, z) for x in (lo[0], hi[0])
                            for y in (lo[1], hi[1]) for z in (0, FLOOR)])
        c8 = trimesh.transform_points(corners, L)
        shadow = c8.copy()
        shadow[:, 2] = 0.0
        placed.append(hull(np.vstack([c8, shadow])))
        wedge_pts.append(shadow[:, :2])

    left, right = union(placed[:2]), union(placed[2:])
    if trimesh.boolean.intersection([left, right], engine="manifold").volume > 1e-6:
        raise RuntimeError("cradles intersect — increase CAM_GAP")

    a, b = lens_axes
    split = np.degrees(np.arccos(np.dot(a[:2], b[:2]) /
                                 (np.linalg.norm(a[:2]) * np.linalg.norm(b[:2]))))
    dip = np.degrees(np.arcsin(-a[2]))
    baseline = float(np.linalg.norm(opt_centers[1] - opt_centers[0]))
    print(f"optical check: yaw split {split:.2f} deg, pitch down {dip:.2f} deg, "
          f"baseline {baseline:.1f} mm")
    extrinsic = {
        "yaw_split_deg": round(split, 3),
        "left_yaw_deg": round(-split / 2, 3),
        "right_yaw_deg": round(split / 2, 3),
        "pitch_deg": round(-dip, 3),
        "roll_deg": 0.0,
        "baseline_mm": round(baseline, 2),
        "optical_center_height_above_base_mm":
            round(float(np.mean([c[2] for c in opt_centers])) + BASE_T, 2),
        "conventions": {
            "yaw": "positive = clockwise seen from above (right camera positive)",
            "pitch": "negative = tilted down",
            "optical_center": "nominal camera front-face center; refine per session",
        },
    }

    # base plate: hull of both wedge footprints + margin, extruded down
    pts2d = np.vstack(wedge_pts)
    ang = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    ring = np.stack([np.cos(ang), np.sin(ang)], axis=1) * BASE_MARGIN
    expanded = (pts2d[:, None, :] + ring[None, :, :]).reshape(-1, 2)
    plate = hull(np.vstack([
        np.column_stack([expanded, np.zeros(len(expanded))]),
        np.column_stack([expanded, np.full(len(expanded), -BASE_T)])]))

    rig = union(placed + [plate])

    # 1/4"-20 mount: through hole + captive hex nut pocket from the bottom
    cx, cy = 0.0, float(pts2d[:, 1].mean())
    hole = trimesh.creation.cylinder(radius=SCREW_D / 2, height=BASE_T + 20,
                                     sections=48)
    hole.apply_translation((cx, cy, -BASE_T / 2))
    nut = trimesh.creation.cylinder(radius=NUT_AF / np.sqrt(3),
                                    height=NUT_T + 0.1, sections=6)
    nut.apply_translation((cx, cy, -BASE_T + NUT_T / 2))
    return difference(rig, [hole, nut] + cutters), extrinsic


def build_rig_lens_inward(cradle, local_cutters):
    """v3 placement: both cameras still aim outward (±yaw/2) but the LEFT
    camera is mounted upside-down (roll 180°). The HERO5 lens sits in a body
    corner (behind-view left), so flipping the left camera puts both lens
    centers on the inboard edges, shortening the baseline. The flipped cradle
    is raised so both lens centers end up at the same height. Cradle spacing
    uses the actual rotated geometry (tightest mirror-symmetric packing)
    instead of the conservative bounding-box formula of v1/v2.
    """
    yaw = np.radians(YAW_SPLIT_DEG / 2)
    pitch = np.radians(PITCH_DOWN_DEG)
    iw, ih, idp = cavity_dims()
    zc = FLOOR + ih / 2  # cavity center height: flip axis for the right side

    # lens center in cradle-local coords (normal orientation)
    lens_local = np.array([[LENS_X, idp / 2,
                            FLOOR + FIT_CLR / 2 + CAM_H - LENS_TOP]])

    flipT = trimesh.transformations.rotation_matrix(np.pi, (0, 1, 0),
                                                    point=(0, 0, zc))
    lo, hi = cradle.bounds
    corners = np.array([(x, y, z) for x in (lo[0], hi[0])
                        for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])

    placed, wedge_pts, lens_axes, lens_pts, cutters = [], [], [], [], []
    lifts, rots = [], []
    for side in (-1, 1):
        R = (trimesh.transformations.rotation_matrix(side * -yaw, (0, 0, 1)) @
             trimesh.transformations.rotation_matrix(-pitch, (1, 0, 0)))
        if side < 0:  # left camera rides upside-down (lens moves inboard)
            R = R @ flipT
        rots.append(R)
        lifts.append(-trimesh.transform_points(cradle.vertices, R)[:, 2].min())

    # equalize lens heights by raising whichever side sits lower
    lz = [trimesh.transform_points(lens_local, rots[i])[0][2] + lifts[i]
          for i in (0, 1)]
    extra = [max(lz) - lz[0], max(lz) - lz[1]]

    for i, side in enumerate((-1, 1)):
        R = rots[i]
        verts = trimesh.transform_points(cradle.vertices, R)
        if side < 0:
            dx = -CAM_GAP / 2 - verts[:, 0].max()
        else:
            dx = CAM_GAP / 2 - verts[:, 0].min()
        L = trimesh.transformations.translation_matrix(
            (dx, 0, lifts[i] + extra[i])) @ R
        m = cradle.copy()
        m.apply_transform(L)
        placed.append(m)
        lens_axes.append(R[:3, :3] @ np.array([0.0, 1.0, 0.0]))
        lens_pts.append(trimesh.transform_points(lens_local, L)[0])
        for c in local_cutters:
            cutters.append(c.copy().apply_transform(L))

        # wedge under the lowest bbox face of the placed cradle
        c8 = trimesh.transform_points(corners, L)
        bottom4 = c8[np.argsort(c8[:, 2])[:4]]
        shadow = bottom4.copy()
        shadow[:, 2] = 0.0
        placed.append(hull(np.vstack([bottom4, shadow])))
        wedge_pts.append(shadow[:, :2])

    left, right = union(placed[:2]), union(placed[2:])
    if trimesh.boolean.intersection([left, right], engine="manifold").volume > 1e-6:
        raise RuntimeError("cradles intersect — increase CAM_GAP")

    a, b = lens_axes
    split = np.degrees(np.arccos(np.dot(a[:2], b[:2]) /
                                 (np.linalg.norm(a[:2]) * np.linalg.norm(b[:2]))))
    dip = np.degrees(np.arcsin(-a[2]))
    baseline = float(np.linalg.norm(lens_pts[1] - lens_pts[0]))
    dz = float(lens_pts[1][2] - lens_pts[0][2])
    print(f"optical check: yaw split {split:.2f} deg, pitch down {dip:.2f} deg, "
          f"baseline {baseline:.1f} mm, lens dz {dz:.3f} mm")
    extrinsic = {
        "yaw_split_deg": round(split, 3),
        "per_camera": {
            "left": {"yaw_deg": round(-split / 2, 3), "pitch_deg": round(-dip, 3),
                     "roll_deg": 180.0,
                     "orientation": "upside-down (set GoPro rotation to Down)"},
            "right": {"yaw_deg": round(split / 2, 3), "pitch_deg": round(-dip, 3),
                      "roll_deg": 0.0, "orientation": "upright"},
        },
        "baseline_mm": round(baseline, 2),
        "lens_height_delta_mm": round(dz, 3),
        "lens_offset_in_body_mm": {"from_side_edge": LENS_EDGE,
                                   "from_top_edge": LENS_TOP},
        "optical_center_height_above_base_mm":
            round(float(np.mean([p[2] for p in lens_pts])) + BASE_T, 2),
        "conventions": {
            "yaw": "positive = clockwise seen from above (right camera positive)",
            "pitch": "negative = tilted down",
            "optical_center": "nominal lens center in the body corner; "
                              "refine per session",
            "lens_side": "lens on the body's left seen from behind "
                         "(front view: right) — confirmed on real camera",
        },
    }

    pts2d = np.vstack(wedge_pts)
    ang = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    ring = np.stack([np.cos(ang), np.sin(ang)], axis=1) * BASE_MARGIN
    expanded = (pts2d[:, None, :] + ring[None, :, :]).reshape(-1, 2)
    plate = hull(np.vstack([
        np.column_stack([expanded, np.zeros(len(expanded))]),
        np.column_stack([expanded, np.full(len(expanded), -BASE_T)])]))

    rig = union(placed + [plate])
    cx, cy = 0.0, float(pts2d[:, 1].mean())
    hole = trimesh.creation.cylinder(radius=SCREW_D / 2, height=BASE_T + 20,
                                     sections=48)
    hole.apply_translation((cx, cy, -BASE_T / 2))
    nut = trimesh.creation.cylinder(radius=NUT_AF / np.sqrt(3),
                                    height=NUT_T + 0.1, sections=6)
    nut.apply_translation((cx, cy, -BASE_T + NUT_T / 2))
    return difference(rig, [hole, nut] + cutters), extrinsic


def render(mesh, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LightSource
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(14, 6))
    views = [(25, -60, "front-left iso"), (25, 240, "back-right iso"),
             (80, -90, "top")]
    for i, (elev, azim, title) in enumerate(views, 1):
        ax = fig.add_subplot(1, 3, i, projection="3d")
        tris = mesh.vertices[mesh.faces]
        pc = Poly3DCollection(tris, facecolors="#8fa8c8", shade=True,
                              lightsource=LightSource(azdeg=315, altdeg=45))
        ax.add_collection3d(pc)
        lo, hi = mesh.bounds
        c, r = (lo + hi) / 2, (hi - lo).max() / 2
        ax.set_xlim(c[0] - r, c[0] + r)
        ax.set_ylim(c[1] - r, c[1] + r)
        ax.set_zlim(c[2] - r, c[2] + r)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def emit(rig_id, loading, builder, rig_builder=None):
    """Write rig.stl + test_cradle.stl + extrinsic.json + preview.png as a
    versioned pair under hardware/rigs/<rig_id>/."""
    import json

    out = OUT_DIR / "rigs" / rig_id
    out.mkdir(parents=True, exist_ok=True)

    cradle, cutters = builder()
    cradle.export(out / "test_cradle.stl")
    print(f"[{rig_id}] test_cradle.stl: "
          f"{cradle.bounds[1] - cradle.bounds[0]} mm, "
          f"watertight={cradle.is_watertight}")
    rig, extrinsic = (rig_builder or build_rig)(cradle, cutters)
    rig.export(out / "rig.stl")
    print(f"[{rig_id}] rig.stl: {rig.bounds[1] - rig.bounds[0]} mm, "
          f"watertight={rig.is_watertight}")

    profile = {
        "rig_id": rig_id,
        "revision_note": loading,
        "camera": {
            "manufacturer": "GoPro",
            "model": "HERO5 Black",
            "count": 2,
            "recording_mode": "4K_16:9_29.97_wide_eis_off",
            "lens_profile": "GoPro_HERO5_Black_Wide_4K_16x9",
        },
        "extrinsic_nominal": extrinsic,
        "mount": "1/4-20 captive hex nut, base center",
        "source_calibration": "yaw_split_deg 68.68-68.83 from project JSONs; "
                              "pitch from devlog/20260717_P01 (~18 deg down)",
        "note": "Design-nominal values — use as the initial alignment; "
                "per-session auto alignment still refines the delta.",
    }
    (out / "extrinsic.json").write_text(json.dumps(profile, indent=2))
    render(rig, out / "preview.png")
    print(f"[{rig_id}] extrinsic.json + preview.png written")


if __name__ == "__main__":
    emit("GP5-DUAL-v1", "toploader: camera drops in from the top",
         build_cradle_toploader)
    emit("GP5-DUAL-v2", "rearload: camera slides in from the back, open rear",
         build_cradle_rearload)
    emit("GP5-DUAL-v3", "rearload cradles, lens-inward: right camera "
         "upside-down, lenses on the inboard edges, minimal baseline",
         build_cradle_rearload, rig_builder=build_rig_lens_inward)
    emit("GP5-DUAL-v4", "v3 + roof plate on both cradles (shutter finger "
         "hole); sun/rain shade and extra stiffness",
         lambda: build_cradle_rearload(roof=True),
         rig_builder=build_rig_lens_inward)
