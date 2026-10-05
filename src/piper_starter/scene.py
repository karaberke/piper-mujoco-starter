"""Build the MuJoCo world: Piper + plate stands + labelled plates + gridded wall.

MuJoCo describes worlds in an XML format called MJCF. Rather than writing
one big XML file, this module loads the Menagerie Piper (which *is* an MJCF
file) and adds everything else with MjSpec, MuJoCo's Python editing API.
That mirrors how the Isaac Sim project builds its USD stage from Python.
"""

import mujoco
import numpy as np

from .config import OUTPUT_DIR, PIPER_DIR, SceneConfig
from .poses import plate_in_holder, quat_wxyz_from_matrix, wall_pose_in_world
from .wall_texture import make_wall_grid


def _look_at_xyaxes(eye, target):
    """Camera x and y axes (MuJoCo's 'xyaxes') for a camera at eye looking at target."""
    fwd = np.asarray(target, float) - np.asarray(eye, float)
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 0, 1])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    return np.concatenate([right, up])


def build_spec(cfg: SceneConfig) -> mujoco.MjSpec:
    scene_xml = PIPER_DIR / "scene.xml"
    if not scene_xml.exists():
        raise FileNotFoundError(f"{scene_xml} is missing. Run: uv run python scripts/fetch_piper.py")
    spec = mujoco.MjSpec.from_file(str(scene_xml))
    spec.option.timestep = 0.002
    spec.visual.global_.offwidth = 1280  # allow larger offscreen renders (videos)
    spec.visual.global_.offheight = 960
    world = spec.worldbody

    # --- Stiffer wrist servos ----------------------------------------------
    # The Menagerie wrist actuators are soft (kp = 10) and every joint has
    # 0.3 Nm of friction, so the wrist stops ~0.03 rad short of its target:
    # ~6 mm at the fingertips. Real Piper servos hold position much better,
    # so raise the gains. A position actuator is force = kp*(ctrl - q) - kv*qdot,
    # stored as gainprm[0] = kp, biasprm[1] = -kp, biasprm[2] = -kv.
    # The gripper has the same problem in a worse form: kp = 40 N/m means a
    # finger stopped 3 mm short by an object squeezes with only 0.1 N, so the
    # plate slips out. With kp = 3000 it hits its 10 N force limit instead.
    for name, kp, kv in [("joint4", 80, 5), ("joint5", 60, 4), ("joint6", 60, 4), ("gripper", 3000, 20)]:
        act = spec.actuator(name)
        act.gainprm[0] = kp
        act.biasprm[1] = -kp
        act.biasprm[2] = -kv

    # --- Tool centre point (TCP) site on the gripper ---------------------
    spec.body("link6").add_site(name="tcp", pos=cfg.tcp_offset, size=[0.006, 0, 0], rgba=[1, 0, 1, 1], group=1)

    # --- Stands with slotted holders, one per plate ----------------------
    st = cfg.stand
    wall_t = 0.01  # thickness of each slot jaw
    for p in cfg.plates:
        ang = np.radians(p.rack_angle_deg)
        stand = world.add_body(
            name=f"stand_{p.label}",
            pos=[p.rack_radius * np.cos(ang), p.rack_radius * np.sin(ang), 0],
            quat=[np.cos(ang / 2), 0, 0, np.sin(ang / 2)],  # turn about z by ang
        )
        # In the stand's frame, x points away from the robot (the plate's normal).
        stand.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[st.size[0] / 2, st.size[1] / 2, st.height / 2],
                       pos=[0, 0, st.height / 2], rgba=[0.45, 0.33, 0.22, 1])
        for side in (-1, 1):
            stand.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX,
                           size=[wall_t / 2, st.size[1] / 2, st.slot_depth / 2],
                           pos=[side * (st.slot_gap / 2 + wall_t / 2), 0, st.height + st.slot_depth / 2],
                           rgba=[0.3, 0.22, 0.15, 1])

    # --- Wall with a coordinate grid ----------------------------------------
    w = cfg.wall
    tex_path = make_wall_grid(w.width, w.height, OUTPUT_DIR / "wall_grid.png")
    spec.add_texture(name="wall_grid", type=mujoco.mjtTexture.mjTEXTURE_2D, file=str(tex_path))
    mat = spec.add_material(name="wall_grid")
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "wall_grid"

    T_world_wall = wall_pose_in_world(w.surface_x, w.left_y)
    wall = world.add_body(name=f"wall_{w.name}", pos=T_world_wall[:3, 3], quat=quat_wxyz_from_matrix(T_world_wall[:3, :3]))
    # The solid wall, behind the surface (wall z < 0), in wall coordinates.
    wall.add_geom(
        name="wall_solid",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[w.width / 2, w.height / 2, w.thickness / 2],
        pos=[w.width / 2, w.height / 2, -w.thickness / 2],
        rgba=[0.8, 0.8, 0.78, 1],
    )
    # A flat quad just in front of the surface that carries the grid picture.
    # Vertices are in wall coordinates (u, v, out-of-wall), texcoords map the
    # picture 1:1 onto u and v.
    spec.add_mesh(
        name="wall_grid_quad",
        uservert=[0, 0, 0, w.width, 0, 0, w.width, w.height, 0, 0, w.height, 0],
        usertexcoord=[0, 1, 1, 1, 1, 0, 0, 0],
        userface=[0, 1, 2, 0, 2, 3],
        inertia=mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL,
    )
    wall.add_geom(
        name="wall_grid",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname="wall_grid_quad",
        material="wall_grid",
        pos=[0, 0, 0.0002],
        contype=0,
        conaffinity=0,
        density=0,
    )

    # --- Plates -------------------------------------------------------------
    for p in cfg.plates:
        T = plate_in_holder(p.rack_radius, np.radians(p.rack_angle_deg), st.height + 0.0005, p.height)
        body = world.add_body(name=f"plate_{p.label}", pos=T[:3, 3], quat=quat_wxyz_from_matrix(T[:3, :3]))
        body.add_freejoint(name=f"plate_{p.label}_free")
        body.add_geom(
            name=f"plate_{p.label}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[p.width / 2, p.height / 2, p.thickness / 2],
            mass=p.mass * 0.95,
            rgba=p.rgba,
        )
        tx, ty, tz = p.tab_size
        body.add_geom(
            name=f"plate_{p.label}_tab",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[tx / 2, ty / 2, tz / 2],
            pos=[0, 0, -(p.thickness / 2 + tz / 2)],  # on the back face
            mass=p.mass * 0.05,
            rgba=[0.2, 0.2, 0.2, 1],
            condim=4,
            friction=[1.5, 0.05, 0.001],
        )
        # A "magnet": a weld between wall and plate, switched on once the
        # plate is in place. The real attachment isn't modelled, this just
        # keeps the plate where the robot put it.
        spec.add_equality(
            name=f"magnet_{p.label}",
            type=mujoco.mjtEq.mjEQ_WELD,
            objtype=mujoco.mjtObj.mjOBJ_BODY,
            name1=f"wall_{w.name}",
            name2=f"plate_{p.label}",
            active=False,
        )

    # The Menagerie "home" keyframe only knows the arm's 8 joints. Append each
    # plate's starting pose (position + quaternion) so the keyframe matches.
    key = spec.key("home")
    qpos = list(key.qpos)
    for p in cfg.plates:
        b = spec.body(f"plate_{p.label}")
        qpos += list(b.pos) + list(b.quat)
    key.qpos = qpos

    # --- Target marker: a see-through ghost plate the scripts can move ------
    ghost = world.add_body(name="target_ghost", mocap=True, pos=[0, 0, -1])
    p0 = cfg.plates[0]
    ghost.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[p0.width / 2, p0.height / 2, p0.thickness / 2 + 0.0005],
        rgba=[0.1, 0.9, 0.2, 0.35],
        contype=0,
        conaffinity=0,
        group=1,
    )

    # --- A fixed camera for recorded videos ---------------------------------
    eye, target = [-0.45, -0.85, 0.75], [0.3, 0.05, 0.25]
    world.add_camera(name="overview", pos=eye, xyaxes=_look_at_xyaxes(eye, target))
    eye2 = [0.05, -0.05, 0.55]
    world.add_camera(name="wall_view", pos=eye2, xyaxes=_look_at_xyaxes(eye2, [w.surface_x, 0.0, 0.3]))
    return spec


def build_model(cfg: SceneConfig | None = None) -> tuple[mujoco.MjModel, mujoco.MjData]:
    cfg = cfg or SceneConfig()
    model = build_spec(cfg).compile()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)
    return model, data
