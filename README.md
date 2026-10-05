# Piper MuJoCo starter

Phase 0 of the wall demo: an AgileX Piper arm picks a labelled A4 metal plate out of a holder and places it on a wall at a requested position and angle, all in MuJoCo on your Mac.

```
uv run mjpython examples/05_pick_and_place.py --plate B --u 0.5 --v 0.3 --theta 15
...
target   u=0.500 v=0.300 theta=15.0
achieved u=0.501 v=0.301 theta=15.0
error    2.0 mm, 0.00 deg  ->  SUCCESS
```

The point of this repo is to learn the robotics (coordinate frames, inverse kinematics, grasping, a task sequence) somewhere fast and free, before Isaac Sim. The pose math in `poses.py` is copied unchanged into the Isaac Sim project.

---

## 1. Set up (once)

You need `uv` and `git` (both installed on your Mac already).

```bash
git clone <this repo> piper-mujoco-starter
cd piper-mujoco-starter
uv sync                                  # creates .venv and installs MuJoCo, mink, etc.
uv run python scripts/fetch_piper.py     # downloads the Piper model (~30 MB) into third_party/
uv run pytest                            # 12 checks, ~10 s; all should pass
```

## 2. Run the examples in order

On macOS, use `mjpython` instead of `python` whenever a window opens (it ships with MuJoCo; macOS requires the viewer to run on the main thread). Add `--headless` to skip the window, or `--video outputs/name.mp4` to record one.

| Step | Command | What you learn |
|---|---|---|
| 1 | `uv run mjpython examples/01_view_robot.py` | The scene, the viewer, what actuators exist |
| 2 | `uv run mjpython examples/02_move_joints.py` | Moving joints directly by writing targets |
| 3 | `uv run mjpython examples/03_move_to_pose.py --xyz 0.35 0.1 0.3 --rpy 0 90 0` | Inverse kinematics: pose in, joint angles out |
| 4 | `uv run python examples/04_reach_map.py --theta 0` | Where on the wall the arm can actually reach |
| 5 | `uv run mjpython examples/05_pick_and_place.py --plate A --u 0.35 --v 0.25 --theta -20` | The full milestone |

**Viewer controls:** left-drag rotates, right-drag pans, scroll zooms. Double-click a body to select it, then Ctrl + right-drag pushes it. Press Tab for the side panel; under *Rendering → Group enable*, turn on group 3 to see the collision shapes.

**Good wall targets:** `u` 0.25 to 0.75 and `v` 0.2 to 0.4 (metres). Step 4 prints the exact map. The arm is fixed in place and only reaches part of the wall; the mobile base comes later.

---

## 3. What's in the repo

```
piper-mujoco-starter/
├── pyproject.toml          dependencies; `uv sync` reads this
├── scripts/fetch_piper.py  downloads the Piper model from MuJoCo Menagerie
├── third_party/            the downloaded model (not committed to git)
├── outputs/                generated files: wall_grid.png, videos (not committed)
├── examples/               the five numbered scripts above, the things you run
├── src/piper_starter/      the library the examples use (read these in this order):
│   ├── config.py           every size and position in the scene
│   ├── poses.py            pose math: wall coords -> plate pose -> gripper pose
│   ├── wall_texture.py     draws the coordinate grid picture for the wall
│   ├── scene.py            builds the MuJoCo world
│   ├── ik.py               inverse kinematics (mink)
│   ├── robot.py            moving the arm: joint moves, straight-line moves, gripper
│   ├── task.py             the pick-and-place sequence
│   └── runtime.py          viewer / headless / video plumbing (safe to skip)
└── tests/                  `uv run pytest`: pose math checks + one full run
```

### `config.py`: the numbers
Plate (0.297 × 0.210 × 0.002 m, 0.2 kg), wall (1.0 × 0.8 m, surface 0.55 m in front of the robot), the two plate stands, and the **TCP** (tool centre point, the point between the fingertips that "the gripper pose" refers to). Change numbers here, not in the other files.

### `poses.py`: the most important file
A **pose** is a position plus an orientation, stored as a 4×4 matrix `T`. `T_a_b` means "frame b, as seen from frame a". Chaining is multiplication: `T_world_tcp = T_world_plate @ T_plate_tcp`. The frames:

- **world**: robot base at the origin, x forward, y left, z up.
- **wall**: origin at the wall's bottom-left corner (as seen from the robot); x = u (right), y = v (up), z points out of the wall. This is the grid painted on the wall.
- **plate**: origin at the plate centre; x along its width, y along its height, z out of its front face.
- **tcp**: the gripper; z is the direction it points ("approach"), y is the direction the fingers close.

The whole task is this chain:

```
T_world_target_plate = T_world_wall @ T_wall_plate(u, v, theta)
T_world_gripper      = T_world_target_plate @ T_plate_tcp
```

Getting signs and axes right here is most of the work in any manipulation project. `tests/test_poses.py` pins the conventions down.

### `scene.py`: building the world
MuJoCo worlds are described in XML (**MJCF**). The Piper comes as an MJCF file from Google DeepMind's MuJoCo Menagerie; `scene.py` loads it and adds everything else with **MjSpec** (MuJoCo's Python API for editing models): the stands and slotted holders, the plates with their grab tabs, the wall with the grid texture, a see-through green "ghost" that marks the target, two cameras, and a weld constraint per plate that acts as a magnet. It also stiffens the wrist and gripper motors (see "Why it's built this way").

### `ik.py`: inverse kinematics
Forward kinematics is easy (joint angles → where the gripper is). IK is the reverse and has no simple formula, so **mink** solves it iteratively: each step finds the small joint change that best reduces the pose error within joint limits. If it gets stuck against a joint limit, it retries from random starting angles.

### `robot.py`: moving the arm
Each Piper joint has a **position actuator**: write the angle you want into `data.ctrl` and a built-in spring-damper drives the joint there. On top of that:

- `move_joints`: all joints interpolate together. Fast and safe in open space, but the gripper's path is curved.
- `move_linear`: the gripper moves along a straight line. Used near objects (approaching the tab, sliding onto the wall).
- `settle`: after each move, measures how far each joint stopped short (gravity, friction) and aims past by that much. It's the same idea as the integral term of a PID controller.

### `task.py`: the milestone
1. **Locate** the plate (read its pose from the simulator; a camera would do this on the real robot).
2. **Plan**: check that every key pose is reachable before moving, and choose which way up to hold the tab.
3. **Pre-grasp, grasp**: hover 8 cm behind the tab, slide in, close.
4. **Lift, retreat**, then measure how the plate actually sits in the hand.
5. **Pre-place**: hold the plate 8 cm out from its target, measure the error, nudge until it lines up.
6. **Place**: slide straight in until it touches the wall.
7. **Attach and release**: switch on the magnet, open, back off, go home. It then reports the error.

---

## 4. Why it's built this way

These came out of making it actually work. Each is a real robotics lesson.

- **The plate has a small tab on its back.** If the gripper pinches the plate's edge, the gripper body ends up between the plate and the wall, so the plate can't go flat against the wall. A tab (like a handle) lets the gripper hold the plate from behind and press it flat. A real system would use a tab, a suction cup or a magnet tool.
- **Plates start upright in holders that face the robot.** The Piper's wrist only bends about ±70°, so it can't point straight down at things near its base, and it reaches best when it approaches straight out from its base. A thin plate lying flat on a table also can't be pinched by a parallel gripper.
- **The motors are stiffer than the Menagerie model.** With the original gains the wrist stopped about 6 mm short and the gripper squeezed with only 0.1 N, so the plate slipped out. `scene.py` explains the numbers.
- **The magnet is a weld constraint switched on at the end.** Attachment isn't modelled for this demo; the weld just holds the plate where the robot put it.

## 5. Things to try

1. Change `theta` and watch how the wrist has to roll. Find a target that's out of reach and read the error message.
2. In `config.py`, move the wall to `surface_x = 0.45`, rerun step 4, and see how the reachable area changes.
3. In `task.py`, delete the pre-place correction loop and compare the final error.
4. In `scene.py`, set the gripper `kp` back to 40 and watch the grasp fail.
5. Add a third plate "C" in `config.py` (copy a line, pick a `rack_angle_deg`).

## 6. Troubleshooting

- **`launch_passive` requires mjpython on macOS**: you ran `python` instead of `mjpython` for a script that opens a window.
- **`third_party/agilex_piper/scene.xml is missing`**: run `uv run python scripts/fetch_piper.py`.
- **"out of the arm's reach"**: pick a target inside the `#` area of step 4's map.

## Credits
Piper model: [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/agilex_piper) (MIT). IK: [mink](https://github.com/kevinzakka/mink).
