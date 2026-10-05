"""Every size and position in the scene, in one place.

Units are SI throughout: metres, kilograms, radians (degrees only at the
command line). The robot base sits at the world origin and faces +x.
"""

from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PIPER_DIR = REPO_ROOT / "third_party" / "agilex_piper"
OUTPUT_DIR = REPO_ROOT / "outputs"


@dataclass
class PlateConfig:
    """An A4 sheet of light metal with a small grab tab on its back face."""

    label: str
    width: float = 0.297  # along the plate's x axis
    height: float = 0.210  # along the plate's y axis
    thickness: float = 0.002  # along the plate's z axis (z = front face normal)
    mass: float = 0.2
    rgba: tuple = (0.75, 0.78, 0.82, 1.0)
    # Where the plate starts: standing upright in a slotted holder on a stand,
    # front face pointing away from the robot. The stand sits rack_radius
    # from the robot base, at rack_angle_deg (0 = straight ahead, + = left).
    # The Piper's wrist only bends +-70 degrees, so it reaches best when it
    # approaches straight out from its base; that's why holders face the base.
    rack_radius: float = 0.35
    rack_angle_deg: float = 60.0
    # The tab the gripper pinches: thin along plate y, sticking out of the back.
    tab_size: tuple = (0.04, 0.003, 0.03)  # (x, y, z) full sizes


@dataclass
class WallConfig:
    """A wall in front of the robot. Its surface faces the robot (-x)."""

    name: str = "W"
    surface_x: float = 0.55  # world x of the wall's front surface
    width: float = 1.0
    height: float = 0.8
    thickness: float = 0.05
    # World y of the wall's left edge as seen from the robot (u = 0 there).
    left_y: float = 0.5


@dataclass
class StandConfig:
    """The pedestal and slotted holder each plate starts in."""

    height: float = 0.12  # top of the pedestal
    size: tuple = (0.10, 0.34)  # depth (towards the robot), width
    slot_depth: float = 0.025  # how far the plate sits down into the slot
    slot_gap: float = 0.003  # slot width; the plate is 0.002 thick


@dataclass
class SceneConfig:
    plates: list = field(
        default_factory=lambda: [
            PlateConfig(label="A", rack_angle_deg=60.0, rgba=(0.80, 0.55, 0.25, 1.0)),
            PlateConfig(label="B", rack_angle_deg=-60.0, rgba=(0.55, 0.65, 0.85, 1.0)),
        ]
    )
    wall: WallConfig = field(default_factory=WallConfig)
    stand: StandConfig = field(default_factory=StandConfig)

    # Tool centre point: a point between the fingertip pads, in the link6 frame.
    # link6's z axis points out of the gripper (the "approach" direction) and
    # its y axis is the direction the fingers close along.
    tcp_offset: tuple = (0.0, 0.0, 0.12)

    # How far beyond the TCP the plate's back face sits once grasped.
    # The pads end 0.015 m past the TCP, so 0.02 leaves 5 mm of clearance.
    grasp_back_face_gap: float = 0.02

    def plate(self, label: str) -> PlateConfig:
        for p in self.plates:
            if p.label == label:
                return p
        raise KeyError(f"No plate labelled {label!r}. Known: {[p.label for p in self.plates]}")
