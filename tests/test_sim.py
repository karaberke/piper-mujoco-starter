"""End-to-end check: the robot really places a plate. Takes ~10 s. Run with: uv run pytest"""

from piper_starter.robot import PiperController
from piper_starter.scene import build_model
from piper_starter.task import pick_and_place


def test_pick_and_place_hits_target():
    model, data = build_model()
    result = pick_and_place(PiperController(model, data), "B", 0.5, 0.3, 15.0)
    assert result.success, result
