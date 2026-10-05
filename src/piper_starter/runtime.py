"""Run a script either in the interactive viewer or headless (optionally recording a video).

On macOS the interactive viewer needs the `mjpython` launcher that ships
with MuJoCo (it runs the viewer on the main thread, which macOS requires):

    uv run mjpython examples/05_pick_and_place.py

On Linux/Windows plain `python` works. `--headless` needs neither.
"""

import argparse
import contextlib
import time
from pathlib import Path

import mujoco

from .config import OUTPUT_DIR


def add_common_args(parser: argparse.ArgumentParser):
    parser.add_argument("--headless", action="store_true", help="no window; run as fast as possible")
    parser.add_argument("--video", type=Path, help="record an mp4 (implies --headless), e.g. outputs/demo.mp4")
    parser.add_argument("--camera", default="overview", help="camera for the video: overview or wall_view")


class _Recorder:
    def __init__(self, model, data, path: Path, camera: str, fps: int = 30):
        import imageio.v2 as imageio  # only needed for videos

        path.parent.mkdir(parents=True, exist_ok=True)
        self.model, self.data, self.camera = model, data, camera
        self.renderer = mujoco.Renderer(model, 720, 1280)
        self.writer = imageio.get_writer(str(path), fps=fps, macro_block_size=8)
        self.every = max(1, round(1 / (fps * model.opt.timestep)))
        self.path = path
        self.n = 0

    def __call__(self):
        self.n += 1
        if self.n % self.every == 0:
            self.renderer.update_scene(self.data, camera=self.camera)
            self.writer.append_data(self.renderer.render())

    def close(self):
        self.writer.close()
        print(f"saved video: {self.path}")


class _RealTimeViewer:
    """Calls viewer.sync() ~60 times a second and sleeps so the sim runs at real speed."""

    def __init__(self, model, data, viewer):
        self.model, self.data, self.viewer = model, data, viewer
        self.every = max(1, round(1 / (60 * model.opt.timestep)))
        self.n = 0
        self.t0_wall = time.perf_counter()
        self.t0_sim = data.time

    def __call__(self):
        self.n += 1
        if self.n % self.every:
            return
        if not self.viewer.is_running():
            raise SystemExit("viewer closed")
        self.viewer.sync()
        ahead = (self.data.time - self.t0_sim) - (time.perf_counter() - self.t0_wall)
        if ahead > 0:
            time.sleep(ahead)


@contextlib.contextmanager
def session(model: mujoco.MjModel, data: mujoco.MjData, args):
    """Yields an on_step callback; keeps the window open at the end until you close it."""
    if args.video:
        rec = _Recorder(model, data, args.video, args.camera)
        try:
            yield rec
        finally:
            rec.close()
    elif args.headless:
        yield lambda: None
    else:
        import mujoco.viewer

        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.azimuth, viewer.cam.elevation, viewer.cam.distance = 210, -25, 1.6
            viewer.cam.lookat[:] = [0.3, 0.0, 0.25]
            yield _RealTimeViewer(model, data, viewer)
            print("done; close the viewer window to exit")
            while viewer.is_running():
                viewer.sync()
                time.sleep(0.05)


def default_video_path(name: str) -> Path:
    return OUTPUT_DIR / f"{name}.mp4"
