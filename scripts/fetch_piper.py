"""Download the AgileX Piper model from Google DeepMind's MuJoCo Menagerie.

Puts it in third_party/agilex_piper (about 30 MB, ignored by git). Only that
one folder is fetched, not the whole Menagerie. Needs git.

    uv run python scripts/fetch_piper.py
"""

import shutil
import subprocess
import tempfile
from pathlib import Path

REPO = "https://github.com/google-deepmind/mujoco_menagerie.git"
FOLDER = "agilex_piper"
DEST = Path(__file__).resolve().parents[1] / "third_party" / FOLDER


def main():
    if (DEST / "piper.xml").exists():
        print(f"already there: {DEST}")
        return
    with tempfile.TemporaryDirectory() as tmp:
        run = lambda *cmd: subprocess.run(cmd, cwd=tmp, check=True)  # noqa: E731
        run("git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", REPO, "menagerie")
        subprocess.run(["git", "sparse-checkout", "set", FOLDER], cwd=Path(tmp) / "menagerie", check=True)
        DEST.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(Path(tmp) / "menagerie" / FOLDER, DEST)
    print(f"saved to {DEST}")


if __name__ == "__main__":
    main()
