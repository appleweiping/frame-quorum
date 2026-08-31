"""Generate the exact demo assets shown in the project README."""

from pathlib import Path

from frame_quorum.cli import main

HERE = Path(__file__).resolve().parent

raise SystemExit(main(["demo", "--output-dir", str(HERE / "output"), "--force"]))
