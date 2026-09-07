from __future__ import annotations

import csv
import io
from pathlib import Path

import pytest
from PIL import Image

from frame_quorum import FrameRate, Shot, render_edl, render_scene_timecodes
from frame_quorum.cli import main
from frame_quorum.errors import ConfigurationError


def test_timing_csv_retains_exact_fraction_and_exclusive_bounds() -> None:
    text = render_scene_timecodes((Shot(0, 0, 30), Shot(1, 30, 60)), FrameRate(30000, 1001), drop_frame=True)
    rows = list(csv.DictReader(io.StringIO(text)))
    assert rows[0]["start_frame"] == "0"
    assert rows[0]["end_frame_exclusive"] == "30"
    assert rows[0]["end_seconds"] == "1001/1000"
    assert rows[1]["start_seconds"] == "1001/1000"
    assert rows[1]["start_timestamp"] == "00:00:01.001"
    assert rows[1]["start_timecode"] == "00:00:01;00"


def test_edl_assembles_disjoint_source_scenes_without_gaps() -> None:
    text = render_edl(
        (Shot(0, 100, 150), Shot(1, 250, 275)), FrameRate(25), reel="SOURCE", title="Cuts", record_start=90000
    )
    assert text.startswith("TITLE: Cuts\nFCM: NON-DROP FRAME\n")
    events = [line.split() for line in text.splitlines() if line[:3].isdigit()]
    assert events == [
        ["001", "SOURCE", "V", "C", "00:00:04:00", "00:00:06:00", "01:00:00:00", "01:00:02:00"],
        ["002", "SOURCE", "V", "C", "00:00:10:00", "00:00:11:00", "01:00:02:00", "01:00:03:00"],
    ]


def test_drop_frame_edl_and_source_start() -> None:
    text = render_edl((Shot(0, 0, 30),), FrameRate(30000, 1001), drop_frame=True, source_start=1800)
    assert "FCM: DROP FRAME" in text
    assert "00:01:00;02 00:01:01;02 00:00:00;00 00:00:01;00" in text


@pytest.mark.parametrize(
    "scenes",
    [
        [],
        ["x"],
        [Shot(1, 0, 1)],
        [Shot(0, 1, 1)],
        [Shot(True, 0, 1)],
        [Shot(0, -1, 1)],
        [Shot(0, 0, 10), Shot(1, 9, 20)],
        "bad",
        1,
    ],
)
def test_invalid_scene_structure(scenes: object) -> None:
    with pytest.raises(ConfigurationError):
        render_edl(scenes, FrameRate(25))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "options",
    [
        {"reel": "TOO_LONG9"},
        {"reel": "a\nb"},
        {"title": "x\nFCM: DROP FRAME"},
        {"title": ""},
        {"record_start": 25 * 3600 * 24},
        {"source_start": 25 * 3600 * 24 - 1},
    ],
)
def test_invalid_editor_metadata_and_day_wrap(options: dict) -> None:
    with pytest.raises(ConfigurationError):
        render_edl((Shot(0, 0, 25),), FrameRate(25), **options)


def test_edl_limits_are_checked_on_streamed_scenes() -> None:
    with pytest.raises(ConfigurationError, match="999"):
        render_edl((Shot(i, i, i + 1) for i in range(1000)), FrameRate(25))
    with pytest.raises(ConfigurationError, match="nominal"):
        render_edl((Shot(0, 0, 25),), FrameRate(60))


def test_cli_exports_timecodes_and_edl_with_consistent_rate(tmp_path: Path) -> None:
    source = tmp_path / "input"
    source.mkdir()
    for i, shade in enumerate([0, 0, 255, 255]):
        Image.new("RGB", (8, 8), (shade, shade, shade)).save(source / f"{i}.png")
    output = tmp_path / "output"
    args = [
        "scenes",
        str(source),
        "-o",
        str(output),
        "--detector",
        "luminance",
        "--timecode-rate",
        "30000/1001",
        "--drop-frame",
        "--edl",
    ]
    assert main(args) == 0
    assert (output / "scenes.edl").is_file()
    rows = list(csv.DictReader(io.StringIO((output / "timecodes.csv").read_text())))
    assert len(rows) == 2 and rows[0]["end_frame_exclusive"] == "2"
    stats = list(csv.DictReader(io.StringIO((output / "statistics.csv").read_text())))
    assert float(stats[1]["timestamp"]) == pytest.approx(1001 / 30000)
    assert main([*args, "--timestamp-mode", "mtime"]) == 2
    assert main(["scenes", str(source), "-o", str(output), "--edl"]) == 2
