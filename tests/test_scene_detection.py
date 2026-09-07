from __future__ import annotations

import csv
import io
import json
import runpy
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image

import frame_quorum.cli as cli_module
from frame_quorum import DetectionConfig, Frame, FrameMetrics, detect_scenes, render_detection_csv
from frame_quorum.cli import main
from frame_quorum.errors import ConfigurationError, OutputError


def frames(values: Sequence[float], *, start: int = 0) -> tuple[Frame, ...]:
    # Flat grayscale frames: structural distance is zero; full black-to-white
    # content distance is 0.25 (RGB) + 0.10 (luminance) = 0.35.
    return tuple(
        Frame(
            index + start,
            Path(f"{index}.png"),
            f"{index}.png",
            index / 25,
            8,
            8,
            20,
            FrameMetrics(0, value, 0, 0, 0, value, value, value),
        )
        for index, value in enumerate(values)
    )


def test_adaptive_finds_abrupt_edit_against_smooth_background_motion() -> None:
    sequence = frames([0, 0.05, 0.1, 0.15, 0.8, 0.85, 0.9, 0.95, 1])
    result = detect_scenes(sequence, DetectionConfig(window_radius=2))
    assert result.cut_indices == (4,)
    assert result.statistics[4].content_score == pytest.approx(0.2275)
    assert result.statistics[4].detector_score == pytest.approx(13)
    assert result.statistics[4].reason == "adaptive_peak"
    assert [scene.frame_count for scene in result.scenes] == [4, 5]
    assert result.statistics[2].reason == "incomplete_window"
    assert result.statistics[-2].detector_score is None
    assert detect_scenes(sequence, DetectionConfig(window_radius=2)) == result


def test_adaptive_zero_baseline_is_finite_and_small_content_floor_suppresses_noise() -> None:
    result = detect_scenes(frames([0] * 5 + [1] * 5))
    assert result.cut_indices == (5,)
    assert result.statistics[5].detector_score == 1_000_000
    assert detect_scenes(frames([0] * 5 + [0.01] * 5)).cut_indices == ()
    assert detect_scenes(frames([0] * 10)).cut_indices == ()
    json.dumps(result.serializable(), allow_nan=False)


def test_adaptive_window_never_invents_missing_neighbor_evidence() -> None:
    assert detect_scenes(frames([0, 1])).cut_indices == ()
    result = detect_scenes(frames([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]))
    assert result.cut_indices == ()
    assert result.statistics[3].detector_score == pytest.approx(1)
    # Two adjacent flash edges each contribute to the other's radius-one mean;
    # neither is a ratio-three isolated cut.
    flash = detect_scenes(frames([0, 0, 0, 1, 0, 0, 0]), DetectionConfig(window_radius=1))
    assert flash.cut_indices == ()
    assert flash.statistics[3].detector_score == pytest.approx(2)


@pytest.mark.parametrize("detector", ["content", "color", "luminance"])
def test_distance_modes_share_scene_minimum_policy(detector: str) -> None:
    config = DetectionConfig(detector=detector, threshold=0.3, min_scene_frames=2)  # type: ignore[arg-type]
    result = detect_scenes(frames([0, 0, 1, 1, 0, 0], start=10), config)
    assert result.cut_indices == (12, 14)
    assert [(scene.start_index, scene.end_index) for scene in result.scenes] == [(10, 12), (12, 14), (14, 16)]
    assert sum(scene.frame_count for scene in result.scenes) == 6
    assert result.statistics[2].reason == "distance_threshold"


def test_minimum_scene_length_includes_first_and_final_scene() -> None:
    config = DetectionConfig(detector="luminance", threshold=0.5, min_scene_frames=2)
    result = detect_scenes(frames([0, 1, 1, 1, 0]), config)
    assert result.cut_indices == ()
    assert result.statistics[1].reason == "short_previous_scene"
    assert result.statistics[4].reason == "short_final_scene"
    assert result.statistics[4].candidate and not result.statistics[4].accepted
    assert len(detect_scenes(frames([0]), config).scenes) == 1


@pytest.mark.parametrize(("bias", "expected"), [(-1, 2), (0, 4), (1, 6), (-0.5, 3)])
def test_fade_bias_places_cut_inside_completed_dark_interval(bias: float, expected: int) -> None:
    result = detect_scenes(
        frames([1, 1, 0, 0, 0, 0, 1, 1]),
        DetectionConfig(detector="threshold", fade_bias=bias),
    )
    assert result.cut_indices == (expected,)
    assert result.statistics[expected].reason == "completed_fade"


def test_fade_hysteresis_requires_bright_return_and_counts_actual_dark_samples() -> None:
    result = detect_scenes(
        frames([1, 0.04, 0.06, 0.07, 0.04, 0.1, 1]),
        DetectionConfig(detector="threshold", dark_threshold=0.05, hysteresis=0.03),
    )
    assert result.cut_indices == (3,)  # dark interval [1, 5), midpoint 3
    assert result.statistics[3].detector_score == 0.07
    no_fade = detect_scenes(
        frames([1, 0.04, 0.06, 0.07, 0.1, 1]),
        DetectionConfig(detector="threshold", dark_threshold=0.05, hysteresis=0.03),
    )
    assert no_fade.cut_indices == ()  # only one sample actually reached darkness


def test_threshold_rejects_single_dark_flash_and_initial_darkness() -> None:
    config = DetectionConfig(detector="threshold")
    assert detect_scenes(frames([1, 1, 0, 1, 1]), config).cut_indices == ()
    assert detect_scenes(frames([0, 0, 1, 1]), config).cut_indices == ()
    assert detect_scenes(frames([0, 0, 0, 0]), config).cut_indices == ()
    assert detect_scenes(frames([1, 0.06, 1]), config).cut_indices == ()
    assert detect_scenes(frames([1, 0.05, 0.05, 0.07, 1]), config).cut_indices == (2,)


def test_final_fade_is_opt_in_and_repeated_fades_are_independent() -> None:
    config = DetectionConfig(detector="threshold")
    sequence = frames([1, 1, 0, 0])
    assert detect_scenes(sequence, config).cut_indices == ()
    result = detect_scenes(sequence, replace(config, include_final_fade=True))
    assert result.cut_indices == (2,)
    assert result.statistics[2].reason == "final_fade"
    assert detect_scenes(frames([1, 0, 0, 1, 1, 0, 0, 1]), config).cut_indices == (2, 6)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("detector", "magic"),
        ("detector", []),
        ("threshold", True),
        ("threshold", float("nan")),
        ("threshold", float("inf")),
        ("threshold", "x"),
        ("threshold", -1),
        ("threshold", 10**1000),
        ("adaptive_ratio", 0.5),
        ("adaptive_ratio", 1_000_001),
        ("min_content", 2),
        ("dark_threshold", 0.99),
        ("hysteresis", 1),
        ("fade_bias", -2),
        ("min_scene_frames", 0),
        ("min_dark_frames", False),
        ("window_radius", 10_001),
        ("window_radius", 1.5),
        ("max_frames", 1_000_001),
        ("include_final_fade", 1),
    ],
)
def test_detector_config_rejects_invalid_values(name: str, value: object) -> None:
    with pytest.raises(ConfigurationError):
        detect_scenes(frames([0, 1]), DetectionConfig(**{name: value}))  # type: ignore[arg-type]


def test_detector_rejects_bad_sequence_before_arithmetic() -> None:
    sequence = frames([0, 1])
    invalid = (
        (),
        (object(),),
        (replace(sequence[0], metrics="x"),),
        (sequence[0], replace(sequence[1], index=0)),
        (sequence[0], replace(sequence[1], timestamp=None)),
        (sequence[0], replace(sequence[1], timestamp=-1)),
        (replace(sequence[0], index=(1 << 63) - 1),),
    )
    for values in invalid:
        with pytest.raises(ConfigurationError):
            detect_scenes(values)  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError):
        detect_scenes(sequence, object())  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError, match="max_frames"):
        detect_scenes(sequence, DetectionConfig(max_frames=1))
    untimed = tuple(replace(frame, timestamp=None) for frame in sequence)
    assert detect_scenes(untimed).statistics[0].timestamp is None


def test_statistics_csv_round_trip_preserves_zero_and_absent_scores() -> None:
    result = detect_scenes(frames([0] * 5 + [1] * 5))
    rows = list(csv.DictReader(io.StringIO(render_detection_csv(result))))
    assert len(rows) == 10
    assert rows[0]["frame_index"] == "0"
    assert rows[0]["timestamp"] == "0.0"
    assert rows[0]["detector_score"] == ""
    assert rows[5]["accepted"] == "true"
    assert rows[5]["candidate"] == "true"
    assert rows[5]["reason"] == "adaptive_peak"
    assert result.serializable()["scenes"][0]["end_index"] == 5
    with pytest.raises(ConfigurationError):
        render_detection_csv(None)  # type: ignore[arg-type]


def test_scene_cli_decodes_real_fade_and_publishes_json_csv_bundle(tmp_path: Path) -> None:
    source = tmp_path / "frames"
    source.mkdir()
    for index, value in enumerate([255, 255, 0, 0, 255, 255]):
        Image.new("RGB", (16, 16), (value,) * 3).save(source / f"{index}.png")
    output = tmp_path / "results"
    assert (
        main(["scenes", str(source), "--detector", "threshold", "-o", str(output), "--frame-rate", "25"]) == 0
    )
    payload = json.loads((output / "scenes.json").read_text(encoding="utf-8"))
    assert payload["cut_indices"] == [3]
    assert payload["statistics"][3]["timestamp"] == 0.12
    rows = list(csv.DictReader((output / "statistics.csv").read_text().splitlines()))
    assert rows[3]["accepted"] == "true"
    assert main(["scenes", str(source), "-o", str(source / "inside")]) == 2
    assert main(["scenes", str(source), "-o", str(output), "--window-radius", "0"]) == 2


def test_scene_cli_staging_failure_preserves_old_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "frames"
    source.mkdir()
    Image.new("RGB", (8, 8)).save(source / "0.png")
    output = tmp_path / "output"
    output.mkdir()
    for name in ("scenes.json", "statistics.csv"):
        (output / name).write_text("old", encoding="utf-8")

    def fail_export(_: object) -> str:
        raise OutputError("simulated CSV export failure")

    monkeypatch.setattr(cli_module, "render_detection_csv", fail_export)
    assert main(["scenes", str(source), "-o", str(output)]) == 2
    assert (output / "scenes.json").read_text() == "old"
    assert (output / "statistics.csv").read_text() == "old"
    assert sorted(path.name for path in output.iterdir()) == ["scenes.json", "statistics.csv"]


def test_detection_example_runs_without_network_or_repository_writes() -> None:
    runpy.run_path(str(Path(__file__).parents[1] / "examples" / "detect_scenes.py"), run_name="__main__")
