from __future__ import annotations

import json
import runpy
from fractions import Fraction
from pathlib import Path

import pytest
from PIL import Image

from frame_quorum import (
    DetectionConfig,
    NativeSceneConfig,
    NativeVideoConfig,
    NativeVideoStatus,
    detect_native_scenes,
)
from frame_quorum.cli import main
from frame_quorum.errors import ScanError
from frame_quorum.metrics import measure_image

av = pytest.importorskip("av", reason="install frame-quorum[video] for genuine native scene tests")
PTS = (5000, 5040, 5110, 5180, 5270, 5310, 5470, 5680)


def video(path, values, pts=PTS):
    with av.open(str(path), "w", format="matroska") as container:
        stream = container.add_stream("ffv1", rate=25)
        stream.width, stream.height, stream.pix_fmt = 16, 12, "bgr0"
        stream.time_base = stream.codec_context.time_base = Fraction(1, 1000)
        for value, timestamp in zip(values, pts, strict=True):
            with Image.new("RGB", (16, 12), (value, value, value)) as image:
                frame = av.VideoFrame.from_image(image)
            frame.pts, frame.time_base = timestamp, Fraction(1, 1000)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


@pytest.mark.parametrize("detector", ["content", "color", "luminance", "adaptive"])
def test_real_lossless_vfr_manual_cut_and_independent_decoded_measurements(tmp_path, detector):
    path = tmp_path / "cut.mkv"
    video(path, [0] * 4 + [255] * 4)
    result = detect_native_scenes(path, NativeSceneConfig(detectors=(DetectionConfig(detector=detector),)))
    assert result.cut_positions == (4,)
    assert result.cut_times == (Fraction(527, 100),)
    assert result.statistics[4].content_score == pytest.approx(0.35)
    assert result.statistics[4].detectors[0].score == pytest.approx(
        {"content": 0.35, "color": 1, "luminance": 1, "adaptive": 1_000_000}[detector]
    )
    assert [scene.sample_count for scene in result.scenes] == [4, 4]
    assert result.scenes[0].end_time == Fraction(527, 100)
    assert result.scenes[-1].last_sample_time == Fraction(142, 25)
    assert result.scenes[-1].end_time is None
    assert result.diagnostics.status is NativeVideoStatus.EOF and result.diagnostics.closed
    with av.open(str(path)) as direct:
        independent = []
        for decoded in direct.decode(video=0):
            with decoded.to_image() as image:
                independent.append((decoded.pts * decoded.time_base, measure_image(image)))
    assert independent == [(row.sample.presentation_time, row.sample.metrics) for row in result.statistics]
    # Native ownership is released by the public operation, not by GC/finalizers.
    path.unlink()


@pytest.mark.parametrize("bias, expected", [(-1, 2), (0, 4), (1, 6)])
def test_real_fade_cut_is_before_selected_pts_not_confirmation_or_interpolated_time(tmp_path, bias, expected):
    path = tmp_path / "fade.mkv"
    video(path, [255, 255, 0, 0, 0, 0, 255, 255])
    result = detect_native_scenes(
        path, NativeSceneConfig(detectors=(DetectionConfig(detector="threshold", fade_bias=bias),))
    )
    assert result.cut_positions == (expected,)
    assert result.cut_times == (Fraction(PTS[expected], 1000),)
    (evidence,) = result.statistics[expected].detectors
    assert evidence.reason == "completed_fade"
    assert evidence.score == (1 if bias == 1 else 0)


def test_real_sampling_counts_supplied_samples_and_keeps_decode_mapping(tmp_path):
    path = tmp_path / "sampled.mkv"
    video(path, [0] * 4 + [255] * 4)
    result = detect_native_scenes(
        path,
        NativeSceneConfig(
            video=NativeVideoConfig(start=Fraction(126, 25), end=Fraction(28, 5), frame_step=2),
            detectors=(DetectionConfig(detector="luminance", threshold=0.5),),
        ),
    )
    assert [row.sample.pts for row in result.statistics] == [5040, 5180, 5310]
    indices = [row.sample.decode_index for row in result.statistics]
    # A backward native seek may begin at an earlier keyframe. Only the
    # generation-local stride, not a guessed global/index-zero origin, is fixed.
    assert [index - indices[0] for index in indices] == [0, 2, 4]
    assert result.cut_positions == (2,) and result.cut_times == (Fraction(531, 100),)
    assert [scene.sample_count for scene in result.scenes] == [2, 1]
    assert result.scenes[-1].end_time == Fraction(28, 5)
    assert result.scenes[-1].end_reason == "requested_end"
    assert result.diagnostics.status is NativeVideoStatus.RANGE_END
    assert result.diagnostics.decoded_frames == indices[-1] + 3  # Stride skip plus end-boundary sample.


@pytest.mark.parametrize("count_kind", ["frame", "decode"])
def test_real_count_limit_stops_without_probing_eof_or_claiming_end(tmp_path, count_kind):
    path = tmp_path / "limited.mkv"
    video(path, [0] * 4 + [255] * 4)
    config = NativeVideoConfig(
        max_frames=4 if count_kind == "frame" else 100,
        max_decoded_frames=4 if count_kind == "decode" else 100,
        end=Fraction(6),
    )
    result = detect_native_scenes(path, NativeSceneConfig(video=config))
    assert result.diagnostics.status.value == f"{count_kind}_limit"
    assert result.diagnostics.returned_frames == result.diagnostics.decoded_frames == 4
    assert len(result.statistics) == 4 and result.cut_positions == ()
    assert result.scenes[-1].end_time is None
    path.unlink()


def test_real_incomplete_fade_policy_is_explicitly_about_supplied_tail(tmp_path):
    path = tmp_path / "incomplete.mkv"
    video(path, [255, 255, 0, 0, 0, 0, 255, 255])
    video_config = NativeVideoConfig(max_frames=4)
    excluded = detect_native_scenes(
        path,
        NativeSceneConfig(
            video=video_config,
            detectors=(DetectionConfig(detector="threshold"),),
        ),
    )
    included = detect_native_scenes(
        path,
        NativeSceneConfig(
            video=video_config,
            detectors=(DetectionConfig(detector="threshold", include_final_fade=True),),
        ),
    )
    assert excluded.cut_positions == ()
    assert included.cut_positions == (2,)
    assert included.statistics[2].detectors[0].reason == "final_fade"
    assert included.scenes[-1].end_time is None
    assert included.diagnostics.status is NativeVideoStatus.FRAME_LIMIT


@pytest.mark.parametrize("bound, expected", [(Fraction(6), "eof"), (Fraction(4), "range_end")])
def test_real_empty_interval_returns_no_fabricated_scene(tmp_path, bound, expected):
    path = tmp_path / "empty-range.mkv"
    video(path, [0] * 4 + [255] * 4)
    config = NativeVideoConfig(start=bound) if expected == "eof" else NativeVideoConfig(end=bound)
    result = detect_native_scenes(path, NativeSceneConfig(video=config))
    assert not result.statistics and not result.scenes
    assert result.diagnostics.status.value == expected
    assert result.diagnostics.closed


@pytest.mark.parametrize("error", [RuntimeError("observer failed"), KeyboardInterrupt()])
def test_real_observer_exception_releases_native_file(tmp_path, error):
    path = tmp_path / "observer.mkv"
    video(path, [0] * 4 + [255] * 4)
    seen = []

    def callback(sample):
        seen.append(sample)
        raise error

    with pytest.raises(type(error)) as caught:
        detect_native_scenes(path, on_sample=callback)
    assert caught.value is error and len(seen) == 1
    path.unlink()


def test_real_scene_failure_leaves_no_successful_cli_report(tmp_path, capsys):
    path = tmp_path / "invalid.mkv"
    path.write_bytes(b"\x1aE\xdf\xa3" + b"bad video" * 12)
    with pytest.raises(ScanError):
        detect_native_scenes(path)
    assert main(["native-scenes", str(path)]) != 0
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err
    path.unlink()


def test_native_scene_cli_ensemble_exact_json_and_limit_diagnostics(tmp_path, capsys):
    path = tmp_path / "report.mkv"
    video(path, [0] * 4 + [255] * 4)
    assert (
        main(
            [
                "native-scenes",
                str(path),
                "--detectors",
                "adaptive",
                "luminance",
                "--minimum-votes",
                "2",
                "--min-scene-samples",
                "2",
                "--start",
                "5",
                "--end",
                "57/10",
                "--max-frames",
                "7",
            ]
        )
        == 0
    )
    data = json.loads(capsys.readouterr().out)
    assert data["kind"] == "frame-quorum-native-scenes" and data["schema_version"] == 1
    assert data["cut_positions"] == [4]
    assert data["cut_times"] == [{"numerator": 527, "denominator": 100}]
    assert data["statistics"][4]["votes"] == 2
    assert data["diagnostics"]["status"] == "frame_limit"
    assert data["scenes"][-1]["end_time"] is None


def test_executable_native_scene_demo_is_generated_offline_and_exact(capsys):
    runpy.run_path(str(Path(__file__).parents[1] / "examples" / "native_scenes.py"), run_name="__main__")
    result = json.loads(capsys.readouterr().out)
    assert result["cut_positions"] == [4]
    assert result["cut_times"] == [{"numerator": 527, "denominator": 100}]
    assert result["scenes"][-1]["end_time"] is None
    assert result["diagnostics"]["status"] == "eof"
