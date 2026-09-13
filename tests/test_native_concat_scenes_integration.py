from __future__ import annotations

import json
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import pytest
from PIL import Image

from frame_quorum import (
    DetectionConfig,
    FrameMetrics,
    NativeConcatClip,
    NativeConcatConfig,
    NativeConcatSceneConfig,
    detect_native_concat_scenes,
)
from tests._concat_scene_oracle import hand_detector

av = pytest.importorskip("av", reason="install frame-quorum[video] for genuine composite scene codecs")


@pytest.mark.parametrize("optimized", [False, True])
def test_real_offline_example_keeps_all_checks_under_optimization(tmp_path, optimized):
    example = Path(__file__).resolve().parents[1] / "examples" / "native_concat_scenes.py"
    process = subprocess.run(
        [sys.executable, "-I", "-B", *(["-O"] if optimized else []), str(example)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert process.returncode == 0, process.stderr
    assert not process.stderr
    assert json.loads(process.stdout) == {
        "sample_count": 8,
        "cut_positions": [4],
        "cut_times": ["1/5"],
        "native_clocks": ["1/1000", "1/48000"],
        "declared_end": "2/5",
        "source_verified": False,
        "coverage": "declared_only",
        "closed": True,
        "complete_native_identity_equal": True,
    }
    assert list(tmp_path.iterdir()) == []


def write_gray(path, values, stamps, clock, rate, format_name):
    with av.open(str(path), "w", format=format_name) as container:
        video = container.add_stream("ffv1", rate=rate)
        video.width, video.height, video.pix_fmt = 16, 12, "bgr0"
        video.time_base = video.codec_context.time_base = clock
        for gray, stamp in zip(values, stamps, strict=True):
            with Image.new("RGB", (16, 12), (gray, gray, gray)) as image:
                frame = av.VideoFrame.from_image(image)
            frame.pts, frame.time_base = stamp, clock
            for packet in video.encode(frame):
                container.mux(packet)
        for packet in video.encode():
            container.mux(packet)


def media(tmp_path, values):
    paths = (tmp_path / "first.mkv", tmp_path / "second.nut")
    write_gray(paths[0], values[:4], (5000, 5040, 5150, 5300), Fraction(1, 1000), 25, "matroska")
    write_gray(paths[1], values[4:], (48000, 49200, 51600, 57600), Fraction(1, 48000), 30, "nut")
    return (
        NativeConcatClip(paths[0], start=5, end=Fraction(27, 5)),
        NativeConcatClip(paths[1], start=1, end=Fraction(13, 10)),
    )


def direct(clips, start=Fraction(0), end=Fraction(7, 10), step=1):
    """Direct codec read, handwritten solid-gray measurements, explicit geometry."""
    rows = []
    offset = Fraction(0)
    for source, clip in enumerate(clips):
        eligible = 0
        with av.open(str(clip.path)) as container:
            for decoded_index, frame in enumerate(container.decode(video=clip.video_stream)):
                t = frame.pts * frame.time_base
                if not clip.start <= t < clip.end:
                    continue
                global_time = offset + t - clip.start
                if start <= global_time < end:
                    with frame.to_image() as image, image.convert("RGB") as rgb:
                        data = rgb.tobytes()
                    assert len(set(data)) == 1  # The exact encoded corpus is solid grayscale.
                    level = round(data[0] / 255, 8)
                    metrics = FrameMetrics(0, level, 0.0, 0.0, 0.0, level, level, level)
                    rows.append(
                        (source, frame.pts, frame.time_base, decoded_index, eligible, global_time, metrics)
                    )
                eligible += 1
        offset += clip.end - clip.start
    return rows[::step]


@pytest.mark.parametrize("detector", ["content", "color", "luminance", "adaptive", "threshold"])
@pytest.mark.parametrize("step", [1, 2])
def test_real_mixed_clocks_seam_context_and_complete_independent_evidence(tmp_path, detector, step):
    values = [255, 255, 0, 0, 0, 0, 255, 255] if detector == "threshold" else [0] * 4 + [255] * 4
    clips = media(tmp_path, values)
    config = DetectionConfig(detector=detector, window_radius=1, min_dark_frames=1)
    result = detect_native_concat_scenes(
        clips, NativeConcatSceneConfig(video=NativeConcatConfig(frame_step=step), detectors=(config,))
    )
    expected = direct(clips, step=step)
    assert [
        (
            r.sample.clip_index,
            r.sample.native.pts,
            r.sample.native.time_base,
            r.sample.native.decode_index,
            r.sample.native.sample_index,
            r.sample.presentation_time,
            r.sample.native.metrics,
        )
        for r in result.statistics
    ] == expected
    content, scores, candidates, decisions = hand_detector([r[-1] for r in expected], config)
    for i, row in enumerate(result.statistics):
        evidence = row.detectors[0]
        assert row.content_score == pytest.approx(content[i], abs=1e-12)
        assert (
            evidence.score is None
            if scores[i] is None
            else evidence.score == pytest.approx(scores[i], abs=1e-12)
        )
        assert (evidence.candidate, evidence.qualified, evidence.reason) == (i in candidates, *decisions[i])
    assert result.cut_positions == (4 // step,)
    assert result.cut_times == (Fraction(2, 5),)
    assert result.scenes[-1].end_time == Fraction(7, 10)
    assert result.scenes[-1].last_sample_time == expected[-1][-2]
    assert result.diagnostics.activations == 4 and result.diagnostics.closed
    assert [m.time_base for m in result.metadata] == [Fraction(1, 1000), Fraction(1, 48000)]
    for clip in clips:
        clip.path.unlink()  # Actual OS handles are released, not delegated to GC.


def test_real_equal_images_do_not_force_cut_at_a_seam_and_repeated_paths_are_occurrences(tmp_path):
    clips = media(tmp_path, [128] * 8)
    clips = (*clips, clips[0])
    result = detect_native_concat_scenes(
        clips, NativeConcatSceneConfig(detectors=(DetectionConfig(detector="color"),))
    )
    assert result.cut_positions == ()
    assert [r.sample.clip_index for r in result.statistics] == [0] * 4 + [1] * 4 + [2] * 4
    assert result.scenes[0].end_time == Fraction(11, 10)
    assert [s.clip_index for s in result.scenes[0].spans] == [0, 1, 2]
    assert result.diagnostics.source_activations == (2, 2, 2)
    assert result.to_dict()["source_verified"] is False


@pytest.mark.parametrize("bias,cut", [(-1.0, 2), (0.0, 4), (1.0, 6)])
def test_real_fade_context_is_not_reset_when_dark_interval_straddles_sources(tmp_path, bias, cut):
    clips = media(tmp_path, [255, 255, 0, 0, 0, 0, 255, 255])
    result = detect_native_concat_scenes(
        clips, NativeConcatSceneConfig(detectors=(DetectionConfig(detector="threshold", fade_bias=bias),))
    )
    assert result.cut_positions == (cut,)
    assert result.statistics[cut].detectors[0].reason == "completed_fade"
    assert result.cut_times == (direct(clips)[cut][-2],)
