"""Actual lossless VFR pixels, independent decoder and online/offline agreement."""

import hashlib
import json
from fractions import Fraction

import pytest
from PIL import Image

from frame_quorum import (
    NativePixelChangeEnd,
    NativePixelChangeStream,
    NativePixelChangeUpdate,
    NativeVideoConfig,
    PixelChangeDetectionConfig,
    PixelChangeWeights,
    analyze_native_pixel_changes,
    capture_native_pixel_changes,
    native_scene_clips,
    write_native_pixel_change_stream,
)
from frame_quorum.cli import main
from frame_quorum.errors import ConfigurationError, OutputError

av = pytest.importorskip("av")
PTS = (5000, 5040, 5110, 5180, 5270, 5310, 5410, 5530, 5590)


def pixel(x, y, active, mode):
    if mode == "hue":
        return (((255, 255, 0), (0, 0, 255)) if active else ((255, 0, 0), (0, 255, 255)))[x % 2]
    value = 255 * ((x % 2 if active else (x + y) % 2) if mode == "edge" else active)
    return (value, value, value)


def make_video(path, mode="value"):
    with av.open(str(path), "w", format="matroska") as container:
        stream = container.add_stream("ffv1", rate=25)
        stream.width, stream.height, stream.pix_fmt = 16, 12, "bgr0"
        stream.time_base = stream.codec_context.time_base = Fraction(1, 1000)
        for index, pts in enumerate(PTS):
            raw = bytes(
                channel for y in range(12) for x in range(16) for channel in pixel(x, y, 3 <= index < 6, mode)
            )
            with Image.frombytes("RGB", (16, 12), raw) as image:
                frame = av.VideoFrame.from_image(image)
            frame.pts, frame.time_base = pts, Fraction(1, 1000)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


@pytest.mark.parametrize("mode", ["value", "hue", "edge"])
@pytest.mark.parametrize("detector", ["content", "adaptive"])
def test_real_pixels_exact_pts_and_independent_scene_oracle(tmp_path, mode, detector, monkeypatch):
    import frame_quorum.native_measurements as measurements
    import frame_quorum.native_scenes as scenes

    source = tmp_path / "input.mkv"
    make_video(source, mode)
    weights = {
        "value": PixelChangeWeights(0, 0, 1, 0),
        "hue": PixelChangeWeights(1, 0, 0, 0),
        "edge": PixelChangeWeights(0, 0, 0, 1),
    }[mode]
    detection = PixelChangeDetectionConfig(
        detector=detector, weights=weights, min_scene_samples=2, window_radius=1, min_content=0.1
    )
    with monkeypatch.context() as isolated:
        isolated.setattr(
            measurements, "_hash_source", lambda *args: pytest.fail("online whole-source hashing")
        )
        isolated.setattr(
            scenes, "_collect_native_records", lambda *args: pytest.fail("online full collection")
        )
        with NativePixelChangeStream(source, detection=detection) as stream:
            events = list(stream)
    rows, end = events[:-1], events[-1]
    assert all(type(row) is NativePixelChangeUpdate for row in rows)
    assert type(end) is NativePixelChangeEnd
    with av.open(str(source)) as container:
        for index, decoded in enumerate(container.decode(video=0)):
            with decoded.to_image() as image:
                assert image.tobytes() == bytes(
                    channel
                    for y in range(12)
                    for x in range(16)
                    for channel in pixel(x, y, 3 <= index < 6, mode)
                )
            assert decoded.pts * decoded.time_base == Fraction(PTS[index], 1000)
    expected_change = {
        "value": (0, 0, 192 * 255, 0),
        "hue": (192 * 256 * 255, 0, 0, 0),
        "edge": (0, 0, 96 * 255, 16 * 11 * 255),
    }[mode]
    for index, row in enumerate(rows):
        assert row.sample.sample.presentation_time == Fraction(PTS[index], 1000)
        change = row.sample.change
        if index == 0:
            assert change is None
        else:
            assert (change.hue_sum, change.saturation_sum, change.value_sum, change.edge_sum) == (
                expected_change if index in (3, 6) else (0, 0, 0, 0)
            )
    assert [row.sample.sample.sample_index for row in rows if row.statistic.accepted] == [3, 6]
    assert [
        (row.closed_scene.start_position, row.closed_scene.end_position) for row in rows if row.closed_scene
    ] == [(0, 3), (3, 6)]
    assert [
        (row.observed_through_sample_index, row.observed_through_time)
        for row in rows
        if row.statistic.accepted
    ] == [(4, Fraction(527, 100)), (7, Fraction(553, 100))]
    assert end.final_scene.end_position == 9 and end.final_scene.end_time is None
    assert end.diagnostics.measurement_pixels == 17 * 192
    offline = analyze_native_pixel_changes(capture_native_pixel_changes(source), detection)
    assert tuple(row.sample.sample for row in rows) == offline.measurements.base.samples
    assert tuple(row.statistic for row in rows) == offline.statistics
    assert (*[row.closed_scene for row in rows if row.closed_scene], end.final_scene) == offline.scenes
    with pytest.raises(ConfigurationError, match="NativeSceneResult"):
        native_scene_clips(end)


@pytest.mark.parametrize(
    "video",
    [
        NativeVideoConfig(start=Fraction(5101, 1000), end=Fraction(552, 100), frame_step=2),
        NativeVideoConfig(max_frames=4),
        NativeVideoConfig(max_decoded_frames=4),
        NativeVideoConfig(end=Fraction(4999, 1000)),
        NativeVideoConfig(start=Fraction(10), end=Fraction(11)),
    ],
)
def test_real_range_stride_empty_and_limits_match_offline(tmp_path, video):
    source = tmp_path / "input.mkv"
    make_video(source)
    detection = PixelChangeDetectionConfig(value_only=True, min_scene_samples=2)
    with NativePixelChangeStream(source, video=video, detection=detection) as stream:
        events = list(stream)
    offline = analyze_native_pixel_changes(capture_native_pixel_changes(source, video), detection)
    assert tuple(row.statistic for row in events[:-1]) == offline.statistics
    assert tuple(row.sample.sample for row in events[:-1]) == offline.measurements.base.samples
    assert events[-1].diagnostics.video == offline.measurements.base.diagnostics
    assert (
        tuple(
            [row.closed_scene for row in events[:-1] if row.closed_scene]
            + ([events[-1].final_scene] if events[-1].final_scene else [])
        )
        == offline.scenes
    )


def test_actual_cli_jsonl_and_atomic_existing_target_refusal(tmp_path, capsys):
    source = tmp_path / "input.mkv"
    make_video(source)
    assert main(["native-change-stream", str(source), "--value-only", "--min-scene-samples", "2"]) == 0
    raw = capsys.readouterr().out.encode("ascii")
    lines = raw.splitlines(keepends=True)
    records = [json.loads(line) for line in lines]
    assert records[0]["kind"] == "frame-quorum-native-pixel-change-events"
    assert records[-1]["sha256"] == hashlib.sha256(b"".join(lines[:-1])).hexdigest()
    assert records[-1]["diagnostics"]["confirmed_samples"] == 9
    assert [row["sample"]["sample_index"] for row in records[1:-1] if row["statistic"]["accepted"]] == [3, 6]
    target = tmp_path / "events"
    path = write_native_pixel_change_stream(NativePixelChangeStream(source), target)
    saved = path.read_bytes()
    with pytest.raises(OutputError, match="exist"):
        write_native_pixel_change_stream(NativePixelChangeStream(source), target)
    assert path.read_bytes() == saved
    assert not list(tmp_path.glob(".frame-quorum-measurements-*"))
    assert main(["native-change-stream", str(source), "-o", str(tmp_path / "cli-report")]) == 0
    assert (tmp_path / "cli-report" / "events.jsonl").is_file()


def test_offline_example_runs_without_downloads_or_workspace_outputs(capsys):
    import runpy
    from pathlib import Path

    runpy.run_path(str(Path(__file__).parents[1] / "examples" / "native_online.py"), run_name="__main__")
    output = capsys.readouterr().out
    assert "cut=259/50, confirmed=527/100" in output
    assert "cut=541/100, confirmed=553/100" in output
    assert "completed=eof, source_verified=False" in output
