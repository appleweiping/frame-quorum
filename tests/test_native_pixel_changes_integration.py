"""Lossless native pixels compared with hand-specified spatial/color oracles."""

import hashlib
import json
import runpy
import subprocess
import sys
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest
from PIL import Image

import frame_quorum.native_pixel_changes as module
import frame_quorum.native_video as native
from frame_quorum import (
    NativeMeasurementLimits,
    NativeVideoConfig,
    PixelChangeConfig,
    PixelChangeDetectionConfig,
    PixelChangeLimits,
    PixelChangeWeights,
    analyze_native_pixel_changes,
    capture_native_measurements,
    capture_native_pixel_changes,
    read_native_pixel_changes,
    write_native_pixel_changes,
)
from frame_quorum.cli import main
from frame_quorum.errors import ConfigurationError, ScanError

av = pytest.importorskip("av")
PTS = (5000, 5040, 5110, 5180, 5270, 5310)


def color(x, y, changed, spatial):
    if spatial:
        value = 255 * (x % 2 if changed else (x + y) % 2)
        return value, value, value
    return ((255, 255, 0), (0, 0, 255))[x % 2] if changed else ((255, 0, 0), (0, 255, 255))[x % 2]


def video(path, *, spatial=False):
    with av.open(str(path), "w", format="matroska") as container:
        stream = container.add_stream("ffv1", rate=25)
        stream.width, stream.height, stream.pix_fmt = 16, 12, "bgr0"
        stream.time_base = stream.codec_context.time_base = Fraction(1, 1000)
        for index, pts in enumerate(PTS):
            with Image.new("RGB", (16, 12)) as image:
                for y in range(12):
                    for x in range(16):
                        image.putpixel((x, y), color(x, y, index in (2, 3), spatial))
                frame = av.VideoFrame.from_image(image)
            frame.pts, frame.time_base = pts, Fraction(1, 1000)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


@pytest.mark.parametrize("spatial", [False, True])
def test_real_vfr_exact_pixel_oracle_and_independent_decoder(tmp_path, monkeypatch, spatial):
    source = tmp_path / "source.mkv"
    video(source, spatial=spatial)
    seen = []
    value = capture_native_pixel_changes(source, on_sample=seen.append)
    assert value.base == capture_native_measurements(source)
    assert value.base.source_sha256 == hashlib.sha256(source.read_bytes()).hexdigest()
    assert [row.change for row in seen] == list(value.changes)
    assert [row.sample for row in seen] == list(value.base.samples)
    assert value.measurement_pixels == 11 * 192
    with av.open(str(source)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            with frame.to_image() as image:
                assert image.size == (16, 12)
                assert image.tobytes() == bytes(
                    channel
                    for y in range(12)
                    for x in range(16)
                    for channel in color(x, y, index in (2, 3), spatial)
                )
            assert frame.pts * frame.time_base == Fraction(PTS[index], 1000)
    for index, change in enumerate(value.changes):
        if index == 0:
            assert change is None
        else:
            active = index in (2, 4)
            expected = (
                ((0, 0, 96 * 255, 16 * 11 * 255) if spatial else (192 * 256 * 255, 0, 0, 0))
                if active
                else (0, 0, 0, 0)
            )
            assert (change.hue_sum, change.saturation_sum, change.value_sum, change.edge_sum) == expected
    cache = write_native_pixel_changes(value, tmp_path / "cache")
    source.unlink()
    monkeypatch.setattr(native, "_load_av", lambda: pytest.fail("offline replay imported decoder"))
    restored = read_native_pixel_changes(cache)
    assert restored == value
    weights = PixelChangeWeights(0, 0, 0, 1) if spatial else PixelChangeWeights(1, 0, 0, 0)
    result = analyze_native_pixel_changes(restored, PixelChangeDetectionConfig(weights=weights))
    assert result.cut_positions == (2, 4)
    assert result.cut_times == (Fraction(511, 100), Fraction(527, 100))
    assert result.source_verified is False and result.scenes[-1].end_time is None


def test_cli_capture_replay_and_no_optional_import_in_fresh_process(tmp_path, capsys):
    source = tmp_path / "source.mkv"
    video(source)
    assert main(["native-change-measure", str(source), "-o", str(tmp_path / "cache")]) == 0
    summary = json.loads(capsys.readouterr().out)
    source.unlink()
    code = """
import importlib.abc, json, sys
class Forbid(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'av' or fullname.startswith('av.'):
            raise RuntimeError('decoder imported during replay')
sys.meta_path.insert(0, Forbid())
from frame_quorum import read_native_pixel_changes, analyze_native_pixel_changes
from frame_quorum import PixelChangeDetectionConfig, PixelChangeWeights
data = read_native_pixel_changes(sys.argv[1])
result = analyze_native_pixel_changes(data, PixelChangeDetectionConfig(weights=PixelChangeWeights(1,0,0,0)))
assert not result.source_verified and 'av' not in sys.modules
print(json.dumps(result.cut_positions))
"""
    process = subprocess.run(
        [sys.executable, "-c", code, summary["cache"]], capture_output=True, text=True, timeout=30
    )
    assert process.returncode == 0, process.stderr
    assert json.loads(process.stdout) == [2, 4]
    for name, flags, expected in (
        ("hue", ["--weights", "1", "0", "0", "0"], [2, 4]),
        ("value", ["--value-only"], []),
        (
            "adaptive",
            ["--detector", "adaptive", "--window-radius", "1", "--weights", "1", "0", "0", "0"],
            [2, 4],
        ),
    ):
        assert main(["native-change-replay", summary["cache"], *flags, "-o", str(tmp_path / name)]) == 0
        assert json.loads(capsys.readouterr().out)["cut_positions"] == expected


def test_range_stride_empty_and_frame_limit(tmp_path):
    source = tmp_path / "source.mkv"
    video(source)
    options = NativeVideoConfig(start=Fraction(126, 25), end=Fraction(53, 10), frame_step=2)
    value = capture_native_pixel_changes(source, options)
    assert [row.pts for row in value.base.samples] == [5040, 5180]
    assert [row.decode_index for row in value.base.samples] == [1, 3]
    assert value.measurement_pixels == 3 * 192
    result = analyze_native_pixel_changes(
        value, PixelChangeDetectionConfig(weights=PixelChangeWeights(1, 0, 0, 0))
    )
    assert result.cut_positions == (1,) and result.scenes[-1].end_time == Fraction(53, 10)
    limited = capture_native_pixel_changes(source, NativeVideoConfig(max_frames=1))
    assert (
        limited.changes == (None,) and limited.base.diagnostics.status is native.NativeVideoStatus.FRAME_LIMIT
    )
    empty = capture_native_pixel_changes(source, NativeVideoConfig(start=Fraction(100)))
    assert empty.changes == () and empty.measurement_pixels == 0


@pytest.mark.parametrize("error", [OSError, KeyboardInterrupt, SystemExit])
def test_callback_cleanup_preserves_controls_and_closes_decoder(tmp_path, error):
    source = tmp_path / "source.mkv"
    video(source)

    def stop(row):
        raise error("stop")

    with pytest.raises(ScanError if error is OSError else error):
        capture_native_pixel_changes(source, on_sample=stop)
    source.unlink()


def test_capture_admission_before_image_allocation_and_dimension_change(tmp_path, monkeypatch):
    source = tmp_path / "source.mkv"
    video(source)
    original = native.NativeVideoFrame.image
    opened = []

    def image(self):
        opened.append(self.sample_index)
        return original(self)

    monkeypatch.setattr(native.NativeVideoFrame, "image", image)
    with pytest.raises(ConfigurationError, match="pixel work"):
        capture_native_pixel_changes(source, pixel_limits=PixelChangeLimits(max_measurement_pixels=575))
    assert opened == [0]
    opened.clear()
    collector = module._collect_native_records

    def resize(path, options, measure, callback):
        def changed(frame):
            if frame.sample_index == 1:
                frame = replace(frame, width=12, height=16)
            return measure(frame)

        return collector(path, options, changed, callback)

    monkeypatch.setattr(module, "_collect_native_records", resize)
    with pytest.raises(ConfigurationError, match="changing frame dimensions"):
        capture_native_pixel_changes(source)
    assert opened == [0]
    source.unlink()


def test_invalid_capture_options_source_mutation_and_callback_returns(tmp_path, monkeypatch):
    source = tmp_path / "source.mkv"
    video(source)
    for options in (
        {"video": object()},
        {"config": object()},
        {"limits": NativeMeasurementLimits(max_samples=1)},
        {"pixel_limits": PixelChangeLimits(max_pair_rgb_bytes=6)},
    ):
        with pytest.raises(ConfigurationError):
            capture_native_pixel_changes(source, **options)
    with pytest.raises(ConfigurationError):
        capture_native_pixel_changes(source, limits=NativeMeasurementLimits(max_cache_bytes=1))

    async def async_callback(row):
        pass

    for callback in (async_callback, lambda row: async_callback(row), lambda row: 1, object()):
        with pytest.raises(ConfigurationError):
            capture_native_pixel_changes(source, on_sample=callback)
    original = module._collect_native_records

    def mutate(*args):
        result = original(*args)
        with source.open("ab") as handle:
            handle.write(b"changed")
        return result

    monkeypatch.setattr(module, "_collect_native_records", mutate)
    with pytest.raises(ScanError, match="changed across"):
        capture_native_pixel_changes(source)
    source.unlink()


def test_acquisition_radius_is_bound_and_example_runs(tmp_path, capsys):
    source = tmp_path / "source.mkv"
    video(source, spatial=True)
    first = capture_native_pixel_changes(source, config=PixelChangeConfig(1))
    second = capture_native_pixel_changes(source, config=PixelChangeConfig(2))
    assert first.digest != second.digest
    assert first.changes[2].edge_sum != second.changes[2].edge_sum
    runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "examples" / "native_pixel_changes.py"), run_name="__main__"
    )
    assert "edge: cuts=['511/100', '527/100']; source_verified=False" in capsys.readouterr().out
