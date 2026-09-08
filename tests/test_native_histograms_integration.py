from __future__ import annotations

import hashlib
import json
import runpy
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import pytest
from PIL import Image

import frame_quorum.native_histograms as module
import frame_quorum.native_video as native
from frame_quorum import (
    HistogramDetectionConfig,
    NativeMeasurementLimits,
    NativeVideoConfig,
    PixelHistogramConfig,
    PixelHistogramLimits,
    analyze_native_histograms,
    capture_native_histograms,
    capture_native_measurements,
    read_native_histograms,
    write_native_histograms,
)
from frame_quorum.cli import main
from frame_quorum.errors import ConfigurationError, ScanError

av = pytest.importorskip("av")
PTS = (5000, 5040, 5110, 5180, 5270, 5310)


def video(path, *, spatial=False):
    with av.open(str(path), "w", format="matroska") as container:
        stream = container.add_stream("ffv1", rate=25)
        stream.width, stream.height, stream.pix_fmt = 16, 12, "bgr0"
        stream.time_base = stream.codec_context.time_base = Fraction(1, 1000)
        for index, pts in enumerate(PTS):
            changed = index in (2, 3)
            with Image.new("RGB", (16, 12)) as image:
                if spatial:
                    image.paste((254, 254, 254), (0, 0 if changed else 6, 16, 6 if changed else 12))
                elif changed:
                    image.paste((127, 127, 127), (0, 0, 16, 12))
                else:
                    image.paste((254, 254, 254), (0, 6, 16, 12))
                frame = av.VideoFrame.from_image(image)
            frame.pts, frame.time_base = pts, Fraction(1, 1000)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


@pytest.mark.parametrize("bins", [8, 32, 256])
@pytest.mark.parametrize("spatial", [False, True])
def test_real_vfr_direct_pixel_oracle_source_hash_and_replay_after_deletion(
    tmp_path, monkeypatch, bins, spatial
):
    path = tmp_path / "source.mkv"
    video(path, spatial=spatial)
    options = NativeVideoConfig(max_frames=20)
    config = PixelHistogramConfig(bins=bins)
    seen = []
    measured = capture_native_histograms(path, options, histogram=config, on_sample=seen.append)
    assert measured.base == capture_native_measurements(path, options)
    assert measured.base.source_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert [row.sample for row in seen] == list(measured.base.samples)
    assert [row.histogram for row in seen] == list(measured.histograms)
    independent = []
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            with frame.to_image() as image, image.convert("RGB") as rgb:
                raw = rgb.tobytes()
            counts = [0] * (4 * 3 * bins)
            for y in range(12):
                for x in range(16):
                    cell = (0 if y < 6 else 1) * 2 + (0 if x < 8 else 1)
                    for channel in range(3):
                        value = raw[(y * 16 + x) * 3 + channel]
                        counts[(cell * 3 + channel) * bins + value * bins // 256] += 1
            independent.append((frame.pts * frame.time_base, tuple(counts)))
    assert independent == [
        (sample.presentation_time, hist.counts)
        for sample, hist in zip(measured.base.samples, measured.histograms, strict=True)
    ]
    assert [sample.presentation_time for sample in measured.base.samples] == [
        Fraction(pts, 1000) for pts in PTS
    ]
    cache = write_native_histograms(measured, tmp_path / "cache")
    path.unlink()
    monkeypatch.setattr(native, "_load_av", lambda: pytest.fail("cached histogram replay loaded decoder"))
    restored = read_native_histograms(cache)
    for mode in ("global", "spatial"):
        result = analyze_native_histograms(restored, HistogramDetectionConfig(mode=mode))
        expected = () if spatial and mode == "global" else (2, 4)
        assert result.cut_positions == expected
        if expected:
            assert result.cut_times == (Fraction(511, 100), Fraction(527, 100))
        assert result.scenes[-1].end_time is None
        assert result.source_verified is False


def test_cli_and_fresh_process_without_any_optional_av_import(tmp_path, capsys):
    source = tmp_path / "source.mkv"
    video(source, spatial=True)
    cache_dir = tmp_path / "cache"
    assert main(["native-histogram-measure", str(source), "--output-dir", str(cache_dir)]) == 0
    summary = json.loads(capsys.readouterr().out)
    source.unlink()
    code = """
import importlib.abc, json, sys
class Forbid(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "av" or fullname.startswith("av."):
            raise RuntimeError("optional decoder imported during replay")
sys.meta_path.insert(0, Forbid())
from frame_quorum import read_native_histograms, analyze_native_histograms
data = read_native_histograms(sys.argv[1])
result = analyze_native_histograms(data)
assert result.source_verified is False
assert "av" not in sys.modules
print(json.dumps(result.cut_positions))
"""
    process = subprocess.run(
        [sys.executable, "-c", code, summary["cache"]], capture_output=True, text=True, timeout=30
    )
    assert process.returncode == 0, process.stderr
    assert json.loads(process.stdout) == [2, 4]
    for mode, expected in (("global", []), ("spatial", [2, 4])):
        assert (
            main(
                [
                    "native-histogram-replay",
                    summary["cache"],
                    "--mode",
                    mode,
                    "--output-dir",
                    str(tmp_path / mode),
                ]
            )
            == 0
        )
        result = json.loads(capsys.readouterr().out)
        assert result["cut_positions"] == expected


def test_exact_stride_range_and_limits_preserve_observed_coordinates(tmp_path):
    source = tmp_path / "vfr.mkv"
    video(source)
    config = NativeVideoConfig(start=Fraction(126, 25), end=Fraction(53, 10), frame_step=2)
    measured = capture_native_histograms(source, config)
    assert [row.pts for row in measured.base.samples] == [5040, 5180]
    assert [row.decode_index for row in measured.base.samples] == [1, 3]
    result = analyze_native_histograms(measured)
    assert result.cut_positions == (1,) and result.scenes[-1].end_time == Fraction(53, 10)
    limited = capture_native_histograms(source, NativeVideoConfig(max_frames=2))
    assert limited.base.diagnostics.status is native.NativeVideoStatus.FRAME_LIMIT
    assert analyze_native_histograms(limited).scenes[-1].end_time is None
    empty = capture_native_histograms(source, NativeVideoConfig(start=Fraction(100)))
    assert analyze_native_histograms(empty).scenes == ()


@pytest.mark.parametrize("exception", [OSError, KeyboardInterrupt, SystemExit])
def test_callback_failure_closes_actual_source_and_owned_images(tmp_path, exception):
    source = tmp_path / "source.mkv"
    video(source)
    seen = []

    def stop(row):
        seen.append(row)
        raise exception("stop")

    expected = ScanError if exception is OSError else exception
    with pytest.raises(expected):
        capture_native_histograms(source, on_sample=stop)
    assert len(seen) == 1
    source.unlink()


def test_capture_aggregate_pixel_budget_stops_before_next_image_allocation(tmp_path, monkeypatch):
    source = tmp_path / "source.mkv"
    video(source)
    original = native.NativeVideoFrame.image
    opened = []

    def image(self):
        opened.append(self.sample_index)
        return original(self)

    monkeypatch.setattr(native.NativeVideoFrame, "image", image)
    with pytest.raises(ConfigurationError, match="pixels"):
        capture_native_histograms(source, pixel_limits=PixelHistogramLimits(max_measurement_pixels=200))
    assert opened == [0]
    source.unlink()


def test_full_source_mutation_checks_and_capture_admission(tmp_path, monkeypatch):
    source = tmp_path / "source.mkv"
    video(source)
    for kwargs in (
        {"limits": NativeMeasurementLimits(max_samples=1)},
        {"pixel_limits": PixelHistogramLimits(max_histogram_values=1)},
        {"histogram": object()},
        {"video": object()},
    ):
        with pytest.raises(ConfigurationError):
            capture_native_histograms(source, **kwargs)
    with pytest.raises(ConfigurationError):
        capture_native_histograms(source, limits=NativeMeasurementLimits(max_cache_bytes=1))
    with pytest.raises(ScanError):
        capture_native_histograms(source, NativeVideoConfig(max_source_bytes=1))
    original = module._collect_native_records

    def changed(*args):
        result = original(*args)
        with source.open("ab") as handle:
            handle.write(b"changed")
        return result

    monkeypatch.setattr(module, "_collect_native_records", changed)
    with pytest.raises(ScanError, match="changed across"):
        capture_native_histograms(source)
    source.unlink()


def test_invalid_async_and_value_callbacks_do_not_leak_resources(tmp_path):
    source = tmp_path / "source.mkv"
    video(source)

    async def callback(row):
        pass

    for value in (callback, lambda row: callback(row), lambda row: 1, object()):
        with pytest.raises(ConfigurationError):
            capture_native_histograms(source, on_sample=value)
    source.unlink()


def test_generated_offline_example(capsys):
    runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "examples" / "native_pixel_histograms.py"),
        run_name="__main__",
    )
    output = capsys.readouterr().out
    assert "global: cuts=[]" in output
    assert "spatial: cuts=['511/100', '527/100']" in output
