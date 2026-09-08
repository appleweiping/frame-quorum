from __future__ import annotations

import hashlib
import json
import runpy
from fractions import Fraction
from pathlib import Path

import pytest
from PIL import Image

import frame_quorum.native_measurements as module
import frame_quorum.native_video as native_video
from frame_quorum import (
    DetectionConfig,
    NativeMeasurementLimits,
    NativeSceneConfig,
    NativeVideoConfig,
    analyze_native_measurements,
    capture_native_measurements,
    detect_native_scenes,
    read_native_measurements,
    write_native_measurements,
)
from frame_quorum.cli import main
from frame_quorum.errors import ConfigurationError, ScanError
from frame_quorum.metrics import measure_image

av = pytest.importorskip("av")
PTS = (5000, 5040, 5110, 5180, 5270, 5310, 5470, 5680)


def video(path, colors=(255, 255, 0, 0, 0, 0, 255, 255)):
    with av.open(str(path), "w", format="matroska") as container:
        stream = container.add_stream("ffv1", rate=25)
        stream.width, stream.height, stream.pix_fmt = 16, 12, "bgr0"
        stream.time_base = stream.codec_context.time_base = Fraction(1, 1000)
        for color, pts in zip(colors, PTS, strict=True):
            with Image.new("RGB", (16, 12), (color, color, color)) as image:
                frame = av.VideoFrame.from_image(image)
            frame.pts, frame.time_base = pts, Fraction(1, 1000)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


@pytest.mark.parametrize("detector", ["content", "color", "luminance", "adaptive", "threshold"])
def test_real_capture_full_independent_measurements_and_no_decoder_replay(tmp_path, monkeypatch, detector):
    path = tmp_path / "source.mkv"
    video(path)
    data = capture_native_measurements(path)
    assert data.source_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert data.metadata.source_bytes == path.stat().st_size
    independent = []
    with av.open(str(path)) as direct:
        for decoded in direct.decode(video=0):
            with decoded.to_image() as image:
                independent.append((decoded.pts * decoded.time_base, measure_image(image)))
    assert independent == [(sample.presentation_time, sample.metrics) for sample in data.samples]
    config = DetectionConfig(detector=detector, window_radius=1)
    fresh = detect_native_scenes(path, NativeSceneConfig(detectors=(config,)))
    cache = write_native_measurements(data, tmp_path / "cache")
    path.unlink()
    monkeypatch.setattr(native_video, "_load_av", lambda: pytest.fail("cached replay loaded PyAV"))
    restored = read_native_measurements(cache)
    replay = analyze_native_measurements(restored, detectors=(config,))
    assert replay.analysis == fresh
    if detector == "threshold":
        assert replay.analysis.cut_positions == (4,)
        assert replay.analysis.cut_times == (Fraction(527, 100),)
    assert replay.source_verified is False
    assert replay.analysis.scenes[-1].end_time is None


def test_cli_real_two_thresholds_after_source_deleted(tmp_path, monkeypatch, capsys):
    path = tmp_path / "source.mkv"
    video(path, (0, 0, 128, 128, 255, 255, 255, 255))
    cache_dir = tmp_path / "cache"
    assert main(["native-measure", str(path), "--output-dir", str(cache_dir)]) == 0
    captured = json.loads(capsys.readouterr().out)
    path.unlink()
    monkeypatch.setattr(native_video, "_load_av", lambda: pytest.fail("decoder was loaded"))
    for threshold, cuts in [("0.4", [2, 4]), ("0.6", [])]:
        output = tmp_path / threshold
        assert (
            main(
                [
                    "native-replay",
                    captured["cache"],
                    "--output-dir",
                    str(output),
                    "--detectors",
                    "luminance",
                    "--threshold",
                    threshold,
                ]
            )
            == 0
        )
        summary = json.loads(capsys.readouterr().out)
        assert summary["source_verified"] is False
        report = json.loads((output / "replay.json").read_bytes())
        assert report["analysis"]["cut_positions"] == cuts
        assert report["execution"] == "cached_measurements"
        assert report["measurement_digest"] == captured["measurement_digest"]
        assert (output / "statistics.csv").is_file()


def test_sampling_and_requested_end_identity(tmp_path):
    path = tmp_path / "vfr.mkv"
    video(path)
    config = NativeVideoConfig(start=Fraction(126, 25), end=Fraction(28, 5), frame_step=2)
    data = capture_native_measurements(path, config)
    assert data.video == config
    assert [sample.pts for sample in data.samples] == [5040, 5180, 5310]
    assert data.diagnostics.status is native_video.NativeVideoStatus.RANGE_END
    assert analyze_native_measurements(data).analysis.scenes[-1].end_time == Fraction(28, 5)


def test_real_source_mutation_after_decode_rejected(tmp_path, monkeypatch):
    path = tmp_path / "source.mkv"
    video(path)
    original = module._collect_native_samples

    def change(*args):
        result = original(*args)
        with path.open("ab") as handle:
            handle.write(b"changed")
        return result

    monkeypatch.setattr(module, "_collect_native_samples", change)
    with pytest.raises(ScanError, match="changed across"):
        capture_native_measurements(path)


def test_capture_checks_bounds_and_callback_cleanup(tmp_path):
    path = tmp_path / "source.mkv"
    video(path)
    with pytest.raises(ConfigurationError, match="sample admission"):
        capture_native_measurements(path, limits=NativeMeasurementLimits(max_samples=1))
    with pytest.raises(ConfigurationError, match="cache"):
        capture_native_measurements(path, limits=NativeMeasurementLimits(max_cache_bytes=1))
    with pytest.raises(ScanError, match="byte limit"):
        capture_native_measurements(path, NativeVideoConfig(max_source_bytes=1))

    def stop(sample):
        raise KeyboardInterrupt("callback stop")

    with pytest.raises(KeyboardInterrupt, match="callback stop"):
        capture_native_measurements(path, on_sample=stop)
    path.unlink()


def test_capture_missing_source_error(tmp_path):
    with pytest.raises(ScanError):
        capture_native_measurements(tmp_path / "missing.mkv")


def test_capture_observes_every_sample_without_decisions(tmp_path):
    path = tmp_path / "source.mkv"
    video(path)
    seen = []
    data = capture_native_measurements(path, on_sample=seen.append)
    assert seen == list(data.samples)
    assert all(not hasattr(sample, "accepted") for sample in seen)


def test_offline_executable_example(capsys):
    runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "examples" / "native_measurement_replay.py"),
        run_name="__main__",
    )
    assert "threshold=0.4: cuts=['511/100', '527/100']" in capsys.readouterr().out
