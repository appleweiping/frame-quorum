from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest

import frame_quorum.native_measurements as module
from frame_quorum import DetectionConfig, FrameMetrics, native_scene_clips
from frame_quorum.errors import ConfigurationError, OutputError, ScanError
from frame_quorum.native_measurements import (
    NativeMeasurementLimits,
    NativeMeasurements,
    NativeReplayResult,
    analyze_native_measurements,
    read_native_measurements,
    render_native_detection_csv,
    write_native_measurements,
    write_native_replay,
)
from frame_quorum.native_scenes import NativeSceneSample
from frame_quorum.native_video import (
    NativeVideoConfig,
    NativeVideoDiagnostics,
    NativeVideoMetadata,
    NativeVideoStatus,
)


def measurements(values=(0, 0.2, 0.8, 0.8, 0.1, 0.1), *, status=NativeVideoStatus.EOF):
    samples = tuple(
        NativeSceneSample(
            5000 + i * i,
            Fraction(1, 1000),
            i,
            i,
            0,
            FrameMetrics(0, value, 0, 0, 0, value, value, value),
        )
        for i, value in enumerate(values)
    )
    return NativeMeasurements(
        NativeVideoConfig(),
        NativeVideoMetadata(
            Path("recorded.mkv"), 4, "matroska", "ffv1", 0, 1, 1, Fraction(1, 1000), 5000, None, None, None
        ),
        NativeVideoDiagnostics(status, len(samples), len(samples), len(samples), 0, True),
        samples,
        hashlib.sha256(b"test").hexdigest(),
    )


def test_manually_calculated_threshold_changes_without_decoder(monkeypatch):
    import frame_quorum.native_video as video

    monkeypatch.setattr(video, "_load_av", lambda: pytest.fail("replay tried to load decoder"))
    data = measurements()
    low = analyze_native_measurements(
        data, detectors=(DetectionConfig(detector="luminance", threshold=0.15),)
    )
    high = analyze_native_measurements(
        data, detectors=(DetectionConfig(detector="luminance", threshold=0.65),)
    )
    assert low.analysis.cut_positions == (1, 2, 4)
    assert high.analysis.cut_positions == (4,)
    assert high.analysis.cut_times == (Fraction(627, 125),)
    assert high.analysis.scenes[-1].end_time is None
    assert low.source_verified is False
    assert low.to_dict()["execution"] == "cached_measurements"
    with pytest.raises(ConfigurationError):
        native_scene_clips(low, final_end=Fraction(6))


def test_cache_canonical_roundtrip_ignores_recorded_source_path(tmp_path):
    data = measurements()
    cache = write_native_measurements(data, tmp_path / "cache")
    assert cache.name == "measurements.fqm.jsonl"
    restored = read_native_measurements(cache)
    assert restored == data
    assert restored.digest == data.digest
    assert replace(data, source_sha256="a" * 64).digest != data.digest


def test_manual_fade_and_quorum_oracles():
    fade = measurements((1, 1, 0, 0, 0, 0, 1, 1))
    assert analyze_native_measurements(
        fade, detectors=(DetectionConfig(detector="threshold", fade_bias=0),)
    ).analysis.cut_positions == (4,)
    data = measurements((0, 1, 1, 0, 0, 1, 0))
    result = analyze_native_measurements(
        data,
        detectors=(
            DetectionConfig(detector="luminance", threshold=0.5),
            DetectionConfig(detector="color", threshold=0.5, min_scene_frames=2),
        ),
        minimum_votes=2,
    )
    assert result.analysis.cut_positions == (3, 5)
    assert result.analysis.statistics[1].votes == 1


def encode(value):
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False) + "\n"
    ).encode("ascii")


def wire(data=None):
    data = measurements() if data is None else data
    return [data.header(), *(sample.to_dict() for sample in data.samples)]


def save_wire(path, records):
    content = b"".join(map(encode, records))
    footer = {"kind": "end", "sample_count": len(records) - 1, "sha256": hashlib.sha256(content).hexdigest()}
    path.write_bytes(content + encode(footer))
    return path


@pytest.mark.parametrize("name", NativeMeasurementLimits.__dataclass_fields__)
@pytest.mark.parametrize("value", [True, 0, -1, 10**1000, float("nan"), "10"])
def test_strict_limits(name, value):
    with pytest.raises(ConfigurationError):
        NativeMeasurementLimits(**{name: value})


@pytest.mark.parametrize(
    "changes",
    [
        {"video": None},
        {"metadata": None},
        {"diagnostics": None},
        {"source_sha256": "A" * 64},
        {"source_sha256": 10},
        {"measurement_version": "future"},
        {"producer_pillow": ""},
        {"producer_pillow": "x" * 65},
        {"producer_pillow": "bad\n"},
        {"producer_pillow": 1},
        {"source_path_label": 1},
        {"source_path_label": ""},
        {"samples": []},
        {"samples": (None,)},
    ],
)
def test_measurement_constructor_rejects_bad_contract(changes):
    with pytest.raises(ConfigurationError):
        replace(measurements(), **changes)


@pytest.mark.parametrize(
    "changes",
    [
        {"closed": False},
        {"cleanup_errors": ("file.close",)},
        {"generation": 1},
        {"status": NativeVideoStatus.ERROR},
        {"decoded_frames": 100001},
        {"decoded_pixels_observed": 1000000001},
        {"returned_frames": 5},
        {"status": NativeVideoStatus.RANGE_END},
        {"status": NativeVideoStatus.FRAME_LIMIT},
        {"status": NativeVideoStatus.DECODE_LIMIT},
    ],
)
def test_constructor_rejects_diagnostic_contradictions(changes):
    data = measurements()
    with pytest.raises(ConfigurationError):
        replace(data, diagnostics=replace(data.diagnostics, **changes))


@pytest.mark.parametrize(
    "changes",
    [
        {"sample_index": 2},
        {"decode_index": 100},
        {"generation": 1},
        {"decode_index": 3},
        {"pts": -100},
    ],
)
def test_constructor_rejects_sample_order(changes):
    data = measurements()
    with pytest.raises(ConfigurationError):
        replace(data, samples=(data.samples[0], replace(data.samples[1], **changes), *data.samples[2:]))


@pytest.mark.parametrize(
    "changes",
    [
        {"start": Fraction(6)},
        {"end": Fraction(5)},
        {"max_frames": 5},
        {"max_source_bytes": 3},
    ],
)
def test_constructor_rejects_video_contradictions(changes):
    data = measurements()
    with pytest.raises(ConfigurationError):
        replace(data, video=replace(data.video, **changes))


@pytest.mark.parametrize(
    "case",
    [
        "kind",
        "version",
        "bool_version",
        "algorithm",
        "header_extra",
        "header_missing",
        "sample_count",
        "count_vs_limit",
        "metadata_path",
        "metadata_codec",
        "metadata_extra",
        "diag_status",
        "diag_cleanup",
        "missing_metric",
        "unknown_metric",
        "hash",
        "metric_nan",
        "metric_bool",
        "time_zero",
        "time_unreduced",
        "time_bool",
        "time_big",
        "sample_extra",
        "sample_missing",
        "sample_generation",
        "derived_time",
        "sample_order",
        "footer_digest",
        "producer",
        "path_long",
        "path_control",
    ],
)
def test_corrupt_cache_rejected_with_recomputed_checksum(tmp_path, case):
    records = wire()
    header, sample = records[0], records[1]
    if case == "kind":
        header["kind"] = "wrong"
    elif case == "version":
        header["schema_version"] = 2
    elif case == "bool_version":
        header["schema_version"] = True
    elif case == "algorithm":
        header["measurement_version"] = "missing-pixel-hsv"
    elif case == "header_extra":
        header["extra"] = None
    elif case == "header_missing":
        del header["video"]
    elif case == "sample_count":
        header["sample_count"] = 100001
    elif case == "count_vs_limit":
        header["video"]["max_frames"] = 1
    elif case == "metadata_path":
        header["metadata"]["path"] = 1
    elif case == "metadata_codec":
        header["metadata"]["codec"] = "bad\n"
    elif case == "metadata_extra":
        header["metadata"]["extra"] = 1
    elif case == "diag_status":
        header["diagnostics"]["status"] = "nonsense"
    elif case == "diag_cleanup":
        header["diagnostics"]["cleanup_errors"] = {}
    elif case == "missing_metric":
        del sample["metrics"]["mean_green"]
    elif case == "unknown_metric":
        sample["metrics"]["histogram"] = []
    elif case == "hash":
        sample["metrics"]["perceptual_hash"] = "00"
    elif case == "metric_nan":
        sample["metrics"]["luminance"] = "NaN"
    elif case == "metric_bool":
        sample["metrics"]["luminance"] = True
    elif case == "time_zero":
        sample["time_base"]["denominator"] = 0
    elif case == "time_unreduced":
        sample["time_base"] = {"numerator": 2, "denominator": 2000}
    elif case == "time_bool":
        sample["time_base"]["numerator"] = True
    elif case == "time_big":
        sample["time_base"]["denominator"] = 10**1000
    elif case == "sample_extra":
        sample["width"] = 1
    elif case == "sample_missing":
        del sample["pts"]
    elif case == "sample_generation":
        sample["generation"] = 1
    elif case == "derived_time":
        sample["presentation_time"]["numerator"] = False
    elif case == "sample_order":
        records[1], records[2] = records[2], records[1]
    elif case == "producer":
        header["producer_pillow"] = "x" * 65
    elif case == "path_long":
        header["metadata"]["path"] = "x" * 4097
    elif case == "path_control":
        header["metadata"]["path"] = "recorded\ninput.mkv"
    path = save_wire(tmp_path / "bad.jsonl", records)
    if case == "footer_digest":
        lines = path.read_bytes().splitlines(keepends=True)
        footer = json.loads(lines[-1])
        footer["sha256"] = "0" * 64
        path.write_bytes(b"".join(lines[:-1]) + encode(footer))
    with pytest.raises(ConfigurationError):
        read_native_measurements(path)


@pytest.mark.parametrize(
    "change",
    [
        "empty",
        "truncated",
        "trailing",
        "spaces",
        "crlf",
        "duplicate",
        "nan",
        "deep",
        "list",
        "invalid_utf8",
        "bool_footer",
    ],
)
def test_strict_canonical_ingress(tmp_path, change):
    path = save_wire(tmp_path / "bad.jsonl", wire())
    raw = path.read_bytes()
    if change == "empty":
        raw = b""
    elif change == "truncated":
        raw = raw[:-1]
    elif change == "trailing":
        raw += b"\n"
    elif change == "spaces":
        raw = raw.replace(b'"kind":', b'"kind": ', 1)
    elif change == "crlf":
        raw = raw.replace(b"\n", b"\r\n")
    elif change == "duplicate":
        raw = b'{"kind":1,"kind":2}\n'
    elif change == "nan":
        raw = b'{"x":NaN}\n'
    elif change == "deep":
        raw = b'{"x":' + b"[" * 2000 + b"0" + b"]" * 2000 + b"}\n"
    elif change == "list":
        raw = b"[]\n"
    elif change == "invalid_utf8":
        raw = b'{"x":"\xff"}\n'
    elif change == "bool_footer":
        lines = raw.splitlines(keepends=True)
        footer = json.loads(lines[-1])
        footer["sample_count"] = True
        raw = b"".join(lines[:-1]) + encode(footer)
    path.write_bytes(raw)
    with pytest.raises(ConfigurationError):
        read_native_measurements(path)


def test_empty_and_exact_equal_pts_sampled_range_roundtrip(tmp_path):
    data = measurements(())
    assert read_native_measurements(write_native_measurements(data, tmp_path / "empty")) == data
    data = measurements((0, 1, 0))
    samples = tuple(
        replace(s, pts=5000 if i < 2 else 6000, decode_index=4 + i * 3) for i, s in enumerate(data.samples)
    )
    data = replace(
        data,
        samples=samples,
        video=replace(data.video, frame_step=3, end=Fraction(7)),
        diagnostics=replace(
            data.diagnostics,
            status=NativeVideoStatus.RANGE_END,
            decoded_frames=11,
            decoded_pixels_observed=11,
        ),
    )
    restored = read_native_measurements(write_native_measurements(data, tmp_path / "sampled"))
    assert restored == data
    replay = analyze_native_measurements(restored, detectors=(DetectionConfig(detector="luminance"),))
    assert replay.analysis.cut_times == (Fraction(5), Fraction(6))
    assert replay.analysis.scenes[-1].end_time == Fraction(7)
    with pytest.raises(ConfigurationError):
        native_scene_clips(replay.analysis)


@pytest.mark.parametrize("status", [NativeVideoStatus.FRAME_LIMIT, NativeVideoStatus.DECODE_LIMIT])
def test_limited_measurements_remain_limited(tmp_path, status):
    data = measurements()
    data = replace(
        data,
        video=replace(data.video, max_frames=6, max_decoded_frames=6),
        diagnostics=replace(data.diagnostics, status=status),
    )
    restored = read_native_measurements(write_native_measurements(data, tmp_path / "limit"))
    replay = analyze_native_measurements(restored)
    assert replay.analysis.diagnostics.status is status
    with pytest.raises(ConfigurationError):
        native_scene_clips(replay.analysis, final_end=Fraction(6))


def test_adaptive_csv_keeps_absent_and_zero_separate(tmp_path):
    result = analyze_native_measurements(
        measurements((0, 0, 0, 1, 1, 1)), detectors=(DetectionConfig(window_radius=1),)
    )
    assert result.analysis.cut_positions == (3,)
    text = render_native_detection_csv(result)
    rows = list(csv.DictReader(io.StringIO(text)))
    assert rows[0]["detector_score"] == rows[1]["detector_score"] == ""
    assert rows[2]["detector_score"] == "0.0"
    assert rows[3]["qualified"] == "true" and rows[3]["pts"] == "5009"
    target = write_native_replay(result, tmp_path / "replay")
    assert (target / "statistics.csv").read_text() == text
    assert json.loads((target / "replay.json").read_text()) == result.to_dict()
    with pytest.raises(OutputError):
        render_native_detection_csv(result, max_bytes=1)


@pytest.mark.parametrize("kind", ["cache_bytes", "line_bytes", "samples", "output_bytes"])
def test_write_bounds_cleanup_and_read_bounds(tmp_path, kind):
    data = measurements()
    name = "max_" + kind
    with pytest.raises((ConfigurationError, OutputError)):
        write_native_measurements(data, tmp_path / "too-small", limits=NativeMeasurementLimits(**{name: 1}))
    assert list(tmp_path.iterdir()) == []
    if kind != "output_bytes":
        path = save_wire(tmp_path / "cache.jsonl", wire(data))
        with pytest.raises(ConfigurationError):
            read_native_measurements(path, limits=NativeMeasurementLimits(**{name: 1}))


@pytest.mark.parametrize("operation", ["analyze", "write", "csv", "report", "limits", "digest", "analysis"])
def test_public_types(operation, tmp_path):
    with pytest.raises(ConfigurationError):
        if operation == "analyze":
            analyze_native_measurements(None)
        elif operation == "write":
            write_native_measurements(None, tmp_path / "x")
        elif operation == "csv":
            render_native_detection_csv(None)
        elif operation == "report":
            write_native_replay(None, tmp_path / "x")
        elif operation == "limits":
            read_native_measurements(tmp_path / "x", limits={})
        elif operation == "digest":
            NativeReplayResult("bad", None)
        else:
            NativeReplayResult("a" * 64, None)


def test_existing_and_racing_target_preserved(tmp_path, monkeypatch):
    target = tmp_path / "exists"
    target.mkdir()
    (target / "user.txt").write_text("preserve")
    with pytest.raises(OutputError):
        write_native_measurements(measurements(), target)
    original = module._publish

    def race(stage, destination):
        destination.mkdir()
        (destination / "other.txt").write_text("racer")
        original(stage, destination)

    monkeypatch.setattr(module, "_publish", race)
    with pytest.raises(OutputError):
        write_native_measurements(measurements(), tmp_path / "race")
    assert (target / "user.txt").read_text() == "preserve"
    assert (tmp_path / "race" / "other.txt").read_text() == "racer"
    assert not list(tmp_path.glob(".frame-quorum-*"))


def test_successful_publish_with_lost_ack_is_not_deleted(tmp_path, monkeypatch):
    original = module._publish

    def lost_ack(stage, target):
        original(stage, target)
        raise OSError("acknowledgment lost")

    monkeypatch.setattr(module, "_publish", lost_ack)
    target = tmp_path / "published"
    with pytest.raises(OutputError) as failure:
        write_native_measurements(measurements(), target)
    assert read_native_measurements(target / "measurements.fqm.jsonl") == measurements()
    assert "inspect" in " ".join(failure.value.__cause__.__notes__)


def test_interrupt_not_masked_by_unlink_error(tmp_path, monkeypatch):
    def interrupted(*args):
        raise KeyboardInterrupt("stop")

    monkeypatch.setattr(module, "_publish", interrupted)
    original = Path.unlink

    def failed_unlink(path, *args, **kwargs):
        if path.name == "measurements.fqm.jsonl":
            raise OSError("cleanup failure")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", failed_unlink)
    with pytest.raises(KeyboardInterrupt) as failure:
        write_native_measurements(measurements(), tmp_path / "out")
    assert "cleanup incomplete" in " ".join(failure.value.__notes__)
    assert not (tmp_path / "out").exists()


def test_unknown_staging_files_preserved(tmp_path, monkeypatch):
    def fail(stage, target):
        (stage / "unknown.txt").write_text("not ours")
        raise OSError("publish failed")

    monkeypatch.setattr(module, "_publish", fail)
    with pytest.raises(OutputError, match="cleanup incomplete"):
        write_native_measurements(measurements(), tmp_path / "out")
    (stage,) = tmp_path.iterdir()
    assert [path.name for path in stage.iterdir()] == ["unknown.txt"]


def test_short_write_failure_cleans_owned_files(tmp_path, monkeypatch):
    original = module._managed_file

    @contextmanager
    def short(path, owned=None):
        with original(path, owned) as handle:

            class Short:
                def write(self, chunk):
                    return handle.write(chunk[:-1])

            yield Short()

    monkeypatch.setattr(module, "_managed_file", short)
    with pytest.raises(OutputError, match="short write"):
        write_native_measurements(measurements(), tmp_path / "out")
    assert list(tmp_path.iterdir()) == []


def test_missing_cache_is_domain_error(tmp_path):
    with pytest.raises(ScanError):
        read_native_measurements(tmp_path / "missing")


def test_parser_and_hash_refuse_special_file_before_open(tmp_path, monkeypatch):
    directory = tmp_path / "directory"
    directory.mkdir()
    with pytest.raises(ConfigurationError, match="regular file"):
        read_native_measurements(directory)
    with pytest.raises(ScanError, match="regular file"):
        module._hash_source(directory, 1000)


def test_file_replaced_between_stat_and_open(tmp_path, monkeypatch):
    path = save_wire(tmp_path / "cache.jsonl", wire())
    original = module._managed_file

    @contextmanager
    def replace_before_open(path, owned=None):
        path.write_bytes(path.read_bytes() + b"\n")
        with original(path, owned) as handle:
            yield handle

    monkeypatch.setattr(module, "_managed_file", replace_before_open)
    with pytest.raises(ConfigurationError, match="changed before reading"):
        read_native_measurements(path)
    with pytest.raises(ScanError, match="changed before hashing"):
        module._hash_source(path, 100000)


@pytest.mark.parametrize("mode", ["truncated", "grew"])
def test_hash_detects_change_during_bounded_read(tmp_path, monkeypatch, mode):
    path = tmp_path / "source"
    path.write_bytes(b"content")
    original = module._managed_file

    @contextmanager
    def changed(path, owned=None):
        with original(path, owned) as handle:

            class Changed:
                first = True

                def fileno(self):
                    return handle.fileno()

                def read(self, count):
                    if mode == "truncated":
                        return b""
                    if self.first:
                        self.first = False
                        return handle.read(count)
                    return b"x"

            yield Changed()

    monkeypatch.setattr(module, "_managed_file", changed)
    with pytest.raises(ScanError, match="changed while hashing"):
        module._hash_source(path, 100)


def test_footer_has_own_byte_budget_and_replay_sample_cap(tmp_path):
    data = measurements()
    body_size = sum(len(encode(record)) for record in wire(data))
    with pytest.raises(ConfigurationError, match="footer"):
        write_native_measurements(
            data, tmp_path / "out", limits=NativeMeasurementLimits(max_cache_bytes=body_size)
        )
    with pytest.raises(ConfigurationError, match="replay count"):
        write_native_replay(
            analyze_native_measurements(data), tmp_path / "out", limits=NativeMeasurementLimits(max_samples=1)
        )
    assert list(tmp_path.iterdir()) == []


def test_canonical_encoder_rejects_nonfinite_and_large_metadata():
    with pytest.raises(ConfigurationError):
        module._canonical({"x": float("inf")})
    data = measurements()
    with pytest.raises(ConfigurationError, match="128"):
        replace(data, metadata=replace(data.metadata, codec_name="x" * 129))


def test_owned_file_replacement_is_preserved(tmp_path, monkeypatch):
    def replacement(stage, target):
        original = stage / "measurements.fqm.jsonl"
        replacement = stage / "replacement"
        replacement.write_bytes(b"belongs to another operation")
        os.replace(replacement, original)
        raise OSError("publication failed")

    monkeypatch.setattr(module, "_publish", replacement)
    with pytest.raises(OutputError, match="cleanup incomplete"):
        write_native_measurements(measurements(), tmp_path / "out")
    (stage,) = tmp_path.iterdir()
    assert (stage / "measurements.fqm.jsonl").read_bytes() == b"belongs to another operation"


@pytest.mark.parametrize(
    "label",
    [
        "/tmp/video.mkv",
        "D:/Company/video.mkv",
        "/tmp/a\\b.mkv",
        "D:\\Company\\video.mkv",
        "/tmp/./nested//video.mkv",
        "/tmp/视频-🔬.mkv",
    ],
)
def test_portable_historical_path_labels_preserve_canonical_bytes(tmp_path, label):
    records = wire()
    records[0]["metadata"]["path"] = label
    source = save_wire(tmp_path / "foreign.jsonl", records)
    loaded = read_native_measurements(source)
    assert loaded.source_path_label == label
    written = write_native_measurements(loaded, tmp_path / "again")
    assert written.read_bytes() == source.read_bytes()
    replay = analyze_native_measurements(loaded)
    assert replay.to_dict()["analysis"]["metadata"]["path"] == label
    assert replay.source_path_label == label


@pytest.mark.parametrize("token", [b"9" * 10000, b"-" + b"9" * 10000, b"0." + b"1" * 100, b"1e" + b"9" * 100])
def test_integer_and_float_preconversion_limits_ignore_global_integer_limit(tmp_path, token):
    path = tmp_path / "huge.jsonl"
    path.write_bytes(b'{"sample_count":' + token + b"}\n")
    previous = sys.get_int_max_str_digits()
    try:
        sys.set_int_max_str_digits(0)
        with pytest.raises(ConfigurationError, match="before conversion"):
            read_native_measurements(path)
    finally:
        sys.set_int_max_str_digits(previous)


def test_structural_preflight_ignores_escaped_brackets_in_strings(tmp_path):
    records = wire()
    label = "/tmp/" + '["\\' * 20 + "video"
    records[0]["metadata"]["path"] = label
    loaded = read_native_measurements(save_wire(tmp_path / "brackets.jsonl", records))
    assert loaded.source_path_label == label


def test_impossible_decoded_pixel_count_rejected():
    data = measurements()
    with pytest.raises(ConfigurationError, match="diagnostics"):
        replace(data, diagnostics=replace(data.diagnostics, decoded_pixels_observed=5))


def test_sample_type_and_replay_path_bounds():
    data = measurements()
    with pytest.raises(ConfigurationError, match="NativeSceneSample"):
        replace(data, samples=(None,) * len(data.samples))
    replay = analyze_native_measurements(data)
    with pytest.raises(ConfigurationError, match="4096"):
        replace(replay, source_path_label="x" * 4097)
    assert NativeReplayResult(replay.measurement_digest, replay.analysis).source_path_label == "recorded.mkv"


def test_detector_work_admission_precedes_kernel(monkeypatch):
    data = measurements()
    data = replace(data, video=replace(data.video, max_frames=250001))
    monkeypatch.setattr(
        module, "_analyze_native_samples", lambda *args: pytest.fail("kernel ran before admission")
    )
    with pytest.raises(ConfigurationError, match="work"):
        analyze_native_measurements(
            data,
            detectors=tuple(
                DetectionConfig(detector=name)
                for name in ("content", "color", "luminance", "adaptive", "threshold")
            ),
        )


def test_identity_binds_configs_metrics_and_is_immutable():
    from dataclasses import FrozenInstanceError

    data = measurements()
    assert replace(data, video=replace(data.video, max_frames=10001)).digest != data.digest
    modified_sample = replace(data.samples[0], metrics=replace(data.samples[0].metrics, sharpness=0.5))
    assert replace(data, samples=(modified_sample, *data.samples[1:])).digest != data.digest
    detached = data.header()
    detached["metadata"]["source_bytes"] = 999
    assert data.header()["metadata"]["source_bytes"] == 4
    with pytest.raises(FrozenInstanceError):
        data.source_sha256 = "b" * 64


def test_cache_replay_in_fresh_process_where_pyav_import_is_forbidden(tmp_path):
    path = write_native_measurements(measurements(), tmp_path / "cache")
    script = """
import importlib.abc
import json
import sys

class NoAV(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'av' or fullname.startswith('av.'):
            raise RuntimeError('PyAV must not be imported for replay')

sys.meta_path.insert(0, NoAV())
from frame_quorum import read_native_measurements, analyze_native_measurements, DetectionConfig
result = analyze_native_measurements(read_native_measurements(sys.argv[1]),
    detectors=(DetectionConfig(detector='luminance', threshold=0.65),))
assert 'av' not in sys.modules
print(json.dumps(list(result.analysis.cut_positions)))
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(path)], capture_output=True, text=True, check=True, timeout=60
    )
    assert json.loads(result.stdout) == [4]


def test_maximum_native_pts_product_roundtrips_without_float(tmp_path):
    data = measurements((0,))
    maximum = (1 << 63) - 1
    sample = replace(data.samples[0], pts=maximum, time_base=Fraction(maximum, 3))
    data = replace(data, samples=(sample,))
    cache = write_native_measurements(data, tmp_path / "large-rational")
    restored = read_native_measurements(cache)
    expected = Fraction(maximum * maximum, 3)
    assert restored.samples[0].presentation_time == expected
    assert analyze_native_measurements(restored).analysis.scenes[0].start_time == expected


@pytest.mark.parametrize("failure", [OSError("stat failed"), KeyboardInterrupt("stat interrupted")])
def test_staging_identity_failure_reports_exact_unremoved_directory(tmp_path, monkeypatch, failure):
    original = Path.lstat

    def broken(path):
        if path.name.startswith(".frame-quorum-measurements-"):
            raise failure
        return original(path)

    monkeypatch.setattr(Path, "lstat", broken)
    with pytest.raises(OutputError if isinstance(failure, Exception) else KeyboardInterrupt) as caught:
        write_native_measurements(measurements(), tmp_path / "out")
    (stage,) = tmp_path.iterdir()
    details = str(caught.value) + " ".join(getattr(caught.value, "__notes__", ()))
    assert str(stage) in details and "inspect" in details
    assert stage.is_dir() and not (tmp_path / "out").exists()
