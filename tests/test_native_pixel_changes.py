"""Native change records with manually specified sums, not self-generated oracles."""

import csv
import hashlib
import json
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest

import frame_quorum.native_pixel_changes as module
from frame_quorum import (
    FrameMetrics,
    NativeMeasurementLimits,
    NativeMeasurements,
    NativeSceneSample,
    NativeVideoConfig,
    NativeVideoDiagnostics,
    NativeVideoMetadata,
    NativeVideoStatus,
    native_scene_clips,
    read_native_measurements,
    write_native_measurements,
)
from frame_quorum.cli import main
from frame_quorum.errors import ConfigurationError, OutputError
from frame_quorum.native_pixel_changes import (
    NativePixelChangeMeasurements,
    NativePixelChangeReplayResult,
    NativePixelChangeSample,
    PixelChangeDetectionConfig,
    analyze_native_pixel_changes,
    read_native_pixel_changes,
    write_native_pixel_change_replay,
    write_native_pixel_changes,
)
from frame_quorum.pixel_changes import PixelChange, PixelChangeConfig, PixelChangeLimits


def data(values=(0, 0, 255, 255, 0, 255)):
    samples, changes = [], []
    config = PixelChangeConfig()
    for index, color in enumerate(values):
        level = color / 255
        samples.append(
            NativeSceneSample(
                5000 + index * index,
                Fraction(1, 1000),
                index,
                index,
                0,
                FrameMetrics(0, level, 0, 0, 0, level, level, level),
            )
        )
        changes.append(
            None if index == 0 else PixelChange(config, 2, 2, 0, 0, 4 * abs(color - values[index - 1]), 0)
        )
    base = NativeMeasurements(
        NativeVideoConfig(max_frames=max(8, len(samples))),
        NativeVideoMetadata(
            Path("unopened.mkv"), 4, "matroska", "ffv1", 0, 2, 2, Fraction(1, 1000), 5000, None, None, None
        ),
        NativeVideoDiagnostics(NativeVideoStatus.EOF, len(samples), len(samples), len(samples) * 4, 0, True),
        tuple(samples),
        hashlib.sha256(b"test").hexdigest(),
        producer_pillow="test-producer",
    )
    return NativePixelChangeMeasurements(
        base, config, 2 if samples else None, 2 if samples else None, tuple(changes)
    )


def test_manual_value_threshold_minimum_and_exact_time_without_decoder(monkeypatch):
    import frame_quorum.native_video as native

    monkeypatch.setattr(native, "_load_av", lambda: pytest.fail("replay imported decoder"))
    result = analyze_native_pixel_changes(data(), PixelChangeDetectionConfig(value_only=True))
    assert result.cut_positions == (2, 4, 5)
    assert result.cut_times == (Fraction(1251, 250), Fraction(627, 125), Fraction(201, 40))
    assert [row.weighted_score for row in result.statistics] == [0, 0, 1, 0, 1, 1]
    minimum = analyze_native_pixel_changes(
        data(), PixelChangeDetectionConfig(value_only=True, min_scene_samples=2)
    )
    assert minimum.cut_positions == (2, 4)
    assert result.scenes[-1].end_time is None and result.source_verified is False
    assert result.to_dict()["execution"] == "cached_pixel_changes"
    with pytest.raises(ConfigurationError):
        native_scene_clips(result, final_end=Fraction(6))


def test_exact_adaptive_peak_reuses_complete_centered_window_policy():
    result = analyze_native_pixel_changes(
        data((0, 0, 0, 255, 255, 255, 255)),
        PixelChangeDetectionConfig(detector="adaptive", value_only=True, window_radius=1, min_content=0.5),
    )
    assert result.cut_positions == (3,)
    assert [row.score for row in result.statistics] == [None, None, 0, 1_000_000, 0, 0, None]


def test_canonical_roundtrip_and_exact_pair_work_admission(tmp_path):
    source = data((0, 255, 0))
    assert source.measurement_pixels == (2 * 3 - 1) * 4 == 20
    cache = write_native_pixel_changes(source, tmp_path / "cache")
    assert read_native_pixel_changes(cache) == source
    with pytest.raises(ConfigurationError, match="pixel work"):
        read_native_pixel_changes(cache, pixel_limits=PixelChangeLimits(max_measurement_pixels=19))
    assert (
        read_native_pixel_changes(cache, pixel_limits=PixelChangeLimits(max_measurement_pixels=20)) == source
    )


def test_empty_and_singleton_keep_absent_pair_evidence(tmp_path):
    for values in ((), (0,)):
        source = data(values)
        assert source.changes == (() if not values else (None,))
        cache = write_native_pixel_changes(source, tmp_path / f"n{len(values)}")
        result = analyze_native_pixel_changes(read_native_pixel_changes(cache))
        assert result.cut_positions == () and len(result.scenes) == len(values)


def records():
    value = data()
    return [
        value.header(),
        *(
            {"sample": s.to_dict(), "change": c.to_dict() if c else None}
            for s, c in zip(value.base.samples, value.changes, strict=True)
        ),
    ]


def save(path, rows):
    def encode(value):
        return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode()

    raw = b"".join(map(encode, rows))
    footer = {"kind": "end", "sample_count": len(rows) - 1, "sha256": hashlib.sha256(raw).hexdigest()}
    path.write_bytes(raw + encode(footer))
    return path


@pytest.mark.parametrize(
    "index,path,value",
    [
        (0, ("kind",), "frame-quorum-native-histograms"),
        (0, ("schema_version",), True),
        (0, ("schema_version",), 2),
        (0, ("measurement_version",), "unknown"),
        (0, ("pixel_config", "edge_radius"), 0),
        (0, ("width",), None),
        (0, ("height",), True),
        (0, ("width",), 16_777_217),
        (0, ("sample_count",), True),
        (0, ("base", "sample_count"), 5),
        (0, ("base", "diagnostics", "decoded_pixels_observed"), 6),
        (1, ("sample", "sample_index"), 1),
        (2, ("change",), None),
        (2, ("change", "hue_sum"), 10**15),
        (2, ("change", "edge_sum"), True),
        (2, ("change", "value_sum"), -1),
        (2, ("change", "width"), 1),
        (2, ("sample", "presentation_time", "numerator"), 7),
    ],
)
def test_semantic_corruption_rejected_even_with_recomputed_checksum(tmp_path, index, path, value):
    rows = records()
    target = rows[index]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ConfigurationError):
        read_native_pixel_changes(save(tmp_path / "bad.jsonl", rows))


@pytest.mark.parametrize(
    "transform",
    [
        lambda raw: raw[:-1],
        lambda raw: raw + b"\n",
        lambda raw: raw.replace(b"\n", b"\r\n"),
        lambda raw: raw.replace(b'"schema_version":1', b'"schema_version":1,"schema_version":1', 1),
        lambda raw: raw.replace(b'"width":2', b'"width":NaN', 1),
        lambda raw: raw.replace(b'"width":2', b'"width":' + b"9" * 10000, 1),
        lambda raw: raw.replace(b'"width":2', b'"width":' + b"[" * 17 + b"0" + b"]" * 17, 1),
        lambda raw: raw.replace(b'"value_sum":0', b'"value_sum":1', 1),
    ],
)
def test_shared_strict_wire_controls(tmp_path, transform):
    path = save(tmp_path / "cache", records())
    path.write_bytes(transform(path.read_bytes()))
    with pytest.raises(ConfigurationError):
        read_native_pixel_changes(path)


def test_native_constructor_cross_fields_and_first_evidence():
    value = data()
    for kwargs in (
        {"base": None},
        {"config": None},
        {"changes": []},
        {"changes": value.changes[:-1]},
        {"changes": (value.changes[1], *value.changes[1:])},
        {"changes": (None, None, *value.changes[2:])},
        {"width": 1},
        {"height": None},
        {"config": PixelChangeConfig(2)},
        {"base": replace(value.base, video=replace(value.base.video, max_frame_pixels=1))},
        {"base": replace(value.base, diagnostics=replace(value.base.diagnostics, decoded_pixels_observed=6))},
    ):
        with pytest.raises(ConfigurationError):
            replace(value, **kwargs)
    with pytest.raises(ConfigurationError):
        replace(data(()), width=2)
    sample = value.base.samples[0]
    for args in (
        (None, 2, 2, None),
        (sample, 2, 2, value.changes[1]),
        (value.base.samples[1], 1, 2, value.changes[1]),
    ):
        with pytest.raises(ConfigurationError):
            NativePixelChangeSample(*args)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"detector": "threshold"},
        {"detector": []},
        {"weights": None},
        {"value_only": 1},
        {"threshold": float("nan")},
        {"threshold": float("inf")},
        {"threshold": -1},
        {"min_scene_samples": True},
        {"window_radius": 0},
        {"adaptive_ratio": 10**1000},
        {"min_content": 2},
    ],
)
def test_detector_options_strict(kwargs):
    with pytest.raises(ConfigurationError):
        PixelChangeDetectionConfig(**kwargs)


def test_results_validate_scores_against_integer_evidence_and_exact_policy():
    result = analyze_native_pixel_changes(data())
    for kwargs in (
        {"measurements": None},
        {"config": None},
        {"statistics": []},
        {"statistics": ()},
        {"statistics": (None, *result.statistics[1:])},
        {"statistics": (replace(result.statistics[0], weighted_score=0.1), *result.statistics[1:])},
    ):
        with pytest.raises(ConfigurationError):
            replace(result, **kwargs)
    assert isinstance(result, NativePixelChangeReplayResult)
    assert analyze_native_pixel_changes(
        data((0, 0)), PixelChangeDetectionConfig(threshold=0)
    ).cut_positions == (1,)
    tail = analyze_native_pixel_changes(data((0, 0, 0, 255)), PixelChangeDetectionConfig(min_scene_samples=2))
    assert tail.cut_positions == () and tail.statistics[-1].reason == "short_final_scene"


def test_budget_and_type_admission_before_side_effects(tmp_path, monkeypatch):
    value = data()
    for options in (
        {"pixel_limits": object()},
        {"limits": object()},
        {"limits": NativeMeasurementLimits(max_samples=1)},
        {"pixel_limits": PixelChangeLimits(max_pair_rgb_bytes=6)},
        {"pixel_limits": PixelChangeLimits(max_measurement_pixels=1)},
    ):
        with pytest.raises(ConfigurationError):
            analyze_native_pixel_changes(value, **options)
    with pytest.raises(ConfigurationError):
        analyze_native_pixel_changes(value.base)
    with pytest.raises(ConfigurationError):
        analyze_native_pixel_changes(value, object())
    with pytest.raises(ConfigurationError):
        write_native_pixel_change_replay(value, tmp_path / "out")
    monkeypatch.setattr(module, "_parse_sample", lambda _: pytest.fail("work admission too late"))
    with pytest.raises(ConfigurationError, match="before record"):
        read_native_pixel_changes(
            save(tmp_path / "cache", records()), pixel_limits=PixelChangeLimits(max_measurement_pixels=1)
        )
    assert not (tmp_path / "out").exists()


def test_new_wire_cannot_upgrade_old_and_old_golden_unchanged(tmp_path):
    old = write_native_measurements(data().base, tmp_path / "old")
    with pytest.raises(ConfigurationError):
        read_native_pixel_changes(old)
    new = write_native_pixel_changes(data(), tmp_path / "new")
    with pytest.raises(ConfigurationError):
        read_native_measurements(new)
    assert len(old.read_bytes()) == 2874
    assert (
        hashlib.sha256(old.read_bytes()).hexdigest()
        == "7f717a62f64de44bf75e2cb0909b604e13be6bbd4ba2da617cbede40231f1e9f"
    )
    assert new.read_bytes() == save(tmp_path / "independent", records()).read_bytes()


def test_exact_csv_and_no_replace_atomic_bundles(tmp_path):
    value = data()
    result = analyze_native_pixel_changes(value, PixelChangeDetectionConfig(value_only=True))
    path = write_native_pixel_change_replay(result, tmp_path / "review")
    report = json.loads((path / "replay.json").read_bytes())
    assert report == result.to_dict() and report["source_verified"] is False
    with (path / "statistics.csv").open(newline="", encoding="ascii") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 6 and rows[2]["value_change"] == "1.0"
    assert rows[2]["pts"] == "5004" and rows[2]["presentation_denominator"] == "250"
    assert rows[0]["weighted_score"] == "0.0" and rows[2]["accepted"] == "true"
    before = (path / "replay.json").read_bytes()
    with pytest.raises(OutputError):
        write_native_pixel_change_replay(result, path)
    assert (path / "replay.json").read_bytes() == before
    for writer, item in ((write_native_pixel_changes, value), (write_native_pixel_change_replay, result)):
        with pytest.raises(OutputError):
            writer(item, tmp_path / "short", limits=NativeMeasurementLimits(max_output_bytes=1))
        assert not (tmp_path / "short").exists()
    assert not list(tmp_path.glob(".*.tmp-*"))


def test_cli_invalid_weights_precede_cache_open(tmp_path, capsys):
    assert (
        main(
            [
                "native-change-replay",
                str(tmp_path / "missing"),
                "--weights",
                "0",
                "0",
                "0",
                "0",
                "-o",
                str(tmp_path / "out"),
            ]
        )
        == 2
    )
    assert "positive" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_histogram_golden_from_signed_baseline_is_unchanged(tmp_path):
    from frame_quorum import (
        NativeHistogramMeasurements,
        PixelHistogram,
        PixelHistogramConfig,
        read_native_histograms,
        write_native_histograms,
    )

    base = data().base
    config = PixelHistogramConfig()
    histograms = tuple(
        PixelHistogram(
            config,
            2,
            2,
            tuple(int(b == (31 if sample.metrics.mean_red else 0)) for _ in range(12) for b in range(32)),
        )
        for sample in base.samples
    )
    value = NativeHistogramMeasurements(base, config, histograms)
    path = write_native_histograms(value, tmp_path / "histogram")
    # Signed b5925bdb modules/fixture unchanged; independently frozen header+rows.
    raw = b"".join(path.read_bytes().splitlines(keepends=True)[:-1])
    assert len(raw) == 7912
    assert (
        hashlib.sha256(raw).hexdigest() == "da8ebbf429f09f95d6b028e0ebf30112965089c7cb13896b234739873e017d05"
    )
    assert read_native_histograms(path) == value
    with pytest.raises(ConfigurationError):
        read_native_pixel_changes(path)
    new = write_native_pixel_changes(data(), tmp_path / "change")
    with pytest.raises(ConfigurationError):
        read_native_histograms(new)


@pytest.mark.parametrize(
    "label", ["/tmp/a\\b.mkv", "D:\\Media\\clip.mkv", "relative//./clip.mkv", "媒体.mkv"]
)
def test_historical_path_label_and_digest_bindings(tmp_path, label):
    value = data()
    value = replace(value, base=replace(value.base, source_path_label=label))
    restored = read_native_pixel_changes(write_native_pixel_changes(value, tmp_path / "cache"))
    assert restored.header()["base"]["metadata"]["path"] == label
    assert replace(value, base=replace(value.base, source_sha256="a" * 64)).digest != value.digest
    header = restored.header()
    header["pixel_config"]["edge_radius"] = 4
    assert value.config.edge_radius == 1


def test_owned_bundle_race_lost_ack_and_control_cleanup(tmp_path, monkeypatch):
    import frame_quorum.native_measurements as transport

    original = transport._publish

    def raced(stage, destination):
        destination.mkdir()
        (destination / "user").write_bytes(b"keep")
        original(stage, destination)

    monkeypatch.setattr(transport, "_publish", raced)
    with pytest.raises(OutputError):
        write_native_pixel_changes(data(), tmp_path / "race")
    assert (tmp_path / "race" / "user").read_bytes() == b"keep"
    assert not (tmp_path / "race" / "pixel-changes.fqm.jsonl").exists()

    def lost(stage, destination):
        original(stage, destination)
        raise OSError("lost acknowledgment")

    monkeypatch.setattr(transport, "_publish", lost)
    with pytest.raises(OutputError):
        write_native_pixel_changes(data(), tmp_path / "lost")
    assert read_native_pixel_changes(tmp_path / "lost" / "pixel-changes.fqm.jsonl") == data()
    lines = module._lines

    def interrupt(value):
        yield next(lines(value))
        raise KeyboardInterrupt("serialization stop")

    monkeypatch.setattr(module, "_lines", interrupt)
    with pytest.raises(KeyboardInterrupt, match="serialization stop"):
        write_native_pixel_changes(data(), tmp_path / "interrupted")
    assert not (tmp_path / "interrupted").exists()
    assert not list(tmp_path.glob(".frame-quorum-measurements-*"))


def test_owned_bundle_preserves_unknown_staged_entries(tmp_path, monkeypatch):
    import frame_quorum.native_measurements as transport

    stages = []

    def fail(stage, destination):
        stages.append(stage)
        (stage / "not-ours").write_bytes(b"keep")
        raise OSError("no publication")

    monkeypatch.setattr(transport, "_publish", fail)
    with pytest.raises(OutputError):
        write_native_pixel_changes(data(), tmp_path / "target")
    assert (stages[0] / "not-ours").read_bytes() == b"keep"
    assert not (stages[0] / "pixel-changes.fqm.jsonl").exists()
    assert not (tmp_path / "target").exists()


def test_header_dimensions_rejected_by_original_frame_budget_before_records(tmp_path, monkeypatch):
    rows = records()
    rows[0]["width"], rows[0]["height"] = 5000, 5000
    monkeypatch.setattr(module, "_parse_sample", lambda _: pytest.fail("pixel admission after records"))
    with pytest.raises(ConfigurationError, match="original frame"):
        read_native_pixel_changes(
            save(tmp_path / "cache", rows),
            pixel_limits=PixelChangeLimits(max_measurement_pixels=1_000_000_000),
        )


def test_compiled_work_ceiling_without_allocating_large_frames():
    value = data((0,) * 8)
    area = 67_108_864
    base = replace(
        value.base,
        video=replace(value.base.video, max_frame_pixels=area, max_total_pixels=2_000_000_000),
        diagnostics=replace(value.base.diagnostics, decoded_pixels_observed=8 * area),
    )
    change = PixelChange(value.config, area, 1, 0, 0, 0, 0)
    with pytest.raises(ConfigurationError, match="compiled ceiling"):
        NativePixelChangeMeasurements(base, value.config, area, 1, (None,) * (1) + (change,) * 7)
