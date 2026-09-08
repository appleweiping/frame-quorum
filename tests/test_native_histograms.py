from __future__ import annotations

import csv
import hashlib
import io
import json
from dataclasses import FrozenInstanceError, replace
from fractions import Fraction
from pathlib import Path

import pytest
from PIL import Image

import frame_quorum.native_histograms as module
from frame_quorum import (
    FrameMetrics,
    HistogramDetectionConfig,
    NativeHistogramMeasurements,
    NativeHistogramSample,
    NativeMeasurementLimits,
    NativeMeasurements,
    NativeSceneSample,
    NativeVideoConfig,
    NativeVideoDiagnostics,
    NativeVideoMetadata,
    NativeVideoStatus,
    PixelHistogramConfig,
    PixelHistogramLimits,
    analyze_native_histograms,
    native_scene_clips,
    read_native_histograms,
    read_native_measurements,
    write_native_histogram_replay,
    write_native_histograms,
    write_native_measurements,
)
from frame_quorum.cli import main
from frame_quorum.errors import ConfigurationError, OutputError, ScanError
from frame_quorum.pixel_histograms import measure_pixel_histogram


def data(values=(0, 0, 255, 255, 0, 255), *, status=NativeVideoStatus.EOF):
    samples, histograms = [], []
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
        with Image.new("RGB", (2, 2), (color, color, color)) as image:
            histograms.append(measure_pixel_histogram(image))
    base = NativeMeasurements(
        NativeVideoConfig(max_frames=max(8, len(samples))),
        NativeVideoMetadata(
            Path("unopened.mkv"), 4, "matroska", "ffv1", 0, 2, 2, Fraction(1, 1000), 5000, None, None, None
        ),
        NativeVideoDiagnostics(status, len(samples), len(samples), len(samples) * 4, 0, True),
        tuple(samples),
        hashlib.sha256(b"test").hexdigest(),
        producer_pillow="test-producer",
    )
    return NativeHistogramMeasurements(base, PixelHistogramConfig(), tuple(histograms))


def encode(value):
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False) + "\n"
    ).encode("ascii")


def records(value=None):
    value = data() if value is None else value
    return [
        value.header(),
        *(
            {"sample": sample.to_dict(), "histogram": histogram.to_dict()}
            for sample, histogram in zip(value.base.samples, value.histograms, strict=True)
        ),
    ]


def save(path, values):
    raw = b"".join(map(encode, values))
    footer = {"kind": "end", "sample_count": len(values) - 1, "sha256": hashlib.sha256(raw).hexdigest()}
    path.write_bytes(raw + encode(footer))
    return path


def test_manual_threshold_minimum_tail_and_exact_times_without_decoder(monkeypatch):
    import frame_quorum.native_video as native

    monkeypatch.setattr(native, "_load_av", lambda: pytest.fail("decoder imported for replay"))
    value = data()
    result = analyze_native_histograms(value)
    assert result.cut_positions == (2, 4, 5)
    assert result.cut_times == (Fraction(1251, 250), Fraction(627, 125), Fraction(201, 40))
    assert [row.score for row in result.statistics] == [0, 0, 1, 0, 1, 1]
    minimum = analyze_native_histograms(value, HistogramDetectionConfig(min_scene_samples=2))
    assert minimum.cut_positions == (2, 4)
    assert minimum.statistics[-1].reason == "short_previous_scene"
    tail = analyze_native_histograms(
        data((0, 0, 0, 0, 0, 255)), HistogramDetectionConfig(min_scene_samples=2)
    )
    assert tail.cut_positions == () and tail.statistics[-1].reason == "short_final_scene"
    assert result.scenes[-1].end_time is None
    assert result.source_verified is False
    assert result.to_dict()["execution"] == "cached_histograms"
    with pytest.raises(ConfigurationError):
        native_scene_clips(result, final_end=Fraction(6))


def test_exact_threshold_equal_and_zero_semantics():
    assert analyze_native_histograms(data(), HistogramDetectionConfig(threshold=1)).cut_positions == (2, 4, 5)
    assert analyze_native_histograms(
        data((0, 0, 0)), HistogramDetectionConfig(threshold=0)
    ).cut_positions == (1, 2)


@pytest.mark.parametrize("values,scene_count", [((), 0), ((0,), 1), ((0, 0, 255), 2)])
def test_complete_canonical_roundtrip_and_independent_checksum(tmp_path, values, scene_count):
    value = data(values)
    cache = write_native_histograms(value, tmp_path / "cache")
    assert cache.name == "histograms.fqm.jsonl"
    assert read_native_histograms(cache) == value
    raw = cache.read_bytes().splitlines(keepends=True)
    assert hashlib.sha256(b"".join(raw[:-1])).hexdigest() == value.digest
    assert json.loads(raw[-1])["sha256"] == value.digest
    expected = save(tmp_path / "independent.jsonl", records(value)).read_bytes()
    assert cache.read_bytes() == expected
    assert len(analyze_native_histograms(value).scenes) == scene_count


def test_range_end_equal_pts_and_stride_preserved(tmp_path):
    value = data((0, 0, 255))
    samples = tuple(
        replace(sample, pts=5000, decode_index=i * 3) for i, sample in enumerate(value.base.samples)
    )
    base = replace(
        value.base,
        video=replace(value.base.video, frame_step=3, end=Fraction(6)),
        samples=samples,
        diagnostics=NativeVideoDiagnostics(NativeVideoStatus.RANGE_END, 8, 3, 32, 0, True),
    )
    value = replace(value, base=base)
    restored = read_native_histograms(write_native_histograms(value, tmp_path / "cache"))
    result = analyze_native_histograms(restored)
    assert result.cut_times == (Fraction(5),)
    assert result.scenes[0].end_time == 5
    assert result.scenes[-1].end_time == 6
    assert result.scenes[-1].end_reason == "requested_end"


@pytest.mark.parametrize(
    "label", ["/tmp/a\\b.mkv", "D:\\Media\\clip.mkv", "relative//./clip.mkv", "媒体.mkv"]
)
def test_historical_label_is_verbatim_not_host_path_normalization(tmp_path, label):
    value = data((0,))
    value = replace(value, base=replace(value.base, source_path_label=label))
    restored = read_native_histograms(write_native_histograms(value, tmp_path / "cache"))
    assert restored.header()["base"]["metadata"]["path"] == label
    assert analyze_native_histograms(restored).to_dict()["capture"]["base"]["metadata"]["path"] == label


def test_old_cache_and_type_not_silently_upgraded(tmp_path):
    value = data()
    old = write_native_measurements(value.base, tmp_path / "old")
    with pytest.raises(ConfigurationError):
        read_native_histograms(old)
    with pytest.raises(ConfigurationError, match="summary-only"):
        analyze_native_histograms(value.base)
    new = write_native_histograms(value, tmp_path / "new")
    with pytest.raises(ConfigurationError):
        read_native_measurements(new)
    assert read_native_measurements(old) == value.base
    # Computed independently with this fixture and the serializer at the signed
    # pre-histogram baseline 801a2b726930ad1c65c12012827845a158b4cf61.
    assert len(old.read_bytes()) == 2874
    assert (
        hashlib.sha256(old.read_bytes()).hexdigest()
        == "7f717a62f64de44bf75e2cb0909b604e13be6bbd4ba2da617cbede40231f1e9f"
    )


@pytest.mark.parametrize(
    "index,path,value",
    [
        (0, ("kind",), "frame-quorum-native-measurements"),
        (0, ("schema_version",), True),
        (0, ("measurement_version",), "future"),
        (0, ("histogram_config", "bins"), 3),
        (0, ("base", "sample_count"), True),
        (0, ("base", "sample_count"), 1),
        (0, ("base", "diagnostics", "decoded_pixels_observed"), 6),
        (1, ("histogram", "width"), 0),
        (1, ("histogram", "counts"), [0]),
        (1, ("histogram", "counts"), "bad"),
        (1, ("histogram", "counts", 0), True),
        (1, ("histogram", "counts", 0), 0),
        (1, ("histogram", "counts", 0), 2),
        (1, ("sample", "sample_index"), 1),
        (1, ("sample", "presentation_time", "numerator"), True),
    ],
)
def test_recomputed_checksum_still_rejects_semantic_corruption(tmp_path, index, path, value):
    values = records()
    target = values[index]
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = value
    with pytest.raises(ConfigurationError):
        read_native_histograms(save(tmp_path / "corrupt.jsonl", values))


@pytest.mark.parametrize(
    "transform",
    [
        lambda raw: raw[:-1],
        lambda raw: raw + b"\n",
        lambda raw: raw.replace(b'"schema_version":1', b'"schema_version":1,"schema_version":1', 1),
        lambda raw: raw.replace(b"\n", b"\r\n"),
        lambda raw: raw.replace(b'"width":2', b'"width":NaN', 1),
        lambda raw: raw.replace(b'"width":2', b'"width":' + b"9" * 10000, 1),
        lambda raw: raw.replace(b'"width":2', b'"width":' + b"[" * 17 + b"0" + b"]" * 17, 1),
        lambda raw: raw.replace(b'"counts":[1', b'"counts":[0', 1),
    ],
)
def test_shared_strict_transport_rejections(tmp_path, transform):
    source = save(tmp_path / "cache", records())
    source.write_bytes(transform(source.read_bytes()))
    with pytest.raises(ConfigurationError):
        read_native_histograms(source)


def test_constructor_immutability_and_cross_field_bounds():
    value = data()
    for changes in (
        {"base": None},
        {"config": None},
        {"histograms": []},
        {"histograms": value.histograms[:-1]},
        {"histograms": (None, *value.histograms[1:])},
        {"config": PixelHistogramConfig(bins=64)},
    ):
        with pytest.raises(ConfigurationError):
            replace(value, **changes)
    with pytest.raises(ConfigurationError, match="compiled"):
        replace(value, base=replace(value.base, video=replace(value.base.video, max_frames=50_000)))
    with pytest.raises(ConfigurationError, match="per-frame"):
        replace(value, base=replace(value.base, video=replace(value.base.video, max_frame_pixels=1)))
    with pytest.raises(ConfigurationError, match="selected"):
        replace(
            value,
            base=replace(value.base, diagnostics=replace(value.base.diagnostics, decoded_pixels_observed=6)),
        )
    with pytest.raises(ConfigurationError, match="unselected"):
        replace(
            value, base=replace(value.base, diagnostics=replace(value.base.diagnostics, decoded_frames=20))
        )
    detached = value.header()
    detached["histogram_config"]["bins"] = 256
    assert value.config.bins == 32
    with pytest.raises(FrozenInstanceError):
        value.config = PixelHistogramConfig()
    assert replace(value, base=replace(value.base, source_sha256="a" * 64)).digest != value.digest
    for sample, histogram in ((None, value.histograms[0]), (value.base.samples[0], None)):
        with pytest.raises(ConfigurationError):
            NativeHistogramSample(sample, histogram)


def test_budget_admission_before_sample_parser_and_distance(tmp_path, monkeypatch):
    value = data()
    source = save(tmp_path / "cache", records(value))
    low_slots = PixelHistogramLimits(max_histogram_values=100)
    monkeypatch.setattr(module, "_parse_sample", lambda raw: pytest.fail("parsed before admission"))
    monkeypatch.setattr(
        module, "histogram_distance", lambda *a, **k: pytest.fail("distance before admission")
    )
    with pytest.raises(ConfigurationError, match="slots"):
        read_native_histograms(source, pixel_limits=low_slots)
    with pytest.raises(ConfigurationError, match="slots"):
        analyze_native_histograms(value, pixel_limits=low_slots)
    with pytest.raises(ConfigurationError, match="sample admission"):
        read_native_histograms(source, limits=NativeMeasurementLimits(max_samples=7))
    with pytest.raises(ConfigurationError, match="pixels"):
        analyze_native_histograms(value, pixel_limits=PixelHistogramLimits(max_measurement_pixels=1))


def test_each_cache_and_output_limit(tmp_path):
    value = data()
    cache = save(tmp_path / "cache", records(value))
    for bounds in (NativeMeasurementLimits(max_cache_bytes=1), NativeMeasurementLimits(max_line_bytes=1)):
        with pytest.raises(ConfigurationError):
            read_native_histograms(cache, limits=bounds)
        with pytest.raises(ConfigurationError):
            write_native_histograms(value, tmp_path / "new", limits=bounds)
        assert not (tmp_path / "new").exists()
    with pytest.raises(ConfigurationError, match="pixels"):
        read_native_histograms(cache, pixel_limits=PixelHistogramLimits(max_measurement_pixels=7))
    with pytest.raises(OutputError):
        write_native_histograms(value, tmp_path / "new", limits=NativeMeasurementLimits(max_output_bytes=1))
    result = analyze_native_histograms(value)
    with pytest.raises(OutputError):
        write_native_histogram_replay(
            result, tmp_path / "new", limits=NativeMeasurementLimits(max_output_bytes=1)
        )
    assert not list(tmp_path.glob(".frame-quorum-measurements-*"))
    with pytest.raises(ScanError):
        read_native_histograms(tmp_path / "missing")


def test_published_report_csv_and_cli_without_native_import(tmp_path, monkeypatch, capsys):
    value = data()
    cache = write_native_histograms(value, tmp_path / "cache")
    import frame_quorum.native_video as native

    monkeypatch.setattr(native, "_load_av", lambda: pytest.fail("decoder loaded"))
    output = tmp_path / "review"
    assert (
        main(["native-histogram-replay", str(cache), "--output-dir", str(output), "--min-scene-samples", "2"])
        == 0
    )
    summary = json.loads(capsys.readouterr().out)
    assert summary["cut_positions"] == [2, 4] and summary["source_verified"] is False
    report = json.loads((output / "replay.json").read_bytes())
    assert report["measurement_digest"] == value.digest
    rows = list(csv.DictReader(io.StringIO((output / "statistics.csv").read_text())))
    assert [int(row["sample_index"]) for row in rows] == list(range(6))
    assert rows[2]["presentation_numerator"] == "1251" and rows[2]["presentation_denominator"] == "250"
    assert rows[-1]["candidate"] == "true" and rows[-1]["accepted"] == "false"
    assert rows[0]["histogram_score"] == "0.0"
    before = (output / "replay.json").read_bytes()
    assert main(["native-histogram-replay", str(cache), "--output-dir", str(output)]) == 2
    assert (output / "replay.json").read_bytes() == before


@pytest.mark.parametrize(
    "changes",
    [
        {"measurements": None},
        {"config": None},
        {"statistics": []},
        {"statistics": ()},
        {"statistics": (None,) * 6},
    ],
)
def test_result_rejects_inconsistent_fields(changes):
    with pytest.raises(ConfigurationError):
        replace(analyze_native_histograms(data()), **changes)


def test_decision_contradictions_invalid_configuration_and_public_types(tmp_path):
    result = analyze_native_histograms(data())
    for row in (replace(result.statistics[0], score=1), replace(result.statistics[0], candidate=True)):
        with pytest.raises(ConfigurationError):
            replace(result, statistics=(row, *result.statistics[1:]))
    for changes in (
        {"score": True},
        {"score": float("nan")},
        {"candidate": 1},
        {"accepted": 1},
        {"reason": "bad"},
    ):
        with pytest.raises(ConfigurationError):
            replace(result.statistics[0], **changes)
    with pytest.raises(ConfigurationError):
        analyze_native_histograms(data(), object())
    with pytest.raises(ConfigurationError):
        write_native_histogram_replay(None, tmp_path / "out")


def test_threshold_consistent_fabricated_scores_are_rejected_against_actual_counts():
    result = analyze_native_histograms(data())
    # These rows are internally consistent with threshold=0.5, but falsely
    # suppress every actual distribution change in the stored histograms.
    fabricated = (
        result.statistics[0],
        *(module.HistogramStatistic(0.0, False, False, "below_threshold") for _ in result.statistics[1:]),
    )
    with pytest.raises(ConfigurationError, match="actual stored pixel counts"):
        replace(result, statistics=fabricated)


def test_legacy_path_resolution_error_keeps_original_exception_boundary(monkeypatch):
    import frame_quorum.native_measurements as legacy

    error = OSError("resolution failed")

    def unresolved(value):
        raise error

    monkeypatch.setattr(legacy, "_source_path", unresolved)
    with pytest.raises(OSError) as captured:
        legacy.capture_native_measurements("unused.mkv")
    assert captured.value is error


def test_huge_declared_pixel_work_rejected_without_allocating_pixel_images():
    from frame_quorum import PixelHistogram

    config = PixelHistogramConfig(bins=8)
    # 16 repeated 8192x8192 descriptors exceed a billion pixels while the test
    # allocates only 96 integer count slots, not any large pixel buffers.
    counts = tuple(number for _ in range(12) for number in (16_777_216, 0, 0, 0, 0, 0, 0, 0))
    histogram = PixelHistogram(config, 8192, 8192, counts)
    base = data((0,) * 16).base
    base = replace(
        base,
        video=NativeVideoConfig(max_frames=16, max_frame_pixels=67_108_864, max_total_pixels=2_000_000_000),
        diagnostics=replace(base.diagnostics, decoded_pixels_observed=16 * 67_108_864),
    )
    with pytest.raises(ConfigurationError, match="ceiling"):
        NativeHistogramMeasurements(base, config, (histogram,) * 16)


def test_publication_race_does_not_replace_existing_target(tmp_path, monkeypatch):
    import frame_quorum.native_measurements as transport

    original = transport._publish
    target = tmp_path / "target"

    def raced(stage, destination):
        destination.mkdir()
        (destination / "sentinel").write_bytes(b"user")
        original(stage, destination)

    monkeypatch.setattr(transport, "_publish", raced)
    with pytest.raises(OutputError):
        write_native_histograms(data(), target)
    assert (target / "sentinel").read_bytes() == b"user"
    assert not (target / "histograms.fqm.jsonl").exists()
    assert not list(tmp_path.glob(".frame-quorum-measurements-*"))


def test_lost_publication_ack_retains_successfully_published_histograms(tmp_path, monkeypatch):
    import frame_quorum.native_measurements as transport

    original = transport._publish
    value = data()
    target = tmp_path / "target"

    def lost(stage, destination):
        original(stage, destination)
        raise OSError("lost acknowledgment")

    monkeypatch.setattr(transport, "_publish", lost)
    with pytest.raises(OutputError) as caught:
        write_native_histograms(value, target)
    assert read_native_histograms(target / "histograms.fqm.jsonl") == value
    assert "publication was attempted" in str(getattr(caught.value.__cause__, "__notes__", ()))


def test_cleanup_preserves_unknown_staged_file_and_reports_residue(tmp_path, monkeypatch):
    import frame_quorum.native_measurements as transport

    stages = []

    def failed(stage, destination):
        stages.append(stage)
        (stage / "unowned").write_bytes(b"not ours")
        raise OSError("publication rejected")

    monkeypatch.setattr(transport, "_publish", failed)
    with pytest.raises(OutputError):
        write_native_histograms(data(), tmp_path / "target")
    assert (stages[0] / "unowned").read_bytes() == b"not ours"
    assert not (stages[0] / "histograms.fqm.jsonl").exists()
    assert not (tmp_path / "target").exists()


def test_cache_production_interrupt_cleans_only_its_own_stage(tmp_path, monkeypatch):
    original = module._histogram_lines

    def interrupt(value):
        yield next(original(value))
        raise KeyboardInterrupt("serialization stop")

    monkeypatch.setattr(module, "_histogram_lines", interrupt)
    with pytest.raises(KeyboardInterrupt, match="serialization stop"):
        write_native_histograms(data(), tmp_path / "target")
    assert not list(tmp_path.iterdir())
