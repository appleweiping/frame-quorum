from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from dataclasses import replace
from itertools import pairwise
from pathlib import Path

import pytest

from frame_quorum._version import VERSION
from frame_quorum.benchmark import (
    BenchmarkConfig,
    BenchmarkProvenance,
    BenchmarkResult,
    BenchmarkRun,
    EvaluationMetrics,
    benchmark_manifest,
    evaluate_selection,
    render_benchmark_svg,
    run_benchmark,
    source_content_digest,
)
from frame_quorum.errors import ConfigurationError, OutputError, ScanError
from frame_quorum.models import Frame, FrameMetrics, ScanConfig, SelectionConfig
from frame_quorum.scanner import scan_frames


def _frames(image_factory: Callable[..., Path], root: Path, count: int = 12) -> tuple[Frame, ...]:
    directory = root / "frames"
    for index in range(count):
        image_factory(
            f"frame_{index:03d}.png",
            color=((index * 29) % 255, (index * 53) % 255, (index * 71) % 255),
            pattern=index + 1,
            directory=directory,
        )
    return scan_frames(directory)


def _synthetic_frame(index: int) -> Frame:
    value = (index * 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
    fraction = (index % 17) / 16
    metrics = FrameMetrics(value, fraction, fraction, fraction, fraction, fraction, 0.5, 1 - fraction)
    return Frame(index, Path(f"{index}.png"), f"{index}.png", float(index), 64, 48, 100, metrics)


def test_benchmark_compares_three_baselines_and_primary_method(
    image_factory: Callable[..., Path], tmp_path: Path
) -> None:
    result = run_benchmark(
        _frames(image_factory, tmp_path),
        SelectionConfig(budget=4, duplicate_threshold=0),
        BenchmarkConfig(random_seed=11, random_trials=3),
    )
    assert [run.method for run in result.runs] == [
        "frame_quorum",
        "time_uniform",
        "change_peaks",
        "seeded_random",
        "seeded_random",
        "seeded_random",
    ]
    assert [run.seed for run in result.runs[-3:]] == [11, 12, 13]
    assert all(len(run.selected_indices) == 4 for run in result.runs)


def test_benchmark_is_byte_stable_without_runtime_measurements(
    image_factory: Callable[..., Path], tmp_path: Path
) -> None:
    frames = _frames(image_factory, tmp_path)
    first = benchmark_manifest(run_benchmark(frames))
    second = benchmark_manifest(run_benchmark(list(reversed(frames))))
    assert first == second
    assert first["protocol"]["measured_record_fingerprint"].startswith("sha256:")
    assert len(first["protocol"]["measured_record_fingerprint"]) == 71
    assert "lossy image metrics" in first["protocol"]["measured_record_fingerprint_algorithm"]
    assert first["aggregates"][-1]["trial_count"] == 8


def test_all_frames_give_complete_content_and_temporal_coverage(
    image_factory: Callable[..., Path], tmp_path: Path
) -> None:
    frames = _frames(image_factory, tmp_path, 4)
    metrics = evaluate_selection(frames, [frame.index for frame in frames])
    assert metrics.content_coverage == pytest.approx(1.0)
    assert metrics.temporal_coverage == pytest.approx(1.0)
    assert metrics.change_coverage == pytest.approx(1.0)
    assert 0 <= metrics.mean_quality <= 1
    assert 0 <= metrics.non_redundancy <= 1


def test_single_frame_sequence_has_defined_metrics(
    image_factory: Callable[..., Path], tmp_path: Path
) -> None:
    frame = _frames(image_factory, tmp_path, 1)[0]
    metrics = evaluate_selection([frame], [frame.index])
    assert metrics.temporal_coverage == 1
    assert metrics.change_coverage == 1
    assert metrics.non_redundancy == 1


def test_flat_duplicate_sequence_has_defined_change_metric() -> None:
    original = _synthetic_frame(0)
    duplicates = tuple(
        replace(
            original,
            index=index,
            path=Path(f"{index}.png"),
            relative_path=f"{index}.png",
            timestamp=float(index),
        )
        for index in range(3)
    )
    metrics = evaluate_selection(duplicates, [0])
    assert metrics.change_coverage == 1
    assert metrics.content_coverage == 1
    benchmark = run_benchmark(
        duplicates,
        SelectionConfig(budget=3),
        BenchmarkConfig(random_trials=1),
    )
    assert all(run.selected_indices == (0,) for run in benchmark.runs)


def test_baselines_share_gap_and_duplicate_constraints() -> None:
    frames = tuple(_synthetic_frame(index) for index in range(10))
    result = run_benchmark(
        frames,
        SelectionConfig(budget=8, min_gap=2.0, duplicate_threshold=0.0),
        BenchmarkConfig(random_trials=2),
    )
    for run in result.runs:
        coordinates = [frames[index].time_coordinate for index in run.selected_indices]
        assert all(right - left >= 2 for left, right in pairwise(coordinates))


def test_time_uniform_baseline_uses_irregular_time_coordinates() -> None:
    timestamps = (0.0, 1.0, 2.0, 100.0)
    frames = tuple(
        replace(_synthetic_frame(index), timestamp=timestamp) for index, timestamp in enumerate(timestamps)
    )
    result = run_benchmark(
        frames,
        SelectionConfig(budget=3, duplicate_threshold=0),
        BenchmarkConfig(random_trials=1),
    )
    uniform = next(run for run in result.runs if run.method == "time_uniform")
    assert uniform.selected_indices == (0, 2, 3)


def test_random_seed_wraps_at_signed_64_boundary() -> None:
    result = run_benchmark(
        tuple(_synthetic_frame(index) for index in range(5)),
        SelectionConfig(budget=2, duplicate_threshold=0),
        BenchmarkConfig(random_seed=(1 << 63) - 1, random_trials=2),
    )
    random_runs = [run for run in result.runs if run.method == "seeded_random"]
    assert [run.seed for run in random_runs] == [(1 << 63) - 1, -(1 << 63)]


def test_svg_is_real_and_refuses_non_file_target(image_factory: Callable[..., Path], tmp_path: Path) -> None:
    result = run_benchmark(_frames(image_factory, tmp_path, 6), SelectionConfig(budget=3))
    destination = render_benchmark_svg(result, tmp_path / "chart.svg")
    rendered = destination.read_text(encoding="utf-8")
    assert rendered.startswith("<svg")
    assert "frame quorum" in rendered
    directory = tmp_path / "directory.svg"
    directory.mkdir()
    with pytest.raises(OutputError, match="regular file"):
        render_benchmark_svg(result, directory)


@pytest.mark.parametrize(
    "config",
    [
        BenchmarkConfig(random_seed=True),
        BenchmarkConfig(random_seed=1 << 63),
        BenchmarkConfig(random_trials=0),
        BenchmarkConfig(random_trials=True),
        BenchmarkConfig(random_trials=1001),
    ],
)
def test_invalid_benchmark_config_is_rejected(config: BenchmarkConfig) -> None:
    with pytest.raises(ConfigurationError):
        run_benchmark([_synthetic_frame(0)], benchmark_config=config)


def test_benchmark_rejects_wrong_config_types() -> None:
    frame = _synthetic_frame(0)
    with pytest.raises(ConfigurationError, match="selection config"):
        run_benchmark([frame], selection_config=object())  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError, match="benchmark config"):
        run_benchmark([frame], benchmark_config=0)  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError, match="scan config"):
        run_benchmark([frame], scan_config=object())  # type: ignore[arg-type]


def test_source_provenance_records_original_bytes_and_scan_config(
    image_factory: Callable[..., Path], tmp_path: Path
) -> None:
    frames = _frames(image_factory, tmp_path, 3)
    digest = source_content_digest(frames)
    scan_config = ScanConfig()
    report = benchmark_manifest(
        run_benchmark(
            frames,
            benchmark_config=BenchmarkConfig(random_trials=1),
            scan_config=scan_config,
            source_digest=digest,
        )
    )
    provenance = report["protocol"]["provenance"]
    assert provenance["source_content_digest"] == digest
    assert provenance["source_file_count"] == 3
    assert provenance["scan_config"]["frame_rate"] == 1.0
    assert provenance["software"]["frame_quorum"] == VERSION
    frames[0].path.write_bytes(b"changed")
    with pytest.raises(ScanError, match="size changed"):
        source_content_digest(frames)


def test_benchmark_provenance_enforces_public_contract() -> None:
    valid = BenchmarkProvenance(source_content_digest="sha256:" + "a" * 64, source_file_count=1)
    valid.validate()
    assert valid.serializable()["source_content_digest_algorithm"].startswith("SHA-256")
    invalid = [
        (replace(valid, frame_quorum_version=""), "non-empty"),
        (replace(valid, scan_config=object()), "scan config"),  # type: ignore[arg-type]
        (replace(valid, source_content_digest="sha256:not-hex"), "64 hex"),
        (BenchmarkProvenance(source_file_count=1), "requires"),
    ]
    for provenance, message in invalid:
        with pytest.raises(ConfigurationError, match=message):
            provenance.validate()


@pytest.mark.parametrize(
    ("selection", "message"),
    [
        ([], "at least one"),
        ([0, 0], "unique"),
        ([99], "reference"),
        ([True], "integer"),
    ],
)
def test_evaluation_rejects_invalid_selections(selection: list[int], message: str) -> None:
    with pytest.raises(ConfigurationError, match=message):
        evaluate_selection([_synthetic_frame(0)], selection)


def test_evaluation_rejects_empty_and_noniterable_inputs() -> None:
    with pytest.raises(ConfigurationError, match="at least one frame"):
        evaluate_selection([], [0])
    with pytest.raises(ConfigurationError, match="iterable"):
        evaluate_selection([_synthetic_frame(0)], None)  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError, match="iterable"):
        run_benchmark(None)  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError, match="iterable"):
        evaluate_selection(None, [0])  # type: ignore[arg-type]


def test_direct_benchmark_records_validate_relationships() -> None:
    frames = tuple(_synthetic_frame(index) for index in range(3))
    result = run_benchmark(frames, SelectionConfig(budget=2), BenchmarkConfig(random_trials=1))
    run = result.runs[0]
    with pytest.raises(ConfigurationError, match="unsupported"):
        replace(run, method="mystery").validate()
    with pytest.raises(ConfigurationError, match="ordered"):
        replace(run, selected_indices=tuple(reversed(run.selected_indices))).validate()
    with pytest.raises(ConfigurationError, match="finite"):
        replace(run, metrics=replace(run.metrics, balanced_score=float("nan"))).validate()
    with pytest.raises(ConfigurationError, match="between"):
        replace(run, metrics=replace(run.metrics, balanced_score=2)).validate()
    with pytest.raises(ConfigurationError, match="must equal"):
        replace(
            run, metrics=replace(run.metrics, balanced_score=run.metrics.balanced_score + 0.01)
        ).validate()
    with pytest.raises(ConfigurationError, match="require a seed"):
        BenchmarkRun("seeded_random", 0, None, (0,), run.metrics).validate()
    with pytest.raises(ConfigurationError, match="trial zero"):
        replace(run, trial=1).validate()
    with pytest.raises(ConfigurationError, match="must be a tuple"):
        replace(run, selected_indices=[0]).validate()  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError, match="unique"):
        replace(run, selected_indices=(0, 0)).validate()
    with pytest.raises(ConfigurationError, match="at least one"):
        replace(run, selected_indices=()).validate()
    with pytest.raises(ConfigurationError, match="EvaluationMetrics"):
        replace(run, metrics=object()).validate()  # type: ignore[arg-type]
    altered_runs = (replace(run, selected_indices=((1 << 63) - 1,)), *result.runs[1:])
    with pytest.raises(ConfigurationError, match="reference"):
        replace(result, runs=altered_runs).validate()
    with pytest.raises(ConfigurationError, match="BenchmarkResult"):
        benchmark_manifest(object())  # type: ignore[arg-type]


def test_direct_benchmark_result_rejects_tampering() -> None:
    frames = tuple(_synthetic_frame(index) for index in range(6))
    result = run_benchmark(frames, SelectionConfig(budget=3), BenchmarkConfig(random_trials=1))
    with pytest.raises(ConfigurationError, match="tuple of Frame"):
        replace(result, frames=list(frames)).validate()  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError, match="unique ordered"):
        replace(result, frames=tuple(reversed(frames))).validate()
    with pytest.raises(ConfigurationError, match="selection config"):
        replace(result, selection_config=object()).validate()  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError, match="benchmark config"):
        replace(result, benchmark_config=object()).validate()  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError, match="provenance"):
        replace(result, provenance=object()).validate()  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError, match="runs must be a tuple"):
        replace(result, runs=list(result.runs)).validate()  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError, match="method and seed protocol"):
        replace(result, runs=result.runs[:-1]).validate()
    wrong_selection = replace(result.runs[0], selected_indices=(0, 1, 5))
    with pytest.raises(ConfigurationError, match="declared method"):
        replace(result, runs=(wrong_selection, *result.runs[1:])).validate()
    changed_metric = replace(
        result.runs[0].metrics,
        mean_quality=result.runs[0].metrics.mean_quality + 0.01,
        balanced_score=result.runs[0].metrics.balanced_score + 0.002,
    )
    with pytest.raises(ConfigurationError, match="selected frames"):
        replace(result, runs=(replace(result.runs[0], metrics=changed_metric), *result.runs[1:])).validate()
    with pytest.raises(ConfigurationError, match="must not exceed"):
        replace(result, selection_config=replace(result.selection_config, budget=1)).validate()


def test_metric_record_rejects_boolean_and_unbounded_values() -> None:
    valid = EvaluationMetrics(0.5, 0.5, 0.5, 0.5, 0.5, 0.5)
    assert valid.serializable()["balanced_score"] == 0.5
    with pytest.raises(ConfigurationError, match="finite"):
        replace(valid, mean_quality=True).validate()  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError, match="between"):
        replace(valid, mean_quality=-0.1).validate()


def test_benchmark_result_rejects_empty_sequence() -> None:
    broken = BenchmarkResult((), SelectionConfig(), BenchmarkConfig(), ())
    with pytest.raises(ConfigurationError, match="at least one"):
        broken.validate()
    with pytest.raises(ConfigurationError, match="at least one frame"):
        run_benchmark([])


def test_one_frame_and_no_endpoint_protocols_are_supported() -> None:
    one = run_benchmark([_synthetic_frame(0)], SelectionConfig(budget=1), BenchmarkConfig(random_trials=1))
    assert all(run.selected_indices == (0,) for run in one.runs)
    no_endpoints = run_benchmark(
        tuple(_synthetic_frame(index) for index in range(4)),
        SelectionConfig(budget=2, keep_endpoints=False, duplicate_threshold=0),
        BenchmarkConfig(random_trials=1),
    )
    assert len(no_endpoints.runs) == 4


def test_benchmark_rejects_unbounded_comparison_workload() -> None:
    frames = tuple(_synthetic_frame(index) for index in range(500))
    with pytest.raises(ConfigurationError, match="workload exceeds"):
        run_benchmark(
            frames,
            SelectionConfig(budget=500),
            BenchmarkConfig(random_trials=1000),
        )


def test_svg_wraps_filesystem_write_error(
    image_factory: Callable[..., Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = run_benchmark(_frames(image_factory, tmp_path, 3), SelectionConfig(budget=2))
    original = Path.write_text

    def fail_write(path: Path, *args: object, **kwargs: object) -> int:
        if path.name == "failed.svg":
            raise OSError("disk full")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_write)
    with pytest.raises(OutputError, match="cannot write"):
        render_benchmark_svg(result, tmp_path / "failed.svg")
    parent_file = tmp_path / "parent-file"
    parent_file.write_text("not a directory", encoding="utf-8")
    with pytest.raises(OutputError, match="cannot write"):
        render_benchmark_svg(result, parent_file / "chart.svg")


def test_evaluation_rejects_non_frame_member() -> None:
    with pytest.raises(ConfigurationError, match="every input"):
        evaluate_selection([object()], [0])  # type: ignore[list-item]
    broken = replace(_synthetic_frame(0), index="zero")
    with pytest.raises(ConfigurationError, match="frame index"):
        evaluate_selection([_synthetic_frame(1), broken], [1])  # type: ignore[list-item]


def test_quality_and_runtime_regression_on_medium_sequence() -> None:
    frames = tuple(_synthetic_frame(index) for index in range(600))
    started = time.perf_counter()
    report = benchmark_manifest(
        run_benchmark(
            frames,
            SelectionConfig(budget=8, duplicate_threshold=0),
            BenchmarkConfig(random_trials=4),
        )
    )
    elapsed = time.perf_counter() - started
    # Wall-clock budgets are only meaningful on a known machine without tracing overhead:
    # branch-coverage instrumentation alone inflates this run several-fold, and CI runners
    # vary widely. The workload above always executes (so coverage is unaffected), but the
    # runtime bound is asserted only when explicitly opted in via FRAME_QUORUM_PERF=1.
    if os.environ.get("FRAME_QUORUM_PERF") == "1":
        assert elapsed < 10.0
    assert len(report["runs"]) == 7
    assert all(0 <= value <= 1 for item in report["aggregates"] for value in item["metrics_mean"].values())


def test_checked_in_benchmark_matches_current_protocol(tmp_path: Path) -> None:
    repository = Path(__file__).parents[1]
    scan_config_path = repository / "examples" / "output" / "frames"
    scan_config = ScanConfig()
    frames = scan_frames(scan_config_path, scan_config)
    result = run_benchmark(
        frames,
        SelectionConfig(budget=6),
        BenchmarkConfig(random_seed=1729, random_trials=8),
        scan_config=scan_config,
        source_digest=source_content_digest(frames),
    )
    expected = json.loads(
        (repository / "examples" / "benchmark" / "benchmark.json").read_text(encoding="utf-8")
    )
    actual = benchmark_manifest(result)
    expected["protocol"]["provenance"]["software"] = actual["protocol"]["provenance"]["software"]
    assert actual == expected
    generated_svg = render_benchmark_svg(result, tmp_path / "benchmark.svg")
    checked_svg = repository / "examples" / "benchmark" / "benchmark.svg"
    assert generated_svg.read_bytes() == checked_svg.read_bytes()
