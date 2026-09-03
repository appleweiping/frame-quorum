"""Reproducible, label-free comparisons for key-frame selectors.

The benchmark deliberately evaluates representation and coverage properties,
not semantic accuracy.  Every method shares the production selector's hard
spacing and duplicate constraints so the comparison isolates candidate order.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import statistics
from dataclasses import asdict, dataclass
from dataclasses import field as dataclass_field
from itertools import combinations, pairwise
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

from PIL import __version__ as pillow_version

from ._version import VERSION
from .errors import ConfigurationError, OutputError, ScanError
from .metrics import content_distance
from .models import (
    _MAX_SIGNED_64,
    Frame,
    ScanConfig,
    SelectionConfig,
    _require_bounded_integer,
    _require_int64,
    _require_safe_text,
)
from .selector import _validate_frames, select_frames

BENCHMARK_SCHEMA_VERSION = "1.1"
_METHODS = ("frame_quorum", "time_uniform", "change_peaks", "seeded_random")
_METRIC_FIELDS = (
    "mean_quality",
    "content_coverage",
    "temporal_coverage",
    "change_coverage",
    "non_redundancy",
    "balanced_score",
)
_MAX_ESTIMATED_COMPARISON_WORK_UNITS = 100_000_000


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    """Controls deterministic baseline replication."""

    random_seed: int = 1729
    random_trials: int = 8

    def validate(self) -> None:
        _require_bounded_integer(
            self.random_seed,
            "benchmark random seed",
            minimum=-(1 << 63),
            maximum=_MAX_SIGNED_64,
        )
        _require_int64(self.random_trials, "benchmark random trials", minimum=1)
        if self.random_trials > 1000:
            raise ConfigurationError("benchmark random trials cannot exceed 1000")


@dataclass(frozen=True, slots=True)
class BenchmarkProvenance:
    """Environment and source facts attached to one benchmark run."""

    frame_quorum_version: str = VERSION
    python_version: str = dataclass_field(default_factory=platform.python_version)
    pillow_version: str = pillow_version
    scan_config: ScanConfig | None = None
    source_content_digest: str | None = None
    source_file_count: int | None = None

    def validate(self) -> None:
        _require_safe_text(self.frame_quorum_version, "benchmark frame-quorum version")
        _require_safe_text(self.python_version, "benchmark Python version")
        _require_safe_text(self.pillow_version, "benchmark Pillow version")
        if self.scan_config is not None:
            if not isinstance(self.scan_config, ScanConfig):
                raise ConfigurationError("benchmark scan config must be ScanConfig or null")
            self.scan_config.validate()
        if self.source_content_digest is None:
            if self.source_file_count is not None:
                raise ConfigurationError("source file count requires a source content digest")
        else:
            if (
                not isinstance(self.source_content_digest, str)
                or not self.source_content_digest.startswith("sha256:")
                or len(self.source_content_digest) != 71
            ):
                raise ConfigurationError("source content digest must be sha256 followed by 64 hex digits")
            try:
                int(self.source_content_digest[7:], 16)
            except ValueError as error:
                raise ConfigurationError(
                    "source content digest must be sha256 followed by 64 hex digits"
                ) from error
            _require_int64(self.source_file_count, "source file count", minimum=1)

    def serializable(self) -> dict[str, Any]:
        self.validate()
        scan_config = None
        if self.scan_config is not None:
            scan_config = asdict(self.scan_config)
            scan_config["extensions"] = list(self.scan_config.extensions)
        return {
            "software": {
                "frame_quorum": self.frame_quorum_version,
                "python": self.python_version,
                "pillow": self.pillow_version,
            },
            "scan_config": scan_config,
            "source_content_digest": self.source_content_digest,
            "source_content_digest_algorithm": (
                "SHA-256 over length-prefixed relative paths and each source file's SHA-256 digest"
                if self.source_content_digest is not None
                else None
            ),
            "source_file_count": self.source_file_count,
        }


@dataclass(frozen=True, slots=True)
class EvaluationMetrics:
    """Label-free measurements; every field is normalized to ``[0, 1]``."""

    mean_quality: float
    content_coverage: float
    temporal_coverage: float
    change_coverage: float
    non_redundancy: float
    balanced_score: float

    def validate(self) -> None:
        values = asdict(self)
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
            for value in values.values()
        ):
            raise ConfigurationError("benchmark metrics must be finite numbers")
        if any(not 0 <= value <= 1 for value in values.values()):
            raise ConfigurationError("benchmark metrics must be between zero and one")
        expected = statistics.fmean(values[field] for field in _METRIC_FIELDS[:-1])
        if not math.isclose(self.balanced_score, expected, rel_tol=0.0, abs_tol=1e-12):
            raise ConfigurationError("balanced score must equal the mean of its component metrics")

    def serializable(self) -> dict[str, float]:
        self.validate()
        return {name: round(value, 6) for name, value in asdict(self).items()}


@dataclass(frozen=True, slots=True)
class BenchmarkRun:
    """One deterministic method/trial outcome."""

    method: str
    trial: int
    seed: int | None
    selected_indices: tuple[int, ...]
    metrics: EvaluationMetrics

    def validate(self) -> None:
        if self.method not in _METHODS:
            raise ConfigurationError(f"unsupported benchmark method: {self.method!r}")
        _require_int64(self.trial, "benchmark trial")
        if self.seed is not None:
            _require_bounded_integer(
                self.seed,
                "benchmark trial seed",
                minimum=-(1 << 63),
                maximum=_MAX_SIGNED_64,
            )
        if self.method == "seeded_random":
            if self.seed is None:
                raise ConfigurationError("seeded-random benchmark runs require a seed")
        elif self.trial != 0 or self.seed is not None:
            raise ConfigurationError("deterministic benchmark methods require trial zero and no seed")
        if not isinstance(self.selected_indices, tuple):
            raise ConfigurationError("benchmark selected indices must be a tuple")
        for index in self.selected_indices:
            _require_int64(index, "benchmark selected frame index")
        if len(self.selected_indices) != len(set(self.selected_indices)):
            raise ConfigurationError("benchmark selected frame indices must be unique")
        if self.selected_indices != tuple(sorted(self.selected_indices)):
            raise ConfigurationError("benchmark selected frame indices must be ordered")
        if not self.selected_indices:
            raise ConfigurationError("benchmark runs must select at least one frame")
        if not isinstance(self.metrics, EvaluationMetrics):
            raise ConfigurationError("benchmark metrics must be EvaluationMetrics")
        self.metrics.validate()

    def serializable(self) -> dict[str, Any]:
        self.validate()
        return {
            "method": self.method,
            "trial": self.trial,
            "seed": self.seed,
            "selected_indices": list(self.selected_indices),
            "selected_count": len(self.selected_indices),
            "metrics": self.metrics.serializable(),
        }


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """Complete deterministic experiment result."""

    frames: tuple[Frame, ...]
    selection_config: SelectionConfig
    benchmark_config: BenchmarkConfig
    runs: tuple[BenchmarkRun, ...]
    provenance: BenchmarkProvenance = dataclass_field(default_factory=BenchmarkProvenance)

    def validate(self) -> None:
        if not isinstance(self.frames, tuple) or any(not isinstance(frame, Frame) for frame in self.frames):
            raise ConfigurationError("benchmark frames must be a tuple of Frame objects")
        if not self.frames:
            raise ConfigurationError("benchmark requires at least one frame")
        validated_frames = _validated_frames(self.frames)
        if validated_frames != self.frames:
            raise ConfigurationError("benchmark frames must have unique ordered indices")
        indices = tuple(frame.index for frame in self.frames)
        if not isinstance(self.selection_config, SelectionConfig):
            raise ConfigurationError("benchmark selection config must be SelectionConfig")
        self.selection_config.validate()
        if not isinstance(self.benchmark_config, BenchmarkConfig):
            raise ConfigurationError("benchmark config must be BenchmarkConfig")
        self.benchmark_config.validate()
        _validate_workload(len(self.frames), self.selection_config, self.benchmark_config)
        if not isinstance(self.provenance, BenchmarkProvenance):
            raise ConfigurationError("benchmark provenance must be BenchmarkProvenance")
        self.provenance.validate()
        if self.provenance.source_file_count is not None and self.provenance.source_file_count != len(
            self.frames
        ):
            raise ConfigurationError("source file count must match benchmark frame count")
        if not isinstance(self.runs, tuple):
            raise ConfigurationError("benchmark runs must be a tuple")
        expected_layout = (
            ("frame_quorum", 0, None),
            ("time_uniform", 0, None),
            ("change_peaks", 0, None),
            *(
                ("seeded_random", trial, _trial_seed(self.benchmark_config.random_seed, trial))
                for trial in range(self.benchmark_config.random_trials)
            ),
        )
        actual_layout = tuple(
            (run.method, run.trial, run.seed) if isinstance(run, BenchmarkRun) else None for run in self.runs
        )
        if actual_layout != expected_layout:
            raise ConfigurationError("benchmark runs must match the declared method and seed protocol")

        quorum_indices = select_frames(self.frames, self.selection_config).selected_indices
        uniform_indices = _baseline_pick(
            self.frames,
            _time_uniform_order(self.frames, self.selection_config.budget),
            self.selection_config,
        )
        change_order = tuple(
            position
            for position, _ in sorted(
                enumerate(_local_changes(self.frames)), key=lambda item: (-item[1], item[0])
            )
        )
        change_indices = _baseline_pick(self.frames, change_order, self.selection_config)
        expected_indices = {
            ("frame_quorum", 0): quorum_indices,
            ("time_uniform", 0): uniform_indices,
            ("change_peaks", 0): change_indices,
        }
        for run in self.runs:
            run.validate()
            if not set(run.selected_indices).issubset(indices):
                raise ConfigurationError("benchmark selections must reference input frames")
            if len(run.selected_indices) > self.selection_config.budget:
                raise ConfigurationError("benchmark selections must not exceed the budget")
            if run.method == "seeded_random":
                random_order = tuple(
                    sorted(
                        range(len(self.frames)),
                        key=lambda position: (
                            _portable_random_key(run.seed, self.frames[position].index),
                            position,
                        ),
                    )
                )
                expected = _baseline_pick(self.frames, random_order, self.selection_config)
            else:
                expected = expected_indices[(run.method, run.trial)]
            if run.selected_indices != expected:
                raise ConfigurationError("benchmark selections must match their declared method")
            if run.metrics != evaluate_selection(self.frames, run.selected_indices):
                raise ConfigurationError("benchmark metrics must match their selected frames")


def evaluate_selection(
    frames: tuple[Frame, ...] | list[Frame], selected_indices: tuple[int, ...] | list[int]
) -> EvaluationMetrics:
    """Measure quality, representation, coverage, and redundancy without labels."""

    ordered = _validated_frames(frames)
    try:
        selected_tuple = tuple(selected_indices)
    except TypeError as error:
        raise ConfigurationError("selected indices must be an iterable of integers") from error
    for index in selected_tuple:
        _require_int64(index, "benchmark selected frame index")
    if not selected_tuple:
        raise ConfigurationError("at least one frame must be selected for evaluation")
    if len(selected_tuple) != len(set(selected_tuple)):
        raise ConfigurationError("selected indices must be unique")
    frame_by_index = {frame.index: frame for frame in ordered}
    if not set(selected_tuple).issubset(frame_by_index):
        raise ConfigurationError("selected indices must reference evaluated frames")
    selected = tuple(frame_by_index[index] for index in sorted(selected_tuple))

    mean_quality = statistics.fmean(_quality(frame) for frame in selected)
    content_coverage = statistics.fmean(
        1.0 - min(content_distance(frame.metrics, chosen.metrics) for chosen in selected) for frame in ordered
    )
    coordinates = tuple(frame.time_coordinate for frame in ordered)
    span = max(coordinates) - min(coordinates)
    if span <= 0:
        temporal_coverage = 1.0
    else:
        temporal_coverage = statistics.fmean(
            1.0 - min(abs(frame.time_coordinate - chosen.time_coordinate) for chosen in selected) / span
            for frame in ordered
        )

    transitions = tuple(
        (right.time_coordinate, content_distance(left.metrics, right.metrics))
        for left, right in pairwise(ordered)
    )
    total_change = sum(weight for _, weight in transitions)
    if total_change <= 0 or span <= 0:
        change_coverage = 1.0
    else:
        change_coverage = (
            sum(
                weight * (1.0 - min(abs(coordinate - chosen.time_coordinate) for chosen in selected) / span)
                for coordinate, weight in transitions
            )
            / total_change
        )

    pairs = tuple(combinations(selected, 2))
    non_redundancy = (
        statistics.fmean(content_distance(left.metrics, right.metrics) for left, right in pairs)
        if pairs
        else 1.0
    )
    positive_metrics = (
        mean_quality,
        content_coverage,
        temporal_coverage,
        change_coverage,
        non_redundancy,
    )
    result = EvaluationMetrics(*positive_metrics, statistics.fmean(positive_metrics))
    result.validate()
    return result


def run_benchmark(
    frames: tuple[Frame, ...] | list[Frame],
    selection_config: SelectionConfig | None = None,
    benchmark_config: BenchmarkConfig | None = None,
    *,
    scan_config: ScanConfig | None = None,
    source_digest: str | None = None,
) -> BenchmarkResult:
    """Compare Frame Quorum with three transparent baselines."""

    options = SelectionConfig() if selection_config is None else selection_config
    experiment = BenchmarkConfig() if benchmark_config is None else benchmark_config
    if not isinstance(options, SelectionConfig):
        raise ConfigurationError("benchmark selection config must be SelectionConfig")
    if not isinstance(experiment, BenchmarkConfig):
        raise ConfigurationError("benchmark config must be BenchmarkConfig")
    if scan_config is not None and not isinstance(scan_config, ScanConfig):
        raise ConfigurationError("benchmark scan config must be ScanConfig or null")
    options.validate()
    experiment.validate()

    ordered_input = _validated_frames(frames)
    _validate_workload(len(ordered_input), options, experiment)
    # The production selector establishes the primary result after the shared
    # record and resource contracts have been checked.
    quorum = select_frames(ordered_input, options)
    ordered = quorum.frames

    runs: list[BenchmarkRun] = [_make_run("frame_quorum", 0, None, ordered, quorum.selected_indices)]
    uniform = _baseline_pick(ordered, _time_uniform_order(ordered, options.budget), options)
    runs.append(_make_run("time_uniform", 0, None, ordered, uniform))
    change_order = tuple(
        position
        for position, _ in sorted(enumerate(_local_changes(ordered)), key=lambda item: (-item[1], item[0]))
    )
    change_peaks = _baseline_pick(ordered, change_order, options)
    runs.append(_make_run("change_peaks", 0, None, ordered, change_peaks))
    for trial in range(experiment.random_trials):
        seed = _trial_seed(experiment.random_seed, trial)
        order = tuple(
            sorted(
                range(len(ordered)),
                key=lambda position: (_portable_random_key(seed, ordered[position].index), position),
            )
        )
        random_selection = _baseline_pick(ordered, order, options)
        runs.append(_make_run("seeded_random", trial, seed, ordered, random_selection))

    provenance = BenchmarkProvenance(
        scan_config=scan_config,
        source_content_digest=source_digest,
        source_file_count=len(ordered) if source_digest is not None else None,
    )
    provenance.validate()
    return BenchmarkResult(ordered, options, experiment, tuple(runs), provenance)


def benchmark_manifest(result: BenchmarkResult) -> dict[str, Any]:
    """Create the stable, machine-readable experiment report."""

    if not isinstance(result, BenchmarkResult):
        raise ConfigurationError("benchmark result must be BenchmarkResult")
    result.validate()
    aggregates = []
    for method in _METHODS:
        method_runs = tuple(run for run in result.runs if run.method == method)
        mean_metrics = {
            field: round(
                statistics.fmean(getattr(run.metrics, field) for run in method_runs),
                6,
            )
            for field in _METRIC_FIELDS
        }
        standard_deviation = {
            field: round(
                statistics.pstdev(getattr(run.metrics, field) for run in method_runs),
                6,
            )
            for field in _METRIC_FIELDS
        }
        aggregates.append(
            {
                "method": method,
                "trial_count": len(method_runs),
                "selected_count_mean": round(
                    statistics.fmean(len(run.selected_indices) for run in method_runs), 6
                ),
                "selected_count_population_stddev": round(
                    statistics.pstdev(len(run.selected_indices) for run in method_runs), 6
                ),
                "metrics_mean": mean_metrics,
                "metrics_population_stddev": standard_deviation,
            }
        )
    return {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "kind": "frame-quorum-benchmark",
        "protocol": {
            "protocol_id": "frame-quorum-label-free-v1",
            "frame_count": len(result.frames),
            "measured_record_fingerprint": _measured_record_fingerprint(result.frames),
            "measured_record_fingerprint_algorithm": (
                "SHA-256 over canonical paths, timestamps, dimensions, byte counts, and lossy image metrics"
            ),
            "provenance": result.provenance.serializable(),
            "methods": list(_METHODS),
            "shared_constraints": ["budget", "min_gap", "duplicate_threshold", "keep_endpoints"],
            "selection_config": asdict(result.selection_config),
            "benchmark_config": asdict(result.benchmark_config),
            "estimated_comparison_work_units": _estimated_work_units(
                len(result.frames), result.selection_config, result.benchmark_config
            ),
            "comparison_work_unit_limit": _MAX_ESTIMATED_COMPARISON_WORK_UNITS,
            "metrics": {
                "mean_quality": "Mean intrinsic quality of selected frames; higher is better.",
                "content_coverage": "Mean similarity to the nearest selected frame; higher is better.",
                "temporal_coverage": "Mean normalized proximity to a selected time; higher is better.",
                "change_coverage": (
                    "Change-weighted proximity of transitions to selections; higher is better."
                ),
                "non_redundancy": "Mean pairwise selected-frame content distance; higher is better.",
                "balanced_score": "Unweighted mean of the five diagnostics; not semantic accuracy.",
            },
        },
        "runs": [run.serializable() for run in result.runs],
        "aggregates": aggregates,
        "limitations": [
            "Metrics are label-free diagnostics and do not measure semantic event recall.",
            "The bundled synthetic sequence is a reproducibility fixture, not an external benchmark.",
            "Random results describe only the declared deterministic seed range.",
        ],
    }


def render_benchmark_svg(result: BenchmarkResult, destination: str | Path) -> Path:
    """Render aggregate metrics as a dependency-free SVG chart."""

    manifest = benchmark_manifest(result)
    aggregates = manifest["aggregates"]
    width = 1120
    row_height = 36
    group_gap = 28
    top = 105
    chart_left = 245
    chart_width = 760
    height = top + len(aggregates) * (len(_METRIC_FIELDS) * row_height + group_gap) + 70
    colors = {
        "mean_quality": "#58d6a9",
        "content_coverage": "#55a8ff",
        "temporal_coverage": "#f2cf57",
        "change_coverage": "#f58b62",
        "non_redundancy": "#bb87f7",
        "balanced_score": "#f3f6fb",
    }
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">Frame Quorum reproducible benchmark</title>',
        '<desc id="desc">Label-free aggregate scores for four key-frame selection methods.</desc>',
        '<rect width="100%" height="100%" fill="#101722"/>',
        '<text x="48" y="48" fill="#ffffff" font-family="sans-serif" font-size="26" '
        'font-weight="700">Reproducible selector comparison</text>',
        '<text x="48" y="76" fill="#9fb2c8" font-family="sans-serif" font-size="14">'
        "Higher is better · synthetic fixture · label-free diagnostics, not semantic accuracy</text>",
    ]
    y = top
    for aggregate in aggregates:
        method = escape(str(aggregate["method"]).replace("_", " "))
        lines.append(
            f'<text x="48" y="{y + 20}" fill="#ffffff" font-family="monospace" '
            f'font-size="17" font-weight="700">{method}</text>'
        )
        means = aggregate["metrics_mean"]
        for field in _METRIC_FIELDS:
            value = float(means[field])
            bar_y = y + 5
            bar_width = round(chart_width * value, 2)
            label = escape(field.replace("_", " "))
            lines.extend(
                (
                    f'<text x="{chart_left - 12}" y="{bar_y + 17}" fill="#aebed0" '
                    f'font-family="sans-serif" font-size="13" text-anchor="end">{label}</text>',
                    f'<rect x="{chart_left}" y="{bar_y}" width="{chart_width}" height="22" '
                    'rx="4" fill="#1f2b3a"/>',
                    f'<rect x="{chart_left}" y="{bar_y}" width="{bar_width}" height="22" '
                    f'rx="4" fill="{colors[field]}"/>',
                    f'<text x="{chart_left + chart_width + 14}" y="{bar_y + 17}" fill="#ffffff" '
                    f'font-family="monospace" font-size="13">{value:.3f}</text>',
                )
            )
            y += row_height
        y += group_gap
    lines.append(
        f'<text x="48" y="{height - 28}" fill="#71859b" font-family="sans-serif" font-size="12">'
        f"records {escape(manifest['protocol']['measured_record_fingerprint'][:16])} · schema "
        f"{BENCHMARK_SCHEMA_VERSION}</text>"
    )
    lines.append("</svg>")
    path = Path(destination)
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise OutputError(f"benchmark chart target must be a regular file or absent: {path}")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    except OSError as error:
        raise OutputError(f"cannot write benchmark chart {path}: {error}") from error
    return path


def _validated_frames(frames: tuple[Frame, ...] | list[Frame]) -> tuple[Frame, ...]:
    try:
        supplied = tuple(frames)
    except TypeError as error:
        raise ConfigurationError("frames must be an iterable of Frame objects") from error
    if not supplied:
        raise ConfigurationError("evaluation requires at least one frame")
    if any(not isinstance(frame, Frame) for frame in supplied):
        raise ConfigurationError("every input must be a Frame")
    for frame in supplied:
        _require_int64(frame.index, "frame index")
    ordered = tuple(sorted(supplied, key=lambda frame: frame.index))
    _validate_frames(ordered)
    return ordered


def _quality(frame: Frame) -> float:
    metrics = frame.metrics
    return min(1.0, 0.45 * metrics.sharpness + 0.35 * metrics.entropy + 0.20 * metrics.colorfulness)


def _local_changes(frames: tuple[Frame, ...]) -> tuple[float, ...]:
    values = [0.0]
    values.extend(content_distance(left.metrics, right.metrics) for left, right in pairwise(frames))
    return tuple(values)


def _time_uniform_order(frames: tuple[Frame, ...], budget: int) -> tuple[int, ...]:
    frame_count = len(frames)
    slots = min(frame_count, budget)
    coordinates = tuple(frame.time_coordinate for frame in frames)
    start = coordinates[0]
    span = coordinates[-1] - start
    if slots == 1:
        anchors = (start + span / 2.0,)
    else:
        anchors = tuple(start + span * (slot / (slots - 1)) for slot in range(slots))
    preferred: list[int] = []
    for anchor in anchors:
        position = min(
            range(frame_count),
            key=lambda candidate: (
                abs(coordinates[candidate] - anchor),
                coordinates[candidate],
                candidate,
            ),
        )
        if position not in preferred:
            preferred.append(position)
    preferred.extend(position for position in range(frame_count) if position not in preferred)
    return tuple(preferred)


def _baseline_pick(
    frames: tuple[Frame, ...], preferred_positions: tuple[int, ...], config: SelectionConfig
) -> tuple[int, ...]:
    selected: list[Frame] = []

    def consider(frame: Frame) -> None:
        if len(selected) >= config.budget or frame in selected:
            return
        if any(abs(frame.time_coordinate - chosen.time_coordinate) < config.min_gap for chosen in selected):
            return
        if any(
            content_distance(frame.metrics, chosen.metrics) <= config.duplicate_threshold
            for chosen in selected
        ):
            return
        selected.append(frame)

    if config.keep_endpoints:
        consider(frames[0])
        if len(frames) > 1:
            consider(frames[-1])
    for position in preferred_positions:
        consider(frames[position])
    return tuple(sorted(frame.index for frame in selected))


def _portable_random_key(seed: int, frame_index: int) -> bytes:
    return hashlib.sha256(f"frame-quorum-random-v1\0{seed}\0{frame_index}".encode()).digest()


def _trial_seed(initial_seed: int, trial: int) -> int:
    seed = initial_seed + trial
    if seed > _MAX_SIGNED_64:
        seed = -(1 << 63) + (seed - _MAX_SIGNED_64 - 1)
    return seed


def _validate_workload(
    frame_count: int, selection_config: SelectionConfig, benchmark_config: BenchmarkConfig
) -> None:
    estimated = _estimated_work_units(frame_count, selection_config, benchmark_config)
    if estimated > _MAX_ESTIMATED_COMPARISON_WORK_UNITS:
        raise ConfigurationError(
            "benchmark workload exceeds the 100000000 conservative comparison-work-unit limit"
        )


def _estimated_work_units(
    frame_count: int, selection_config: SelectionConfig, benchmark_config: BenchmarkConfig
) -> int:
    """Conservatively cover construction plus one full integrity validation."""

    budget = min(frame_count, selection_config.budget)
    runs = benchmark_config.random_trials + 3
    return 8 * frame_count * budget * (budget + runs) + 4 * runs * budget * budget


def _make_run(
    method: str,
    trial: int,
    seed: int | None,
    frames: tuple[Frame, ...],
    selected_indices: tuple[int, ...],
) -> BenchmarkRun:
    run = BenchmarkRun(
        method,
        trial,
        seed,
        selected_indices,
        evaluate_selection(frames, selected_indices),
    )
    run.validate()
    return run


def _measured_record_fingerprint(frames: tuple[Frame, ...]) -> str:
    canonical = json.dumps(
        [frame.serializable() for frame in frames],
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(canonical).hexdigest()}"


def source_content_digest(frames: tuple[Frame, ...] | list[Frame]) -> str:
    """Hash source bytes with unambiguous relative-path framing."""

    ordered = _validated_frames(frames)
    aggregate = hashlib.sha256(b"frame-quorum-source-set-v1\0")
    for frame in ordered:
        path = frame.path
        if path.is_symlink():
            raise ScanError(f"refusing to hash a symbolic-link frame source: {path}")
        try:
            relative = frame.relative_path.encode("utf-8")
            file_digest = hashlib.sha256()
            byte_count = 0
            with path.open("rb") as stream:
                opened_size = os.fstat(stream.fileno()).st_size
                if opened_size != frame.byte_size:
                    raise ScanError(f"frame source size changed before provenance hashing: {path}")
                while chunk := stream.read(1024 * 1024):
                    file_digest.update(chunk)
                    byte_count += len(chunk)
            if byte_count != frame.byte_size:
                raise ScanError(f"frame source size changed during provenance hashing: {path}")
        except OSError as error:
            raise ScanError(f"cannot hash frame source {path}: {error}") from error
        aggregate.update(len(relative).to_bytes(8, "big"))
        aggregate.update(relative)
        aggregate.update(file_digest.digest())
    return f"sha256:{aggregate.hexdigest()}"
