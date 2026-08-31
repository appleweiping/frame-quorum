"""Constraint-aware, explainable key-frame selection."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

from .errors import ConfigurationError
from .metrics import content_distance
from .models import (
    Frame,
    FrameDecision,
    FrameMetrics,
    ScoreBreakdown,
    SelectionConfig,
    SelectionResult,
    _is_finite_number,
    _require_safe_text,
)


@dataclass(frozen=True, slots=True)
class _Intrinsic:
    quality: float
    change: float


def select_frames(
    frames: tuple[Frame, ...] | list[Frame], config: SelectionConfig | None = None
) -> SelectionResult:
    """Choose up to ``budget`` frames and retain a reason for every outcome.

    Selection is deterministic. Endpoints can be reserved first, then a greedy
    pass balances visual quality, change from the preceding frame, and temporal
    coverage relative to frames already chosen. Near duplicates and minimum-gap
    violations are hard constraints.
    """

    options = config or SelectionConfig()
    options.validate()
    try:
        supplied = tuple(frames)
    except TypeError as error:
        raise ConfigurationError("frames must be an iterable of Frame objects") from error
    if any(not isinstance(frame, Frame) for frame in supplied):
        raise ConfigurationError("every input must be a Frame")
    ordered = tuple(sorted(supplied, key=lambda frame: frame.index))
    coordinate_span = _validate_frames(ordered)
    if not ordered:
        return SelectionResult((), (), (), options)

    frame_map = {frame.index: frame for frame in ordered}
    intrinsic = _intrinsic_scores(ordered)
    selected: list[int] = []
    selection_codes: dict[int, str] = {}
    selection_scores: dict[int, ScoreBreakdown] = {}
    if options.keep_endpoints:
        _reserve_endpoint(
            ordered[0],
            "endpoint_start",
            frame_map,
            selected,
            selection_codes,
            selection_scores,
            intrinsic,
            options,
            coordinate_span,
        )
        if len(ordered) > 1 and len(selected) < options.budget:
            _reserve_endpoint(
                ordered[-1],
                "endpoint_end",
                frame_map,
                selected,
                selection_codes,
                selection_scores,
                intrinsic,
                options,
                coordinate_span,
            )

    while len(selected) < options.budget:
        candidates: list[tuple[float, float, int, ScoreBreakdown]] = []
        for frame in ordered:
            if frame.index in selected or not _passes_constraints(frame, frame_map, selected, options):
                continue
            scores = _score(frame, frame_map, selected, intrinsic, options, coordinate_span)
            candidates.append((scores.utility, scores.change, -frame.index, scores))
        if not candidates:
            break
        _, _, negative_index, chosen_scores = max(candidates)
        chosen = -negative_index
        selected.append(chosen)
        selection_codes[chosen] = "highest_utility"
        selection_scores[chosen] = chosen_scores

    rank_by_index = {index: rank for rank, index in enumerate(selected, start=1)}
    selected_sorted = tuple(sorted(selected))
    decisions = tuple(
        _decision(
            frame,
            frame_map,
            selected_sorted,
            rank_by_index,
            selection_codes,
            selection_scores,
            intrinsic,
            options,
            coordinate_span,
        )
        for frame in ordered
    )
    return SelectionResult(ordered, selected_sorted, decisions, options)


def _validate_frames(frames: tuple[Frame, ...]) -> float:
    if any(not isinstance(frame, Frame) for frame in frames):
        raise ConfigurationError("every input must be a Frame")
    indices = [frame.index for frame in frames]
    if any(isinstance(index, bool) or not isinstance(index, int) for index in indices):
        raise ConfigurationError("frame indices must be integers")
    if len(indices) != len(set(indices)):
        raise ConfigurationError("frame indices must be unique")
    timestamps = [frame.timestamp for frame in frames if frame.timestamp is not None]
    if timestamps and len(timestamps) != len(frames):
        raise ConfigurationError("timestamps must be present for every frame or for none")
    if any(not _is_finite_number(timestamp) for timestamp in timestamps):
        raise ConfigurationError("frame timestamps must be finite numbers")
    if any(right < left for left, right in pairwise(timestamps)):
        raise ConfigurationError("frame timestamps must be non-decreasing")
    span = _coordinate_span(frames)
    for frame in frames:
        _require_safe_text(frame.relative_path, "frame relative path")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < minimum
            for value, minimum in (
                (frame.width, 1),
                (frame.height, 1),
                (frame.byte_size, 0),
            )
        ):
            raise ConfigurationError("frame dimensions and byte sizes must be valid integers")
        metrics = frame.metrics
        if not isinstance(metrics, FrameMetrics):
            raise ConfigurationError("frame metrics must be FrameMetrics")
        if (
            isinstance(metrics.perceptual_hash, bool)
            or not isinstance(metrics.perceptual_hash, int)
            or not 0 <= metrics.perceptual_hash < 2**64
        ):
            raise ConfigurationError("perceptual hashes must be unsigned 64-bit integers")
        normalized_metrics = (
            metrics.luminance,
            metrics.entropy,
            metrics.sharpness,
            metrics.colorfulness,
            metrics.mean_red,
            metrics.mean_green,
            metrics.mean_blue,
        )
        if any(not _is_finite_number(value) or not 0 <= value <= 1 for value in normalized_metrics):
            raise ConfigurationError("frame metrics must be finite values between zero and one")
    return span


def _coordinate_span(frames: tuple[Frame, ...]) -> float:
    try:
        coordinates = [frame.time_coordinate for frame in frames]
        span = max(coordinates, default=0.0) - min(coordinates, default=0.0)
    except (OverflowError, ValueError) as error:
        raise ConfigurationError("frame time coordinates must be finite") from error
    if any(not _is_finite_number(coordinate) for coordinate in coordinates) or not _is_finite_number(span):
        raise ConfigurationError("frame time coordinates and their span must be finite")
    return span


def _intrinsic_scores(frames: tuple[Frame, ...]) -> dict[int, _Intrinsic]:
    scores: dict[int, _Intrinsic] = {}
    previous: Frame | None = None
    for frame in frames:
        metrics = frame.metrics
        quality = 0.45 * metrics.sharpness + 0.35 * metrics.entropy + 0.20 * metrics.colorfulness
        change = content_distance(previous.metrics, metrics) if previous else 0.0
        scores[frame.index] = _Intrinsic(quality=min(1.0, quality), change=change)
        previous = frame
    return scores


def _reserve_endpoint(
    frame: Frame,
    code: str,
    frame_map: dict[int, Frame],
    selected: list[int],
    codes: dict[int, str],
    scores: dict[int, ScoreBreakdown],
    intrinsic: dict[int, _Intrinsic],
    config: SelectionConfig,
    coordinate_span: float,
) -> None:
    if not selected or _passes_constraints(frame, frame_map, selected, config):
        scores[frame.index] = _score(frame, frame_map, selected, intrinsic, config, coordinate_span)
        selected.append(frame.index)
        codes[frame.index] = code


def _passes_constraints(
    candidate: Frame,
    frame_map: dict[int, Frame],
    selected: list[int] | tuple[int, ...],
    config: SelectionConfig,
) -> bool:
    selected_frames = [frame_map[index] for index in selected]
    if any(
        abs(candidate.time_coordinate - frame.time_coordinate) < config.min_gap for frame in selected_frames
    ):
        return False
    return not any(
        content_distance(candidate.metrics, frame.metrics) <= config.duplicate_threshold
        for frame in selected_frames
    )


def _score(
    frame: Frame,
    frame_map: dict[int, Frame],
    selected: list[int] | tuple[int, ...],
    intrinsic: dict[int, _Intrinsic],
    config: SelectionConfig,
    coordinate_span: float,
) -> ScoreBreakdown:
    values = intrinsic[frame.index]
    coverage = _coverage(frame, frame_map, selected, coordinate_span)
    weight_sum = config.quality_weight + config.change_weight + config.coverage_weight
    utility = (
        config.quality_weight * values.quality
        + config.change_weight * values.change
        + config.coverage_weight * coverage
    ) / weight_sum
    return ScoreBreakdown(values.quality, values.change, coverage, utility)


def _coverage(
    frame: Frame,
    frame_map: dict[int, Frame],
    selected: list[int] | tuple[int, ...],
    coordinate_span: float,
) -> float:
    if not selected:
        return 1.0
    if coordinate_span <= 0:
        return 0.0
    nearest = min(abs(frame.time_coordinate - frame_map[index].time_coordinate) for index in selected)
    return min(1.0, nearest / coordinate_span * 2.0)


def _decision(
    frame: Frame,
    frame_map: dict[int, Frame],
    selected: tuple[int, ...],
    ranks: dict[int, int],
    codes: dict[int, str],
    selection_scores: dict[int, ScoreBreakdown],
    intrinsic: dict[int, _Intrinsic],
    config: SelectionConfig,
    coordinate_span: float,
) -> FrameDecision:
    nearest = _nearest_selected(frame, frame_map, selected)
    if frame.index in ranks:
        scores = selection_scores[frame.index]
        code = codes[frame.index]
        if code == "endpoint_start":
            reason = "Selected to preserve the beginning of the sequence."
        elif code == "endpoint_end":
            reason = "Selected to preserve the end of the sequence."
        else:
            reason = (
                f"Selected at rank {ranks[frame.index]} with utility {scores.utility:.3f}; "
                f"quality={scores.quality:.3f}, change={scores.change:.3f}, coverage={scores.coverage:.3f}."
            )
        return FrameDecision(
            frame.index,
            frame.relative_path,
            True,
            ranks[frame.index],
            code,
            reason,
            nearest,
            scores,
        )

    scores = _score(frame, frame_map, selected, intrinsic, config, coordinate_span)
    selected_others = [frame_map[index] for index in selected if index != frame.index]
    nearest_duplicate = min(
        selected_others,
        key=lambda item: (
            content_distance(frame.metrics, item.metrics),
            abs(frame.index - item.index),
            item.index,
        ),
        default=None,
    )
    nearest_in_time = min(
        selected_others,
        key=lambda item: (abs(frame.time_coordinate - item.time_coordinate), item.index),
        default=None,
    )
    if (
        nearest_duplicate
        and content_distance(frame.metrics, nearest_duplicate.metrics) <= config.duplicate_threshold
    ):
        code = "near_duplicate"
        nearest = nearest_duplicate.index
        reason = f"Skipped as a near duplicate of selected frame {nearest}."
    elif nearest_in_time and abs(frame.time_coordinate - nearest_in_time.time_coordinate) < config.min_gap:
        code = "min_gap"
        nearest = nearest_in_time.index
        reason = f"Skipped because it is inside the minimum gap around selected frame {nearest}."
    elif len(selected) >= config.budget:
        code = "budget_exhausted"
        reason = f"Skipped after the selection budget of {config.budget} was filled."
    else:
        code = "constraint_limited"
        reason = "Skipped because no remaining admissible selection slot was available."
    return FrameDecision(frame.index, frame.relative_path, False, None, code, reason, nearest, scores)


def _nearest_selected(frame: Frame, frame_map: dict[int, Frame], selected: tuple[int, ...]) -> int | None:
    others = [index for index in selected if index != frame.index]
    if not others:
        return None
    return min(
        others,
        key=lambda index: (
            content_distance(frame.metrics, frame_map[index].metrics),
            abs(frame.index - index),
            index,
        ),
    )
