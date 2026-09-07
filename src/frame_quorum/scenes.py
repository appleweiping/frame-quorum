"""Deterministic shot-boundary analysis and per-shot budget allocation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .errors import ConfigurationError
from .metrics import content_distance
from .models import Frame


@dataclass(frozen=True, slots=True)
class Shot:
    """A contiguous half-open slice of the ordered frame sequence."""

    ordinal: int
    start_index: int
    end_index: int
    boundary_distance: float | None = None

    @property
    def frame_count(self) -> int:
        return self.end_index - self.start_index

    def serializable(self) -> dict[str, object]:
        return {
            "ordinal": self.ordinal,
            "start_index": self.start_index,
            "end_index": self.end_index,
            "frame_count": self.frame_count,
            "boundary_distance": self.boundary_distance,
        }


@dataclass(frozen=True, slots=True)
class SceneAnalysis:
    """Shot segmentation plus stable allocation diagnostics."""

    shots: tuple[Shot, ...]
    threshold: float
    allocated_budget: tuple[int, ...]

    def serializable(self) -> dict[str, object]:
        return {
            "threshold": self.threshold,
            "shots": [shot.serializable() for shot in self.shots],
            "allocated_budget": list(self.allocated_budget),
        }


def detect_shots(
    frames: Sequence[Frame], *, threshold: float = 0.30, min_frames: int = 1
) -> tuple[Shot, ...]:
    """Split frames at content-distance peaks above ``threshold``.

    This is a representation-level detector, not a semantic scene classifier.
    Boundaries are evaluated in frame order and tie-breaking is therefore
    reproducible across platforms.
    """

    if not isinstance(frames, Sequence) or not frames:
        raise ConfigurationError("frames must be a non-empty sequence")
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool) or not 0 <= threshold <= 1:
        raise ConfigurationError("threshold must be between zero and one")
    if type(min_frames) is not int or min_frames < 1:
        raise ConfigurationError("min_frames must be a positive integer")
    ordered = tuple(frames)
    for frame in ordered:
        if not isinstance(frame, Frame):
            raise ConfigurationError("every item must be a Frame")
        frame.validate()
    boundaries: list[tuple[int, float]] = []
    last_start = 0
    for position in range(1, len(ordered)):
        distance = content_distance(ordered[position - 1].metrics, ordered[position].metrics)
        if distance >= threshold and position - last_start >= min_frames:
            boundaries.append((position, distance))
            last_start = position
    shots: list[Shot] = []
    start = 0
    previous_distance: float | None = None
    for ordinal, (end, distance) in enumerate(boundaries):
        shots.append(Shot(ordinal, ordered[start].index, ordered[end - 1].index + 1, previous_distance))
        start = end
        previous_distance = distance
    shots.append(
        Shot(
            len(shots),
            ordered[start].index,
            ordered[-1].index + 1,
            previous_distance,
        )
    )
    return tuple(shots)


def allocate_budget(shots: Sequence[Shot], budget: int) -> tuple[int, ...]:
    """Allocate at least one slot per shot when the budget permits it."""

    if not shots or type(budget) is not int or budget < 1:
        raise ConfigurationError("shots must be non-empty and budget must be positive")
    if budget < len(shots):
        # Deterministically favor longer shots when the budget is too small.
        ranked = sorted(range(len(shots)), key=lambda index: (-shots[index].frame_count, index))
        result = [0] * len(shots)
        for index in ranked[:budget]:
            result[index] = 1
        return tuple(result)
    result = [1] * len(shots)
    remaining = budget - len(shots)
    order = sorted(range(len(shots)), key=lambda index: (-shots[index].frame_count, index))
    cursor = 0
    while remaining:
        index = order[cursor % len(order)]
        if result[index] < shots[index].frame_count:
            result[index] += 1
            remaining -= 1
        cursor += 1
        if cursor > budget * max(1, len(shots)) * 2:
            break
    return tuple(result)


def analyze_scenes(
    frames: Sequence[Frame], *, threshold: float = 0.30, budget: int = 8, min_frames: int = 1
) -> SceneAnalysis:
    """Detect shots and return a budget plan suitable for downstream selection."""

    shots = detect_shots(frames, threshold=threshold, min_frames=min_frames)
    allocation = allocate_budget(shots, budget)
    return SceneAnalysis(shots, float(threshold), allocation)


__all__ = ["SceneAnalysis", "Shot", "allocate_budget", "analyze_scenes", "detect_shots"]
