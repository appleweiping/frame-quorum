"""How well a selection stands in for the frames it left out.

The selector reports why each frame was chosen or rejected, which answers "was
this decision defensible" and not "did the result miss anything". Those are
different questions. A budget of eight can be filled with eight defensible
choices and still leave a whole event unrepresented, and nothing in the manifest
would say so.

Representation error is the distance from each dropped frame to the nearest
frame that was kept, in the same content metric the selector already uses to
detect duplicates. A dropped frame close to a kept one is covered by it. A
dropped frame far from every kept one is a moment the selection did not
capture, and a run of them is an event that fell in a gap.

The metric is the selector's own, deliberately. Introducing a second notion of
"similar" would let a selection look well represented under one measure while
the selector rejected duplicates under another, and the disagreement would be
invisible.

None of this says the selection is wrong. A budget of eight cannot represent a
sequence that contains twenty distinct moments, and the worst gap is then a
fact about the budget rather than about the choices. The budget curve exists to
make that distinction visible: if the worst gap keeps falling as frames are
added, the budget is binding, and if it flattens, it is not.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import islice
from typing import Any

from frame_quorum.errors import ConfigurationError
from frame_quorum.metrics import content_distance
from frame_quorum.models import (
    Frame,
    SelectionConfig,
    SelectionResult,
    _is_stable_json_number,
    _require_int64,
    _require_safe_text,
)

#: Worst-represented frames named individually in a report. A long gap is one
#: finding, and listing every frame inside it would bury the others.
MAX_REPORTED_GAPS = 10

#: Maximum number of budgets accepted from a programmatic iterable. This keeps
#: accidental or hostile infinite generators from being materialized without a
#: bound before validation.
MAX_BUDGET_POINTS = 4_096

_MAX_REPORT_ITEMS = 100_000
_MAX_BUDGET_WORK_UNITS = 100_000_000

#: Distance below which a dropped frame counts as represented by its nearest
#: kept neighbour. Matched to the selector's own duplicate threshold, so a
#: frame the selector would have called a duplicate is one this calls covered.
DEFAULT_COVERED_DISTANCE = SelectionConfig().duplicate_threshold


@dataclass(frozen=True, slots=True)
class FrameRepresentation:
    """One dropped frame and the kept frame standing in for it."""

    index: int
    relative_path: str
    nearest_index: int
    distance: float

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        _require_int64(self.index, "represented frame index")
        _require_safe_text(self.relative_path, "represented frame path")
        _require_int64(self.nearest_index, "nearest selected frame index")
        if self.index == self.nearest_index:
            raise ConfigurationError("a dropped frame cannot be its own selected representative")
        _require_unit_interval(self.distance, "representation distance")

    def as_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "index": self.index,
            "relative_path": self.relative_path,
            "nearest_index": self.nearest_index,
            "distance": self.distance,
        }


@dataclass(frozen=True, slots=True)
class RepresentationGap:
    """A run of consecutive frames that no kept frame represents closely."""

    start_index: int
    end_index: int
    worst_index: int
    worst_distance: float

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        _require_int64(self.start_index, "gap start index")
        _require_int64(self.end_index, "gap end index")
        _require_int64(self.worst_index, "gap worst index")
        if self.end_index < self.start_index:
            raise ConfigurationError("gap end index must not precede its start index")
        if not self.start_index <= self.worst_index <= self.end_index:
            raise ConfigurationError("gap worst index must lie inside the gap")
        _require_unit_interval(self.worst_distance, "gap worst distance")

    @property
    def length(self) -> int:
        self.validate()
        return self.end_index - self.start_index + 1

    def as_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "start_index": self.start_index,
            "end_index": self.end_index,
            "length": self.length,
            "worst_index": self.worst_index,
            "worst_distance": self.worst_distance,
        }


@dataclass(frozen=True, slots=True)
class RepresentationReport:
    """What a selection covers, and what it does not."""

    selected: int
    dropped: int
    covered_distance: float
    worst_distance: float
    mean_distance: float
    covered_fraction: float
    gaps: tuple[RepresentationGap, ...]
    worst_frames: tuple[FrameRepresentation, ...]

    def __post_init__(self) -> None:
        gaps = tuple(
            _snapshot_gap(item)
            for item in _bounded_tuple(self.gaps, _MAX_REPORT_ITEMS, "representation gaps")
        )
        worst_frames = tuple(
            _snapshot_frame_representation(item)
            for item in _bounded_tuple(
                self.worst_frames,
                MAX_REPORTED_GAPS,
                "worst represented frames",
            )
        )
        object.__setattr__(self, "gaps", gaps)
        object.__setattr__(self, "worst_frames", worst_frames)
        self.validate()

    def validate(self) -> None:
        _require_int64(self.selected, "selected frame count", minimum=1)
        _require_int64(self.dropped, "dropped frame count")
        if self.selected + self.dropped > (1 << 63) - 1:
            raise ConfigurationError("total frame count must fit the signed 64-bit range")
        _require_unit_interval(self.covered_distance, "covered_distance")
        _require_unit_interval(self.worst_distance, "worst representation distance")
        _require_unit_interval(self.mean_distance, "mean representation distance")
        _require_unit_interval(self.covered_fraction, "covered fraction")
        if self.mean_distance > self.worst_distance:
            raise ConfigurationError("mean representation distance cannot exceed the worst distance")
        if not isinstance(self.gaps, tuple) or len(self.gaps) > _MAX_REPORT_ITEMS:
            raise ConfigurationError(
                f"representation gaps must be a tuple of at most {_MAX_REPORT_ITEMS} records"
            )
        if not isinstance(self.worst_frames, tuple) or len(self.worst_frames) > MAX_REPORTED_GAPS:
            raise ConfigurationError(
                f"worst represented frames must be a tuple of at most {MAX_REPORTED_GAPS} records"
            )
        for gap in self.gaps:
            if not isinstance(gap, RepresentationGap):
                raise ConfigurationError("representation gaps must contain RepresentationGap records")
            gap.validate()
        for item in self.worst_frames:
            if not isinstance(item, FrameRepresentation):
                raise ConfigurationError("worst represented frames must contain FrameRepresentation records")
            item.validate()
        if any(
            earlier.end_index >= later.start_index
            for earlier, later in zip(self.gaps, self.gaps[1:], strict=False)
        ):
            raise ConfigurationError("representation gaps must be ordered and non-overlapping")
        if any(gap.worst_distance <= self.covered_distance for gap in self.gaps):
            raise ConfigurationError("every representation gap must exceed the covered distance")
        if any(gap.worst_distance > self.worst_distance for gap in self.gaps):
            raise ConfigurationError("a representation gap cannot exceed the report's worst distance")
        ordered_worst = tuple(sorted(self.worst_frames, key=lambda item: (-item.distance, item.index)))
        if self.worst_frames != ordered_worst:
            raise ConfigurationError("worst represented frames must be ordered by distance")
        worst_indices = tuple(item.index for item in self.worst_frames)
        if len(worst_indices) != len(set(worst_indices)):
            raise ConfigurationError("worst represented frame indices must be unique")
        if any(item.distance > self.worst_distance for item in self.worst_frames):
            raise ConfigurationError("a represented frame cannot exceed the report's worst distance")
        if self.dropped == 0:
            if (
                self.worst_distance != 0.0
                or self.mean_distance != 0.0
                or self.covered_fraction != 1.0
                or self.gaps
                or self.worst_frames
            ):
                raise ConfigurationError("a report without dropped frames must be complete and empty")
        else:
            expected_worst_frames = min(self.dropped, MAX_REPORTED_GAPS)
            if len(self.worst_frames) != expected_worst_frames:
                raise ConfigurationError(
                    f"a report with dropped frames must identify its {expected_worst_frames} worst frames"
                )
            if self.worst_frames[0].distance != self.worst_distance:
                raise ConfigurationError("the first worst frame must match the report's worst distance")
            if bool(self.gaps) == (self.covered_fraction == 1.0):
                raise ConfigurationError("gap presence must agree with the covered fraction")
            covered_count = round(self.covered_fraction * self.dropped)
            if not math.isclose(
                self.covered_fraction,
                covered_count / self.dropped,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ConfigurationError("covered fraction must describe a whole number of dropped frames")
            uncovered_count = self.dropped - covered_count
            if sum(gap.length for gap in self.gaps) != uncovered_count:
                raise ConfigurationError(
                    "representation gaps must account for every uncovered frame exactly once"
                )

    @property
    def complete(self) -> bool:
        """Whether every dropped frame is close to something that was kept."""

        self.validate()
        return not self.gaps

    def as_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": 1,
            "selected": self.selected,
            "dropped": self.dropped,
            "covered_distance": self.covered_distance,
            "worst_distance": self.worst_distance,
            "mean_distance": self.mean_distance,
            "covered_fraction": self.covered_fraction,
            "complete": self.complete,
            "gaps": [gap.as_dict() for gap in self.gaps],
            "worst_frames": [item.as_dict() for item in self.worst_frames],
        }


@dataclass(frozen=True, slots=True)
class BudgetPoint:
    """What one budget achieved."""

    budget: int
    selected: int
    worst_distance: float
    mean_distance: float
    covered_fraction: float

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        _require_int64(self.budget, "curve budget", minimum=1)
        _require_int64(self.selected, "curve selected frame count", minimum=1)
        if self.selected > self.budget:
            raise ConfigurationError("curve selected frame count cannot exceed its budget")
        _require_unit_interval(self.worst_distance, "curve worst distance")
        _require_unit_interval(self.mean_distance, "curve mean distance")
        _require_unit_interval(self.covered_fraction, "curve covered fraction")
        if self.mean_distance > self.worst_distance:
            raise ConfigurationError("curve mean distance cannot exceed its worst distance")

    def as_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "budget": self.budget,
            "selected": self.selected,
            "worst_distance": self.worst_distance,
            "mean_distance": self.mean_distance,
            "covered_fraction": self.covered_fraction,
        }


@dataclass(frozen=True, slots=True)
class BudgetCurve:
    """Representation against budget, for choosing one deliberately."""

    points: tuple[BudgetPoint, ...]
    covered_distance: float

    def __post_init__(self) -> None:
        points = tuple(
            _snapshot_budget_point(item)
            for item in _bounded_tuple(self.points, MAX_BUDGET_POINTS, "budget curve points")
        )
        object.__setattr__(self, "points", points)
        self.validate()

    def validate(self) -> None:
        if not self.points:
            raise ConfigurationError("a budget curve needs at least one point")
        if not isinstance(self.points, tuple) or len(self.points) > MAX_BUDGET_POINTS:
            raise ConfigurationError(
                f"budget curve points must be a tuple of at most {MAX_BUDGET_POINTS} records"
            )
        for point in self.points:
            if not isinstance(point, BudgetPoint):
                raise ConfigurationError("budget curve points must contain BudgetPoint records")
            point.validate()
        budgets = tuple(point.budget for point in self.points)
        if budgets != tuple(sorted(set(budgets))):
            raise ConfigurationError("budget curve points must have unique increasing budgets")
        if any(
            earlier.selected > later.selected
            for earlier, later in zip(self.points, self.points[1:], strict=False)
        ):
            raise ConfigurationError("selected frame counts must not decrease as the budget grows")
        if any(
            earlier.worst_distance < later.worst_distance
            for earlier, later in zip(self.points, self.points[1:], strict=False)
        ):
            raise ConfigurationError("worst representation distance must not increase as the budget grows")
        if any(
            earlier.covered_fraction > later.covered_fraction
            for earlier, later in zip(self.points, self.points[1:], strict=False)
        ):
            raise ConfigurationError("covered fraction must not decrease as the budget grows")
        _require_unit_interval(self.covered_distance, "covered_distance")

    def knee(self, tolerance: float = 0.02) -> int | None:
        """Smallest budget past which the worst gap stops improving much.

        Returns None when the worst gap was still falling at the largest budget
        tried, which means the budget is still binding and the answer lies
        outside the range that was searched.
        """

        self.validate()
        _require_unit_interval(tolerance, "knee tolerance")
        if len(self.points) < 2:
            return None
        best = self.points[-1].worst_distance
        for point in self.points:
            if point.worst_distance - best <= tolerance:
                return point.budget if point is not self.points[-1] else None
        return None

    def as_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": 1,
            "covered_distance": self.covered_distance,
            "knee": self.knee(),
            "points": [point.as_dict() for point in self.points],
        }


def _nearest(frame: Frame, kept: Sequence[Frame]) -> tuple[int, float]:
    best_index = kept[0].index
    best = content_distance(frame.metrics, kept[0].metrics)
    for candidate in kept[1:]:
        distance = content_distance(frame.metrics, candidate.metrics)
        if distance < best:
            best, best_index = distance, candidate.index
    return best_index, best


def _gaps(ordered: Sequence[FrameRepresentation], covered_distance: float) -> tuple[RepresentationGap, ...]:
    """Group consecutive under-represented frames into runs.

    One event that the selection missed shows up as a stretch of neighbouring
    frames, all far from anything kept. Reporting them individually would
    describe one absence many times.
    """

    gaps: list[RepresentationGap] = []
    run: list[FrameRepresentation] = []
    for item in ordered:
        if item.distance <= covered_distance:
            continue
        if run and item.index != run[-1].index + 1:
            gaps.append(_gap_from(run))
            run = []
        run.append(item)
    if run:
        gaps.append(_gap_from(run))
    return tuple(gaps)


def _gap_from(run: Sequence[FrameRepresentation]) -> RepresentationGap:
    worst = max(run, key=lambda item: (item.distance, -item.index))
    return RepresentationGap(
        start_index=run[0].index,
        end_index=run[-1].index,
        worst_index=worst.index,
        worst_distance=worst.distance,
    )


def analyze_representation(
    result: SelectionResult, *, covered_distance: float = DEFAULT_COVERED_DISTANCE
) -> RepresentationReport:
    """Measure how far every dropped frame sits from the nearest kept one."""

    _require_unit_interval(covered_distance, "covered_distance")
    if not isinstance(result, SelectionResult):
        raise ConfigurationError("representation input must be a SelectionResult")
    result.validate()
    if len(result.frames) > _MAX_REPORT_ITEMS:
        raise ConfigurationError(f"representation input cannot contain more than {_MAX_REPORT_ITEMS} frames")
    selected = set(result.selected_indices)
    kept = [frame for frame in result.frames if frame.index in selected]
    if not kept:
        raise ConfigurationError("a selection with no frames cannot represent anything")
    dropped = [frame for frame in result.frames if frame.index not in selected]
    if not dropped:
        return RepresentationReport(
            selected=len(kept),
            dropped=0,
            covered_distance=covered_distance,
            worst_distance=0.0,
            mean_distance=0.0,
            covered_fraction=1.0,
            gaps=(),
            worst_frames=(),
        )

    measured = []
    for frame in dropped:
        nearest_index, distance = _nearest(frame, kept)
        measured.append(
            FrameRepresentation(
                index=frame.index,
                relative_path=frame.relative_path,
                nearest_index=nearest_index,
                distance=distance,
            )
        )
    ordered = sorted(measured, key=lambda item: item.index)
    covered = sum(item.distance <= covered_distance for item in ordered)
    return RepresentationReport(
        selected=len(kept),
        dropped=len(ordered),
        covered_distance=covered_distance,
        worst_distance=max(item.distance for item in ordered),
        mean_distance=sum(item.distance for item in ordered) / len(ordered),
        covered_fraction=covered / len(ordered),
        gaps=_gaps(ordered, covered_distance),
        worst_frames=tuple(
            sorted(measured, key=lambda item: (-item.distance, item.index))[:MAX_REPORTED_GAPS]
        ),
    )


def budget_curve(
    frames: tuple[Frame, ...],
    config: SelectionConfig,
    budgets: Iterable[int],
    *,
    covered_distance: float = DEFAULT_COVERED_DISTANCE,
) -> BudgetCurve:
    """Re-select at several budgets and report what each one represents.

    The budget is the one parameter a caller picks with no basis, and this is
    the basis. A worst gap that keeps falling as frames are added says the
    budget is binding; one that flattens says it is not, and the extra frames
    are being spent on moments already covered.
    """

    from frame_quorum.selector import select_frames

    _require_unit_interval(covered_distance, "covered_distance")
    if not isinstance(config, SelectionConfig):
        raise ConfigurationError("budget curve config must be SelectionConfig")
    config.validate()
    supplied = _bounded_tuple(budgets, MAX_BUDGET_POINTS, "budgets")
    if not supplied:
        raise ConfigurationError("a budget curve needs at least one budget")
    for budget in supplied:
        _require_int64(budget, "curve budget", minimum=1)
    wanted = sorted(set(supplied))
    if not isinstance(frames, tuple) or any(not isinstance(frame, Frame) for frame in frames):
        raise ConfigurationError("budget curve frames must be a tuple of Frame objects")
    if len(frames) > _MAX_REPORT_ITEMS:
        raise ConfigurationError(f"budget curve input cannot contain more than {_MAX_REPORT_ITEMS} frames")
    if len(frames) * len(wanted) > _MAX_BUDGET_WORK_UNITS:
        raise ConfigurationError(
            f"budget curve work cannot exceed {_MAX_BUDGET_WORK_UNITS} frame-budget units"
        )
    points = []
    for budget in wanted:
        candidate = SelectionConfig(
            budget=budget,
            min_gap=config.min_gap,
            duplicate_threshold=config.duplicate_threshold,
            quality_weight=config.quality_weight,
            change_weight=config.change_weight,
            coverage_weight=config.coverage_weight,
            keep_endpoints=config.keep_endpoints,
        )
        result = select_frames(frames, candidate)
        report = analyze_representation(result, covered_distance=covered_distance)
        points.append(
            BudgetPoint(
                budget=budget,
                # A budget larger than the sequence, or one the constraints cut
                # short, selects fewer frames than it asked for. Reporting the
                # request alone would make the curve look like it flattened
                # because more frames stopped helping.
                selected=report.selected,
                worst_distance=report.worst_distance,
                mean_distance=report.mean_distance,
                covered_fraction=report.covered_fraction,
            )
        )
    return BudgetCurve(points=tuple(points), covered_distance=covered_distance)


def _require_unit_interval(value: float | int, label: str) -> None:
    if not _is_stable_json_number(value) or not 0.0 <= value <= 1.0:
        raise ConfigurationError(f"{label} must be a finite number between zero and one")


def _bounded_tuple(values: Iterable[Any], limit: int, label: str) -> tuple[Any, ...]:
    if isinstance(values, (str, bytes)):
        raise ConfigurationError(f"{label} must be an iterable of records")
    try:
        items = tuple(islice(iter(values), limit + 1))
    except TypeError as exc:
        raise ConfigurationError(f"{label} must be iterable") from exc
    if len(items) > limit:
        raise ConfigurationError(f"{label} cannot contain more than {limit} items")
    return items


def _snapshot_frame_representation(value: Any) -> FrameRepresentation:
    if not isinstance(value, FrameRepresentation):
        raise ConfigurationError("worst represented frames must contain FrameRepresentation records")
    value.validate()
    return FrameRepresentation(value.index, value.relative_path, value.nearest_index, value.distance)


def _snapshot_gap(value: Any) -> RepresentationGap:
    if not isinstance(value, RepresentationGap):
        raise ConfigurationError("representation gaps must contain RepresentationGap records")
    value.validate()
    return RepresentationGap(value.start_index, value.end_index, value.worst_index, value.worst_distance)


def _snapshot_budget_point(value: Any) -> BudgetPoint:
    if not isinstance(value, BudgetPoint):
        raise ConfigurationError("budget curve points must contain BudgetPoint records")
    value.validate()
    return BudgetPoint(
        value.budget,
        value.selected,
        value.worst_distance,
        value.mean_distance,
        value.covered_fraction,
    )
