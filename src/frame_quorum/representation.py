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

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from frame_quorum.errors import ConfigurationError
from frame_quorum.metrics import content_distance
from frame_quorum.models import Frame, SelectionConfig, SelectionResult

#: Worst-represented frames named individually in a report. A long gap is one
#: finding, and listing every frame inside it would bury the others.
MAX_REPORTED_GAPS = 10

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

    def as_dict(self) -> dict[str, Any]:
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

    @property
    def length(self) -> int:
        return self.end_index - self.start_index + 1

    def as_dict(self) -> dict[str, Any]:
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

    @property
    def complete(self) -> bool:
        """Whether every dropped frame is close to something that was kept."""

        return not self.gaps

    def as_dict(self) -> dict[str, Any]:
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

    def as_dict(self) -> dict[str, Any]:
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

    def knee(self, tolerance: float = 0.02) -> int | None:
        """Smallest budget past which the worst gap stops improving much.

        Returns None when the worst gap was still falling at the largest budget
        tried, which means the budget is still binding and the answer lies
        outside the range that was searched.
        """

        if len(self.points) < 2:
            return None
        best = self.points[-1].worst_distance
        for point in self.points:
            if point.worst_distance - best <= tolerance:
                return point.budget if point is not self.points[-1] else None
        return None

    def as_dict(self) -> dict[str, Any]:
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

    if not 0.0 <= covered_distance <= 1.0:
        raise ConfigurationError("covered_distance must lie between 0 and 1")
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

    wanted = sorted({int(budget) for budget in budgets})
    if not wanted:
        raise ConfigurationError("a budget curve needs at least one budget")
    if any(budget < 1 for budget in wanted):
        raise ConfigurationError("every budget must be at least one")
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
