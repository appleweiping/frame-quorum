"""How well a selection stands in for the frames it left out."""

from __future__ import annotations

import itertools
import json
from dataclasses import replace
from pathlib import Path

import pytest

from frame_quorum.cli import main
from frame_quorum.demo import create_demo_sequence
from frame_quorum.errors import ConfigurationError
from frame_quorum.models import Frame, ScanConfig, SelectionConfig
from frame_quorum.representation import (
    DEFAULT_COVERED_DISTANCE,
    MAX_BUDGET_POINTS,
    MAX_REPORTED_GAPS,
    BudgetCurve,
    BudgetPoint,
    FrameRepresentation,
    RepresentationGap,
    RepresentationReport,
    analyze_representation,
    budget_curve,
)
from frame_quorum.scanner import scan_frames
from frame_quorum.selector import select_frames


@pytest.fixture(scope="module")
def frames(tmp_path_factory) -> tuple[Frame, ...]:
    directory = create_demo_sequence(tmp_path_factory.mktemp("demo") / "frames")
    return scan_frames(directory, ScanConfig())


def report_for(frames: tuple[Frame, ...], budget: int, **kwargs):
    config = SelectionConfig(budget=budget)
    return analyze_representation(select_frames(frames, config), **kwargs)


# ---------------------------------------------------------------------------
# Measuring one selection.
# ---------------------------------------------------------------------------


def test_every_dropped_frame_is_accounted_for(frames: tuple[Frame, ...]) -> None:
    report = report_for(frames, 6)
    assert report.selected == 6
    assert report.selected + report.dropped == len(frames)


def test_the_worst_distance_bounds_every_other(frames: tuple[Frame, ...]) -> None:
    report = report_for(frames, 6)
    assert report.mean_distance <= report.worst_distance
    assert all(item.distance <= report.worst_distance for item in report.worst_frames)


def test_the_nearest_kept_frame_is_actually_the_nearest(frames: tuple[Frame, ...]) -> None:
    """The number is only useful if it names the right neighbour."""

    from frame_quorum.metrics import content_distance

    config = SelectionConfig(budget=5)
    result = select_frames(frames, config)
    report = analyze_representation(result)
    kept = {frame.index: frame for frame in frames if frame.index in result.selected_indices}
    by_index = {frame.index: frame for frame in frames}
    for item in report.worst_frames:
        dropped = by_index[item.index]
        best = min(content_distance(dropped.metrics, candidate.metrics) for candidate in kept.values())
        assert item.distance == pytest.approx(best)
        assert content_distance(dropped.metrics, kept[item.nearest_index].metrics) == pytest.approx(best)


def test_a_selection_that_keeps_everything_has_nothing_to_represent(
    frames: tuple[Frame, ...],
) -> None:
    report = report_for(frames, len(frames))
    assert report.dropped == 0
    assert report.worst_distance == 0.0
    assert report.covered_fraction == 1.0
    assert report.complete


def test_worst_frames_are_ordered_by_how_badly_they_are_represented(
    frames: tuple[Frame, ...],
) -> None:
    distances = [item.distance for item in report_for(frames, 4).worst_frames]
    assert distances == sorted(distances, reverse=True)
    assert len(distances) <= MAX_REPORTED_GAPS


# ---------------------------------------------------------------------------
# Gaps.
# ---------------------------------------------------------------------------


def test_neighbouring_unrepresented_frames_form_one_gap(frames: tuple[Frame, ...]) -> None:
    """One missed event is one finding, not one finding per frame in it."""

    report = report_for(frames, 6)
    assert report.gaps
    for gap in report.gaps:
        assert gap.start_index <= gap.worst_index <= gap.end_index
        assert gap.length == gap.end_index - gap.start_index + 1


def test_gaps_do_not_overlap_and_run_in_order(frames: tuple[Frame, ...]) -> None:
    gaps = report_for(frames, 4).gaps
    for earlier, later in itertools.pairwise(gaps):
        assert earlier.end_index < later.start_index


def test_a_generous_threshold_covers_everything(frames: tuple[Frame, ...]) -> None:
    report = report_for(frames, 6, covered_distance=1.0)
    assert report.covered_fraction == 1.0
    assert report.gaps == ()
    assert report.complete


def test_a_strict_threshold_covers_nothing(frames: tuple[Frame, ...]) -> None:
    report = report_for(frames, 6, covered_distance=0.0)
    assert report.covered_fraction == 0.0
    assert not report.complete


def test_the_demo_sequence_has_no_near_duplicates_to_cover(
    frames: tuple[Frame, ...],
) -> None:
    """Worth pinning, because it explains a zero that looks like a bug.

    Every demo frame is a distinct scene, so at the selector's own duplicate
    threshold no dropped frame is covered at any budget. The zero is a fact
    about the sequence rather than about the selection.
    """

    assert report_for(frames, 6).covered_fraction == 0.0
    assert report_for(frames, 12).covered_fraction == 0.0


@pytest.mark.parametrize("distance", [-0.1, 1.5])
def test_an_out_of_range_threshold_is_refused(frames: tuple[Frame, ...], distance: float) -> None:
    with pytest.raises(ConfigurationError, match="covered_distance"):
        report_for(frames, 6, covered_distance=distance)


def test_an_empty_selection_cannot_represent_anything(frames: tuple[Frame, ...]) -> None:
    selected = select_frames(frames, SelectionConfig(budget=3))
    decisions = tuple(
        replace(decision, selected=False, rank=None, nearest_selected_index=None)
        for decision in selected.decisions
    )
    result = replace(selected, selected_indices=(), decisions=decisions)
    with pytest.raises(ConfigurationError, match="cannot represent"):
        analyze_representation(result)


# ---------------------------------------------------------------------------
# The budget curve.
# ---------------------------------------------------------------------------


def test_more_frames_never_make_the_worst_gap_worse(frames: tuple[Frame, ...]) -> None:
    curve = budget_curve(frames, SelectionConfig(), [2, 4, 6, 8, 12])
    worst = [point.worst_distance for point in curve.points]
    assert worst == sorted(worst, reverse=True)


def test_the_curve_reports_frames_kept_not_frames_asked_for(
    frames: tuple[Frame, ...],
) -> None:
    """A budget past the sequence length would otherwise look like a plateau."""

    curve = budget_curve(frames, SelectionConfig(), [len(frames) + 50])
    assert curve.points[0].budget == len(frames) + 50
    assert curve.points[0].selected <= len(frames)


def test_budgets_are_deduplicated_and_ordered(frames: tuple[Frame, ...]) -> None:
    curve = budget_curve(frames, SelectionConfig(), [8, 2, 8, 4])
    assert [point.budget for point in curve.points] == [2, 4, 8]


def test_the_knee_is_where_the_worst_gap_stops_improving(
    frames: tuple[Frame, ...],
) -> None:
    curve = budget_curve(frames, SelectionConfig(), [2, 4, 6, 8, 12, 16])
    knee = curve.knee()
    assert knee is not None
    point = next(item for item in curve.points if item.budget == knee)
    assert point.worst_distance - curve.points[-1].worst_distance <= 0.02


def test_a_curve_still_improving_at_the_end_reports_no_knee(
    frames: tuple[Frame, ...],
) -> None:
    """Reporting the largest budget tried would invent a plateau."""

    curve = budget_curve(frames, SelectionConfig(), [2, 3])
    assert curve.points[0].worst_distance > curve.points[-1].worst_distance + 0.02
    assert curve.knee() is None


def test_a_single_budget_has_no_knee_to_find(frames: tuple[Frame, ...]) -> None:
    assert budget_curve(frames, SelectionConfig(), [6]).knee() is None


def test_the_curve_keeps_every_other_setting(frames: tuple[Frame, ...]) -> None:
    # Only the budget varies; changing the weights too would measure something
    # else entirely.
    config = SelectionConfig(budget=4, quality_weight=0.8, change_weight=0.1, coverage_weight=0.1)
    baseline = analyze_representation(select_frames(frames, config))
    curve = budget_curve(frames, config, [4])
    assert curve.points[0].worst_distance == pytest.approx(baseline.worst_distance)


@pytest.mark.parametrize("budgets", [[], [0], [-1], [3, 0]])
def test_an_invalid_budget_list_is_refused(frames: tuple[Frame, ...], budgets: list[int]) -> None:
    with pytest.raises(ConfigurationError, match="budget"):
        budget_curve(frames, SelectionConfig(), budgets)


def test_the_curve_serializes(frames: tuple[Frame, ...]) -> None:
    payload = budget_curve(frames, SelectionConfig(), [2, 4, 8]).as_dict()
    assert payload["schema_version"] == 1
    assert len(payload["points"]) == 3
    json.dumps(payload, allow_nan=False)


def test_a_report_serializes(frames: tuple[Frame, ...]) -> None:
    payload = report_for(frames, 6).as_dict()
    assert payload["selected"] == 6
    assert payload["gaps"]
    json.dumps(payload, allow_nan=False)


def test_the_default_threshold_matches_the_selectors_own(frames: tuple[Frame, ...]) -> None:
    # A frame the selector would call a duplicate is one this calls covered.
    assert SelectionConfig().duplicate_threshold == DEFAULT_COVERED_DISTANCE


def test_representation_records_reject_impossible_direct_construction() -> None:
    with pytest.raises(ConfigurationError, match="own selected representative"):
        FrameRepresentation(index=2, relative_path="2.png", nearest_index=2, distance=0.1)
    with pytest.raises(ConfigurationError, match="inside the gap"):
        RepresentationGap(start_index=3, end_index=5, worst_index=6, worst_distance=0.4)
    with pytest.raises(ConfigurationError, match="cannot exceed its budget"):
        BudgetPoint(
            budget=2,
            selected=3,
            worst_distance=0.4,
            mean_distance=0.2,
            covered_fraction=0.5,
        )

    with pytest.raises(ConfigurationError, match="path"):
        FrameRepresentation(index=2, relative_path="", nearest_index=1, distance=0.1)
    with pytest.raises(ConfigurationError, match="finite number"):
        FrameRepresentation(index=2, relative_path="2.png", nearest_index=1, distance=float("nan"))
    with pytest.raises(ConfigurationError, match="precede"):
        RepresentationGap(start_index=5, end_index=3, worst_index=4, worst_distance=0.4)
    with pytest.raises(ConfigurationError, match="worst distance"):
        BudgetPoint(2, 1, 0.2, 0.3, 0.5)


def test_report_rejects_forged_counts_types_and_aggregates() -> None:
    worst = FrameRepresentation(1, "1.png", 0, 0.6)
    gap = RepresentationGap(1, 1, 1, 0.6)
    report = RepresentationReport(1, 1, 0.1, 0.6, 0.6, 0.0, (gap,), (worst,))

    with pytest.raises(ConfigurationError, match="selected frame count"):
        replace(report, selected=0)
    with pytest.raises(ConfigurationError, match="mean representation"):
        replace(report, mean_distance=0.7)
    with pytest.raises(ConfigurationError, match="RepresentationGap"):
        replace(report, gaps=(worst,))  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError, match="FrameRepresentation"):
        replace(report, worst_frames=(gap,))  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError, match="gap cannot exceed"):
        replace(report, worst_distance=0.55, mean_distance=0.5)
    with pytest.raises(ConfigurationError, match="whole number"):
        replace(report, covered_fraction=0.25)
    with pytest.raises(ConfigurationError, match="without dropped frames"):
        replace(report, dropped=0)


def test_report_gaps_account_for_each_uncovered_frame_once() -> None:
    worst = (
        FrameRepresentation(1, "1.png", 0, 0.6),
        FrameRepresentation(2, "2.png", 0, 0.5),
    )
    with pytest.raises(ConfigurationError, match="account for every uncovered"):
        RepresentationReport(
            selected=1,
            dropped=2,
            covered_distance=0.1,
            worst_distance=0.6,
            mean_distance=0.55,
            covered_fraction=0.0,
            gaps=(RepresentationGap(1, 1, 1, 0.6),),
            worst_frames=worst,
        )


def test_report_rejects_overlapping_gaps_and_incomplete_worst_list() -> None:
    worst = FrameRepresentation(1, "1.png", 0, 0.6)
    with pytest.raises(ConfigurationError, match="ordered and non-overlapping"):
        RepresentationReport(
            selected=1,
            dropped=2,
            covered_distance=0.1,
            worst_distance=0.6,
            mean_distance=0.55,
            covered_fraction=0.0,
            gaps=(RepresentationGap(1, 1, 1, 0.6), RepresentationGap(1, 2, 1, 0.6)),
            worst_frames=(worst, FrameRepresentation(2, "2.png", 0, 0.5)),
        )
    with pytest.raises(ConfigurationError, match="2 worst frames"):
        RepresentationReport(1, 2, 0.1, 0.6, 0.55, 0.0, (RepresentationGap(1, 2, 1, 0.6),), (worst,))


def test_report_replace_rechecks_cross_record_invariants(frames: tuple[Frame, ...]) -> None:
    report = report_for(frames, 6)
    assert report.gaps
    with pytest.raises(ConfigurationError, match="gap presence"):
        replace(report, covered_fraction=1.0)
    with pytest.raises(ConfigurationError, match="ordered by distance"):
        replace(report, worst_frames=tuple(reversed(report.worst_frames)))


def test_budget_curve_snapshots_and_validates_programmatic_points() -> None:
    first = BudgetPoint(1, 1, 0.5, 0.4, 0.0)
    second = BudgetPoint(2, 2, 0.3, 0.2, 0.5)
    caller_owned = [first, second]
    curve = BudgetCurve(points=caller_owned, covered_distance=0.1)  # type: ignore[arg-type]
    caller_owned.clear()
    assert curve.points == (first, second)
    with pytest.raises(ConfigurationError, match="unique increasing"):
        replace(curve, points=(second, first))
    with pytest.raises(ConfigurationError, match="knee tolerance"):
        curve.knee(float("nan"))


def test_reports_and_curves_deep_snapshot_nested_records() -> None:
    worst = FrameRepresentation(1, "1.png", 0, 0.6)
    gap = RepresentationGap(1, 1, 1, 0.6)
    report = RepresentationReport(1, 1, 0.1, 0.6, 0.6, 0.0, (gap,), (worst,))
    point = BudgetPoint(1, 1, 0.5, 0.4, 0.0)
    curve = BudgetCurve((point,), 0.1)

    object.__setattr__(worst, "distance", float("nan"))
    object.__setattr__(gap, "worst_distance", float("nan"))
    object.__setattr__(point, "worst_distance", float("nan"))

    assert report.as_dict()["worst_distance"] == 0.6
    assert curve.as_dict()["points"][0]["worst_distance"] == 0.5


def test_serialization_revalidates_nested_records_after_forced_mutation() -> None:
    worst = FrameRepresentation(1, "1.png", 0, 0.6)
    gap = RepresentationGap(1, 1, 1, 0.6)
    report = RepresentationReport(1, 1, 0.1, 0.6, 0.6, 0.0, (gap,), (worst,))
    curve = BudgetCurve((BudgetPoint(1, 1, 0.5, 0.4, 0.0),), 0.1)

    object.__setattr__(report.worst_frames[0], "distance", float("nan"))
    with pytest.raises(ConfigurationError, match="finite number"):
        report.as_dict()

    object.__setattr__(curve.points[0], "budget", 0)
    with pytest.raises(ConfigurationError, match="between 1"):
        curve.as_dict()


def test_serialization_rejects_forced_unbounded_outer_collections() -> None:
    report = RepresentationReport(1, 0, 0.1, 0.0, 0.0, 1.0, (), ())
    curve = BudgetCurve((BudgetPoint(1, 1, 0.0, 0.0, 1.0),), 0.1)

    object.__setattr__(report, "gaps", itertools.repeat(RepresentationGap(1, 1, 1, 0.6)))
    with pytest.raises(ConfigurationError, match="must be a tuple"):
        report.as_dict()

    object.__setattr__(curve, "points", itertools.repeat(BudgetPoint(1, 1, 0.0, 0.0, 1.0)))
    with pytest.raises(ConfigurationError, match="must be a tuple"):
        curve.as_dict()


def test_budget_curve_rejects_cross_point_regressions() -> None:
    first = BudgetPoint(1, 1, 0.5, 0.4, 0.2)
    with pytest.raises(ConfigurationError, match="selected frame counts"):
        BudgetCurve((BudgetPoint(2, 2, 0.5, 0.4, 0.2), BudgetPoint(3, 1, 0.4, 0.3, 0.3)), 0.1)
    with pytest.raises(ConfigurationError, match="worst representation"):
        BudgetCurve((first, BudgetPoint(2, 2, 0.6, 0.4, 0.3)), 0.1)
    with pytest.raises(ConfigurationError, match="covered fraction"):
        BudgetCurve((first, BudgetPoint(2, 2, 0.4, 0.3, 0.1)), 0.1)


def test_budget_curve_rejects_empty_wrong_and_non_iterable_points() -> None:
    with pytest.raises(ConfigurationError, match="at least one point"):
        BudgetCurve((), 0.1)
    with pytest.raises(ConfigurationError, match="iterable of records"):
        BudgetCurve("not-points", 0.1)  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError, match="must be iterable"):
        BudgetCurve(7, 0.1)  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError, match="BudgetPoint"):
        BudgetCurve((object(),), 0.1)  # type: ignore[arg-type]


def test_budget_iterables_are_bounded_before_materialization(frames: tuple[Frame, ...]) -> None:
    with pytest.raises(ConfigurationError, match=str(MAX_BUDGET_POINTS)):
        budget_curve(frames, SelectionConfig(), itertools.repeat(1))
    with pytest.raises(ConfigurationError, match="integer"):
        budget_curve(frames, SelectionConfig(), [True])
    with pytest.raises(ConfigurationError, match="SelectionConfig"):
        budget_curve(frames, object(), [1])  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError, match="tuple of Frame"):
        budget_curve(list(frames), SelectionConfig(), [1])  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError, match="covered_distance"):
        budget_curve(frames, SelectionConfig(), [1], covered_distance=float("inf"))


def test_representation_requires_a_selection_result() -> None:
    with pytest.raises(ConfigurationError, match="SelectionResult"):
        analyze_representation(object())  # type: ignore[arg-type]


def test_representation_revalidates_the_result_before_reading_its_fields(
    frames: tuple[Frame, ...],
) -> None:
    result = select_frames(frames, SelectionConfig(budget=3))
    object.__setattr__(result, "frames", itertools.repeat(frames[0]))

    with pytest.raises(ConfigurationError, match="frames must be a tuple"):
        analyze_representation(result)


# ---------------------------------------------------------------------------
# The command.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sequence(tmp_path_factory) -> Path:
    return create_demo_sequence(tmp_path_factory.mktemp("cli") / "frames")


def test_the_command_reports_gaps(sequence: Path, capsys) -> None:
    assert main(["coverage", str(sequence), "--budget", "6"]) == 0
    printed = capsys.readouterr().out
    assert "worst gap" in printed
    assert "represented no closer than" in printed


def test_the_command_writes_json_on_request(sequence: Path, tmp_path: Path) -> None:
    output = tmp_path / "coverage.json"
    assert main(["coverage", str(sequence), "--budget", "6", "--output", str(output)]) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["representation"]["selected"] == 6
    assert "budget_curve" not in payload


def test_the_command_adds_a_curve_when_budgets_are_given(sequence: Path, tmp_path: Path, capsys) -> None:
    output = tmp_path / "curve.json"
    assert (
        main(
            [
                "coverage",
                str(sequence),
                "--budget",
                "6",
                "--budgets",
                "2",
                "4",
                "8",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert "budget" in capsys.readouterr().out
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert len(payload["budget_curve"]["points"]) == 3


def test_the_command_says_when_the_budget_is_still_binding(sequence: Path, capsys) -> None:
    assert main(["coverage", str(sequence), "--budget", "3", "--budgets", "2", "3"]) == 0
    assert "still binding" in capsys.readouterr().out
