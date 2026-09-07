"""How well a selection stands in for the frames it left out."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import pytest

from frame_quorum.cli import main
from frame_quorum.demo import create_demo_sequence
from frame_quorum.errors import ConfigurationError
from frame_quorum.models import Frame, ScanConfig, SelectionConfig
from frame_quorum.representation import (
    DEFAULT_COVERED_DISTANCE,
    MAX_REPORTED_GAPS,
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
    from dataclasses import replace

    result = replace(select_frames(frames, SelectionConfig(budget=3)), selected_indices=())
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
