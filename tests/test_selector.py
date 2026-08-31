from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

import frame_quorum.selector as selector_module
from frame_quorum.errors import ConfigurationError
from frame_quorum.models import Frame, FrameMetrics, SelectionConfig
from frame_quorum.selector import select_frames


def _frame(
    index: int,
    *,
    hash_value: int | None = None,
    quality: float = 0.5,
    timestamp: float | None = None,
) -> Frame:
    hash_value = (index * 0x9E3779B97F4A7C15) & ((1 << 64) - 1) if hash_value is None else hash_value
    metrics = FrameMetrics(
        perceptual_hash=hash_value,
        luminance=0.5,
        entropy=quality,
        sharpness=quality,
        colorfulness=quality,
        mean_red=(index % 3) / 3,
        mean_green=((index + 1) % 3) / 3,
        mean_blue=((index + 2) % 3) / 3,
    )
    return Frame(index, Path(f"{index}.png"), f"{index}.png", timestamp, 100, 80, 100, metrics)


def test_empty_sequence_returns_empty_result() -> None:
    result = select_frames([])
    assert result.selected_indices == ()
    assert result.decisions == ()


def test_selection_respects_budget() -> None:
    result = select_frames([_frame(index) for index in range(10)], SelectionConfig(budget=4))
    assert len(result.selected_indices) == 4


def test_endpoints_are_reserved() -> None:
    result = select_frames([_frame(index) for index in range(6)], SelectionConfig(budget=3))
    assert 0 in result.selected_indices
    assert 5 in result.selected_indices


def test_budget_one_keeps_start_endpoint() -> None:
    result = select_frames([_frame(index) for index in range(4)], SelectionConfig(budget=1))
    assert result.selected_indices == (0,)
    assert result.decision_for(0).reason_code == "endpoint_start"
    assert result.decision_for(0).scores.coverage == 1.0


def test_coverage_prefers_middle_after_endpoints() -> None:
    frames = [_frame(index, hash_value=1 << (index * 10), quality=0.5) for index in range(5)]
    result = select_frames(frames, SelectionConfig(budget=3, coverage_weight=2.0))
    assert result.selected_indices == (0, 2, 4)


def test_exact_duplicate_is_suppressed() -> None:
    first = _frame(0, hash_value=0)
    duplicate = replace(first, index=1, path=Path("1.png"), relative_path="1.png")
    changed = _frame(2, hash_value=(1 << 64) - 1)
    result = select_frames([first, duplicate, changed], SelectionConfig(budget=3))
    assert result.selected_indices == (0, 2)
    assert result.decision_for(1).reason_code == "near_duplicate"


def test_minimum_gap_is_a_hard_constraint() -> None:
    frames = [_frame(index, timestamp=float(index)) for index in range(4)]
    result = select_frames(frames, SelectionConfig(budget=4, min_gap=1.5, duplicate_threshold=0))
    assert result.selected_indices == (0, 3)
    assert result.decision_for(1).reason_code == "min_gap"


def test_index_is_used_when_timestamps_are_absent() -> None:
    frames = [_frame(index) for index in range(4)]
    result = select_frames(frames, SelectionConfig(budget=4, min_gap=1.5, duplicate_threshold=0))
    assert result.selected_indices == (0, 3)


def test_no_endpoints_selects_highest_utility() -> None:
    frames = [_frame(0, quality=0.0), _frame(1, quality=1.0), _frame(2, quality=0.0)]
    result = select_frames(
        frames,
        SelectionConfig(
            budget=1,
            keep_endpoints=False,
            quality_weight=10,
            change_weight=0,
            coverage_weight=0,
        ),
    )
    assert result.selected_indices == (1,)


def test_selection_is_deterministic() -> None:
    frames = [_frame(index) for index in range(12)]
    config = SelectionConfig(budget=5)
    assert select_frames(frames, config) == select_frames(frames, config)


def test_sequence_span_is_computed_once(monkeypatch: pytest.MonkeyPatch) -> None:
    frames = [_frame(index, timestamp=float(index)) for index in range(12)]
    original = selector_module._coordinate_span
    calls = 0

    def counted_span(ordered: tuple[Frame, ...]) -> float:
        nonlocal calls
        calls += 1
        return original(ordered)

    monkeypatch.setattr(selector_module, "_coordinate_span", counted_span)
    select_frames(frames, SelectionConfig(budget=8, duplicate_threshold=0))
    assert calls == 1


def test_every_frame_receives_reason_and_scores() -> None:
    result = select_frames([_frame(index) for index in range(7)], SelectionConfig(budget=3))
    assert len(result.decisions) == 7
    assert all(decision.reason for decision in result.decisions)
    assert all(0 <= decision.scores.utility <= 1 for decision in result.decisions)


def test_selected_ranks_are_unique_and_contiguous() -> None:
    result = select_frames([_frame(index) for index in range(7)], SelectionConfig(budget=4))
    ranks = sorted(decision.rank for decision in result.decisions if decision.rank is not None)
    assert ranks == [1, 2, 3, 4]


def test_input_order_does_not_change_result() -> None:
    frames = [_frame(index) for index in range(7)]
    assert select_frames(frames).selected_indices == select_frames(list(reversed(frames))).selected_indices


def test_duplicate_indices_are_rejected() -> None:
    with pytest.raises(ConfigurationError, match="unique"):
        select_frames([_frame(0), _frame(0)])


def test_non_frame_input_is_rejected() -> None:
    with pytest.raises(ConfigurationError, match="every input"):
        select_frames([object()])  # type: ignore[list-item]


def test_boolean_frame_index_is_rejected() -> None:
    with pytest.raises(ConfigurationError, match="indices"):
        select_frames([replace(_frame(0), index=True)])


def test_partial_timestamps_are_rejected() -> None:
    with pytest.raises(ConfigurationError, match="every frame"):
        select_frames([_frame(0, timestamp=0.0), _frame(1)])


def test_decreasing_timestamps_are_rejected() -> None:
    with pytest.raises(ConfigurationError, match="non-decreasing"):
        select_frames([_frame(0, timestamp=2.0), _frame(1, timestamp=1.0)])


@pytest.mark.parametrize(
    "config",
    [
        SelectionConfig(budget=0),
        SelectionConfig(min_gap=-1),
        SelectionConfig(duplicate_threshold=2),
        SelectionConfig(quality_weight=-1),
        SelectionConfig(min_gap=float("nan")),
        SelectionConfig(duplicate_threshold=float("inf")),
        SelectionConfig(quality_weight=float("nan")),
        SelectionConfig(budget=True),
        SelectionConfig(keep_endpoints=1),
        SelectionConfig(quality_weight=0, change_weight=0, coverage_weight=0),
        SelectionConfig(quality_weight=1e308, change_weight=1e308, coverage_weight=1e308),
        SelectionConfig(min_gap=10**400),
        SelectionConfig(quality_weight=10**400),
    ],
)
def test_invalid_selection_config(config: SelectionConfig) -> None:
    with pytest.raises(ConfigurationError):
        select_frames([_frame(0)], config)


def test_nonfinite_frame_timestamp_is_rejected() -> None:
    with pytest.raises(ConfigurationError, match="finite"):
        select_frames([_frame(0, timestamp=float("nan"))])


def test_unbounded_frame_index_is_rejected_as_time_coordinate() -> None:
    with pytest.raises(ConfigurationError, match="coordinates"):
        select_frames([_frame(10**400)])


def test_finite_timestamps_with_overflowing_span_are_rejected() -> None:
    with pytest.raises(ConfigurationError, match="span"):
        select_frames([_frame(0, timestamp=-1e308), _frame(1, timestamp=1e308)])


def test_nonfinite_frame_metric_is_rejected() -> None:
    frame = _frame(0)
    broken = replace(
        frame,
        metrics=replace(frame.metrics, sharpness=float("nan")),
    )
    with pytest.raises(ConfigurationError, match="metrics"):
        select_frames([broken])


def test_unbounded_frame_metric_is_rejected() -> None:
    frame = _frame(0)
    broken = replace(frame, metrics=replace(frame.metrics, sharpness=10**400))
    with pytest.raises(ConfigurationError, match="metrics"):
        select_frames([broken])


@pytest.mark.parametrize("path", ["bad\ud800.png", "bad\x00.png"])
def test_unsafe_relative_path_is_rejected(path: str) -> None:
    with pytest.raises(ConfigurationError):
        select_frames([replace(_frame(0), relative_path=path)])
