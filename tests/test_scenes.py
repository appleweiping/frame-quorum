from __future__ import annotations

from pathlib import Path

from frame_quorum import Frame, FrameMetrics, allocate_budget, analyze_scenes, detect_shots


def _frame(index: int, value: int) -> Frame:
    metrics = FrameMetrics(
        perceptual_hash=value,
        luminance=value / 255,
        entropy=0.5,
        sharpness=0.5,
        colorfulness=0.5,
        mean_red=value / 255,
        mean_green=0.2,
        mean_blue=0.2,
    )
    return Frame(index, Path(f"{index}.png"), f"{index}.png", float(index), 2, 2, 10, metrics)


def test_detect_shots_and_allocate_budget_are_deterministic() -> None:
    frames = tuple(_frame(i, 0 if i < 3 else 255) for i in range(6))
    shots = detect_shots(frames, threshold=0.1)
    assert len(shots) == 2
    assert [shot.frame_count for shot in shots] == [3, 3]
    assert allocate_budget(shots, 4) == (2, 2)
    assert allocate_budget(shots, 1) in {(1, 0), (0, 1)}


def test_scene_analysis_serializes_contract() -> None:
    analysis = analyze_scenes(tuple(_frame(i, i * 10) for i in range(4)), budget=3)
    payload = analysis.serializable()
    assert payload["threshold"] == 0.3
    assert sum(payload["allocated_budget"]) == 3
