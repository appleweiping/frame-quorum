from __future__ import annotations

from pathlib import Path

import pytest

from frame_quorum import (
    Frame,
    FrameMetrics,
    SelectionConfig,
    detect_transitions,
    render_decision_csv,
    select_frames,
    write_decision_csv,
)
from frame_quorum.errors import ConfigurationError


def _frame(index: int, *, luminance: float, red: float = 0.2) -> Frame:
    return Frame(
        index=index,
        path=Path(f"frame-{index}.png"),
        relative_path=f"frame-{index}.png",
        timestamp=float(index),
        width=16,
        height=16,
        byte_size=10,
        metrics=FrameMetrics(
            perceptual_hash=index * 17,
            luminance=luminance,
            entropy=0.4,
            sharpness=0.5,
            colorfulness=0.3,
            mean_red=red,
            mean_green=0.2,
            mean_blue=0.2,
        ),
    )


def test_detector_family_is_deterministic() -> None:
    frames = (_frame(0, luminance=0.1), _frame(1, luminance=0.9), _frame(2, luminance=0.9))
    transitions = detect_transitions(frames, detector="luminance", threshold=0.5)
    assert [item.after_index for item in transitions] == [1]
    assert transitions[0].serializable()["detector"] == "luminance"
    assert detect_transitions(frames, detector="luminance", threshold=0.5) == transitions


def test_detector_rejects_invalid_configuration() -> None:
    frames = (_frame(0, luminance=0.1), _frame(1, luminance=0.9))
    with pytest.raises(ConfigurationError):
        detect_transitions([], detector="content")
    with pytest.raises(ConfigurationError):
        detect_transitions(frames, detector="semantic")  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError):
        detect_transitions(frames, threshold=2)
    with pytest.raises(ConfigurationError):
        detect_transitions(frames, threshold=True)  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError):
        detect_transitions(frames, min_frames=0)
    with pytest.raises(ConfigurationError):
        detect_transitions([object()], detector="content")  # type: ignore[list-item]


def test_detector_content_and_color_paths() -> None:
    frames = (_frame(0, luminance=0.1, red=0.0), _frame(1, luminance=0.9, red=1.0))
    assert detect_transitions(frames, detector="content", threshold=0)  # content path
    assert detect_transitions(frames, detector="color", threshold=0)  # color path


def test_decision_csv_round_trip_and_atomic_write(tmp_path: Path) -> None:
    frames = (_frame(0, luminance=0.1), _frame(1, luminance=0.9))
    result = select_frames(frames, SelectionConfig(budget=1, keep_endpoints=False))
    csv_text = render_decision_csv(result)
    assert csv_text.splitlines()[0].startswith("index,path,selected")
    output = write_decision_csv(result, tmp_path / "decisions.csv")
    assert output.read_text(encoding="utf-8") == csv_text
    with pytest.raises(ConfigurationError):
        render_decision_csv(object())  # type: ignore[arg-type]
