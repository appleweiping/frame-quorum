"""Multiple deterministic representation-level transition detectors.

The production selector works on frame representations rather than decoded
video semantics.  This module makes that boundary explicit while exposing a
small detector family comparable to the content/adaptive/threshold choices in
larger scene-analysis tools.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from .errors import ConfigurationError
from .metrics import content_distance
from .models import Frame

DetectorName = Literal["content", "luminance", "color"]


@dataclass(frozen=True, slots=True)
class Transition:
    """A boundary between ``before_index`` and ``after_index``."""

    before_index: int
    after_index: int
    score: float
    detector: DetectorName

    def serializable(self) -> dict[str, object]:
        return {
            "before_index": self.before_index,
            "after_index": self.after_index,
            "score": round(self.score, 8),
            "detector": self.detector,
        }


def detect_transitions(
    frames: Sequence[Frame],
    *,
    detector: DetectorName = "content",
    threshold: float = 0.30,
    min_frames: int = 1,
) -> tuple[Transition, ...]:
    """Return boundaries whose selected representation distance crosses a threshold.

    ``content`` combines hash, RGB and luminance distance; ``luminance`` is
    useful for fades; ``color`` isolates palette changes in otherwise stable
    structure.  Inputs are validated and sorted by their supplied order only:
    callers must provide the same ordered sequence used by selection.
    """

    if not isinstance(frames, Sequence) or not frames:
        raise ConfigurationError("frames must be a non-empty sequence")
    if detector not in {"content", "luminance", "color"}:
        raise ConfigurationError("detector must be content, luminance, or color")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise ConfigurationError("threshold must be a finite number between zero and one")
    threshold_value = float(threshold)
    if not math.isfinite(threshold_value) or not 0 <= threshold_value <= 1:
        raise ConfigurationError("threshold must be a finite number between zero and one")
    if type(min_frames) is not int or min_frames < 1:
        raise ConfigurationError("min_frames must be a positive integer")
    for frame in frames:
        if not isinstance(frame, Frame):
            raise ConfigurationError("every item must be a Frame")
        frame.validate()
    transitions: list[Transition] = []
    last_boundary = -min_frames
    for position in range(1, len(frames)):
        score = _distance(frames[position - 1], frames[position], detector)
        if score >= threshold_value and position - last_boundary >= min_frames:
            transitions.append(
                Transition(
                    before_index=frames[position - 1].index,
                    after_index=frames[position].index,
                    score=score,
                    detector=detector,
                )
            )
            last_boundary = position
    return tuple(transitions)


def _distance(left: Frame, right: Frame, detector: DetectorName) -> float:
    if detector == "content":
        return content_distance(left.metrics, right.metrics)
    if detector == "luminance":
        return min(1.0, abs(left.metrics.luminance - right.metrics.luminance))
    return min(
        1.0,
        math.sqrt(
            (left.metrics.mean_red - right.metrics.mean_red) ** 2
            + (left.metrics.mean_green - right.metrics.mean_green) ** 2
            + (left.metrics.mean_blue - right.metrics.mean_blue) ** 2
        )
        / math.sqrt(3.0),
    )


__all__ = ["DetectorName", "Transition", "detect_transitions"]
