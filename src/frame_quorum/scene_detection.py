"""Auditable scene detection with adaptive contrast and fade-through-dark modes."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from itertools import pairwise
from typing import Literal

from .errors import ConfigurationError
from .metrics import content_distance
from .models import Frame, FrameMetrics
from .scenes import Shot

SceneDetectorName = Literal["content", "luminance", "color", "adaptive", "threshold"]
_DETECTORS = {"content", "luminance", "color", "adaptive", "threshold"}
_MAX_FRAMES = 1_000_000
_MAX_WINDOW_RADIUS = 10_000
_RATIO_FLOOR = 1e-12
_RATIO_CAP = 1_000_000.0


@dataclass(frozen=True, slots=True)
class DetectionConfig:
    """Detector settings; all lengths count supplied frames, including sampled video.

    ``threshold`` applies to the three adjacent-distance modes. Adaptive mode
    uses ``adaptive_ratio`` and ``min_content``. Threshold mode uses the dark
    threshold, hysteresis, minimum dark samples, bias and final-fade policy.
    ``min_scene_frames`` applies to every emitted scene, including the tail;
    an input shorter than this limit remains one unsplit scene.
    """

    detector: SceneDetectorName = "adaptive"
    threshold: float = 0.30
    min_scene_frames: int = 1
    window_radius: int = 2
    adaptive_ratio: float = 3.0
    min_content: float = 0.15
    dark_threshold: float = 0.05
    hysteresis: float = 0.02
    min_dark_frames: int = 2
    fade_bias: float = 0.0
    include_final_fade: bool = False
    max_frames: int = _MAX_FRAMES

    def validate(self) -> None:
        if not isinstance(self.detector, str) or self.detector not in _DETECTORS:
            raise ConfigurationError("unknown scene detector")
        for name in ("threshold", "min_content", "dark_threshold", "hysteresis"):
            _number(getattr(self, name), name, minimum=0, maximum=1)
        _number(self.adaptive_ratio, "adaptive_ratio", minimum=1, maximum=_RATIO_CAP)
        _number(self.fade_bias, "fade_bias", minimum=-1, maximum=1)
        if self.dark_threshold + self.hysteresis >= 1:
            raise ConfigurationError("dark_threshold + hysteresis must be below one")
        for name, maximum in (
            ("min_scene_frames", _MAX_FRAMES),
            ("window_radius", _MAX_WINDOW_RADIUS),
            ("min_dark_frames", _MAX_FRAMES),
            ("max_frames", _MAX_FRAMES),
        ):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= maximum:
                raise ConfigurationError(f"{name} must be an integer between 1 and {maximum}")
        if type(self.include_final_fade) is not bool:
            raise ConfigurationError("include_final_fade must be a boolean")


def _number(value: object, name: str, *, minimum: float, maximum: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not minimum <= value <= maximum:
        raise ConfigurationError(f"{name} must be a finite number between {minimum} and {maximum}")


@dataclass(frozen=True, slots=True)
class FrameStatistic:
    """One diagnostic per frame; candidate/accepted refers to a cut before it.

    Adaptive scores are absent where a complete centered window is unavailable.
    For threshold mode the score is absolute luminance, and the candidate can
    be placed earlier than the frame that confirmed the completed fade.
    """

    position: int
    frame_index: int
    timestamp: float | None
    content_score: float
    luminance: float
    detector_score: float | None
    candidate: bool
    accepted: bool
    reason: str

    def serializable(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DetectionResult:
    """An ordered half-open scene partition and the evidence for every boundary."""

    config: DetectionConfig
    scenes: tuple[Shot, ...]
    statistics: tuple[FrameStatistic, ...]

    @property
    def cut_indices(self) -> tuple[int, ...]:
        return tuple(item.frame_index for item in self.statistics if item.accepted)

    def serializable(self) -> dict[str, object]:
        return {
            "kind": "frame-quorum-scenes",
            "schema_version": 1,
            "config": asdict(self.config),
            "cut_indices": list(self.cut_indices),
            "scenes": [scene.serializable() for scene in self.scenes],
            "statistics": [item.serializable() for item in self.statistics],
        }


def detect_scenes(frames: Sequence[Frame], config: DetectionConfig | None = None) -> DetectionResult:
    """Detect scenes from immutable measurements in O(n) time and O(n) memory.

    Adaptive contrast compares each content distance with the mean of the
    ``window_radius`` distances on each side, excluding itself. A complete
    window is required; the first/last radius distances have no adaptive score.
    A zero baseline uses a 1e-12 denominator, with ratios capped at 1e6.

    Threshold detection requires an established bright scene, enters a fade
    at luminance <= dark_threshold, and confirms it only at luminance strictly
    greater than dark_threshold + hysteresis. Enough truly dark samples must
    occur within that interval. The boundary lies within the dark interval
    according to fade_bias (-1=start, 0=midpoint, +1=bright return).
    """

    options = DetectionConfig() if config is None else config
    if not isinstance(options, DetectionConfig):
        raise ConfigurationError("config must be DetectionConfig")
    options.validate()
    _validate_frames(frames, options.max_frames)
    content = [0.0] + [
        content_distance(frames[index - 1].metrics, frames[index].metrics) for index in range(1, len(frames))
    ]
    scores, candidates = _candidates([frame.metrics for frame in frames], content, options)
    accepted: list[int] = []
    statistics: list[FrameStatistic] = []
    decisions = _decisions(scores, candidates, options)
    for position, frame in enumerate(frames):
        candidate = position in candidates
        kept, reason = decisions[position]
        if kept:
            accepted.append(position)
        statistics.append(
            FrameStatistic(
                position,
                frame.index,
                frame.timestamp,
                content[position],
                frame.metrics.luminance,
                scores[position],
                candidate,
                kept,
                reason,
            )
        )
    starts = [0, *accepted]
    ends = [*accepted, len(frames)]
    scenes = tuple(
        Shot(ordinal, frames[start].index, frames[end - 1].index + 1, None if start == 0 else content[start])
        for ordinal, (start, end) in enumerate(zip(starts, ends, strict=True))
    )
    return DetectionResult(options, scenes, tuple(statistics))


def _validate_frames(frames: Sequence[Frame], maximum: int) -> None:
    if not isinstance(frames, Sequence) or not frames:
        raise ConfigurationError("frames must be a non-empty sequence")
    if len(frames) > maximum:
        raise ConfigurationError(f"frame count exceeds max_frames={maximum}")
    for position, frame in enumerate(frames):
        if not isinstance(frame, Frame):
            raise ConfigurationError("every item must be a Frame")
        frame.validate()
        if position:
            previous = frames[position - 1]
            if frame.index != previous.index + 1:
                raise ConfigurationError("frame indices must be contiguous and increasing")
            if (frame.timestamp is None) != (previous.timestamp is None):
                raise ConfigurationError("timestamps must be present for every frame or none")
            if (
                frame.timestamp is not None
                and previous.timestamp is not None
                and frame.timestamp < previous.timestamp
            ):
                raise ConfigurationError("frame timestamps must not decrease")
    if frames[-1].index == (1 << 63) - 1:
        raise ConfigurationError("last frame index leaves no signed-64 end index")


def _decisions(
    scores: Sequence[float | None], candidates: dict[int, str], config: DetectionConfig
) -> list[tuple[bool, str]]:
    """Shared sample-count policy, independent of image paths and time coordinates."""
    decisions: list[tuple[bool, str]] = []
    previous = 0
    for position, score in enumerate(scores):
        reason = "below_threshold" if score is not None else "incomplete_window"
        if config.detector == "threshold":
            reason = "no_completed_fade"
        if position == 0:
            reason = "sequence_start"
        kept = False
        if position in candidates:
            if position - previous < config.min_scene_frames:
                reason = "short_previous_scene"
            elif len(scores) - position < config.min_scene_frames:
                reason = "short_final_scene"
            else:
                kept, reason, previous = True, candidates[position], position
        decisions.append((kept, reason))
    return decisions


def _candidates(
    metrics: Sequence[FrameMetrics],
    content: list[float],
    config: DetectionConfig,
) -> tuple[list[float | None], dict[int, str]]:
    if not metrics:
        return [], {}
    if config.detector == "adaptive":
        scores = _adaptive_scores(content, config.window_radius)
        return scores, {
            position: "adaptive_peak"
            for position, score in enumerate(scores)
            if score is not None
            and score >= config.adaptive_ratio
            and content[position] >= config.min_content
        }
    if config.detector == "threshold":
        return [item.luminance for item in metrics], _fade_candidates(metrics, config)
    distances = content
    if config.detector == "luminance":
        distances = [0.0] + [
            abs(metrics[position].luminance - metrics[position - 1].luminance)
            for position in range(1, len(metrics))
        ]
    elif config.detector == "color":
        distances = [0.0]
        for left, right in pairwise(metrics):
            distances.append(
                math.sqrt(
                    (left.mean_red - right.mean_red) ** 2
                    + (left.mean_green - right.mean_green) ** 2
                    + (left.mean_blue - right.mean_blue) ** 2
                )
                / math.sqrt(3.0)
            )
    return list(distances), {
        position: "distance_threshold"
        for position in range(1, len(metrics))
        if distances[position] >= config.threshold
    }


def _adaptive_scores(content: list[float], radius: int) -> list[float | None]:
    scores: list[float | None] = [None] * len(content)
    first = radius + 1
    stop = len(content) - radius
    if first >= stop:
        return scores
    # Distance zero belongs to no image pair, so it is never baseline evidence.
    window_sum = math.fsum(content[1 : 2 * radius + 2])
    for position in range(first, stop):
        baseline = max(0.0, (window_sum - content[position]) / (2 * radius))
        scores[position] = min(_RATIO_CAP, content[position] / max(baseline, _RATIO_FLOOR))
        if position + 1 < stop:
            window_sum = math.fsum((window_sum, -content[position - radius], content[position + radius + 1]))
    return scores


def _fade_candidates(metrics: Sequence[FrameMetrics], config: DetectionConfig) -> dict[int, str]:
    candidates: dict[int, str] = {}
    armed = False
    start: int | None = None
    dark_samples = 0
    release = config.dark_threshold + config.hysteresis
    for position, item in enumerate(metrics):
        luminance = item.luminance
        if luminance > release:
            if start is not None and dark_samples >= config.min_dark_frames:
                boundary = start + math.floor((position - start) * (1 + config.fade_bias) / 2)
                candidates[boundary] = "completed_fade"
            armed, start, dark_samples = True, None, 0
        elif armed:
            if luminance <= config.dark_threshold:
                if start is None:
                    start = position
                dark_samples += 1
    if config.include_final_fade and start is not None and dark_samples >= config.min_dark_frames:
        candidates[start] = "final_fade"
    return candidates


__all__ = ["DetectionConfig", "DetectionResult", "FrameStatistic", "SceneDetectorName", "detect_scenes"]
