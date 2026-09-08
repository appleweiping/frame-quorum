"""Exact-PTS native-video scene analysis with bounded, offline measurements.

Decoding is incremental; adaptive/fade decisions and the complete diagnostic
result retain O(samples * detectors) data. This is not O(1) streaming detection.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Sequence
from contextlib import suppress
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal, TypeVar, cast

from .errors import ConfigurationError
from .metrics import content_distance
from .models import FrameMetrics
from .native_video import (
    NativeVideoConfig,
    NativeVideoDiagnostics,
    NativeVideoFrame,
    NativeVideoMetadata,
    NativeVideoStatus,
    NativeVideoStream,
    _fraction,
    _integer,
    _pair,
)
from .scene_detection import DetectionConfig, SceneDetectorName, _candidates, _decisions, _number

_WORK_LIMIT = 1_000_000
_INT64 = (1 << 63) - 1
_DETECTORS = {"content", "color", "luminance", "adaptive", "threshold"}
_REASONS = {
    "sequence_start",
    "below_threshold",
    "incomplete_window",
    "no_completed_fade",
    "short_previous_scene",
    "short_final_scene",
    "adaptive_peak",
    "distance_threshold",
    "completed_fade",
    "final_fade",
}
_ACCEPTED_REASONS = {"adaptive_peak", "distance_threshold", "completed_fade", "final_fade"}
_Record = TypeVar("_Record")


def _time(value: Fraction, name: str) -> None:
    # A native int64 PTS multiplied by a bounded rational can exceed int64.
    if type(value) is not Fraction or abs(value.numerator) > (1 << 127) - 1 or value.denominator > _INT64:
        raise ConfigurationError(f"{name} must be an exact supported presentation time")


@dataclass(frozen=True, slots=True)
class NativeSceneConfig:
    video: NativeVideoConfig = field(default_factory=NativeVideoConfig)
    detectors: tuple[DetectionConfig, ...] = (DetectionConfig(),)
    minimum_votes: int = 1
    min_scene_samples: int = 1

    def __post_init__(self) -> None:
        if type(self.video) is not NativeVideoConfig:
            raise ConfigurationError("video must be NativeVideoConfig")
        if type(self.detectors) is not tuple or not 1 <= len(self.detectors) <= 5:
            raise ConfigurationError("detectors must be an immutable tuple of one to five configurations")
        names = []
        for detector in self.detectors:
            if type(detector) is not DetectionConfig:
                raise ConfigurationError("every detector must be DetectionConfig")
            detector.validate()
            if self.video.max_frames > detector.max_frames:
                raise ConfigurationError("video.max_frames must not exceed a detector's max_frames")
            names.append(detector.detector)
        if len(set(names)) != len(names):
            raise ConfigurationError("each detector type may be configured only once")
        if self.video.max_frames * len(self.detectors) > _WORK_LIMIT:
            raise ConfigurationError("sample times detector work exceeds the 1000000 ceiling")
        _integer(self.minimum_votes, "minimum_votes", 1, len(self.detectors))
        _integer(self.min_scene_samples, "min_scene_samples", 1, _WORK_LIMIT)

    def to_dict(self) -> dict[str, Any]:
        video = asdict(self.video)
        video["start"], video["end"] = _pair(self.video.start), _pair(self.video.end)
        return {
            "video": video,
            "detectors": [asdict(item) for item in self.detectors],
            "minimum_votes": self.minimum_votes,
            "min_scene_samples": self.min_scene_samples,
        }


@dataclass(frozen=True, slots=True)
class NativeSceneSample:
    """One selected sample, without an image path, synthetic frame number or RGB."""

    pts: int
    time_base: Fraction
    decode_index: int
    sample_index: int
    generation: int
    metrics: FrameMetrics

    def __post_init__(self) -> None:
        _integer(self.pts, "pts", -_INT64 - 1, _INT64)
        object.__setattr__(self, "time_base", _fraction(self.time_base, "time_base", positive=True))
        for name in ("decode_index", "sample_index", "generation"):
            _integer(getattr(self, name), name, 0, _INT64)
        if type(self.metrics) is not FrameMetrics:
            raise ConfigurationError("metrics must be FrameMetrics")
        self.metrics.validate()

    @property
    def presentation_time(self) -> Fraction:
        return self.pts * self.time_base

    def to_dict(self) -> dict[str, Any]:
        return {
            "pts": self.pts,
            "time_base": _pair(self.time_base),
            "presentation_time": _pair(self.presentation_time),
            "decode_index": self.decode_index,
            "sample_index": self.sample_index,
            "generation": self.generation,
            "metrics": self.metrics.serializable(),
        }


@dataclass(frozen=True, slots=True)
class NativeDetectorStatistic:
    detector: SceneDetectorName
    score: float | None
    candidate: bool
    qualified: bool
    reason: str

    def __post_init__(self) -> None:
        if type(self.detector) is not str or self.detector not in _DETECTORS:
            raise ConfigurationError("unknown detector statistic")
        if self.score is not None:
            _number(self.score, "score", minimum=0, maximum=1_000_000 if self.detector == "adaptive" else 1)
        elif self.detector != "adaptive":
            raise ConfigurationError("only an adaptive score may be absent")
        if type(self.candidate) is not bool or type(self.qualified) is not bool:
            raise ConfigurationError("candidate and qualified must be booleans")
        if type(self.reason) is not str or self.reason not in _REASONS:
            raise ConfigurationError("unknown detector reason")
        if self.qualified and (not self.candidate or self.score is None):
            raise ConfigurationError("qualified evidence requires a scored candidate")
        if self.candidate and self.score is None:
            raise ConfigurationError("candidate evidence requires a score")
        if self.qualified != (self.reason in _ACCEPTED_REASONS):
            raise ConfigurationError("qualified flag contradicts the detector reason")
        if self.candidate != (self.qualified or self.reason in {"short_previous_scene", "short_final_scene"}):
            raise ConfigurationError("candidate flag contradicts the detector reason")


@dataclass(frozen=True, slots=True)
class NativeSceneStatistic:
    sample: NativeSceneSample
    content_score: float
    detectors: tuple[NativeDetectorStatistic, ...]
    accepted: bool
    reason: Literal[
        "sequence_start", "insufficient_votes", "short_previous_scene", "short_final_scene", "quorum"
    ]

    def __post_init__(self) -> None:
        if type(self.sample) is not NativeSceneSample:
            raise ConfigurationError("sample must be NativeSceneSample")
        _number(self.content_score, "content_score", minimum=0, maximum=1)
        if (
            type(self.detectors) is not tuple
            or not 1 <= len(self.detectors) <= 5
            or any(type(item) is not NativeDetectorStatistic for item in self.detectors)
        ):
            raise ConfigurationError("statistics need an immutable detector tuple")
        if len({item.detector for item in self.detectors}) != len(self.detectors):
            raise ConfigurationError("statistic detector names must be unique")
        if (
            type(self.accepted) is not bool
            or type(self.reason) is not str
            or self.reason
            not in {
                "sequence_start",
                "insufficient_votes",
                "short_previous_scene",
                "short_final_scene",
                "quorum",
            }
        ):
            raise ConfigurationError("invalid aggregate decision")
        if self.accepted != (self.reason == "quorum") or (self.accepted and not self.votes):
            raise ConfigurationError("aggregate decision contradicts its evidence")

    @property
    def votes(self) -> int:
        return sum(item.qualified for item in self.detectors)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample": self.sample.to_dict(),
            "content_score": self.content_score,
            "detectors": [asdict(item) for item in self.detectors],
            "votes": self.votes,
            "accepted": self.accepted,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class NativeScene:
    """Half-open supplied-sample partition; an unknown final time stays None."""

    ordinal: int
    start_position: int
    end_position: int
    start_time: Fraction
    last_sample_time: Fraction
    end_time: Fraction | None
    end_reason: Literal["cut", "requested_end", "unknown"]

    def __post_init__(self) -> None:
        _integer(self.ordinal, "ordinal", 0, _WORK_LIMIT - 1)
        _integer(self.start_position, "start_position", 0, _WORK_LIMIT - 1)
        _integer(self.end_position, "end_position", self.start_position + 1, _WORK_LIMIT)
        _time(self.start_time, "start_time")
        _time(self.last_sample_time, "last_sample_time")
        if self.last_sample_time < self.start_time:
            raise ConfigurationError("scene sample times must not decrease")
        if type(self.end_reason) is not str or self.end_reason not in {"cut", "requested_end", "unknown"}:
            raise ConfigurationError("invalid scene endpoint reason")
        if (self.end_time is None) != (self.end_reason == "unknown"):
            raise ConfigurationError("unknown endpoint must be represented by None")
        if self.end_time is not None:
            _time(self.end_time, "end_time")
            if self.end_time < self.last_sample_time:
                raise ConfigurationError("scene endpoint precedes the last sample")

    @property
    def sample_count(self) -> int:
        return self.end_position - self.start_position

    def to_dict(self) -> dict[str, Any]:
        return {
            "ordinal": self.ordinal,
            "start_position": self.start_position,
            "end_position": self.end_position,
            "sample_count": self.sample_count,
            "start_time": _pair(self.start_time),
            "last_sample_time": _pair(self.last_sample_time),
            "end_time": _pair(self.end_time),
            "end_reason": self.end_reason,
        }


@dataclass(frozen=True, slots=True)
class NativeSceneResult:
    config: NativeSceneConfig
    metadata: NativeVideoMetadata
    diagnostics: NativeVideoDiagnostics
    statistics: tuple[NativeSceneStatistic, ...]
    scenes: tuple[NativeScene, ...]

    def __post_init__(self) -> None:
        if type(self.config) is not NativeSceneConfig or type(self.metadata) is not NativeVideoMetadata:
            raise ConfigurationError("result requires native scene configuration and metadata")
        if type(self.diagnostics) is not NativeVideoDiagnostics or not self.diagnostics.closed:
            raise ConfigurationError("result requires closed native diagnostics")
        if (
            self.diagnostics.status
            not in {
                NativeVideoStatus.EOF,
                NativeVideoStatus.RANGE_END,
                NativeVideoStatus.FRAME_LIMIT,
                NativeVideoStatus.DECODE_LIMIT,
            }
            or self.diagnostics.cleanup_errors
            or self.diagnostics.generation != 0
        ):
            raise ConfigurationError("only successfully terminated single-generation analysis has a result")
        if type(self.statistics) is not tuple or len(self.statistics) > self.config.video.max_frames:
            raise ConfigurationError("result statistics must be an immutable bounded tuple")
        if type(self.scenes) is not tuple or len(self.scenes) > len(self.statistics):
            raise ConfigurationError("result scenes must be an immutable bounded tuple")
        if self.diagnostics.returned_frames != len(self.statistics):
            raise ConfigurationError("sample count disagrees with decode diagnostics")
        if self.diagnostics.decoded_frames > self.config.video.max_decoded_frames:
            raise ConfigurationError("decode count exceeds the configured budget")
        if self.diagnostics.status is NativeVideoStatus.RANGE_END and self.config.video.end is None:
            raise ConfigurationError("range termination requires a configured end")
        if (
            self.diagnostics.status is NativeVideoStatus.FRAME_LIMIT
            and len(self.statistics) != self.config.video.max_frames
        ) or (
            self.diagnostics.status is NativeVideoStatus.DECODE_LIMIT
            and self.diagnostics.decoded_frames != self.config.video.max_decoded_frames
        ):
            raise ConfigurationError("limit termination disagrees with its configured count")
        previous: NativeSceneSample | None = None
        last_cut = 0
        names = tuple(item.detector for item in self.config.detectors)
        for position, row in enumerate(self.statistics):
            if type(row) is not NativeSceneStatistic:
                raise ConfigurationError("every statistic must be NativeSceneStatistic")
            sample = row.sample
            if sample.sample_index != position or sample.generation != 0:
                raise ConfigurationError("samples require contiguous zero-based positions in generation zero")
            if sample.decode_index >= self.diagnostics.decoded_frames:
                raise ConfigurationError("sample decode index exceeds observed work")
            if previous is not None and (
                sample.decode_index - previous.decode_index != self.config.video.frame_step
                or sample.presentation_time < previous.presentation_time
            ):
                raise ConfigurationError("sample stride or presentation order disagrees with configuration")
            start, end = self.config.video.start, self.config.video.end
            if (start is not None and sample.presentation_time < start) or (
                end is not None and sample.presentation_time >= end
            ):
                raise ConfigurationError("sample is outside the configured presentation interval")
            if tuple(item.detector for item in row.detectors) != names:
                raise ConfigurationError("statistic detector order disagrees with configuration")
            accepted, reason = _aggregate(position, len(self.statistics), last_cut, row.votes, self.config)
            if row.accepted != accepted or row.reason != reason:
                raise ConfigurationError("aggregate decision disagrees with quorum or scene minimum")
            if accepted:
                last_cut = position
            if position == 0 and row.content_score != 0:
                raise ConfigurationError("first sample has no prior content distance")
            previous = sample
        expected = _scenes(self.statistics, self.config, self.diagnostics)
        if any(type(item) is not NativeScene for item in self.scenes) or self.scenes != expected:
            raise ConfigurationError("scene partition or exact endpoints disagree with statistics")

    @property
    def cut_positions(self) -> tuple[int, ...]:
        return tuple(row.sample.sample_index for row in self.statistics if row.accepted)

    @property
    def cut_times(self) -> tuple[Fraction, ...]:
        return tuple(row.sample.presentation_time for row in self.statistics if row.accepted)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "frame-quorum-native-scenes",
            "schema_version": 1,
            "config": self.config.to_dict(),
            "metadata": self.metadata.to_dict(),
            "diagnostics": self.diagnostics.to_dict(),
            "sample_count": len(self.statistics),
            "cut_positions": list(self.cut_positions),
            "cut_times": [_pair(value) for value in self.cut_times],
            "statistics": [row.to_dict() for row in self.statistics],
            "scenes": [scene.to_dict() for scene in self.scenes],
        }


def _aggregate(
    position: int, count: int, previous: int, votes: int, config: NativeSceneConfig
) -> tuple[
    bool,
    Literal["sequence_start", "insufficient_votes", "short_previous_scene", "short_final_scene", "quorum"],
]:
    if position == 0:
        return False, "sequence_start"
    if votes < config.minimum_votes:
        return False, "insufficient_votes"
    if position - previous < config.min_scene_samples:
        return False, "short_previous_scene"
    if count - position < config.min_scene_samples:
        return False, "short_final_scene"
    return True, "quorum"


def _scenes(
    statistics: tuple[NativeSceneStatistic, ...],
    config: NativeSceneConfig,
    diagnostics: NativeVideoDiagnostics,
) -> tuple[NativeScene, ...]:
    return _partition_native_samples(
        tuple(row.sample for row in statistics),
        [row.sample.sample_index for row in statistics if row.accepted],
        config.video,
        diagnostics,
    )


def _partition_native_samples(
    samples: Sequence[NativeSceneSample],
    cuts: list[int],
    video: NativeVideoConfig,
    diagnostics: NativeVideoDiagnostics,
) -> tuple[NativeScene, ...]:
    """Shared exact-PTS partition, independent of the detector's measurement type."""
    if not samples:
        return ()
    scenes = []
    for ordinal, (start, end) in enumerate(zip([0, *cuts], [*cuts, len(samples)], strict=True)):
        endpoint = None
        reason: Literal["cut", "requested_end", "unknown"] = "unknown"
        if end < len(samples):
            endpoint, reason = samples[end].presentation_time, "cut"
        elif diagnostics.status is NativeVideoStatus.RANGE_END and video.end is not None:
            endpoint, reason = Fraction(video.end), "requested_end"
        scenes.append(
            NativeScene(
                ordinal,
                start,
                end,
                samples[start].presentation_time,
                samples[end - 1].presentation_time,
                endpoint,
                reason,
            )
        )
    return tuple(scenes)


def detect_native_scenes(
    path: str | Path,
    config: NativeSceneConfig | None = None,
    *,
    on_sample: Callable[[NativeSceneSample], None] | None = None,
) -> NativeSceneResult:
    """Decode under owned context, then analyze immutable measurements offline.

    The optional synchronous callback observes one measured sample at a time,
    not a final scene decision. It must return None. Exceptions and interrupts
    propagate after context cleanup; there is no successful partial result.
    """
    options = NativeSceneConfig() if config is None else config
    if type(options) is not NativeSceneConfig:
        raise ConfigurationError("config must be NativeSceneConfig")
    metadata, diagnostics, samples = _collect_native_samples(path, options, on_sample)
    return _analyze_native_samples(options, metadata, diagnostics, samples)


def _collect_native_samples(
    path: str | Path,
    options: NativeSceneConfig,
    on_sample: Callable[[NativeSceneSample], None] | None,
) -> tuple[NativeVideoMetadata, NativeVideoDiagnostics, tuple[NativeSceneSample, ...]]:
    return _collect_native_records(path, options.video, _measure_native_sample, on_sample)


def _measure_native_sample(frame: NativeVideoFrame) -> NativeSceneSample:
    return NativeSceneSample(
        frame.pts, frame.time_base, frame.decode_index, frame.sample_index, frame.generation, frame.measure()
    )


def _collect_native_records(
    path: str | Path,
    video: NativeVideoConfig,
    measure: Callable[[NativeVideoFrame], _Record],
    on_sample: Callable[[_Record], None] | None,
) -> tuple[NativeVideoMetadata, NativeVideoDiagnostics, tuple[_Record, ...]]:
    """Trusted internal measurement hook; not a public plugin/import mechanism."""
    if on_sample is not None and (
        not callable(on_sample)
        or inspect.iscoroutinefunction(on_sample)
        # Callable-instance async detection, not a substitute for callable().
        or inspect.iscoroutinefunction(getattr(on_sample, "__call__", None))  # noqa: B004
    ):
        raise ConfigurationError("on_sample must be a synchronous callable")
    samples: list[_Record] = []
    with NativeVideoStream(path, video) as stream:
        metadata = stream.metadata
        for frame in stream:
            sample = measure(frame)
            del frame  # The collection/callback retains measurements, not earlier RGB snapshots.
            samples.append(sample)
            if on_sample is not None:
                returned = cast(Callable[[_Record], object], on_sample)(sample)
                if returned is not None:
                    if inspect.iscoroutine(returned):
                        # Ordinary invalid-coroutine cleanup cannot hide the
                        # callback contract failure; genuine interrupts propagate.
                        with suppress(Exception):
                            returned.close()
                    raise ConfigurationError("on_sample must return None, not a value or awaitable")
    return metadata, stream.diagnostics, tuple(samples)


def _analyze_native_samples(
    options: NativeSceneConfig,
    metadata: NativeVideoMetadata,
    diagnostics: NativeVideoDiagnostics,
    samples: tuple[NativeSceneSample, ...],
) -> NativeSceneResult:
    """Single shared measurement-only kernel for fresh decoding and explicit replay."""
    metrics = [sample.metrics for sample in samples]
    content = [0.0] + [content_distance(left, right) for left, right in pairwise(metrics)]
    per_detector: list[list[NativeDetectorStatistic]] = []
    for detector in options.detectors:
        scores, candidates = _candidates(metrics, content if samples else [], detector)
        decisions = _decisions(scores, candidates, detector)
        per_detector.append(
            [
                NativeDetectorStatistic(detector.detector, scores[i], i in candidates, kept, reason)
                for i, (kept, reason) in enumerate(decisions)
            ]
        )
    statistics = []
    previous = 0
    for position, sample in enumerate(samples):
        evidence = tuple(rows[position] for rows in per_detector)
        kept, reason = _aggregate(
            position, len(samples), previous, sum(row.qualified for row in evidence), options
        )
        statistics.append(NativeSceneStatistic(sample, content[position], evidence, kept, reason))
        if kept:
            previous = position
    rows = tuple(statistics)
    return NativeSceneResult(options, metadata, diagnostics, rows, _scenes(rows, options, diagnostics))


__all__ = [
    "NativeDetectorStatistic",
    "NativeScene",
    "NativeSceneConfig",
    "NativeSceneResult",
    "NativeSceneSample",
    "NativeSceneStatistic",
    "detect_native_scenes",
]
