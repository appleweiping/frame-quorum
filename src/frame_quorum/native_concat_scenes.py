"""Complete offline detector analysis on an exact caller-declared concat timeline.

Only scalar measurements are retained. Declared endpoints are not physical media
coverage, and internally consistent evidence is not source-content verification.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from contextlib import suppress
from dataclasses import asdict, dataclass, field, fields, replace
from fractions import Fraction
from pathlib import Path, PosixPath, WindowsPath
from typing import Any, Literal, cast

from .errors import ConfigurationError, ScanError
from .models import FrameMetrics
from .native_concat import (
    NativeConcatClip,
    NativeConcatConfig,
    NativeConcatDiagnostics,
    NativeConcatSpan,
    NativeConcatStream,
    NativeConcatTimeline,
    _time,
)
from .native_scenes import (
    NativeDetectorStatistic,
    NativeSceneSample,
    NativeSceneStatistic,
    _aggregate_policy,
    _detector_evidence,
    _measure_native_sample,
)
from .native_video import NativeVideoMetadata, _integer, _pair
from .scene_detection import DetectionConfig

_WORK = 1_000_000
_INT64 = (1 << 63) - 1
_PATH_TYPES = (Path, PosixPath, WindowsPath)
_Reason = Literal[
    "sequence_start", "insufficient_votes", "short_previous_scene", "short_final_scene", "quorum"
]


def _exact_number(value: object, name: str) -> None:
    if type(value) not in (int, float):
        raise ConfigurationError(f"{name} must be an exact numeric scalar")


def _detector(value: DetectionConfig) -> DetectionConfig:
    if type(value) is not DetectionConfig or type(value.detector) is not str:
        raise ConfigurationError("detectors require exact DetectionConfig and detector name")
    for name in ("threshold", "adaptive_ratio", "min_content", "dark_threshold", "hysteresis", "fade_bias"):
        _exact_number(getattr(value, name), name)
    value.validate()
    return replace(value)


def _native_sample(value: NativeSceneSample) -> NativeSceneSample:
    if type(value) is not NativeSceneSample or type(value.metrics) is not FrameMetrics:
        raise ConfigurationError("sample requires exact native sample and metrics")
    for item in fields(FrameMetrics):
        scalar = getattr(value.metrics, item.name)
        if item.name == "perceptual_hash":
            _integer(scalar, item.name, 0, (1 << 64) - 1)
        else:
            _exact_number(scalar, item.name)
    metrics = replace(value.metrics)
    metrics.validate()
    return replace(value, metrics=metrics)


def _tuple(value: object, name: str, maximum: int) -> None:
    if type(value) is not tuple or len(value) > maximum:
        raise ConfigurationError(f"{name} must be an exact bounded tuple")


@dataclass(frozen=True, slots=True)
class NativeConcatSceneConfig:
    video: NativeConcatConfig = field(default_factory=NativeConcatConfig)
    detectors: tuple[DetectionConfig, ...] = (DetectionConfig(),)
    minimum_votes: int = 1
    min_scene_samples: int = 1

    def __post_init__(self) -> None:
        if type(self.video) is not NativeConcatConfig:
            raise ConfigurationError("video must be NativeConcatConfig")
        video = replace(self.video)
        _tuple(self.detectors, "detectors", 5)
        if not self.detectors:
            raise ConfigurationError("at least one detector is required")
        detectors = tuple(_detector(item) for item in self.detectors)
        if len({item.detector for item in detectors}) != len(detectors):
            raise ConfigurationError("each detector may be configured only once")
        if any(video.limits.max_frames > item.max_frames for item in detectors):
            raise ConfigurationError("video max_frames exceeds detector capacity")
        if video.limits.max_frames * len(detectors) > _WORK:
            raise ConfigurationError("sample times detector work exceeds 1000000")
        _integer(self.minimum_votes, "minimum_votes", 1, len(detectors))
        _integer(self.min_scene_samples, "min_scene_samples", 1, _WORK)
        object.__setattr__(self, "video", video)
        object.__setattr__(self, "detectors", detectors)

    def to_dict(self) -> dict[str, Any]:
        checked = replace(self)
        return {
            "video": {
                "start": _pair(Fraction(checked.video.start)),
                "end": _pair(None if checked.video.end is None else Fraction(checked.video.end)),
                "frame_step": checked.video.frame_step,
                "limits": asdict(checked.video.limits),
            },
            "detectors": [asdict(item) for item in checked.detectors],
            "minimum_votes": checked.minimum_votes,
            "min_scene_samples": checked.min_scene_samples,
        }


@dataclass(frozen=True, slots=True)
class NativeConcatSceneSample:
    timeline_digest: str
    clip_index: int
    activation: int
    generation: int
    sample_index: int
    presentation_time: Fraction
    native: NativeSceneSample

    def __post_init__(self) -> None:
        if (
            type(self.timeline_digest) is not str
            or len(self.timeline_digest) != 64
            or any(c not in "0123456789abcdef" for c in self.timeline_digest)
        ):
            raise ConfigurationError("timeline_digest must be 64 lowercase hexadecimal characters")
        _integer(self.clip_index, "clip_index", 0, 127)
        for name in ("activation", "generation", "sample_index"):
            _integer(getattr(self, name), name, 0, _INT64)
        if (
            type(self.presentation_time) is not Fraction
            or _time(self.presentation_time, "presentation_time") < 0
        ):
            raise ConfigurationError("presentation_time must be an exact nonnegative Fraction")
        object.__setattr__(self, "native", _native_sample(self.native))

    def to_dict(self) -> dict[str, Any]:
        checked = replace(self)
        return {
            "timeline_digest": checked.timeline_digest,
            "clip_index": checked.clip_index,
            "activation": checked.activation,
            "generation": checked.generation,
            "sample_index": checked.sample_index,
            "presentation_time": _pair(checked.presentation_time),
            "native": checked.native.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class NativeConcatSceneStatistic:
    sample: NativeConcatSceneSample
    content_score: float
    detectors: tuple[NativeDetectorStatistic, ...]
    accepted: bool
    reason: _Reason

    def __post_init__(self) -> None:
        if type(self.sample) is not NativeConcatSceneSample:
            raise ConfigurationError("statistic requires NativeConcatSceneSample")
        sample = replace(self.sample)
        _exact_number(self.content_score, "content_score")
        _tuple(self.detectors, "detector evidence", 5)
        evidence = []
        for item in self.detectors:
            if type(item) is not NativeDetectorStatistic:
                raise ConfigurationError("evidence requires NativeDetectorStatistic")
            if item.score is not None:
                _exact_number(item.score, "detector score")
            evidence.append(replace(item))
        # Reuse the established local evidence/reason contract, not its native identity.
        NativeSceneStatistic(sample.native, self.content_score, tuple(evidence), self.accepted, self.reason)
        object.__setattr__(self, "sample", sample)
        object.__setattr__(self, "detectors", tuple(evidence))

    @property
    def votes(self) -> int:
        return sum(item.qualified for item in self.detectors)

    def to_dict(self) -> dict[str, Any]:
        checked = replace(self)
        return {
            "sample": checked.sample.to_dict(),
            "content_score": checked.content_score,
            "detectors": [asdict(item) for item in checked.detectors],
            "votes": checked.votes,
            "accepted": checked.accepted,
            "reason": checked.reason,
        }


@dataclass(frozen=True, slots=True)
class NativeConcatScene:
    ordinal: int
    start_position: int
    end_position: int
    start_time: Fraction
    last_sample_time: Fraction
    end_time: Fraction
    end_reason: Literal["cut", "declared_end"]
    spans: tuple[NativeConcatSpan, ...]

    def __post_init__(self) -> None:
        _integer(self.ordinal, "ordinal", 0, _WORK - 1)
        _integer(self.start_position, "start_position", 0, _WORK - 1)
        _integer(self.end_position, "end_position", self.start_position + 1, _WORK)
        for name in ("start_time", "last_sample_time", "end_time"):
            value = getattr(self, name)
            if type(value) is not Fraction or _time(value, name) < 0:
                raise ConfigurationError("scene times must be exact nonnegative Fractions")
        if not self.start_time <= self.last_sample_time <= self.end_time:
            raise ConfigurationError("scene sample/endpoint times disagree")
        if type(self.end_reason) is not str or self.end_reason not in {"cut", "declared_end"}:
            raise ConfigurationError("invalid declared scene endpoint reason")
        _tuple(self.spans, "spans", 128)
        if bool(self.spans) != (self.start_time < self.end_time):
            raise ConfigurationError("only positive-duration scenes have declared spans")
        for span in self.spans:
            if type(span) is not NativeConcatSpan or type(span.timeline) is not NativeConcatTimeline:
                raise ConfigurationError("scene spans require exact NativeConcatSpan and timeline")
            _integer(span.clip_index, "span.clip_index", 0, 127)
            for name in ("global_start", "global_end"):
                value = getattr(span, name)
                if type(value) is not Fraction:
                    raise ConfigurationError("span positions must be exact Fractions")
                _time(value, name)
            if not self.start_time <= span.global_start < span.global_end <= self.end_time:
                raise ConfigurationError("span lies outside scene")
        # Whole timeline/partition binding is checked once by the containing result.

    @property
    def sample_count(self) -> int:
        return self.end_position - self.start_position

    def to_dict(self) -> dict[str, Any]:
        checked = replace(self)
        if checked.spans:
            timeline = _timeline(checked.spans[0].timeline)
            _bind_spans(checked, timeline, 128, checked.spans[0].timeline)
        return _scene_document(checked)


def _timeline(value: NativeConcatTimeline) -> NativeConcatTimeline:
    if type(value) is not NativeConcatTimeline:
        raise ConfigurationError("result requires NativeConcatTimeline")
    _tuple(value.clips, "timeline clips", 128)
    _tuple(value.offsets, "timeline offsets", 129)
    for clip in value.clips:
        if type(clip) is not NativeConcatClip or type(clip.path) not in _PATH_TYPES:
            raise ConfigurationError("timeline requires normalized native clips and exact local paths")
    for offset in value.offsets:
        if type(offset) is not Fraction:
            raise ConfigurationError("timeline offsets require exact Fractions")
        _time(offset, "offset")
    if type(value.digest) is not str or len(value.digest) != 64:
        raise ConfigurationError("invalid timeline digest")
    checked = NativeConcatTimeline(value.clips)
    if checked.offsets != value.offsets or checked.digest != value.digest:
        raise ConfigurationError("timeline offsets/digest contradict its declared clips")
    return checked


def _window(
    config: NativeConcatSceneConfig, timeline: NativeConcatTimeline
) -> tuple[Fraction, Fraction, tuple[int, ...]]:
    start = Fraction(config.video.start)
    end = timeline.duration if config.video.end is None else Fraction(config.video.end)
    if not 0 <= start <= end <= timeline.duration or (
        start == end and (config.video.end is not None or start != timeline.duration)
    ):
        raise ConfigurationError("concat scene range is outside the declared timeline or empty")
    count = len(timeline.clips)
    limits = config.video.limits
    if count > limits.max_sources or limits.max_frames * count > _WORK:
        raise ConfigurationError("concat scene source/mapping work exceeds capacity")
    activations = tuple(
        1 + int(max(start, timeline.offsets[i]) < min(end, timeline.offsets[i + 1])) for i in range(count)
    )
    if sum(activations) > limits.max_activations:
        raise ConfigurationError("complete analysis requires more activations than the configured capacity")
    return start, end, activations


def _bind_spans(
    scene: NativeConcatScene,
    timeline: NativeConcatTimeline,
    maximum: int,
    source_timeline: NativeConcatTimeline | None,
) -> None:
    expected = timeline.map_span(scene.start_time, scene.end_time, max_parts=maximum)
    if len(scene.spans) != len(expected):
        raise ConfigurationError("scene spans do not cover its declared interval")
    for actual, wanted in zip(scene.spans, expected, strict=True):
        # No per-row or per-span path resolution/reconstruction of the timeline.
        if actual.timeline is not source_timeline:
            raise ConfigurationError("scene spans must share one declared timeline")
        if (actual.clip_index, actual.global_start, actual.global_end) != (
            wanted.clip_index,
            wanted.global_start,
            wanted.global_end,
        ):
            raise ConfigurationError("scene span geometry contradicts the declared map")


def _scene_document(scene: NativeConcatScene) -> dict[str, Any]:
    return {
        "ordinal": scene.ordinal,
        "start_position": scene.start_position,
        "end_position": scene.end_position,
        "sample_count": scene.sample_count,
        "start_time": _pair(scene.start_time),
        "last_sample_time": _pair(scene.last_sample_time),
        "end_time": _pair(scene.end_time),
        "end_reason": scene.end_reason,
        "spans": [span.to_dict() for span in scene.spans],
    }


def _statistics(
    samples: tuple[NativeConcatSceneSample, ...], config: NativeConcatSceneConfig
) -> tuple[NativeConcatSceneStatistic, ...]:
    content, evidence = _detector_evidence([sample.native.metrics for sample in samples], config.detectors)
    result = []
    previous = 0
    for position, sample in enumerate(samples):
        rows = tuple(items[position] for items in evidence)
        accepted, reason = _aggregate_policy(
            position,
            len(samples),
            previous,
            sum(item.qualified for item in rows),
            config.minimum_votes,
            config.min_scene_samples,
        )
        result.append(NativeConcatSceneStatistic(sample, content[position], rows, accepted, reason))
        if accepted:
            previous = position
    return tuple(result)


def _scenes(
    rows: tuple[NativeConcatSceneStatistic, ...],
    timeline: NativeConcatTimeline,
    end: Fraction,
    max_parts: int,
) -> tuple[NativeConcatScene, ...]:
    if not rows:
        return ()
    cuts = [i for i, row in enumerate(rows) if row.accepted]
    scenes = []
    for ordinal, (a, b) in enumerate(zip([0, *cuts], [*cuts, len(rows)], strict=True)):
        right = rows[b].sample.presentation_time if b < len(rows) else end
        left = rows[a].sample.presentation_time
        scenes.append(
            NativeConcatScene(
                ordinal,
                a,
                b,
                left,
                rows[b - 1].sample.presentation_time,
                right,
                "cut" if b < len(rows) else "declared_end",
                timeline.map_span(left, right, max_parts=max_parts),
            )
        )
    return tuple(scenes)


@dataclass(frozen=True, slots=True)
class NativeConcatSceneResult:
    config: NativeConcatSceneConfig
    timeline: NativeConcatTimeline
    metadata: tuple[NativeVideoMetadata, ...]
    diagnostics: NativeConcatDiagnostics
    statistics: tuple[NativeConcatSceneStatistic, ...]
    scenes: tuple[NativeConcatScene, ...]

    def __post_init__(self) -> None:
        if type(self.config) is not NativeConcatSceneConfig:
            raise ConfigurationError("result requires NativeConcatSceneConfig")
        config = replace(self.config)
        timeline = _timeline(self.timeline)
        start, end, activations = _window(config, timeline)
        _tuple(self.metadata, "metadata", len(timeline.clips))
        if len(self.metadata) != len(timeline.clips):
            raise ConfigurationError("one metadata record is required per source occurrence")
        metadata: list[NativeVideoMetadata] = []
        for clip, item in zip(timeline.clips, self.metadata, strict=True):
            if type(item) is not NativeVideoMetadata or type(item.path) not in _PATH_TYPES:
                raise ConfigurationError("metadata requires exact NativeVideoMetadata and local path")
            checked = replace(item)
            if checked.path != clip.path:
                raise ConfigurationError("metadata path does not match its source occurrence")
            if metadata and (checked.width, checked.height) != (metadata[0].width, metadata[0].height):
                raise ConfigurationError("concat source dimensions must match")
            metadata.append(checked)
        if type(self.diagnostics) is not NativeConcatDiagnostics or type(
            self.diagnostics.limit_scope
        ) not in (str, type(None)):
            raise ConfigurationError("result requires exact concat diagnostics")
        diagnostics = replace(self.diagnostics)
        playback = tuple(i for i, count in enumerate(activations) if count == 2)
        next_clip = playback[-1] + 1 if playback else len(timeline.clips)
        child_statuses = {"eof", "range_end"} if playback else {"closed"}
        if (
            not diagnostics.closed
            or diagnostics.status != ("range_end" if end < timeline.duration else "intervals_exhausted")
            or diagnostics.last_child_status not in child_statuses
            or diagnostics.current_clip != (next_clip if next_clip < len(timeline.clips) else None)
            or diagnostics.generation != 0
            or diagnostics.seeks != 0
            or diagnostics.cleanup_errors
            or diagnostics.limit_scope is not None
        ):
            raise ConfigurationError("result requires clean complete no-seek concat diagnostics")
        _tuple(self.statistics, "statistics", config.video.limits.max_frames)
        _tuple(self.scenes, "scenes", len(self.statistics))
        if (
            diagnostics.returned_frames != len(self.statistics)
            or diagnostics.returned_frames
            != (diagnostics.owned_rgb_frames + config.video.frame_step - 1) // config.video.frame_step
            or diagnostics.source_activations != activations
        ):
            raise ConfigurationError("sample/activation counts disagree with complete result")
        _budget_diagnostics(config, tuple(metadata), diagnostics)
        rgb_offsets = [0]
        for count in diagnostics.source_rgb_frames:
            rgb_offsets.append(rgb_offsets[-1] + count)
        rows = []
        previous: NativeConcatSceneSample | None = None
        for position, row in enumerate(self.statistics):
            if type(row) is not NativeConcatSceneStatistic:
                raise ConfigurationError("statistics require exact NativeConcatSceneStatistic")
            checked_row = replace(row)
            sample = checked_row.sample
            if sample.clip_index >= len(timeline.clips):
                raise ConfigurationError("sample source occurrence is outside timeline")
            clip = timeline.clips[sample.clip_index]
            if (
                sample.timeline_digest != timeline.digest
                or sample.sample_index != position
                or sample.activation != 1
                or sample.generation != 0
                or sample.native.generation != 0
                or not clip.start <= sample.native.presentation_time < clip.end
                or sample.presentation_time
                != timeline.offsets[sample.clip_index] + sample.native.presentation_time - clip.start
                or not start <= sample.presentation_time < end
            ):
                raise ConfigurationError("sample native identity or derived timeline position disagrees")
            if (
                sample.native.decode_index >= diagnostics.source_decoded_frames[sample.clip_index]
                or sample.native.sample_index >= diagnostics.source_rgb_frames[sample.clip_index]
                or rgb_offsets[sample.clip_index] + sample.native.sample_index
                != position * config.video.frame_step
            ):
                raise ConfigurationError("native sample indices contradict observed work or global stride")
            if previous is not None and (
                sample.clip_index < previous.clip_index
                or sample.presentation_time < previous.presentation_time
                or (
                    sample.clip_index == previous.clip_index
                    and (
                        sample.native.decode_index <= previous.native.decode_index
                        or sample.native.sample_index <= previous.native.sample_index
                    )
                )
            ):
                raise ConfigurationError("sample source/time/native index order is inconsistent")
            previous = sample
            rows.append(checked_row)
        expected_rows = _statistics(tuple(row.sample for row in rows), config)
        if tuple(rows) != expected_rows:
            raise ConfigurationError("detector/quorum evidence does not match bounded measurement replay")
        scenes = []
        parts = 0
        span_timeline: NativeConcatTimeline | None = None
        for scene in self.scenes:
            if type(scene) is not NativeConcatScene:
                raise ConfigurationError("scenes require exact NativeConcatScene")
            checked_scene = replace(scene)
            if checked_scene.spans and span_timeline is None:
                span_timeline = checked_scene.spans[0].timeline
                if span_timeline is not self.timeline and _timeline(span_timeline) != timeline:
                    raise ConfigurationError("scene spans refer to another declared timeline")
            _bind_spans(checked_scene, timeline, config.video.limits.max_span_parts, span_timeline)
            parts += len(checked_scene.spans)
            scenes.append(checked_scene)
        if parts > _WORK or parts > max(0, len(scenes) + len(timeline.clips) - 1):
            raise ConfigurationError("total mapped scene pieces exceed the disjoint partition bound")
        expected_scenes = _scenes(expected_rows, timeline, end, config.video.limits.max_span_parts)
        if len(scenes) != len(expected_scenes) or any(
            (
                actual.ordinal,
                actual.start_position,
                actual.end_position,
                actual.start_time,
                actual.last_sample_time,
                actual.end_time,
                actual.end_reason,
            )
            != (
                wanted.ordinal,
                wanted.start_position,
                wanted.end_position,
                wanted.start_time,
                wanted.last_sample_time,
                wanted.end_time,
                wanted.end_reason,
            )
            for actual, wanted in zip(scenes, expected_scenes, strict=True)
        ):
            raise ConfigurationError("scenes contradict detector cuts or declared endpoints")
        object.__setattr__(self, "config", config)
        object.__setattr__(self, "timeline", timeline)
        object.__setattr__(self, "metadata", tuple(metadata))
        object.__setattr__(self, "diagnostics", diagnostics)
        object.__setattr__(self, "statistics", expected_rows)
        object.__setattr__(self, "scenes", expected_scenes)

    @property
    def cut_positions(self) -> tuple[int, ...]:
        return tuple(row.sample.sample_index for row in self.statistics if row.accepted)

    @property
    def cut_times(self) -> tuple[Fraction, ...]:
        return tuple(row.sample.presentation_time for row in self.statistics if row.accepted)

    def to_dict(self) -> dict[str, Any]:
        checked = replace(self)
        return {
            "kind": "frame-quorum-native-concat-scenes",
            "schema_version": 1,
            "execution": "offline_declared_concat",
            "source_verified": False,
            "coverage": "declared_only",
            "config": checked.config.to_dict(),
            "timeline": checked.timeline.to_dict(),
            "timeline_digest": checked.timeline.digest,
            "metadata": [item.to_dict() for item in checked.metadata],
            "diagnostics": checked.diagnostics.to_dict(),
            "sample_count": len(checked.statistics),
            "cut_positions": list(checked.cut_positions),
            "cut_times": [_pair(t) for t in checked.cut_times],
            "statistics": [row.to_dict() for row in checked.statistics],
            "scenes": [_scene_document(scene) for scene in checked.scenes],
        }


def _budget_diagnostics(
    config: NativeConcatSceneConfig,
    metadata: tuple[NativeVideoMetadata, ...],
    diagnostic: NativeConcatDiagnostics,
) -> None:
    limits = config.video.limits
    if diagnostic.opened_source_bytes != sum(
        item.source_bytes * count for item, count in zip(metadata, diagnostic.source_activations, strict=True)
    ):
        raise ConfigurationError("opened source bytes disagree with probe/playback activations")
    if sum(item.source_bytes for item in metadata) > limits.max_manifest_source_bytes:
        raise ConfigurationError("manifest source bytes exceed capacity")
    # Returned/RGB ceilings stop on the last accepted frame, without an EOF
    # peek. Decode/pixel equality can instead end at a decoded range sentinel.
    if (
        diagnostic.returned_frames >= limits.max_frames
        or diagnostic.owned_rgb_frames >= limits.max_rgb_frames
    ):
        raise ConfigurationError("complete analysis cannot exhaust returned/RGB capacity")
    for value, maximum in (
        (diagnostic.opened_source_bytes, limits.max_opened_source_bytes),
        (diagnostic.decoded_frames, limits.max_decoded_frames),
        (diagnostic.owned_rgb_frames, limits.max_rgb_frames),
        (diagnostic.returned_frames, limits.max_frames),
        (diagnostic.decoded_pixels_observed, limits.max_total_pixels),
    ):
        if value > maximum:
            raise ConfigurationError("aggregate source work exceeds configured capacity")
    for i, item in enumerate(metadata):
        decoded, rgb, pixels = (
            diagnostic.source_decoded_frames[i],
            diagnostic.source_rgb_frames[i],
            diagnostic.source_pixels[i],
        )
        if (
            item.source_bytes > limits.max_source_bytes
            or item.width * item.height > limits.max_frame_pixels
            or decoded > limits.max_source_decoded_frames
            or rgb >= limits.max_source_rgb_frames
            or pixels > limits.max_source_total_pixels
            or rgb > decoded
            or not rgb * item.width * item.height + decoded - rgb
            <= pixels
            <= decoded * limits.max_frame_pixels
            or (diagnostic.source_activations[i] == 1 and (decoded != 0 or rgb != 0 or pixels != 0))
        ):
            raise ConfigurationError("per-source metadata/work exceeds configured capacity")


def detect_native_concat_scenes(
    clips: tuple[NativeConcatClip, ...],
    config: NativeConcatSceneConfig | None = None,
    *,
    on_sample: Callable[[NativeConcatSceneSample], None] | None = None,
) -> NativeConcatSceneResult:
    """Own decoding, close it, then return a complete bounded measurement analysis.

    On post-construction failure, ``error.native_concat_cleanup`` retains the
    same stream for same-thread close retry (ordinary attribute-capable errors).
    Callback effects are not rolled back. No partial successful result is returned.
    """
    if config is not None and type(config) is not NativeConcatSceneConfig:
        raise ConfigurationError("config must be NativeConcatSceneConfig")
    options = NativeConcatSceneConfig() if config is None else replace(config)
    if on_sample is not None and (
        not callable(on_sample)
        or inspect.iscoroutinefunction(on_sample)
        or inspect.iscoroutinefunction(getattr(on_sample, "__call__", None))  # noqa: B004
    ):
        raise ConfigurationError("on_sample must be a synchronous callable")
    stream = NativeConcatStream(clips, options.video)
    try:
        _, end, _ = _window(options, stream.timeline)
        samples = []
        with stream:
            metadata = stream.metadata
            for frame in stream:
                sample = NativeConcatSceneSample(
                    stream.timeline.digest,
                    frame.clip_index,
                    frame.activation,
                    frame.generation,
                    frame.sample_index,
                    frame.presentation_time,
                    _measure_native_sample(frame.native),
                )
                del frame
                samples.append(sample)
                if on_sample is not None:
                    returned = cast(Callable[[NativeConcatSceneSample], object], on_sample)(sample)
                    if returned is not None:
                        if inspect.iscoroutine(returned):
                            with suppress(Exception):
                                returned.close()
                        raise ConfigurationError("on_sample must return None, not a value or awaitable")
        diagnostics = stream.diagnostics
        if (
            diagnostics.status not in {"intervals_exhausted", "range_end"}
            or not diagnostics.closed
            or diagnostics.cleanup_errors
        ):
            raise ScanError("incomplete concat scene analysis: declared range did not complete cleanly")
        rows = _statistics(tuple(samples), options)
        scenes = _scenes(rows, stream.timeline, end, options.video.limits.max_span_parts)
        return NativeConcatSceneResult(options, stream.timeline, metadata, diagnostics, rows, scenes)
    except BaseException as error:
        # A selected native control/error must not be replaced by attribute hooks.
        with suppress(BaseException):
            error.native_concat_cleanup = stream  # type: ignore[attr-defined]
        raise


__all__ = [
    "NativeConcatScene",
    "NativeConcatSceneConfig",
    "NativeConcatSceneResult",
    "NativeConcatSceneSample",
    "NativeConcatSceneStatistic",
    "detect_native_concat_scenes",
]
