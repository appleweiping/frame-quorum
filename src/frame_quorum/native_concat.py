"""Bounded real multi-source decoding on an exact caller-declared timeline.

Declared clip endpoints are coordinates, not verified physical sample durations.
Original native PTS/RGB remain separate from composite presentation positions.
"""

from __future__ import annotations

import hashlib
import json
import stat
import threading
from bisect import bisect_right
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field, replace
from fractions import Fraction
from pathlib import Path
from types import TracebackType
from typing import Any

from PIL import Image

from .errors import ConfigurationError, ScanError
from .native_video import (
    NativeVideoConfig,
    NativeVideoFrame,
    NativeVideoMetadata,
    NativeVideoStatus,
    NativeVideoStream,
    _fraction,
    _integer,
    _pair,
    _source_path,
)

_INT64 = (1 << 63) - 1
_STATES = frozenset(
    {
        "new",
        "open",
        "intervals_exhausted",
        "range_end",
        "frame_limit",
        "decode_limit",
        "rgb_limit",
        "pixel_limit",
        "source_limit",
        "activation_limit",
        "seek_limit",
        "closed",
        "error",
        "interrupted",
    }
)
_BUDGET_FIELDS = (
    ("max_sources", 128),
    ("max_source_bytes", 1 << 40),
    ("max_manifest_source_bytes", 1 << 40),
    ("max_opened_source_bytes", 1 << 40),
    ("max_activations", 10_000),
    ("max_seeks", 10_000),
    ("max_decoded_frames", 10_000_000),
    ("max_source_decoded_frames", 10_000_000),
    ("max_rgb_frames", 1_000_000),
    ("max_source_rgb_frames", 1_000_000),
    ("max_frames", 1_000_000),
    ("max_frame_pixels", 67_108_864),
    ("max_total_pixels", 1 << 40),
    ("max_source_total_pixels", 1 << 40),
    ("max_span_parts", 128),
)


def _time(value: Any, name: str) -> Fraction:
    if type(value) not in (int, Fraction):
        raise ConfigurationError(f"{name} must be an exact integer or Fraction")
    if type(value) is int and value.bit_length() > 127:
        raise ConfigurationError(f"{name} exceeds composite rational bounds")
    result = Fraction(value)
    if abs(result.numerator) > (1 << 127) - 1 or result.denominator > _INT64:
        raise ConfigurationError(f"{name} exceeds composite rational bounds")
    return result


def _canonical(value: Any) -> bytes:
    raw = json.dumps(
        value, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(",", ":")
    ).encode("utf-8")
    if len(raw) > 1 << 20:
        raise ConfigurationError("concat manifest exceeds the 1 MiB canonical limit")
    return raw


@dataclass(frozen=True, slots=True)
class NativeConcatClip:
    path: str | Path
    start: Fraction | int = field(kw_only=True)
    end: Fraction | int = field(kw_only=True)
    video_stream: int = field(default=0, kw_only=True)

    def __post_init__(self) -> None:
        if not isinstance(self.path, (str, Path)):
            raise ConfigurationError("concat clip path must be local and at most 4096 UTF-8 bytes")
        if len(str(self.path)) > 4096:
            raise ConfigurationError("concat clip path must be local and at most 4096 UTF-8 bytes")
        try:
            path_size = len(str(self.path).encode("utf-8"))
        except UnicodeEncodeError as error:
            raise ConfigurationError("concat clip path must be valid Unicode") from error
        if path_size > 4096:
            raise ConfigurationError("concat clip path must be local and at most 4096 UTF-8 bytes")
        path = _source_path(self.path)
        if len(path.as_posix()) > 4096 or len(path.as_posix().encode("utf-8")) > 4096:
            raise ConfigurationError("resolved concat clip path exceeds 4096 UTF-8 bytes")
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "start", _fraction(self.start, "clip.start"))
        object.__setattr__(self, "end", _fraction(self.end, "clip.end"))
        if self.end <= self.start:
            raise ConfigurationError("concat clip end must be strictly greater than start")
        _integer(self.video_stream, "clip.video_stream", 0, 1023)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "start": _pair(Fraction(self.start)),
            "end": _pair(Fraction(self.end)),
            "video_stream": self.video_stream,
        }


@dataclass(frozen=True, slots=True)
class NativeConcatLimits:
    max_sources: int = 32
    max_source_bytes: int = 1_000_000_000
    max_manifest_source_bytes: int = 4_000_000_000
    max_opened_source_bytes: int = 16_000_000_000
    max_activations: int = 256
    max_seeks: int = 128
    max_decoded_frames: int = 100_000
    max_source_decoded_frames: int = 100_000
    max_rgb_frames: int = 100_000
    max_source_rgb_frames: int = 100_000
    max_frames: int = 10_000
    max_frame_pixels: int = 16_777_216
    max_total_pixels: int = 1_000_000_000
    max_source_total_pixels: int = 1_000_000_000
    max_span_parts: int = 32

    def __post_init__(self) -> None:
        for name, maximum in _BUDGET_FIELDS:
            _integer(getattr(self, name), name, 1, maximum)


@dataclass(frozen=True, slots=True, kw_only=True)
class NativeConcatConfig:
    start: Fraction | int = 0
    end: Fraction | int | None = None
    frame_step: int = 1
    limits: NativeConcatLimits = field(default_factory=NativeConcatLimits)

    def __post_init__(self) -> None:
        object.__setattr__(self, "start", _time(self.start, "start"))
        if self.start < 0:
            raise ConfigurationError("composite start cannot be negative")
        if self.end is not None:
            object.__setattr__(self, "end", _time(self.end, "end"))
            if self.end <= self.start:
                raise ConfigurationError("composite end must be strictly greater than start")
        _integer(self.frame_step, "frame_step", 1, 1_000_000)
        if type(self.limits) is not NativeConcatLimits:
            raise ConfigurationError("limits must be NativeConcatLimits")
        object.__setattr__(self, "limits", replace(self.limits))


@dataclass(frozen=True, slots=True)
class NativeConcatTimeline:
    clips: tuple[NativeConcatClip, ...]
    offsets: tuple[Fraction, ...] = field(init=False)
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            type(self.clips) is not tuple
            or not 1 <= len(self.clips) <= 128
            or any(type(c) is not NativeConcatClip for c in self.clips)
        ):
            raise ConfigurationError("concat clips must be an exact tuple of 1..128 NativeConcatClip values")
        owned = tuple(replace(clip) for clip in self.clips)
        object.__setattr__(self, "clips", owned)
        offsets = [Fraction(0)]
        for clip in owned:
            offsets.append(_time(offsets[-1] + clip.end - clip.start, "declared offset"))
        object.__setattr__(self, "offsets", tuple(offsets))
        object.__setattr__(self, "digest", hashlib.sha256(_canonical(self.to_dict())).hexdigest())

    @property
    def duration(self) -> Fraction:
        return self.offsets[-1]

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "frame-quorum-native-concat-timeline",
            "schema_version": 1,
            "policy": "caller_declared",
            "clips": [c.to_dict() for c in self.clips],
            "offsets": [_pair(t) for t in self.offsets],
        }

    def map_span(
        self, start: Fraction | int, end: Fraction | int, *, max_parts: int = 128
    ) -> tuple[NativeConcatSpan, ...]:
        left, right = _time(start, "span.start"), _time(end, "span.end")
        _integer(max_parts, "max_parts", 1, 128)
        if not 0 <= left <= right <= self.duration:
            raise ConfigurationError("span must satisfy 0 <= start <= end <= declared duration")
        if left == right:
            return ()
        pieces: list[NativeConcatSpan] = []
        for index in range(len(self.clips)):
            a, b = max(left, self.offsets[index]), min(right, self.offsets[index + 1])
            if a < b:
                if len(pieces) >= max_parts:
                    raise ConfigurationError("span mapping exceeds its piece limit")
                pieces.append(NativeConcatSpan(self, index, a, b))
        return tuple(pieces)


@dataclass(frozen=True, slots=True)
class NativeConcatSpan:
    timeline: NativeConcatTimeline = field(repr=False)
    clip_index: int
    global_start: Fraction
    global_end: Fraction

    def __post_init__(self) -> None:
        if type(self.timeline) is not NativeConcatTimeline:
            raise ConfigurationError("span requires an exact declared timeline")
        _integer(self.clip_index, "clip_index", 0, len(self.timeline.clips) - 1)
        a, b = _time(self.global_start, "global_start"), _time(self.global_end, "global_end")
        if not self.timeline.offsets[self.clip_index] <= a < b <= self.timeline.offsets[self.clip_index + 1]:
            raise ConfigurationError("span must lie inside exactly one declared source interval")
        object.__setattr__(self, "global_start", a)
        object.__setattr__(self, "global_end", b)
        _time(self.local_start, "local_start")
        _time(self.local_end, "local_end")

    @property
    def path(self) -> Path:
        return Path(self.timeline.clips[self.clip_index].path)

    @property
    def local_start(self) -> Fraction:
        return (
            Fraction(self.timeline.clips[self.clip_index].start)
            + self.global_start
            - self.timeline.offsets[self.clip_index]
        )

    @property
    def local_end(self) -> Fraction:
        return (
            Fraction(self.timeline.clips[self.clip_index].start)
            + self.global_end
            - self.timeline.offsets[self.clip_index]
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "timeline_digest": self.timeline.digest,
            "clip_index": self.clip_index,
            "path": self.path.as_posix(),
            "global_start": _pair(self.global_start),
            "global_end": _pair(self.global_end),
            "local_start": _pair(self.local_start),
            "local_end": _pair(self.local_end),
            "coverage": "declared_only",
        }


@dataclass(frozen=True, slots=True)
class NativeConcatFrame:
    timeline: NativeConcatTimeline = field(repr=False)
    clip_index: int
    activation: int
    generation: int
    sample_index: int
    native: NativeVideoFrame = field(repr=False)

    def __post_init__(self) -> None:
        if type(self.timeline) is not NativeConcatTimeline or type(self.native) is not NativeVideoFrame:
            raise ConfigurationError("concat frame requires an exact timeline and native frame")
        _integer(self.clip_index, "clip_index", 0, len(self.timeline.clips) - 1)
        for name in ("activation", "generation", "sample_index"):
            _integer(getattr(self, name), name, 0, _INT64)
        clip = self.timeline.clips[self.clip_index]
        if not clip.start <= self.native.presentation_time < clip.end:
            raise ConfigurationError("native frame is outside its declared clip")
        _time(self.presentation_time, "composite presentation_time")

    @property
    def presentation_time(self) -> Fraction:
        return (
            self.timeline.offsets[self.clip_index]
            + self.native.presentation_time
            - self.timeline.clips[self.clip_index].start
        )

    @property
    def rgb(self) -> bytes:
        return self.native.rgb

    def image(self) -> Image.Image:
        return self.native.image()

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "frame-quorum-native-concat-frame",
            "schema_version": 1,
            "timeline_digest": self.timeline.digest,
            "clip_index": self.clip_index,
            "activation": self.activation,
            "generation": self.generation,
            "sample_index": self.sample_index,
            "presentation_time": _pair(self.presentation_time),
            "native": self.native.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class NativeConcatDiagnostics:
    status: str
    closed: bool
    generation: int
    seeks: int
    activations: int
    opened_source_bytes: int
    decoded_frames: int
    owned_rgb_frames: int
    returned_frames: int
    decoded_pixels_observed: int
    source_decoded_frames: tuple[int, ...]
    source_rgb_frames: tuple[int, ...]
    source_pixels: tuple[int, ...]
    source_activations: tuple[int, ...]
    current_clip: int | None
    limit_scope: str | None = None
    cleanup_errors: tuple[str, ...] = ()
    last_child_status: str | None = None

    def __post_init__(self) -> None:
        if type(self.status) is not str or self.status not in _STATES or type(self.closed) is not bool:
            raise ConfigurationError("invalid concat diagnostic status/closed flag")
        for name in (
            "generation",
            "seeks",
            "activations",
            "opened_source_bytes",
            "decoded_frames",
            "owned_rgb_frames",
            "returned_frames",
            "decoded_pixels_observed",
        ):
            _integer(getattr(self, name), name, 0, _INT64)
        if (
            not self.returned_frames <= self.owned_rgb_frames <= self.decoded_frames
            or self.generation != self.seeks
        ):
            raise ConfigurationError("concat output/work/generation counters disagree")
        if type(self.source_decoded_frames) is not tuple:
            raise ConfigurationError("per-source diagnostics require exact tuples")
        count = len(self.source_decoded_frames)
        for name, expected in (
            ("source_decoded_frames", self.decoded_frames),
            ("source_rgb_frames", self.owned_rgb_frames),
            ("source_pixels", self.decoded_pixels_observed),
            ("source_activations", self.activations),
        ):
            values = getattr(self, name)
            if type(values) is not tuple or not 1 <= len(values) == count <= 128:
                raise ConfigurationError("per-source diagnostics require equally sized bounded tuples")
            for value in values:
                _integer(value, name, 0, _INT64)
            if sum(values) != expected:
                raise ConfigurationError("per-source and aggregate diagnostic counts disagree")
        if self.current_clip is not None:
            _integer(self.current_clip, "current_clip", 0, count - 1)
        if self.limit_scope not in (None, "source", "total"):
            raise ConfigurationError("invalid limit scope")
        if self.last_child_status is not None and (
            type(self.last_child_status) is not str
            or self.last_child_status not in {status.value for status in NativeVideoStatus}
        ):
            raise ConfigurationError("invalid last native child status")
        if (
            type(self.cleanup_errors) is not tuple
            or len(self.cleanup_errors) > 3
            or any(
                type(v) is not str or v not in ("frames.close", "container.close", "file.close")
                for v in self.cleanup_errors
            )
            or len(set(self.cleanup_errors)) != len(self.cleanup_errors)
        ):
            raise ConfigurationError("invalid concat cleanup resource inventory")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        for name in (
            "source_decoded_frames",
            "source_rgb_frames",
            "source_pixels",
            "source_activations",
            "cleanup_errors",
        ):
            result[name] = list(result[name])
        return result


@dataclass(slots=True)
class _Work:
    decoded: int = 0
    rgb: int = 0
    pixels: int = 0
    activations: int = 0


class _BudgetStop(Exception):
    def __init__(self, status: str, scope: str):
        self.status, self.scope = status, scope


def _primary_error(primary: BaseException | None, secondary: BaseException) -> BaseException:
    if primary is not None and (not isinstance(primary, Exception) or isinstance(secondary, Exception)):
        primary.add_note("concat cleanup or work observation also failed")
        return primary
    return secondary


def _identity(path: Path) -> tuple[int, int, int, int]:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
        raise ScanError("concat source must be a regular nonsymlink/nonreparse file")
    if (
        type(info.st_dev) is not int
        or type(info.st_ino) is not int
        or not 0 <= info.st_dev < 1 << 128
        or not 0 < info.st_ino < 1 << 128
    ):
        raise ScanError("concat source requires an available stable file identity")
    _integer(info.st_size, "source bytes", 0, 1 << 40)
    _integer(info.st_mtime_ns, "source mtime", -_INT64 - 1, _INT64)
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns


class NativeConcatStream(Iterator[NativeConcatFrame]):
    """Single-owner real decoding over a frozen caller-declared interval map."""

    def __init__(self, clips: tuple[NativeConcatClip, ...], config: NativeConcatConfig | None = None):
        self._timeline = NativeConcatTimeline(clips)
        if config is not None and type(config) is not NativeConcatConfig:
            raise ConfigurationError("config must be NativeConcatConfig")
        self._config = NativeConcatConfig() if config is None else replace(config)
        self._limits = self.config.limits
        if len(clips) > self._limits.max_sources:
            raise ConfigurationError("concat source occurrence limit exceeded")
        self._metadata: tuple[NativeVideoMetadata, ...] | None = None
        self._identities: tuple[tuple[int, int, int, int], ...] = ()
        self._work = [_Work() for _ in clips]
        self._child: NativeVideoStream | None = None
        self._accounted = (0, 0, 0)
        self._index = 0
        self._status = "new"
        self._scope: str | None = None
        self._cleanup_errors: tuple[str, ...] = ()
        self._last_child_status: NativeVideoStatus | None = None
        self._active = self._manual_closed = self._busy = False
        self._owner: int | None = None
        self._generation = self._seeks = self._activations = self._opened_bytes = 0
        self._decoded = self._rgb = self._pixels = self._returned = self._eligible = 0
        self._start, self._end = self._window(self.config.start, self.config.end)

    def _window(self, start: Fraction | int, end: Fraction | int | None) -> tuple[Fraction, Fraction]:
        left, right = _time(start, "start"), self.timeline.duration if end is None else _time(end, "end")
        if (
            not 0 <= left <= self.timeline.duration
            or not left <= right <= self.timeline.duration
            or (left == right and (end is not None or left != self.timeline.duration))
        ):
            raise ConfigurationError("concat window is outside the declared timeline or empty")
        for index in range(len(self.timeline.clips)):
            bounds = self._bounds(index, left, right)
            if bounds is not None and self._metadata is not None:
                _integer(
                    bounds[0] // self._metadata[index].time_base, "native seek offset", -_INT64 - 1, _INT64
                )
        return left, right

    def _bounds(self, index: int, left: Fraction, right: Fraction) -> tuple[Fraction, Fraction] | None:
        offset = self.timeline.offsets[index]
        a, b = max(left, offset), min(right, self.timeline.offsets[index + 1])
        if a >= b:
            return None
        origin = self.timeline.clips[index].start
        return _fraction(origin + a - offset, "native window start"), _fraction(
            origin + b - offset, "native window end"
        )

    @property
    def timeline(self) -> NativeConcatTimeline:
        return self._timeline

    @property
    def config(self) -> NativeConcatConfig:
        return self._config

    @property
    def metadata(self) -> tuple[NativeVideoMetadata, ...]:
        if self._metadata is None:
            raise ConfigurationError("concat metadata requires successful complete source probing")
        return self._metadata

    @property
    def diagnostics(self) -> NativeConcatDiagnostics:
        return NativeConcatDiagnostics(
            self._status,
            self._child is None or self._child.diagnostics.closed,
            self._generation,
            self._seeks,
            self._activations,
            self._opened_bytes,
            self._decoded,
            self._rgb,
            self._returned,
            self._pixels,
            tuple(w.decoded for w in self._work),
            tuple(w.rgb for w in self._work),
            tuple(w.pixels for w in self._work),
            tuple(w.activations for w in self._work),
            self._index if self._index < len(self._work) else None,
            self._scope,
            self._cleanup_errors,
            None if self._last_child_status is None else self._last_child_status.value,
        )

    def _check_owner(self) -> None:
        if self._owner is not None and self._owner != threading.get_ident():
            raise ConfigurationError("concat stream is single-owner, not thread-safe")
        if self._busy:
            raise ConfigurationError("concat stream is not reentrant")

    def _account(self) -> None:
        if self._child is None:
            return
        observed = self._child.diagnostics
        now = (observed.decoded_frames, observed.returned_frames, observed.decoded_pixels_observed)
        delta = tuple(a - b for a, b in zip(now, self._accounted, strict=True))
        if any(v < 0 for v in delta):
            raise ScanError("child lifetime work counters decreased")
        work = self._work[self._index]
        work.decoded += delta[0]
        work.rgb += delta[1]
        work.pixels += delta[2]
        self._decoded += delta[0]
        self._rgb += delta[1]
        self._pixels += delta[2]
        self._accounted = now
        self._last_child_status = observed.status
        self._cleanup_errors = tuple(dict.fromkeys((*self._cleanup_errors, *observed.cleanup_errors)))

    def _account_after(self, primary: BaseException | None = None) -> None:
        try:
            self._account()
        except BaseException as error:
            if _primary_error(primary, error) is not primary:
                raise

    def _close_child(self, primary: BaseException | None = None) -> None:
        if self._child is None:
            return
        failure: BaseException | None = None
        try:
            self._child.close()
        except BaseException as cleanup:
            failure = cleanup
        try:
            self._account_after(failure if failure is not None else primary)
        except BaseException as observation:
            failure = observation
        try:
            if self._child.diagnostics.closed:
                self._child = None
        except BaseException as acknowledgement:
            failure = _primary_error(failure, acknowledgement)
        if failure is not None and _primary_error(primary, failure) is not primary:
            raise failure

    def _remaining(self, index: int) -> tuple[int, int, int]:
        limit, work = self._limits, self._work[index]
        quantities = (
            (
                "decode_limit",
                limit.max_decoded_frames - self._decoded,
                limit.max_source_decoded_frames - work.decoded,
            ),
            ("rgb_limit", limit.max_rgb_frames - self._rgb, limit.max_source_rgb_frames - work.rgb),
            (
                "pixel_limit",
                limit.max_total_pixels - self._pixels,
                limit.max_source_total_pixels - work.pixels,
            ),
        )
        for status, total, source in quantities:
            if total <= 0 or source <= 0:
                raise _BudgetStop(status, "total" if total <= 0 else "source")
        return min(quantities[0][1:]), min(quantities[1][1:]), min(quantities[2][1:])

    def _admit_activation(self, index: int) -> tuple[int, int, int]:
        if self._activations >= self._limits.max_activations:
            raise _BudgetStop("activation_limit", "total")
        if self._opened_bytes + self._identities[index][2] > self._limits.max_opened_source_bytes:
            raise _BudgetStop("source_limit", "total")
        return self._remaining(index)

    def _activate(self, index: int, bounds: tuple[Fraction, Fraction] | None) -> NativeVideoStream:
        if self._child is not None:
            raise ScanError("cannot acquire a child while previous closure is unknown")
        decoded, rgb, pixels = self._admit_activation(index)
        self._index = index
        if _identity(Path(self.timeline.clips[index].path)) != self._identities[index]:
            raise ScanError("concat source changed between probe/playback activations")
        clip = self.timeline.clips[index]
        config = NativeVideoConfig(
            video_stream=clip.video_stream,
            start=None if bounds is None else bounds[0],
            end=None if bounds is None else bounds[1],
            frame_step=1,
            max_frames=rgb,
            max_decoded_frames=decoded,
            max_source_bytes=self._limits.max_source_bytes,
            max_frame_pixels=self._limits.max_frame_pixels,
            max_total_pixels=pixels,
        )
        child = NativeVideoStream(clip.path, config)
        self._child = child  # Own setup failures before __enter__ can raise.
        self._accounted = (0, 0, 0)
        self._activations += 1
        self._work[index].activations += 1
        self._opened_bytes += self._identities[index][2]
        primary: BaseException | None = None
        try:
            child.__enter__()
            if (
                child._captured_identity() != self._identities[index]
                or _identity(Path(clip.path)) != self._identities[index]
            ):
                raise ScanError("concat source identity changed while opening")
            if self._metadata is not None and child.metadata != self._metadata[index]:
                raise ScanError("concat source metadata changed between activations")
        except BaseException as error:
            primary = error
            raise
        finally:
            self._account_after(primary)
        return child

    def _finish(self, status: str, primary: BaseException | None = None) -> None:
        self._status = status
        try:
            self._close_child(primary)
        except BaseException as error:
            self._status = "interrupted" if not isinstance(error, Exception) else "error"
            raise
        if self._child is not None:
            self._status = (
                "interrupted" if primary is not None and not isinstance(primary, Exception) else "error"
            )

    def __enter__(self) -> NativeConcatStream:
        self._check_owner()
        if self._status != "new":
            raise ConfigurationError("concat stream context cannot be entered twice")
        self._owner = threading.get_ident()
        self._active = self._busy = True
        try:
            identities = tuple(_identity(Path(clip.path)) for clip in self.timeline.clips)
            if (
                any(info[2] > self._limits.max_source_bytes for info in identities)
                or sum(info[2] for info in identities) > self._limits.max_manifest_source_bytes
            ):
                raise _BudgetStop("source_limit", "total")
            self._identities = identities
            metadata: list[NativeVideoMetadata] = []
            for index in range(len(identities)):
                child = self._activate(index, None)
                item = child.metadata
                if metadata and (item.width, item.height) != (metadata[0].width, metadata[0].height):
                    raise ScanError("concat source dimensions must match")
                metadata.append(item)
                self._close_child()
            self._metadata = tuple(metadata)
            self._window(self._start, None if self.config.end is None else self._end)
            self._index = bisect_right(self.timeline.offsets, self._start) - 1
            self._status = "intervals_exhausted" if self._start == self.timeline.duration else "open"
        except BaseException as error:
            self._active = False
            if isinstance(error, _BudgetStop):
                self._scope = error.scope
            self._finish(
                error.status
                if isinstance(error, _BudgetStop)
                else "error"
                if isinstance(error, Exception)
                else "interrupted",
                error,
            )
            if isinstance(error, (ConfigurationError, ScanError)) or not isinstance(error, Exception):
                raise
            raise ScanError("concat source setup failed") from error
        finally:
            self._busy = False
        return self

    def __iter__(self) -> NativeConcatStream:
        self._check_owner()
        if not self._active:
            raise ConfigurationError("concat iteration requires an active context")
        return self

    def _child_limit(self) -> _BudgetStop:
        assert self._child is not None
        status = self._last_child_status
        work, limits = self._work[self._index], self._limits
        if status is NativeVideoStatus.FRAME_LIMIT:
            return _BudgetStop("rgb_limit", "total" if self._rgb >= limits.max_rgb_frames else "source")
        if status is NativeVideoStatus.DECODE_LIMIT:
            return _BudgetStop(
                "decode_limit", "total" if self._decoded >= limits.max_decoded_frames else "source"
            )
        if status is NativeVideoStatus.PIXEL_LIMIT:
            return _BudgetStop(
                "pixel_limit", "total" if self._pixels >= limits.max_total_pixels else "source"
            )
        raise ScanError(f"child stopped without logical exhaustion: {status}; observed={work.decoded}")

    def __next__(self) -> NativeConcatFrame:
        self._check_owner()
        if not self._active:
            raise ConfigurationError("concat iteration requires an active context")
        if self._status != "open":
            raise StopIteration
        self._busy = True
        try:
            return self._next_frame()
        except StopIteration:
            raise
        except _BudgetStop as error:
            self._scope = error.scope
            self._finish(error.status)
            raise StopIteration from None
        except BaseException as error:
            status = "error" if isinstance(error, Exception) else "interrupted"
            if self._child is not None and self._last_child_status is NativeVideoStatus.PIXEL_LIMIT:
                limit = self._child_limit()
                status, self._scope = limit.status, limit.scope
            self._finish(status, error)
            if isinstance(error, (ConfigurationError, ScanError)) or not isinstance(error, Exception):
                raise
            raise ScanError("concat frame decode or allocation failed") from error
        finally:
            self._busy = False

    def _next_frame(self) -> NativeConcatFrame:
        while self._index < len(self.timeline.clips):
            if self._returned >= self._limits.max_frames:
                raise _BudgetStop("frame_limit", "total")
            bounds = self._bounds(self._index, self._start, self._end)
            if bounds is None:
                break
            if self._child is None:
                self._activate(self._index, bounds)
            assert self._child is not None
            primary: BaseException | None = None
            try:
                try:
                    native = next(self._child)
                except BaseException as error:
                    # Logical exhaustion must not suppress failed work accounting.
                    primary = None if isinstance(error, StopIteration) else error
                    raise
                finally:
                    self._account_after(primary)
            except StopIteration:
                if self._last_child_status not in (NativeVideoStatus.EOF, NativeVideoStatus.RANGE_END):
                    raise self._child_limit() from None
                self._close_child()
                self._index += 1
                continue
            expected = self.metadata[self._index]
            if (native.width, native.height) != (expected.width, expected.height):
                raise ScanError("concat in-range frame dimensions changed")
            selected = self._eligible % self.config.frame_step == 0
            self._eligible += 1
            result = None
            if selected:
                sample_index = self._returned
                self._returned += 1  # Charge the allocation attempt, including a failed materialization.
                result = NativeConcatFrame(
                    self.timeline,
                    self._index,
                    self._work[self._index].activations - 1,
                    self._generation,
                    sample_index,
                    native,
                )
                if not self._start <= result.presentation_time < self._end:
                    raise ScanError("child returned a frame outside the requested composite window")
            if self._returned >= self._limits.max_frames:
                self._scope = "total"
                self._finish("frame_limit")
            elif self._last_child_status in (
                NativeVideoStatus.FRAME_LIMIT,
                NativeVideoStatus.DECODE_LIMIT,
            ):
                limit = self._child_limit()
                self._scope = limit.scope
                self._finish(limit.status)
            if result is not None:
                return result
            if self._status != "open":
                raise StopIteration
        self._finish("range_end" if self._end < self.timeline.duration else "intervals_exhausted")
        raise StopIteration

    def seek(self, start: Fraction | int, *, end: Fraction | int | None = None) -> None:
        self._check_owner()
        if (
            not self._active
            or self._manual_closed
            or self._status not in ("open", "range_end", "intervals_exhausted")
        ):
            raise ConfigurationError("cannot seek a closed, failed or exhausted concat stream")
        left, right = self._window(start, end)
        if self._seeks >= self._limits.max_seeks:
            raise ScanError("concat seek lifetime budget is exhausted")
        if self._returned >= self._limits.max_frames:
            raise ScanError("concat frame lifetime budget is exhausted")
        index = bisect_right(self.timeline.offsets, left) - 1
        if index < len(self._work):
            try:
                self._admit_activation(index)
            except _BudgetStop as error:
                raise ScanError(f"concat {error.status} budget is exhausted") from error
        self._busy = True
        try:
            self._close_child()
            self._seeks += 1
            self._generation += 1
            self._eligible = 0
            self._index = index
            self._start, self._end = left, right
            self._status = "intervals_exhausted" if left == self.timeline.duration else "open"
            if index < len(self._work):
                self._activate(index, self._bounds(index, left, right))
        except BaseException as error:
            self._finish("error" if isinstance(error, Exception) else "interrupted", error)
            raise
        finally:
            self._busy = False

    def reset(self) -> None:
        self.seek(Fraction(0))

    def map_span(self, start: Fraction | int, end: Fraction | int) -> tuple[NativeConcatSpan, ...]:
        return self.timeline.map_span(start, end, max_parts=self._limits.max_span_parts)

    def close(self) -> None:
        self._check_owner()
        self._busy = True
        try:
            self._manual_closed = True
            self._finish("closed" if self._status in ("new", "open") else self._status)
        finally:
            self._busy = False

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, traceback: TracebackType | None
    ) -> None:
        self._check_owner()
        self._busy = True
        try:
            if exc is None:
                self._manual_closed = True
                self._finish("closed" if self._status in ("new", "open") else self._status)
            else:
                status = (
                    "interrupted"
                    if not isinstance(exc, Exception)
                    else self._status
                    if self._status.endswith("_limit")
                    else "error"
                )
                self._finish(status, exc)
        finally:
            self._active = False
            self._busy = False


__all__ = [
    "NativeConcatClip",
    "NativeConcatConfig",
    "NativeConcatDiagnostics",
    "NativeConcatFrame",
    "NativeConcatLimits",
    "NativeConcatSpan",
    "NativeConcatStream",
    "NativeConcatTimeline",
]
