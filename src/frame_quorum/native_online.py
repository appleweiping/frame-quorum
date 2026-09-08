"""Pull-driven native pixel decisions with bounded lookahead, not a full-source certificate."""

from __future__ import annotations

import hashlib
import threading
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from types import TracebackType
from typing import Any

from .errors import ConfigurationError, OutputError, ScanError
from .native_measurements import _bundle, _canonical, _video_dict
from .native_pixel_changes import (
    NativePixelChangeSample,
    PixelChangeDetectionConfig,
    PixelChangeStatistic,
    _PixelChangeMeasurer,
    _policy,
)
from .native_scenes import NativeScene, NativeSceneSample, _native_scene, _time
from .native_splitting import _target_path
from .native_video import (
    NativeVideoConfig,
    NativeVideoDiagnostics,
    NativeVideoMetadata,
    NativeVideoStatus,
    NativeVideoStream,
    _integer,
    _pair,
)
from .pixel_changes import PixelChangeConfig, PixelChangeLimits, _change_config, _change_limits
from .scene_detection import _AdaptiveWindow, _decision_at

_SUCCESS = frozenset(
    {
        NativeVideoStatus.EOF,
        NativeVideoStatus.RANGE_END,
        NativeVideoStatus.FRAME_LIMIT,
        NativeVideoStatus.DECODE_LIMIT,
    }
)
_PHASES = frozenset({"new", "open", "draining", "finished", "cancelled", "closed", "error", "interrupted"})


@dataclass(frozen=True, slots=True)
class NativeOnlineLimits:
    max_buffered_samples: int = 4096
    max_line_bytes: int = 16 * 1024
    max_output_bytes: int = 256 * 1024 * 1024

    def __post_init__(self) -> None:
        _integer(self.max_buffered_samples, "max_buffered_samples", 1, 65_536)
        _integer(self.max_line_bytes, "max_line_bytes", 1, 65_536)
        _integer(self.max_output_bytes, "max_output_bytes", 1, 1024 * 1024 * 1024)


@dataclass(frozen=True, slots=True)
class NativeOnlineDiagnostics:
    status: str
    observed_samples: int
    confirmed_samples: int
    measurement_pixels: int
    peak_buffer_slots: int
    video: NativeVideoDiagnostics

    def __post_init__(self) -> None:
        if type(self.status) is not str or self.status not in _PHASES:
            raise ConfigurationError("unknown online status")
        _integer(self.observed_samples, "observed_samples", 0, 1_000_000)
        _integer(self.confirmed_samples, "confirmed_samples", 0, self.observed_samples)
        _integer(self.measurement_pixels, "measurement_pixels", 0, 1_000_000_000)
        _integer(self.peak_buffer_slots, "peak_buffer_slots", 0, 65_536)
        if (
            type(self.video) is not NativeVideoDiagnostics
            or self.observed_samples > self.video.returned_frames
        ):
            raise ConfigurationError("online samples contradict native diagnostics")

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "observed_samples": self.observed_samples,
            "confirmed_samples": self.confirmed_samples,
            "measurement_pixels": self.measurement_pixels,
            "peak_buffer_slots": self.peak_buffer_slots,
            "video": self.video.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class NativePixelChangeUpdate:
    sample: NativePixelChangeSample
    statistic: PixelChangeStatistic
    observed_through_sample_index: int
    observed_through_time: Fraction
    closed_scene: NativeScene | None

    def __post_init__(self) -> None:
        if (
            type(self.sample) is not NativePixelChangeSample
            or type(self.statistic) is not PixelChangeStatistic
        ):
            raise ConfigurationError("online update requires typed pixel evidence and statistics")
        _integer(
            self.observed_through_sample_index,
            "observed_through_sample_index",
            self.sample.sample.sample_index,
            999_999,
        )
        _time(self.observed_through_time, "observed_through_time")
        if self.observed_through_time < self.sample.sample.presentation_time:
            raise ConfigurationError("online confirmation time precedes evidence")
        if self.statistic.accepted != (self.closed_scene is not None):
            raise ConfigurationError("accepted online cut must close exactly one scene")
        if self.closed_scene is not None and (
            type(self.closed_scene) is not NativeScene
            or self.closed_scene.end_position != self.sample.sample.sample_index
            or self.closed_scene.end_time != self.sample.sample.presentation_time
            or self.closed_scene.end_reason != "cut"
        ):
            raise ConfigurationError("closed scene contradicts its cut")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "sample",
            "sample": self.sample.sample.to_dict(),
            "width": self.sample.width,
            "height": self.sample.height,
            "change": None if self.sample.change is None else self.sample.change.to_dict(),
            "statistic": asdict(self.statistic),
            "observed_through_sample_index": self.observed_through_sample_index,
            "observed_through_time": _pair(self.observed_through_time),
            "closed_scene": None if self.closed_scene is None else self.closed_scene.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class NativePixelChangeEnd:
    diagnostics: NativeOnlineDiagnostics
    final_scene: NativeScene | None

    def __post_init__(self) -> None:
        if type(self.diagnostics) is not NativeOnlineDiagnostics:
            raise ConfigurationError("online end requires typed diagnostics")
        info = self.diagnostics
        if (
            info.status != "finished"
            or not info.video.closed
            or info.video.cleanup_errors
            or info.video.status not in _SUCCESS
            or info.confirmed_samples != info.observed_samples
            or info.video.returned_frames != info.observed_samples
        ):
            raise ConfigurationError("online end requires successful closed and fully finalized diagnostics")
        if (self.final_scene is None) != (info.observed_samples == 0):
            raise ConfigurationError("online final scene contradicts empty/nonempty stream")
        if self.final_scene is not None and (
            type(self.final_scene) is not NativeScene
            or self.final_scene.end_position != info.observed_samples
            or self.final_scene.end_reason == "cut"
        ):
            raise ConfigurationError("online final scene contradicts observed extent")

    @property
    def source_verified(self) -> bool:
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "end",
            "execution": "online_native_pixels",
            "source_verified": False,
            "diagnostics": self.diagnostics.to_dict(),
            "final_scene": None if self.final_scene is None else self.final_scene.to_dict(),
        }


@dataclass(slots=True)
class _Pending:
    sample: NativePixelChangeSample
    weighted: float
    score: float | None


class NativePixelChangeStream(Iterator[NativePixelChangeUpdate | NativePixelChangeEnd]):
    """Single-owner synchronous iterator. Break requires close/context exit; cancel is cooperative.

    Native calls are not preemptible. No RGB, scene list or statistics history is
    retained after delivery. Already delivered objects are caller-owned.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        video: NativeVideoConfig | None = None,
        measurement: PixelChangeConfig | None = None,
        detection: PixelChangeDetectionConfig | None = None,
        pixel_limits: PixelChangeLimits | None = None,
        limits: NativeOnlineLimits | None = None,
    ) -> None:
        self._video = NativeVideoConfig() if video is None else video
        self._measurement = _change_config(measurement)
        self._detection = PixelChangeDetectionConfig() if detection is None else detection
        self._pixel_limits = _change_limits(pixel_limits)
        self._limits = NativeOnlineLimits() if limits is None else limits
        if (
            type(self.video) is not NativeVideoConfig
            or type(self.detection) is not PixelChangeDetectionConfig
        ):
            raise ConfigurationError(
                "online detection requires native video and pixel detection configuration"
            )
        if type(self.limits) is not NativeOnlineLimits:
            raise ConfigurationError("online limits must be NativeOnlineLimits")
        radius = self.detection.window_radius if self.detection.detector == "adaptive" else 0
        self._confirmation_lag = max(radius, self.detection.min_scene_samples - 1)
        # Pending L+1; score ring 2r+1; eight fixed/overlap slots cover current
        # frame, measured row, returned update, scene start/last/latest and End.
        self._required_buffer_slots = (2 * radius + 1 if radius else 0) + self.confirmation_lag + 1 + 8
        if self.required_buffer_slots > self.limits.max_buffered_samples:
            raise ConfigurationError("online buffer capacity cannot cover configured confirmation lookahead")
        if 6 * self.video.max_frame_pixels > self.pixel_limits.max_pair_rgb_bytes:
            raise ConfigurationError("declared maximum two-frame RGB capacity exceeds pair byte limit")
        self._source = NativeVideoStream(path, self.video)
        self._measure = _PixelChangeMeasurer(self.measurement, self.pixel_limits)
        self._policy = _policy(self.detection)
        self._window = _AdaptiveWindow(radius) if radius else None
        self._pending: dict[int, _Pending] = {}
        self._observed = self._confirmed = self._previous_cut = self._ordinal = self._peak = 0
        self._last: NativeSceneSample | None = None
        self._start: NativeSceneSample | None = None
        self._latest: NativeSceneSample | None = None
        self._phase = "new"
        self._owner: int | None = None
        self._active = self._busy = False
        self._cancel = threading.Event()

    @property
    def video(self) -> NativeVideoConfig:
        return self._video

    @property
    def measurement(self) -> PixelChangeConfig:
        return self._measurement

    @property
    def detection(self) -> PixelChangeDetectionConfig:
        return self._detection

    @property
    def pixel_limits(self) -> PixelChangeLimits:
        return self._pixel_limits

    @property
    def limits(self) -> NativeOnlineLimits:
        return self._limits

    @property
    def confirmation_lag(self) -> int:
        return self._confirmation_lag

    @property
    def required_buffer_slots(self) -> int:
        return self._required_buffer_slots

    @property
    def metadata(self) -> NativeVideoMetadata:
        return self._source.metadata

    @property
    def diagnostics(self) -> NativeOnlineDiagnostics:
        return NativeOnlineDiagnostics(
            self._phase,
            self._observed,
            self._confirmed,
            self._measure.spent,
            self._peak,
            self._source.diagnostics,
        )

    def header(self) -> dict[str, Any]:
        return {
            "kind": "frame-quorum-native-pixel-change-events",
            "schema_version": 1,
            "measurement_version": "frame-quorum-circular-hsv-gradient-v1",
            "execution": "online_native_pixels",
            "source_verified": False,
            "metadata": self.metadata.to_dict(),
            "video": _video_dict(self.video),
            "measurement": asdict(self.measurement),
            "detection": asdict(self.detection),
            "pixel_limits": asdict(self.pixel_limits),
            "limits": asdict(self.limits),
            "confirmation_lag": self.confirmation_lag,
            "required_buffer_slots": self.required_buffer_slots,
        }

    def _check(self) -> None:
        if self._owner is not None and self._owner != threading.get_ident():
            raise ConfigurationError("online stream is single-owner")
        if self._busy:
            raise ConfigurationError("online stream is not reentrant")

    def __enter__(self) -> NativePixelChangeStream:
        self._check()
        if self._phase != "new" or self._cancel.is_set():
            raise ConfigurationError("online stream cannot be entered twice or after cancellation")
        self._owner = threading.get_ident()
        self._busy = True
        try:
            self._source.__enter__()
            self._phase, self._active = "open", True
        except BaseException as error:
            self._fail(error)
            raise
        finally:
            self._busy = False
        return self

    def __iter__(self) -> NativePixelChangeStream:
        self._check()
        if not self._active:
            raise ConfigurationError("online iteration requires an active with context")
        return self

    def __next__(self) -> NativePixelChangeUpdate | NativePixelChangeEnd:
        self._check()
        if not self._active:
            raise ConfigurationError("online iteration requires an active with context")
        self._busy = True
        try:
            while self._phase in {"open", "draining"}:
                if self._cancel.is_set():
                    self._stop("cancelled")
                    raise StopIteration
                if self._pending and (
                    self._phase == "draining" or self._confirmed < self._observed - self.confirmation_lag
                ):
                    return self._emit()
                if self._phase == "draining":
                    return self._end()
                try:
                    frame = next(self._source)
                except StopIteration:
                    self._source.close()
                    self._measure.close()
                    if (
                        self._source.diagnostics.status not in _SUCCESS
                        or self._source.diagnostics.cleanup_errors
                        or not self._source.diagnostics.closed
                    ):
                        raise ScanError("online source did not terminate successfully") from None
                    self._phase = "draining"
                    continue
                if self._cancel.is_set():
                    del frame
                    continue
                row = self._measure.measure(frame)
                del frame
                if row.sample.sample_index != self._observed or row.sample.generation != 0:
                    raise ScanError("online source changed sample identity or generation")
                self._observed += 1
                self._latest = row.sample
                change = row.change
                weighted = (
                    0.0
                    if change is None
                    else change.components[2]
                    if self.detection.value_only
                    else change.score(self.detection.weights)
                )
                self._pending[row.sample.sample_index] = _Pending(
                    row,
                    weighted,
                    weighted if self._window is None else None,
                )
                if self._window is not None:
                    result = self._window.push(weighted)
                    if result is not None:
                        index, score = result
                        self._pending[index].score = score
                slots = len(self._pending) + (len(self._window.values) if self._window is not None else 0) + 8
                self._peak = max(self._peak, slots)
                if slots > self.limits.max_buffered_samples:
                    raise ConfigurationError("online buffer limit exceeded")
            raise StopIteration
        except StopIteration:
            raise
        except BaseException as error:
            self._fail(error)
            raise
        finally:
            self._busy = False

    def _emit(self) -> NativePixelChangeUpdate:
        item = self._pending[self._confirmed]
        position, score = self._confirmed, item.score
        candidate = None
        if position > 0 and score is not None:
            if self.detection.detector == "adaptive":
                if score >= self.detection.adaptive_ratio and item.weighted >= self.detection.min_content:
                    candidate = "adaptive_peak"
            elif score >= self.detection.threshold:
                candidate = "distance_threshold"
        accepted, reason = _decision_at(
            position, score, candidate, self._previous_cut, self._observed, self._policy
        )
        statistic = PixelChangeStatistic(item.weighted, score, candidate is not None, accepted, reason)
        scene = None
        start = item.sample.sample if self._start is None else self._start
        if accepted:
            if self._last is None:
                raise ConfigurationError("online cut cannot precede its first scene")
            scene = _native_scene(
                self._ordinal,
                start,
                self._last,
                position,
                self.video,
                self._source.diagnostics,
                item.sample.sample.presentation_time,
            )
        latest = self._latest
        if latest is None:
            raise ConfigurationError("online update requires observed source evidence")
        result = NativePixelChangeUpdate(
            item.sample, statistic, latest.sample_index, latest.presentation_time, scene
        )
        if self._cancel.is_set():
            self._stop("cancelled")
            raise StopIteration
        del self._pending[position]
        self._confirmed += 1
        self._last = item.sample.sample
        self._start = item.sample.sample if accepted else start
        if accepted:
            self._previous_cut = position
            self._ordinal += 1
        return result

    def _end(self) -> NativePixelChangeEnd:
        scene = None
        if self._start is not None and self._last is not None:
            scene = _native_scene(
                self._ordinal, self._start, self._last, self._observed, self.video, self._source.diagnostics
            )
        info = NativeOnlineDiagnostics(
            "finished",
            self._observed,
            self._confirmed,
            self._measure.spent,
            self._peak,
            self._source.diagnostics,
        )
        result = NativePixelChangeEnd(info, scene)
        if self._cancel.is_set():
            self._stop("cancelled")
            raise StopIteration
        self._phase = "finished"
        self._release()
        return result

    def _release(self) -> None:
        self._measure.close()
        self._pending.clear()
        if self._window is not None:
            self._window.values.clear()
        self._last = self._start = self._latest = None

    def _stop(self, phase: str) -> None:
        self._phase = phase
        self._cleanup()

    def _fail(self, primary: BaseException) -> None:
        self._phase = "error" if isinstance(primary, Exception) else "interrupted"
        self._cleanup(primary)

    def _cleanup(self, primary: BaseException | None = None) -> None:
        failure = primary
        for cleanup in (self._source.close, self._release):
            try:
                cleanup()
            except BaseException as error:
                if failure is not None:
                    failure.add_note("online cleanup failed; inspect native diagnostics before reuse")
                if failure is None or (isinstance(failure, Exception) and not isinstance(error, Exception)):
                    failure = error
        if failure is not None and failure is not primary:
            if not isinstance(failure, Exception):
                self._phase = "interrupted"
            raise failure

    def cancel(self) -> None:
        """Request cancellation without touching owner-thread native resources or preempting calls."""
        self._cancel.set()

    def close(self) -> None:
        self._check()
        self._busy = True
        try:
            self._stop("closed" if self._phase in {"new", "open", "draining"} else self._phase)
        except BaseException as error:
            self._phase = "error" if isinstance(error, Exception) else "interrupted"
            raise
        finally:
            self._busy = False

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, traceback: TracebackType | None
    ) -> None:
        self._check()
        try:
            if exc is not None:
                self._busy = True
                try:
                    self._fail(exc)
                finally:
                    self._busy = False
            else:
                self.close()
        finally:
            self._active = False


def iter_native_pixel_change_jsonl(stream: NativePixelChangeStream) -> Iterator[bytes]:
    """Consume a fresh active stream as bounded canonical lines. The caller owns its context.

    Each line includes its newline. The final line hashes preceding bytes; it is
    emitted only after native cleanup and full line/total-byte admission. A
    missing final line always means incomplete output, never successful EOF.
    """
    if type(stream) is not NativePixelChangeStream:
        raise ConfigurationError("JSONL requires a NativePixelChangeStream")
    stream._check()
    if not stream._active or stream._phase != "open" or stream._observed:
        raise ConfigurationError("JSONL requires a fresh active unconsumed stream")
    digest = hashlib.sha256()
    total = 0

    def encode(value: dict[str, Any]) -> bytes:
        nonlocal total
        line = _canonical(value)
        if len(line) > stream.limits.max_line_bytes or total + len(line) > stream.limits.max_output_bytes:
            raise OutputError("online JSONL exceeds line or total output byte budget")
        total += len(line)
        digest.update(line)
        return line

    yield encode(stream.header())
    for expected, event in enumerate(stream):
        value = event.to_dict()
        if isinstance(event, NativePixelChangeEnd):
            if event.diagnostics.confirmed_samples != expected:
                raise ConfigurationError("JSONL stream was consumed outside its line iterator")
            value["sha256"] = digest.hexdigest()
            yield encode(value)
            return
        if event.sample.sample.sample_index != expected:
            raise ConfigurationError("JSONL stream was consumed outside its line iterator")
        yield encode(value)
    raise ScanError("online JSONL ended without successful finalization")


def write_native_pixel_change_stream(
    stream: NativePixelChangeStream,
    output_dir: str | Path,
) -> Path:
    """Own a new stream during no-replace publication; never silently publish cancelled prefixes."""
    if type(stream) is not NativePixelChangeStream:
        raise ConfigurationError("writer requires a NativePixelChangeStream")
    target = _target_path(output_dir)
    # The outer context guarantees immediate decoder cleanup if the file sink
    # fails while a line iterator is suspended. It does not depend on generator GC.
    with stream:
        return (
            _bundle(
                target,
                (("events.jsonl", iter_native_pixel_change_jsonl(stream)),),
                stream.limits.max_output_bytes,
            )
            / "events.jsonl"
        )


__all__ = [
    "NativeOnlineDiagnostics",
    "NativeOnlineLimits",
    "NativePixelChangeEnd",
    "NativePixelChangeStream",
    "NativePixelChangeUpdate",
    "iter_native_pixel_change_jsonl",
    "write_native_pixel_change_stream",
]
