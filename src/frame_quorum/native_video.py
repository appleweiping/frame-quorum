"""Optional local-file PyAV decoding with exact PTS and explicit ownership.

Limits are checked at the Python/native boundary, not enforced by a codec
sandbox. No timestamps are synthesized from a nominal frame rate.
"""

from __future__ import annotations

import importlib
import os
import re
import stat
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import StrEnum
from fractions import Fraction
from pathlib import Path
from types import TracebackType
from typing import Any, BinaryIO

from PIL import Image

from .errors import ConfigurationError, ScanError
from .metrics import measure_image
from .models import FrameMetrics, _require_safe_text

_INT64 = 2**63 - 1


def _integer(value: Any, name: str, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ConfigurationError(f"{name} must be an integer between {minimum} and {maximum}")
    return value


def _fraction(value: Any, name: str, *, positive: bool = False) -> Fraction:
    if type(value) not in (int, Fraction):
        raise ConfigurationError(f"{name} must be an exact Fraction or integer, not a float")
    result = Fraction(value)
    if abs(result.numerator) > _INT64 or result.denominator > _INT64 or (positive and result <= 0):
        raise ConfigurationError(f"{name} is outside the supported rational range")
    return result


def _range(start: Fraction | None, end: Fraction | None) -> None:
    if start is not None:
        _fraction(start, "start")
    if end is not None:
        _fraction(end, "end")
    if start is not None and end is not None and end <= start:
        raise ConfigurationError("end must be strictly greater than start")


def _pair(value: Fraction | None) -> dict[str, int] | None:
    return None if value is None else {"numerator": value.numerator, "denominator": value.denominator}


class NativeVideoStatus(StrEnum):
    NEW = "new"
    OPEN = "open"
    EOF = "eof"
    RANGE_END = "range_end"
    FRAME_LIMIT = "frame_limit"
    DECODE_LIMIT = "decode_limit"
    PIXEL_LIMIT = "pixel_limit"
    SOURCE_LIMIT = "source_limit"
    CLOSED = "closed"
    INTERRUPTED = "interrupted"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class NativeVideoConfig:
    video_stream: int = 0
    start: Fraction | None = None
    end: Fraction | None = None
    frame_step: int = 1
    max_frames: int = 10_000
    max_decoded_frames: int = 100_000
    max_source_bytes: int = 1_000_000_000
    max_frame_pixels: int = 16_777_216
    max_total_pixels: int = 1_000_000_000

    def __post_init__(self) -> None:
        for name, minimum, maximum in (
            ("video_stream", 0, 1_023),
            ("frame_step", 1, 1_000_000),
            ("max_frames", 1, 1_000_000),
            ("max_decoded_frames", 1, 10_000_000),
            ("max_source_bytes", 1, 1 << 40),
            ("max_frame_pixels", 1, 67_108_864),
            ("max_total_pixels", 1, 1 << 40),
        ):
            _integer(getattr(self, name), name, minimum, maximum)
        _range(self.start, self.end)


@dataclass(frozen=True, slots=True)
class NativeVideoMetadata:
    path: Path
    source_bytes: int
    format_name: str
    codec_name: str
    stream_index: int
    width: int
    height: int
    time_base: Fraction
    start_pts: int | None
    duration_pts: int | None
    average_rate: Fraction | None
    base_rate: Fraction | None

    def __post_init__(self) -> None:
        if not isinstance(self.path, Path):
            raise ConfigurationError("metadata.path must be a pathlib.Path")
        _integer(self.source_bytes, "source_bytes", 0, 1 << 40)
        _integer(self.stream_index, "stream_index", 0, _INT64)
        for name in ("format_name", "codec_name"):
            value = getattr(self, name)
            if type(value) is not str:
                raise ConfigurationError(f"metadata.{name} must be text")
            _require_safe_text(value, name)
        _integer(self.width, "width", 1, 67_108_864)
        _integer(self.height, "height", 1, 67_108_864)
        if self.width * self.height > 67_108_864:
            raise ConfigurationError("metadata exceeds the compiled pixel ceiling")
        object.__setattr__(self, "time_base", _fraction(self.time_base, "time_base", positive=True))
        for name in ("average_rate", "base_rate"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _fraction(value, name, positive=True))
        if self.start_pts is not None:
            _integer(self.start_pts, "start_pts", -_INT64 - 1, _INT64)
        if self.duration_pts is not None:
            _integer(self.duration_pts, "duration_pts", 0, _INT64)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path.as_posix(),
            "source_bytes": self.source_bytes,
            "format": self.format_name,
            "codec": self.codec_name,
            "stream_index": self.stream_index,
            "width": self.width,
            "height": self.height,
            "time_base": _pair(self.time_base),
            "start_pts": self.start_pts,
            "duration_pts": self.duration_pts,
            "average_rate": _pair(self.average_rate),
            "base_rate": _pair(self.base_rate),
        }


@dataclass(frozen=True, slots=True)
class NativeVideoFrame:
    """Owned immutable RGB bytes. ``image()`` returns a new caller-owned Pillow image.

    decode_index is local to one seek generation and includes pre-range/stride
    discarded frames. sample_index is lifetime-wide. Neither is a global video
    frame number after seeking. pts/time_base remain the native decoded values.
    """

    pts: int
    time_base: Fraction
    decode_index: int
    sample_index: int
    generation: int
    width: int
    height: int
    rgb: bytes = field(repr=False)

    def __post_init__(self) -> None:
        _integer(self.pts, "pts", -_INT64 - 1, _INT64)
        object.__setattr__(self, "time_base", _fraction(self.time_base, "time_base", positive=True))
        for name in ("decode_index", "sample_index", "generation"):
            _integer(getattr(self, name), name, 0, _INT64)
        _integer(self.width, "width", 1, 67_108_864)
        _integer(self.height, "height", 1, 67_108_864)
        if self.width * self.height > 67_108_864:
            raise ConfigurationError("frame exceeds the compiled pixel ceiling")
        if type(self.rgb) is not bytes or len(self.rgb) != self.width * self.height * 3:
            raise ConfigurationError("rgb must be an exact owned RGB byte snapshot")

    @property
    def presentation_time(self) -> Fraction:
        return self.pts * self.time_base

    def image(self) -> Image.Image:
        return Image.frombytes("RGB", (self.width, self.height), self.rgb)

    def measure(self) -> FrameMetrics:
        with self.image() as image:
            return measure_image(image)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pts": self.pts,
            "time_base": _pair(self.time_base),
            "presentation_time": _pair(self.presentation_time),
            "decode_index": self.decode_index,
            "sample_index": self.sample_index,
            "generation": self.generation,
            "width": self.width,
            "height": self.height,
        }


@dataclass(frozen=True, slots=True)
class NativeVideoDiagnostics:
    status: NativeVideoStatus
    decoded_frames: int
    returned_frames: int
    decoded_pixels_observed: int
    generation: int
    closed: bool
    cleanup_errors: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.status) is not NativeVideoStatus or type(self.closed) is not bool:
            raise ConfigurationError("diagnostics require a NativeVideoStatus and boolean closed flag")
        for name in ("decoded_frames", "returned_frames", "decoded_pixels_observed", "generation"):
            _integer(getattr(self, name), name, 0, _INT64)
        if self.returned_frames > self.decoded_frames:
            raise ConfigurationError("returned frame count cannot exceed decoded frame count")
        if (
            type(self.cleanup_errors) is not tuple
            or len(self.cleanup_errors) > 3
            or any(
                type(value) is not str or value not in ("frames.close", "container.close", "file.close")
                for value in self.cleanup_errors
            )
            or len(set(self.cleanup_errors)) != len(self.cleanup_errors)
        ):
            raise ConfigurationError("cleanup_errors must be a unique immutable tuple of resource names")

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "decoded_frames": self.decoded_frames,
            "returned_frames": self.returned_frames,
            "decoded_pixels_observed": self.decoded_pixels_observed,
            "generation": self.generation,
            "closed": self.closed,
            "cleanup_errors": list(self.cleanup_errors),
        }


def _load_av() -> Any:
    try:
        return importlib.import_module("av")
    except ImportError as exc:
        raise ConfigurationError("native video requires the optional frame-quorum[video] extra") from exc


def _deny_secondary(url: str, flags: int, options: dict[str, Any]) -> BinaryIO:
    raise PermissionError("secondary native video file/protocol opens are forbidden")


def _source_path(value: str | Path) -> Path:
    if not isinstance(value, (str, Path)):
        raise ConfigurationError("native video source must be a local file path")
    _local_path_text(str(value))
    supplied = Path(value).expanduser()
    if supplied.is_symlink():
        raise ScanError("native video refuses a symbolic-link input")
    resolved = supplied.resolve()
    # Relative paths/ancestor links can resolve into a forbidden namespace even
    # when the original spelling was local. Check before native/file setup.
    _local_path_text(str(resolved))
    return resolved


def _local_path_text(text: str) -> None:
    _require_safe_text(text, "native video source")
    # Permit a Windows absolute drive prefix, but no URLs, UNC paths or device names.
    remainder = text[2:] if re.match(r"^[A-Za-z]:[\\/]", text) else text
    if ":" in remainder or text.startswith(("\\\\", "//")):
        raise ConfigurationError("native video accepts local files only, not protocols or UNC paths")


def _format(header: bytes) -> str:
    if header.startswith(b"nut/multimedia container\x00"):
        return "nut"
    if header.startswith(b"\x1aE\xdf\xa3"):
        return "matroska"
    if header[:4] == b"RIFF" and header[8:12] == b"AVI ":
        return "avi"
    if header[4:8] == b"ftyp":
        return "mov"
    raise ScanError("native video supports only NUT, Matroska/WebM, AVI and ftyp-stamped MP4/MOV files")


class _LocalReader:
    """Keep FFmpeg on the already-open bounded file, rejecting observed mutation."""

    def __init__(self, handle: BinaryIO, fingerprint: tuple[int, int, int, int]) -> None:
        self.handle = handle
        self.fingerprint = fingerprint

    def read(self, size: int = -1) -> bytes:
        info = os.fstat(self.handle.fileno())
        if (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns) != self.fingerprint:
            raise OSError("native video source changed during reading")
        remaining = max(0, self.fingerprint[2] - self.handle.tell())
        return self.handle.read(remaining if size < 0 else min(size, remaining))

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        return self.handle.seek(offset, whence)

    def tell(self) -> int:
        return self.handle.tell()

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True


class NativeVideoStream(Iterator[NativeVideoFrame]):
    """One context-managed local decoder; lifetime budgets survive seeks/replay.

    Read ownership belongs to the thread entering the context. Closing early
    requires ``close()`` or exiting ``with``; breaking iteration alone does not
    close an otherwise live stream. No native decoder is exposed to callers.
    """

    def __init__(self, path: str | Path, config: NativeVideoConfig | None = None) -> None:
        self.path = _source_path(path)
        self.config = NativeVideoConfig() if config is None else config
        if type(self.config) is not NativeVideoConfig:
            raise ConfigurationError("config must be NativeVideoConfig")
        self._file: BinaryIO | None = None
        self._container: Any = None
        self._stream: Any = None
        self._frames: Any = None
        self._metadata: NativeVideoMetadata | None = None
        self._fingerprint: tuple[int, int, int, int] | None = None
        self._owner: int | None = None
        self._active = False
        self._manual_closed = False
        self._busy = False
        self._status = NativeVideoStatus.NEW
        self._cleanup_errors: tuple[str, ...] = ()
        self._decoded = self._returned = self._pixels = self._generation = 0
        self._decode_index = self._eligible = 0
        self._last_time: Fraction | None = None
        self._start, self._end = self.config.start, self.config.end

    @property
    def metadata(self) -> NativeVideoMetadata:
        if self._metadata is None:
            raise ConfigurationError("native video metadata is available only after successful setup")
        return self._metadata

    @property
    def diagnostics(self) -> NativeVideoDiagnostics:
        return NativeVideoDiagnostics(
            self._status,
            self._decoded,
            self._returned,
            self._pixels,
            self._generation,
            self._frames is None and self._container is None and self._file is None,
            self._cleanup_errors,
        )

    def _check_owner(self) -> None:
        if self._owner is not None and self._owner != threading.get_ident():
            raise ConfigurationError("native video stream is single-owner, not thread-safe")
        if self._busy:
            raise ConfigurationError("native video stream is not reentrant")

    def __enter__(self) -> NativeVideoStream:
        if self._status is not NativeVideoStatus.NEW:
            raise ConfigurationError("native video stream contexts cannot be entered twice")
        self._owner = threading.get_ident()
        self._active = True
        try:
            self._open()
        except BaseException:
            self._active = False
            raise
        return self

    def _open(self) -> None:
        try:
            av = _load_av()
            if self.path.is_symlink() or not self.path.is_file():
                raise ScanError("native video source must be a regular nonsymlink file")
            self._file = self.path.open("rb")
            info = os.fstat(self._file.fileno())
            if not stat.S_ISREG(info.st_mode):
                raise ScanError("native video source must be a regular file")
            if info.st_size > self.config.max_source_bytes:
                self._status = NativeVideoStatus.SOURCE_LIMIT
                raise ScanError("native video source exceeds byte limit")
            fingerprint = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
            if self._fingerprint is not None and fingerprint != self._fingerprint:
                raise ScanError("native video source changed before seek/replay")
            self._fingerprint = fingerprint
            format_name = _format(self._file.read(32))
            self._file.seek(0)
            self._container = av.open(
                _LocalReader(self._file, fingerprint),
                mode="r",
                format=format_name,
                io_open=_deny_secondary,
                options={"protocol_whitelist": "file", "enable_drefs": "0", "use_absolute_path": "0"},
            )
            streams = self._container.streams.video
            if self.config.video_stream >= len(streams):
                raise ScanError("requested video stream does not exist")
            self._stream = streams[self.config.video_stream]
            codec = self._stream.codec_context
            codec.thread_count = 1
            self._dimensions(codec.width, codec.height)
            time_base = _fraction(self._stream.time_base, "native stream time_base", positive=True)
            start_pts, duration_pts = self._stream.start_time, self._stream.duration
            for value, name in ((start_pts, "start_pts"), (duration_pts, "duration_pts")):
                if value is not None:
                    _integer(value, name, -_INT64 - 1, _INT64)
            average = self._stream.average_rate
            base = self._stream.base_rate
            self._metadata = NativeVideoMetadata(
                self.path,
                info.st_size,
                format_name,
                codec.name,
                self._stream.index,
                codec.width,
                codec.height,
                time_base,
                start_pts,
                duration_pts,
                None if average is None else _fraction(average, "average_rate", positive=True),
                None if base is None else _fraction(base, "base_rate", positive=True),
            )
            if self._start is not None:
                offset = self._start // time_base
                _integer(offset, "seek offset", -_INT64 - 1, _INT64)
                self._container.seek(offset, stream=self._stream, backward=True, any_frame=False)
            self._frames = iter(self._container.decode(video=self.config.video_stream))
            self._status = NativeVideoStatus.OPEN
        except BaseException as exc:
            self._finish(
                self._status
                if self._status in (NativeVideoStatus.SOURCE_LIMIT, NativeVideoStatus.PIXEL_LIMIT)
                else NativeVideoStatus.ERROR
                if isinstance(exc, Exception)
                else NativeVideoStatus.INTERRUPTED,
                suppress_errors=True,
            )
            if isinstance(exc, (ConfigurationError, ScanError)) or not isinstance(exc, Exception):
                raise
            raise ScanError("native video setup failed") from exc

    def _dimensions(self, width: Any, height: Any) -> int:
        if type(width) is not int or type(height) is not int or width <= 0 or height <= 0:
            raise ScanError("native video has invalid or unknown frame dimensions")
        pixels = width * height
        if pixels > self.config.max_frame_pixels:
            self._status = NativeVideoStatus.PIXEL_LIMIT
            raise ScanError("native video frame exceeds pixel limit")
        return pixels

    def __iter__(self) -> NativeVideoStream:
        self._check_owner()
        if not self._active:
            raise ConfigurationError("native video iteration requires an active with context")
        return self

    def __next__(self) -> NativeVideoFrame:
        self._check_owner()
        if not self._active:
            raise ConfigurationError("native video iteration requires an active with context")
        if self._status is not NativeVideoStatus.OPEN:
            raise StopIteration
        self._busy = True
        try:
            return self._next_frame()
        except StopIteration:
            raise
        except BaseException as exc:
            status = (
                NativeVideoStatus.PIXEL_LIMIT
                if self.diagnostics.status is NativeVideoStatus.PIXEL_LIMIT
                else (
                    NativeVideoStatus.ERROR if isinstance(exc, Exception) else NativeVideoStatus.INTERRUPTED
                )
            )
            self._finish(status, suppress_errors=True)
            if isinstance(exc, (ConfigurationError, ScanError)) or not isinstance(exc, Exception):
                raise
            raise ScanError("native video decode failed") from exc
        finally:
            self._busy = False

    def _next_frame(self) -> NativeVideoFrame:
        while True:
            if self._decoded >= self.config.max_decoded_frames:
                self._finish(NativeVideoStatus.DECODE_LIMIT)
                raise StopIteration
            try:
                frame = next(self._frames)
            except StopIteration:
                self._finish(NativeVideoStatus.EOF)
                raise
            index = self._decode_index
            self._decode_index += 1
            self._decoded += 1
            pixels = self._dimensions(frame.width, frame.height)
            self._pixels += pixels
            if self._pixels > self.config.max_total_pixels:
                self._status = NativeVideoStatus.PIXEL_LIMIT
                raise ScanError("native video observed decoded pixels exceed total limit")
            if frame.pts is None or frame.time_base is None:
                raise ScanError("native video frame lacks PTS/time_base; timestamps are never synthesized")
            pts = _integer(frame.pts, "native frame pts", -_INT64 - 1, _INT64)
            time_base = _fraction(frame.time_base, "native frame time_base", positive=True)
            instant = pts * time_base
            if self._last_time is not None and instant < self._last_time:
                raise ScanError("native video presentation timestamps decreased")
            self._last_time = instant
            if self._end is not None and instant >= self._end:
                self._finish(NativeVideoStatus.RANGE_END)
                raise StopIteration
            if self._start is not None and instant < self._start:
                continue
            eligible = self._eligible
            self._eligible += 1
            if eligible % self.config.frame_step:
                continue
            with frame.to_image() as image, image.convert("RGB") as rgb:
                snapshot = rgb.tobytes()
            result = NativeVideoFrame(
                pts,
                time_base,
                index,
                self._returned,
                self._generation,
                frame.width,
                frame.height,
                snapshot,
            )
            self._returned += 1
            if self._returned >= self.config.max_frames:
                self._finish(NativeVideoStatus.FRAME_LIMIT)
            elif self._decoded >= self.config.max_decoded_frames:
                self._finish(NativeVideoStatus.DECODE_LIMIT)
            return result

    def seek(self, start: Fraction | None, *, end: Fraction | None = None) -> None:
        """Start a new decode generation; None replays from the beginning.

        Lifetime work/output budgets are not reset. Native keyframe seek is
        followed by exact presentation-time filtering, never index estimation.
        """
        self._check_owner()
        if (
            not self._active
            or self._manual_closed
            or self._status
            in (
                NativeVideoStatus.CLOSED,
                NativeVideoStatus.ERROR,
                NativeVideoStatus.INTERRUPTED,
                NativeVideoStatus.SOURCE_LIMIT,
                NativeVideoStatus.PIXEL_LIMIT,
            )
        ):
            raise ConfigurationError("cannot seek a closed, failed or inactive stream")
        _range(start, end)
        if self._decoded >= self.config.max_decoded_frames or self._returned >= self.config.max_frames:
            raise ScanError("native video lifetime frame budget is exhausted")
        self._finish(NativeVideoStatus.CLOSED)
        self._start, self._end = start, end
        self._generation += 1
        self._decode_index = self._eligible = 0
        self._last_time = None
        self._open()

    def _finish(self, status: NativeVideoStatus, *, suppress_errors: bool = False) -> None:
        self._status = status
        failures: list[tuple[str, BaseException]] = []
        # Attempt every resource even if an earlier close failed. Keep failed
        # references so closure remains unknown and the owner can retry close().
        for attribute, name in (
            ("_frames", "frames.close"),
            ("_container", "container.close"),
            ("_file", "file.close"),
        ):
            resource = getattr(self, attribute)
            if resource is None:
                continue
            try:
                close = getattr(resource, "close", None)
                if close is not None:
                    close()
            except BaseException as exc:
                failures.append((name, exc))
            else:
                setattr(self, attribute, None)
                if attribute == "_container":
                    self._stream = None
        if failures:
            self._cleanup_errors = tuple(
                dict.fromkeys((*self._cleanup_errors, *(name for name, _ in failures)))
            )
            interrupts = [error for _, error in failures if not isinstance(error, Exception)]
            self._status = (
                NativeVideoStatus.INTERRUPTED
                if interrupts or status is NativeVideoStatus.INTERRUPTED
                else NativeVideoStatus.ERROR
            )
            if interrupts:
                raise interrupts[0]
            if not suppress_errors:
                raise ScanError(
                    "native video cleanup failed: " + ", ".join(name for name, _ in failures)
                ) from failures[0][1]

    def close(self) -> None:
        self._check_owner()
        self._manual_closed = True
        self._finish(
            NativeVideoStatus.CLOSED
            if self._status in (NativeVideoStatus.NEW, NativeVideoStatus.OPEN)
            else self._status
        )

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, traceback: TracebackType | None
    ) -> None:
        self._check_owner()
        try:
            if exc_type is not None:
                self._finish(
                    NativeVideoStatus.INTERRUPTED if self._status is NativeVideoStatus.OPEN else self._status,
                    suppress_errors=True,
                )
            else:
                self.close()
        finally:
            self._active = False


__all__ = [
    "NativeVideoConfig",
    "NativeVideoDiagnostics",
    "NativeVideoFrame",
    "NativeVideoMetadata",
    "NativeVideoStatus",
    "NativeVideoStream",
]
