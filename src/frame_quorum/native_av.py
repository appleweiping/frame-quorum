"""Bounded PCM16 decoding and exact audio-sample lattice operations."""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from types import TracebackType
from typing import Any, BinaryIO

from .errors import ConfigurationError, ScanError
from .native_splitting import NativeSplitConfig, _close_all
from .native_video import _INT64, _format, _fraction, _integer, _open_native_container, _source_path


@dataclass(frozen=True, slots=True)
class NativeAVSplitConfig(NativeSplitConfig):
    audio_stream: int = 0
    max_streams: int = 32
    max_audio_blocks: int = 100_000
    max_audio_block_samples: int = 1_048_576
    max_source_audio_samples: int = 48_000_000
    max_audio_samples: int = 48_000_000
    max_verification_audio_samples: int = 48_000_000
    max_verification_audio_blocks: int = 100_000
    max_pending_packet_bytes: int = 64 * 1024 * 1024

    def __post_init__(self) -> None:
        NativeSplitConfig.__post_init__(self)
        for name, minimum, maximum in (
            ("audio_stream", 0, 1023),
            ("max_streams", 2, 1024),
            ("max_audio_blocks", 1, 1_000_000),
            ("max_audio_block_samples", 1, 4_194_304),
            ("max_source_audio_samples", 1, 1 << 36),
            ("max_audio_samples", 1, 1 << 36),
            ("max_verification_audio_samples", 1, 1 << 36),
            ("max_verification_audio_blocks", 1, 1_000_000),
            ("max_pending_packet_bytes", 1, 256 * 1024 * 1024),
        ):
            _integer(getattr(self, name), name, minimum, maximum)


def _audio_epoch(start: Fraction, anchor: Fraction, rate: int) -> Fraction:
    return anchor + ((start - anchor) * rate // 1) / Fraction(rate)


def _ceil(value: Fraction) -> int:
    return -(-value.numerator // value.denominator)


def _sample_slice(
    start: Fraction, rate: int, samples: int, clip_start: Fraction, clip_end: Fraction
) -> tuple[int, int]:
    return (
        min(samples, max(0, _ceil((clip_start - start) * rate))),
        min(samples, max(0, _ceil((clip_end - start) * rate))),
    )


@dataclass(frozen=True, slots=True)
class _AudioBlock:
    start: Fraction
    rate: int
    channels: int
    samples: int
    pcm: bytes

    @property
    def end(self) -> Fraction:
        return self.start + Fraction(self.samples, self.rate)

    def trim(self, start: Fraction, end: Fraction, *, maximum: int) -> _AudioBlock | None:
        left, right = _sample_slice(self.start, self.rate, self.samples, start, end)
        if left == right:
            return None
        if right - left > maximum:
            raise ScanError("native AV selected audio sample limit exceeded")
        width = self.channels * 2
        return _AudioBlock(
            self.start + Fraction(left, self.rate),
            self.rate,
            self.channels,
            right - left,
            self.pcm[left * width : right * width],
        )


def _audio_shape(frame: Any, maximum: int) -> tuple[int, int, int, Fraction]:
    samples = _integer(frame.samples, "audio block samples", 1, maximum)
    rate = _integer(frame.sample_rate, "audio sample rate", 8000, 192000)
    channels = len(frame.layout.channels)
    if (channels, frame.layout.name) not in (
        (1, "mono"),
        (2, "stereo"),
        (1, "1 channels"),
        (2, "2 channels"),
    ):
        raise ScanError("native audio supports only mono or stereo")
    if frame.format.name not in ("s16", "s16p"):
        raise ScanError("native audio requires decoded PCM16 without conversion or resampling")
    pts = _integer(frame.pts, "audio pts", -_INT64 - 1, _INT64)
    time_base = _fraction(frame.time_base, "audio time base", positive=True)
    return samples, rate, channels, pts * time_base


def _pcm_bytes(frame: Any, samples: int, channels: int) -> bytes:
    """Copy only the admitted samples, never native padding or an entire arbitrary plane."""
    planar = frame.format.name == "s16p"
    if len(frame.planes) != (channels if planar else 1):
        raise ScanError("native PCM16 plane count is inconsistent")
    needed = samples * 2 * (1 if planar else channels)
    planes: list[memoryview] = []
    try:
        for plane in frame.planes:
            planes.append(memoryview(plane))
        if any(len(plane) < needed for plane in planes):
            raise ScanError("native PCM16 plane is shorter than its admitted samples")
        if not planar or channels == 1:
            with planes[0][:needed] as selected:
                return bytes(selected)
        output = bytearray(samples * channels * 2)
        for index in range(samples):
            with planes[0][index * 2 : index * 2 + 2] as left, planes[1][index * 2 : index * 2 + 2] as right:
                output[index * 4 : index * 4 + 2] = left
                output[index * 4 + 2 : index * 4 + 4] = right
        return bytes(output)
    finally:
        for plane in reversed(planes):
            plane.release()


class _AudioReader:
    """Private single-pass audio decoder. The enclosing operation owns its lifetime."""

    def __init__(self, path: str | Path, config: NativeAVSplitConfig, av: Any) -> None:
        self.path, self.config, self.av = _source_path(path), config, av
        self.handle: BinaryIO | None = None
        self.container: Any = None
        self.frames: Any = None
        self.fingerprint: tuple[int, int, int, int] | None = None
        self.anchor: Fraction | None = None
        self.rate = self.channels = self.samples = self.blocks = 0
        self.status = "new"
        self.stream_index = -1

    def __enter__(self) -> _AudioReader:
        try:
            if self.path.is_symlink() or not self.path.is_file():
                raise ScanError("native audio source must be a regular nonsymlink file")
            self.handle = self.path.open("rb")
            info = os.fstat(self.handle.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size > self.config.max_source_bytes:
                raise ScanError("native audio source is not regular or exceeds its byte limit")
            self.fingerprint = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
            format_name = _format(self.handle.read(32))
            self.handle.seek(0)
            self.container = _open_native_container(self.av, self.handle, self.fingerprint, format_name)
            if len(self.container.streams) > self.config.max_streams:
                raise ScanError("native audio container exceeds stream limit")
            audio = self.container.streams.audio
            if self.config.audio_stream >= len(audio):
                raise ScanError("requested audio stream does not exist")
            selected = audio[self.config.audio_stream]
            selected.codec_context.thread_count = 1
            self.stream_index = selected.index
            self.frames = iter(self.container.decode(audio=self.config.audio_stream))
            self.status = "open"
            return self
        except BaseException as error:
            self.status = "error"
            self.close(error)
            raise

    def next(self) -> _AudioBlock | None:
        if self.status == "eof":
            return None
        if self.status != "open":
            raise ConfigurationError("native audio reader is not open")
        if self.blocks >= self.config.max_audio_blocks:
            raise ScanError("native audio decoded block limit reached before EOF confirmation")
        try:
            frame = next(self.frames)
        except StopIteration:
            self.status = "eof"
            return None
        self.blocks += 1
        samples, rate, channels, start = _audio_shape(frame, self.config.max_audio_block_samples)
        if self.samples + samples > self.config.max_source_audio_samples:
            raise ScanError("native source audio sample limit exceeded")
        if self.anchor is None:
            self.anchor, self.rate, self.channels = start, rate, channels
        elif (
            rate != self.rate
            or channels != self.channels
            or start != self.anchor + Fraction(self.samples, self.rate)
        ):
            raise ScanError("native audio sample grid is discontinuous or format changed")
        self.samples += samples
        return _AudioBlock(start, rate, channels, samples, _pcm_bytes(frame, samples, channels))

    def close(self, primary: BaseException | None = None) -> None:
        frames, container, handle = self.frames, self.container, self.handle
        self.frames = self.container = self.handle = None
        if self.status == "open":
            self.status = "closed"
        check_error: BaseException | None = None
        if handle is not None and self.fingerprint is not None:
            try:
                current, named = os.fstat(handle.fileno()), self.path.lstat()
                if any(
                    (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns) != self.fingerprint
                    or not stat.S_ISREG(info.st_mode)
                    for info in (current, named)
                ):
                    raise ScanError("native audio source changed before closure")
            except BaseException as error:
                if primary is None or (isinstance(primary, Exception) and not isinstance(error, Exception)):
                    check_error = error
                else:
                    primary.add_note("native audio final source-identity check failed")
        closing_primary = primary if check_error is None else check_error
        try:
            _close_all(
                (
                    ("audio.iterator.close", frames),
                    ("audio.container.close", container),
                    ("audio.file.close", handle),
                ),
                closing_primary,
            )
        except BaseException as cleanup_error:
            if closing_primary is not None and not isinstance(closing_primary, Exception):
                closing_primary.add_note("native audio closure also raised; first control exception retained")
                if closing_primary is primary:
                    return
                raise closing_primary from cleanup_error
            raise
        if check_error is not None:
            raise check_error

    def __exit__(
        self, kind: type[BaseException] | None, value: BaseException | None, traceback: TracebackType | None
    ) -> None:
        self.close(value)


__all__ = ["NativeAVSplitConfig"]
