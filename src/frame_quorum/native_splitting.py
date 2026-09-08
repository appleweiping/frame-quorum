"""Verified lossless local clip transcoding with exact rational presentation times."""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import stat
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from tempfile import mkdtemp
from typing import Any, BinaryIO

from .errors import ConfigurationError, OutputError, ScanError
from .native_scenes import NativeSceneResult
from .native_video import (
    _INT64,
    NativeVideoConfig,
    NativeVideoFrame,
    NativeVideoStatus,
    NativeVideoStream,
    _deny_secondary,
    _fraction,
    _integer,
    _load_av,
    _local_path_text,
    _pair,
)


@dataclass(frozen=True, slots=True)
class NativeClip:
    start: Fraction
    end: Fraction

    def __post_init__(self) -> None:
        object.__setattr__(self, "start", _fraction(self.start, "clip start"))
        object.__setattr__(self, "end", _fraction(self.end, "clip end"))
        if self.end <= self.start:
            raise ConfigurationError("clip end must be strictly greater than start")


@dataclass(frozen=True, slots=True)
class NativeSplitConfig:
    video_stream: int = 0
    max_clips: int = 100
    max_frames: int = 10_000
    max_decoded_frames: int = 100_000
    max_source_bytes: int = 1_000_000_000
    max_frame_pixels: int = 16_777_216
    max_source_pixels: int = 1_000_000_000
    max_verification_pixels: int = 1_000_000_000
    max_output_bytes: int = 1_000_000_000
    max_manifest_bytes: int = 16 * 1024 * 1024

    def __post_init__(self) -> None:
        for name, minimum, maximum in (
            ("video_stream", 0, 1023),
            ("max_clips", 1, 1000),
            ("max_frames", 1, 100_000),
            ("max_decoded_frames", 1, 1_000_000),
            ("max_source_bytes", 1, 1 << 40),
            ("max_frame_pixels", 1, 67_108_864),
            ("max_source_pixels", 1, 1 << 40),
            ("max_verification_pixels", 1, 1 << 40),
            ("max_output_bytes", 1, 1 << 40),
            ("max_manifest_bytes", 1, 64 * 1024 * 1024),
        ):
            _integer(getattr(self, name), name, minimum, maximum)


@dataclass(frozen=True, slots=True)
class NativeSplitResult:
    output_dir: Path
    clip_count: int
    frame_count: int
    total_output_bytes: int
    manifest_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.output_dir, Path):
            raise ConfigurationError("output_dir must be a pathlib.Path")
        _integer(self.clip_count, "clip_count", 1, 1000)
        _integer(self.frame_count, "frame_count", self.clip_count, 100_000)
        _integer(self.total_output_bytes, "total_output_bytes", 1, 1 << 40)
        if (
            type(self.manifest_sha256) is not str
            or len(self.manifest_sha256) != 64
            or any(char not in "0123456789abcdef" for char in self.manifest_sha256)
        ):
            raise ConfigurationError("manifest_sha256 must be a SHA-256 hex digest")

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_dir": self.output_dir.as_posix(),
            "clip_count": self.clip_count,
            "frame_count": self.frame_count,
            "total_output_bytes": self.total_output_bytes,
            "manifest_sha256": self.manifest_sha256,
        }


def native_scene_clips(
    result: NativeSceneResult, *, final_end: Fraction | None = None
) -> tuple[NativeClip, ...]:
    """Convert complete unsampled decisions; never infer an unknown final duration."""
    if type(result) is not NativeSceneResult:
        raise ConfigurationError("result must be NativeSceneResult")
    result.__post_init__()
    if result.config.video.frame_step != 1 or result.diagnostics.status not in (
        NativeVideoStatus.EOF,
        NativeVideoStatus.RANGE_END,
    ):
        raise ConfigurationError("scene splitting requires complete, unsampled analysis")
    if not result.scenes:
        raise ConfigurationError("scene result has no clips")
    tail = result.scenes[-1]
    if final_end is not None:
        final_end = _fraction(final_end, "final_end")
        if final_end <= tail.last_sample_time:
            raise ConfigurationError("final_end must be after the final sample")
        if result.config.video.end is not None and final_end > result.config.video.end:
            raise ConfigurationError("final_end exceeds the analyzed interval")
        if tail.end_time is not None and final_end != tail.end_time:
            raise ConfigurationError("final_end contradicts the known scene endpoint")
    endpoint = tail.end_time if tail.end_time is not None else final_end
    if endpoint is None:
        raise ConfigurationError("unknown final scene endpoint requires explicit final_end")
    intervals = []
    for index, scene in enumerate(result.scenes):
        end = endpoint if index == len(result.scenes) - 1 else scene.end_time
        if end is None:
            raise ConfigurationError("scene has an unknown interior endpoint")
        intervals.append(NativeClip(scene.start_time, end))
    return tuple(intervals)


class _ByteBudget:
    def __init__(self, maximum: int) -> None:
        self.maximum = maximum
        self.total = 0

    def grow(self, amount: int) -> None:
        if self.total + amount > self.maximum:
            raise OutputError("native split exceeds total output byte limit")
        self.total += amount


class _Writer:
    """Bound file growth before writes, including native headers and trailers."""

    def __init__(self, handle: BinaryIO, budget: _ByteBudget, maximum: int) -> None:
        self.handle, self.budget, self.maximum = handle, budget, maximum
        self.size = 0
        self.failed = False

    def write(self, data: bytes) -> int:
        try:
            extent = max(self.size, self.handle.tell() + len(data))
            if extent > self.maximum:
                raise OutputError("native split file exceeds byte limit")
            self.budget.grow(extent - self.size)
            self.size = extent
            written = self.handle.write(data)
            if written != len(data):
                raise OutputError("native split encountered a short output write")
            return written
        except BaseException:
            self.failed = True
            raise

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        if whence not in (os.SEEK_SET, os.SEEK_CUR, os.SEEK_END):
            self.failed = True
            raise OSError("unsupported native output seek")
        position = offset + (
            0 if whence == os.SEEK_SET else self.handle.tell() if whence == os.SEEK_CUR else self.size
        )
        if not 0 <= position <= self.maximum:
            self.failed = True
            raise OutputError("native output seek exceeds byte limit")
        return self.handle.seek(position)

    def tell(self) -> int:
        return self.handle.tell()

    def writable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True


def _close_all(resources: tuple[tuple[str, Any], ...], primary: BaseException | None = None) -> None:
    failures: list[tuple[str, BaseException]] = []
    for name, resource in resources:
        if resource is not None:
            try:
                resource.close()
            except BaseException as error:
                failures.append((name, error))
    if failures:
        detail = "native split resource cleanup failed: " + ", ".join(name for name, _ in failures)
        for _, failure in failures:
            if not isinstance(failure, Exception):
                failure.add_note(detail)
                raise failure
        if primary is None:
            raise OutputError(detail) from failures[0][1]
        primary.add_note(detail)


def _identity(info: os.stat_result) -> tuple[int, int]:
    return info.st_dev, info.st_ino


def _open_new(path: Path, owned: dict[Path, tuple[int, int]]) -> BinaryIO:
    handle = path.open("xb")
    try:
        owned[path] = _identity(os.fstat(handle.fileno()))
    except BaseException as error:
        _close_all((("file.close", handle),), error)
        raise
    return handle


@contextmanager
def _managed_file(path: Path, owned: dict[Path, tuple[int, int]] | None = None) -> Iterator[BinaryIO]:
    handle = path.open("rb") if owned is None else _open_new(path, owned)
    primary: BaseException | None = None
    try:
        yield handle
    except BaseException as error:
        primary = error
        raise
    finally:
        _close_all((("file.close", handle),), primary)


@dataclass(frozen=True, slots=True)
class _Expected:
    pts: int
    time_base: Fraction
    decode_index: int
    width: int
    height: int
    sha256: str

    @property
    def time(self) -> Fraction:
        return self.pts * self.time_base


class _Encoder:
    def __init__(
        self,
        path: Path,
        first: NativeVideoFrame,
        budget: _ByteBudget,
        av: Any,
        owned: dict[Path, tuple[int, int]] | None = None,
    ) -> None:
        self.path, self.av = path, av
        self.expected: list[_Expected] = []
        self.origin = first.presentation_time
        # AVRational components are signed 32-bit. Leave room for muxer rescaling.
        _integer(first.time_base.denominator, "output tick denominator", 1, 1_000_000_000)
        self.time_base = Fraction(1, first.time_base.denominator)
        self.width, self.height = first.width, first.height
        self.handle: BinaryIO | None = None
        self.container: Any = None
        self.stream: Any = None
        try:
            self.handle = _open_new(path, {} if owned is None else owned)
            self.writer = _Writer(self.handle, budget, budget.maximum)
            self.container = av.open(
                self.writer,
                "w",
                format="nut",
                io_open=_deny_secondary,
                options={"protocol_whitelist": "file", "avoid_negative_ts": "disabled"},
            )
            self.stream = self.container.add_stream("ffv1")
            self.stream.width, self.stream.height, self.stream.pix_fmt = first.width, first.height, "bgr0"
            self.stream.time_base = self.stream.codec_context.time_base = self.time_base
            self.stream.codec_context.thread_count = 1
        except BaseException as error:
            self.close(error)
            raise

    def write(self, frame: NativeVideoFrame) -> None:
        if (frame.width, frame.height) != (self.width, self.height):
            raise ScanError("native clip dimensions changed within one clip")
        if self.expected and frame.presentation_time <= self.expected[-1].time:
            raise ScanError("native clip requires strictly increasing selected timestamps")
        ticks = (frame.presentation_time - self.origin) / self.time_base
        if ticks.denominator != 1:
            raise ScanError("native frame timestamp is not representable in this clip's output tick")
        _integer(ticks.numerator, "rebased output pts", 0, _INT64)
        with frame.image() as image:
            encoded = self.av.VideoFrame.from_image(image)
        encoded.pts, encoded.time_base = ticks.numerator, self.time_base
        for packet in self.stream.encode(encoded):
            self.container.mux(packet)
        if self.writer.failed:
            raise OutputError("native output writer failed during encoding")
        self.expected.append(
            _Expected(
                frame.pts,
                frame.time_base,
                frame.decode_index,
                frame.width,
                frame.height,
                hashlib.sha256(frame.rgb).hexdigest(),
            )
        )

    def finish(self) -> None:
        primary: BaseException | None = None
        try:
            for packet in self.stream.encode():
                self.container.mux(packet)
        except BaseException as error:
            primary = error
            raise
        finally:
            self.close(primary)
        if self.writer.failed:
            raise OutputError("native output writer failed while finishing clip")

    def close(self, primary: BaseException | None = None) -> None:
        _close_all((("container.close", self.container), ("file.close", self.handle)), primary)


def _hash_file(path: Path, maximum: int) -> str:
    digest = hashlib.sha256()
    total = 0
    with _managed_file(path) as handle:
        while chunk := handle.read(min(1024 * 1024, maximum - total + 1)):
            total += len(chunk)
            if total > maximum:
                raise OutputError("staged clip exceeds output byte limit")
            digest.update(chunk)
    return digest.hexdigest()


def _verify(
    encoder: _Encoder, config: NativeSplitConfig, remaining_pixels: int
) -> tuple[list[dict[str, Any]], int]:
    if remaining_pixels <= 0:
        raise OutputError("native split verification pixel budget exhausted")
    expected = encoder.expected
    rows = []
    with NativeVideoStream(
        encoder.path,
        NativeVideoConfig(
            max_frames=len(expected) + 1,
            max_decoded_frames=len(expected) + 1,
            max_source_bytes=config.max_output_bytes,
            max_frame_pixels=config.max_frame_pixels,
            max_total_pixels=remaining_pixels,
        ),
    ) as verified:
        if verified.metadata.codec_name != "ffv1" or verified.metadata.format_name != "nut":
            raise OutputError("native output codec/container verification failed")
        for index, frame in enumerate(verified):
            if index >= len(expected):
                raise OutputError("native output has an extra frame")
            original = expected[index]
            if (
                frame.presentation_time != original.time - encoder.origin
                or (frame.width, frame.height) != (original.width, original.height)
                or hashlib.sha256(frame.rgb).hexdigest() != original.sha256
            ):
                raise OutputError("native output frame PTS, dimensions or RGB verification failed")
            rows.append(
                {
                    "source_pts": original.pts,
                    "source_time_base": _pair(original.time_base),
                    "source_decode_index": original.decode_index,
                    "output_pts": frame.pts,
                    "output_time_base": _pair(frame.time_base),
                    "width": frame.width,
                    "height": frame.height,
                    "rgb_sha256": original.sha256,
                }
            )
    if len(rows) != len(expected) or verified.diagnostics.status is not NativeVideoStatus.EOF:
        raise OutputError("native output frame count or completion verification failed")
    return rows, verified.diagnostics.decoded_pixels_observed


def _target_path(value: str | Path) -> Path:
    if not isinstance(value, (str, Path)):
        raise ConfigurationError("output directory must be a local path")
    _local_path_text(str(value))
    supplied = Path(value).expanduser()
    if supplied.is_symlink():
        raise OutputError("native split refuses a symbolic-link output directory")
    try:
        parent = supplied.parent.resolve(strict=True)
    except OSError as error:
        raise OutputError("native split needs an existing output parent") from error
    _local_path_text(str(parent))
    target = parent / supplied.name
    if not parent.is_dir() or target.exists() or target.is_symlink():
        raise OutputError("native split needs an existing parent and a new output directory")
    return target


def _publish(stage: Path, target: Path) -> None:
    if sys.platform == "win32":
        os.rename(stage, target)  # Windows fails if target exists, even an empty directory.
    elif sys.platform == "linux":
        library = ctypes.CDLL(None, use_errno=True)
        rename = getattr(library, "renameat2", None)
        if rename is None:
            raise OutputError("atomic no-replace directory publication is unavailable")
        rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        if rename(-100, os.fsencode(stage), -100, os.fsencode(target), 1) != 0:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error), str(target))
    else:
        raise OutputError("atomic no-replace publication supports only Windows and Linux")


def _cleanup(
    stage: Path,
    primary: BaseException,
    owned: dict[Path, tuple[int, int]],
    identity: tuple[int, int],
) -> None:
    errors: list[str] = []
    interrupts: list[BaseException] = []
    try:
        info = stage.lstat()
        if not stat.S_ISDIR(info.st_mode) or _identity(info) != identity:
            raise OSError("staging directory was replaced")
    except FileNotFoundError:
        return
    except BaseException as error:
        errors.append("validate staging directory identity")
        if not isinstance(error, Exception):
            interrupts.append(error)
    for path, expected_identity in () if errors else owned.items():
        try:
            info = path.lstat()
            if not stat.S_ISREG(info.st_mode) or _identity(info) != expected_identity:
                raise OSError("owned staging file was replaced")
            path.unlink()
        except FileNotFoundError:
            continue
        except BaseException as error:
            errors.append(path.name)
            if not isinstance(error, Exception):
                interrupts.append(error)
    try:
        # Recheck the owned directory after file cleanup. Unknown entries are
        # never enumerated/deleted; they make rmdir fail and remain for inspection.
        info = stage.lstat()
        if not stat.S_ISDIR(info.st_mode) or _identity(info) != identity:
            raise OSError("staging directory was replaced")
        stage.rmdir()
    except BaseException as error:
        errors.append("remove staging directory")
        if not isinstance(error, Exception):
            interrupts.append(error)
    if errors:
        detail = (
            f"native split cleanup incomplete; inspect owned staging directory {stage}: {', '.join(errors)}"
        )
        if interrupts:
            interrupts[0].add_note(detail)
            raise interrupts[0]
        if isinstance(primary, Exception):
            raise OutputError(detail) from primary
        primary.add_note(detail)


def split_native_video(
    source: str | Path,
    output_dir: str | Path,
    clips: tuple[NativeClip, ...],
    config: NativeSplitConfig | None = None,
) -> NativeSplitResult:
    options = NativeSplitConfig() if config is None else config
    if type(options) is not NativeSplitConfig:
        raise ConfigurationError("config must be NativeSplitConfig")
    options.__post_init__()
    if type(clips) is not tuple or not 1 <= len(clips) <= options.max_clips:
        raise ConfigurationError("clips must be a nonempty bounded tuple")
    for index, clip in enumerate(clips):
        if type(clip) is not NativeClip:
            raise ConfigurationError("clips must contain NativeClip values")
        clip.__post_init__()
        if index and clip.start < clips[index - 1].end:
            raise ConfigurationError("clips must be sorted and non-overlapping")
    target = _target_path(output_dir)
    budget = _ByteBudget(options.max_output_bytes)
    stage: Path | None = None
    active: _Encoder | None = None
    encoders: list[_Encoder] = []
    selected = 0
    publication_attempted = False
    owned: dict[Path, tuple[int, int]] = {}
    stage_identity: tuple[int, int] | None = None
    try:
        av = _load_av()
        av.codec.Codec("ffv1", "w")  # Fail before staging if this optional build has no encoder.
        with NativeVideoStream(
            source,
            NativeVideoConfig(
                video_stream=options.video_stream,
                end=clips[-1].end,
                max_frames=options.max_decoded_frames,
                max_decoded_frames=options.max_decoded_frames,
                max_source_bytes=options.max_source_bytes,
                max_frame_pixels=options.max_frame_pixels,
                max_total_pixels=options.max_source_pixels,
            ),
        ) as stream:
            metadata = stream.metadata
            stage = Path(mkdtemp(prefix=".frame-quorum-split-", dir=target.parent))
            stage_identity = _identity(stage.lstat())
            index = 0
            for frame in stream:
                while frame.presentation_time >= clips[index].end:
                    if active is None:
                        raise ScanError("requested native clip contains no frames")
                    active.finish()
                    active = None
                    index += 1
                if frame.presentation_time < clips[index].start:
                    continue
                if selected >= options.max_frames:
                    raise ScanError("native split exceeds retained frame/hash count limit")
                if active is None:
                    active = _Encoder(stage / f"clip-{index:06d}.nut", frame, budget, av, owned)
                    encoders.append(active)
                active.write(frame)
                selected += 1
            if stream.diagnostics.status not in (NativeVideoStatus.EOF, NativeVideoStatus.RANGE_END):
                raise ScanError("native split refuses truncated source decoding")
            if active is None or len(encoders) != len(clips):
                raise ScanError("requested native clip contains no frames")
            active.finish()
            active = None
        rows = []
        verification_pixels = 0
        for ordinal, (encoder, clip) in enumerate(zip(encoders, clips, strict=True)):
            frames, pixels = _verify(encoder, options, options.max_verification_pixels - verification_pixels)
            verification_pixels += pixels
            rows.append(
                {
                    "ordinal": ordinal,
                    "file": encoder.path.name,
                    "start": _pair(clip.start),
                    "end": _pair(clip.end),
                    "origin": _pair(encoder.origin),
                    "frame_count": len(frames),
                    "byte_size": encoder.writer.size,
                    "sha256": _hash_file(encoder.path, options.max_output_bytes),
                    "frames": frames,
                }
            )
        document = {
            "kind": "frame-quorum-native-split",
            "schema_version": 1,
            "source": metadata.to_dict(),
            "source_diagnostics": stream.diagnostics.to_dict(),
            "codec": "ffv1",
            "pixel_format": "bgr0",
            "container": "nut",
            "video_only": True,
            "rebasing": "first-selected-frame",
            "pyav_version": av.__version__,
            "native_libraries": {key: list(value) for key, value in av.library_versions.items()},
            "frame_count": selected,
            "clip_count": len(clips),
            "clips": rows,
            "verification_pixels": verification_pixels,
            "limits": asdict(options),
        }
        with _managed_file(stage / "manifest.json", owned) as handle:
            writer = _Writer(handle, budget, options.max_manifest_bytes)
            for text in json.JSONEncoder(ensure_ascii=True, allow_nan=False, sort_keys=True).iterencode(
                document
            ):
                writer.write(text.encode("utf-8"))
        result = NativeSplitResult(
            target,
            len(clips),
            selected,
            budget.total,
            _hash_file(stage / "manifest.json", options.max_manifest_bytes),
        )
        publication_attempted = True
        _publish(stage, target)
        stage = None  # No fallible cleanup operation follows successful publication.
        return result
    except BaseException as primary:
        if publication_attempted:
            primary.add_note(
                f"Publication did not acknowledge success; inspect destination {target}. "
                "A moved destination is never removed by failure cleanup."
            )
        cleanup_primary = primary
        try:
            if active is not None:
                active.close(primary)
        except BaseException as error:
            cleanup_primary = error
        if stage is not None:
            if stage_identity is None:
                cleanup_primary.add_note(f"Staging identity unavailable; inspect {stage}; not removed.")
            else:
                _cleanup(stage, cleanup_primary, owned, stage_identity)
        if cleanup_primary is not primary:
            raise cleanup_primary from primary
        if isinstance(primary, (ConfigurationError, ScanError, OutputError)) or not isinstance(
            primary, Exception
        ):
            raise
        message = (
            f"native split publication did not acknowledge success; inspect {target}"
            if publication_attempted
            else "native video splitting failed before publication"
        )
        raise OutputError(message) from primary


__all__ = ["NativeClip", "NativeSplitConfig", "NativeSplitResult", "native_scene_clips", "split_native_video"]
