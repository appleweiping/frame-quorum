"""Optional, resource-bounded FFmpeg adapter for image-sequence extraction."""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import IO

from .errors import ConfigurationError, OutputError, ScanError
from .models import ScanConfig, _is_stable_json_number, _require_int64, _require_safe_text
from .reporting import write_json
from .scanner import scan_frames

_MAX_OUTPUT_BYTES = 1 << 40
_STDERR_RETAIN_BYTES = 16 * 1024
_VERSION_RETAIN_BYTES = 4 * 1024
_POLL_SECONDS = 0.02
_STOP_GRACE_SECONDS = 2.0


@dataclass(frozen=True, slots=True)
class VideoExtractionConfig:
    """Resource and sampling limits for an external FFmpeg process.

    ``max_output_bytes`` counts generated PNG bytes only; the small
    ``extraction.json`` provenance record is not part of that limit.
    ``timeout_seconds`` limits the FFmpeg subprocess, not input hashing,
    version probing, or validation after decoding.
    """

    frame_rate: float = 2.0
    max_frames: int = 10_000
    max_output_bytes: int = 1_000_000_000
    timeout_seconds: float = 300.0
    executable: str = "ffmpeg"

    def validate(self) -> None:
        if not _is_stable_json_number(self.frame_rate) or self.frame_rate <= 0:
            raise ConfigurationError("video extraction frame_rate must be greater than zero")
        _require_int64(self.max_frames, "video extraction max_frames", minimum=1)
        if self.max_frames > 1_000_000:
            raise ConfigurationError("video extraction max_frames cannot exceed 1000000")
        _require_int64(self.max_output_bytes, "video extraction max_output_bytes", minimum=1)
        if self.max_output_bytes > _MAX_OUTPUT_BYTES:
            raise ConfigurationError(f"video extraction max_output_bytes cannot exceed {_MAX_OUTPUT_BYTES}")
        if not _is_stable_json_number(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ConfigurationError("video extraction timeout_seconds must be greater than zero")
        if self.timeout_seconds > 86_400:
            raise ConfigurationError("video extraction timeout_seconds cannot exceed 86400")
        _require_safe_text(self.executable, "video extraction executable")


@dataclass(frozen=True, slots=True)
class VideoExtractionResult:
    """Summary of a completed, atomically published extraction.

    ``total_output_bytes`` and ``max_output_bytes`` both refer to generated
    PNG bytes and exclude ``extraction.json``.
    """

    output_dir: Path
    frame_count: int
    frame_rate: float
    max_frames: int
    total_output_bytes: int
    max_output_bytes: int
    input_sha256: str
    ffmpeg_version: str

    def validate(self) -> None:
        if not isinstance(self.output_dir, Path):
            raise ConfigurationError("video extraction output_dir must be pathlib.Path")
        _require_int64(self.frame_count, "video extraction frame_count", minimum=1)
        if not _is_stable_json_number(self.frame_rate) or self.frame_rate <= 0:
            raise ConfigurationError("video extraction frame_rate must be greater than zero")
        _require_int64(self.max_frames, "video extraction max_frames", minimum=1)
        if self.max_frames > 1_000_000:
            raise ConfigurationError("video extraction max_frames cannot exceed 1000000")
        if self.frame_count > self.max_frames:
            raise ConfigurationError("video extraction frame_count cannot exceed max_frames")
        _require_int64(self.total_output_bytes, "video extraction total_output_bytes", minimum=1)
        _require_int64(self.max_output_bytes, "video extraction max_output_bytes", minimum=1)
        if self.max_output_bytes > _MAX_OUTPUT_BYTES:
            raise ConfigurationError(f"video extraction max_output_bytes cannot exceed {_MAX_OUTPUT_BYTES}")
        if self.total_output_bytes > self.max_output_bytes:
            raise ConfigurationError("video extraction output cannot exceed max_output_bytes")
        _require_sha256(self.input_sha256, "video extraction input_sha256")
        _require_safe_text(self.ffmpeg_version, "video extraction FFmpeg version")


@dataclass(frozen=True, slots=True)
class _ProcessOutcome:
    total_output_bytes: int
    stderr_text: str
    stderr_truncated: bool


@dataclass(frozen=True, slots=True)
class _FileSnapshot:
    sha256: str
    byte_size: int
    device: int
    inode: int
    modified_ns: int


class _BoundedCollector:
    def __init__(self, retain_bytes: int) -> None:
        self._retain_bytes = retain_bytes
        self.buffer = bytearray()
        self.truncated = False
        self.error: OSError | None = None

    def drain(self, stream: IO[bytes] | None) -> None:
        if stream is None:
            return
        try:
            while chunk := stream.read(4096):
                remaining = self._retain_bytes - len(self.buffer)
                if remaining > 0:
                    self.buffer.extend(chunk[:remaining])
                if len(chunk) > remaining:
                    self.truncated = True
        except OSError as error:
            self.error = error

    def text(self) -> str:
        rendered = self.buffer.decode("utf-8", "replace")
        detail = _safe_process_detail(rendered)
        if self.truncated:
            detail = f"{detail} [diagnostic truncated]"
        return detail


def extract_video_frames(
    input_video: str | Path,
    output_dir: str | Path,
    config: VideoExtractionConfig | None = None,
) -> VideoExtractionResult:
    """Decode to a new directory while enforcing time, count, and byte limits."""

    options = VideoExtractionConfig() if config is None else config
    if not isinstance(options, VideoExtractionConfig):
        raise ConfigurationError("video extraction config must be VideoExtractionConfig")
    options.validate()

    supplied_source = Path(input_video).expanduser()
    supplied_target = Path(output_dir).expanduser()
    if supplied_source.is_symlink():
        raise ScanError(f"refusing to extract a symbolic-link video input: {supplied_source}")
    if supplied_target.is_symlink():
        raise OutputError(f"refusing to use a symbolic-link extraction output: {supplied_target}")
    source = supplied_source.resolve()
    target = supplied_target.resolve()
    if not source.exists() or not source.is_file():
        raise ScanError(f"input video is not a regular file: {source}")
    if target.exists():
        raise OutputError(f"video extraction output must not already exist: {target}")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise OutputError(f"cannot create video extraction parent {target.parent}: {error}") from error

    input_snapshot = _snapshot_file(source)
    ffmpeg_version = _probe_ffmpeg_version(options.executable)
    command_arguments = [
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-i",
        "<input>",
        "-vf",
        f"fps={float(options.frame_rate):.17g}",
        "-frames:v",
        str(options.max_frames),
        "-fs",
        str(options.max_output_bytes),
        "-start_number",
        "0",
        "<output>/frame_%09d.png",
    ]
    try:
        with TemporaryDirectory(prefix=".frame-quorum-extract-", dir=target.parent) as temporary:
            stage = Path(temporary) / "frames"
            stage.mkdir()
            pattern = stage / "frame_%09d.png"
            command = [
                options.executable,
                *command_arguments[:5],
                str(source),
                *command_arguments[6:-1],
                str(pattern),
            ]
            outcome = _execute_ffmpeg(command, stage, options)
            if supplied_source.is_symlink():
                raise ScanError("input video became a symbolic link while FFmpeg extraction was running")
            verified_snapshot = _snapshot_file(source)
            if verified_snapshot != input_snapshot:
                raise ScanError("input video changed while FFmpeg extraction was running")
            extracted = tuple(sorted(stage.glob("frame_*.png")))
            if not extracted:
                raise OutputError("FFmpeg completed without producing any frames")
            if len(extracted) > options.max_frames:
                raise OutputError("FFmpeg produced more frames than the configured maximum")
            expected_names = tuple(f"frame_{index:09d}.png" for index in range(len(extracted)))
            entries = tuple(sorted(stage.iterdir(), key=lambda path: path.name))
            if (
                tuple(path.name for path in extracted) != expected_names
                or entries != extracted
                or any(path.is_symlink() or not path.is_file() for path in extracted)
            ):
                raise OutputError("FFmpeg output must be one contiguous sequence of regular PNG files")
            try:
                scanned = scan_frames(
                    stage,
                    ScanConfig(frame_rate=options.frame_rate, extensions=(".png",)),
                )
            except ScanError as error:
                raise OutputError(f"FFmpeg produced an invalid image sequence: {error}") from error
            if len(scanned) != len(extracted):
                raise OutputError("FFmpeg output contains an unexpected image layout")
            total_output_bytes = sum(path.stat().st_size for path in extracted)
            if total_output_bytes != outcome.total_output_bytes:
                raise OutputError("FFmpeg output changed after resource monitoring completed")
            write_json(
                {
                    "schema_version": "1.1",
                    "kind": "frame-quorum-video-extraction",
                    "input": {
                        "sha256": input_snapshot.sha256,
                        "byte_size": input_snapshot.byte_size,
                    },
                    "ffmpeg": {
                        "executable": options.executable,
                        "version": ffmpeg_version,
                        "arguments": command_arguments,
                        "stderr_retained_bytes": _STDERR_RETAIN_BYTES,
                        "stderr_truncated": outcome.stderr_truncated,
                    },
                    "sampling": {
                        "frame_rate": options.frame_rate,
                        "filename_pattern": "frame_%09d.png",
                    },
                    "limits": {
                        "max_frames": options.max_frames,
                        "max_output_bytes": options.max_output_bytes,
                        "max_output_bytes_scope": "generated PNG files; extraction.json excluded",
                        "timeout_seconds": options.timeout_seconds,
                        "timeout_scope": "FFmpeg subprocess runtime only",
                    },
                    "result": {
                        "frame_count": len(extracted),
                        "total_output_bytes": total_output_bytes,
                    },
                },
                stage / "extraction.json",
            )
            if target.exists() or supplied_target.is_symlink():
                raise OutputError(f"video extraction output appeared during extraction: {target}")
            stage.replace(target)
    except OSError as error:
        raise OutputError(f"cannot publish extracted frames to {target}: {error}") from error

    result = VideoExtractionResult(
        target,
        len(extracted),
        options.frame_rate,
        options.max_frames,
        total_output_bytes,
        options.max_output_bytes,
        input_snapshot.sha256,
        ffmpeg_version,
    )
    result.validate()
    return result


def ffmpeg_available(executable: str = "ffmpeg") -> bool:
    """Return whether an FFmpeg executable can be found without running it."""

    _require_safe_text(executable, "video extraction executable")
    return shutil.which(executable) is not None


def _execute_ffmpeg(command: list[str], stage: Path, options: VideoExtractionConfig) -> _ProcessOutcome:
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as error:
        raise OutputError(f"FFmpeg executable was not found: {options.executable}") from error
    except OSError as error:
        raise OutputError(f"cannot start FFmpeg: {error}") from error

    collector = _BoundedCollector(_STDERR_RETAIN_BYTES)
    reader = threading.Thread(target=collector.drain, args=(process.stderr,), name="ffmpeg-stderr")
    reader.start()
    deadline = time.monotonic() + float(options.timeout_seconds)
    failure: OutputError | None = None
    returncode: int | None = None
    total_output_bytes = 0
    try:
        while True:
            total_output_bytes = _directory_output_bytes(stage, options.max_output_bytes)
            if total_output_bytes > options.max_output_bytes:
                failure = OutputError(
                    "FFmpeg output exceeded the configured max_output_bytes during decoding"
                )
                break
            returncode = process.poll()
            if returncode is not None:
                break
            if time.monotonic() >= deadline:
                failure = OutputError(
                    f"FFmpeg exceeded the {float(options.timeout_seconds):g} second timeout"
                )
                break
            time.sleep(_POLL_SECONDS)
    except OSError as error:
        failure = OutputError(f"cannot monitor FFmpeg output: {error}")
    finally:
        if failure is not None or process.poll() is None:
            _stop_process(process)
        else:
            process.wait()
        _finish_reader(process.stderr, reader)

    if collector.error is not None:
        raise OutputError(f"cannot read FFmpeg diagnostics: {collector.error}") from collector.error
    if failure is not None:
        raise failure
    if returncode != 0:
        raise OutputError(f"FFmpeg extraction failed with exit code {returncode}: {collector.text()}")
    total_output_bytes = _directory_output_bytes(stage, options.max_output_bytes)
    if total_output_bytes > options.max_output_bytes:
        raise OutputError("FFmpeg output exceeded the configured max_output_bytes during decoding")
    return _ProcessOutcome(total_output_bytes, collector.text(), collector.truncated)


def _probe_ffmpeg_version(executable: str) -> str:
    try:
        process = subprocess.Popen(
            [executable, "-version"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    except FileNotFoundError as error:
        raise OutputError(f"FFmpeg executable was not found: {executable}") from error
    except OSError as error:
        raise OutputError(f"cannot query FFmpeg version: {error}") from error
    collector = _BoundedCollector(_VERSION_RETAIN_BYTES)
    reader = threading.Thread(target=collector.drain, args=(process.stdout,), name="ffmpeg-version")
    reader.start()
    try:
        process.wait(timeout=10.0)
    except subprocess.TimeoutExpired as error:
        _stop_process(process)
        _finish_reader(process.stdout, reader)
        raise OutputError("FFmpeg version query exceeded 10 seconds") from error
    _finish_reader(process.stdout, reader)
    if collector.error is not None:
        raise OutputError(f"cannot read FFmpeg version: {collector.error}") from collector.error
    if process.returncode != 0:
        raise OutputError(f"FFmpeg version query failed with exit code {process.returncode}")
    lines = collector.buffer.decode("utf-8", "replace").splitlines()
    version = lines[0].strip() if lines else ""
    _require_safe_text(version, "FFmpeg version")
    return version


def _directory_output_bytes(stage: Path, limit: int) -> int:
    total = 0
    with os.scandir(stage) as entries:
        for entry in entries:
            metadata = entry.stat(follow_symlinks=False)
            if stat.S_ISREG(metadata.st_mode):
                total += metadata.st_size
                if total > limit:
                    return total
    return total


def _stop_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        process.wait()
        return
    try:
        process.terminate()
        process.wait(timeout=_STOP_GRACE_SECONDS)
    except (OSError, subprocess.TimeoutExpired):
        process.kill()
        process.wait()


def _finish_reader(stream: IO[bytes] | None, reader: threading.Thread) -> None:
    reader.join(timeout=_STOP_GRACE_SECONDS)
    if reader.is_alive():
        if stream is not None:
            stream.close()
        reader.join(timeout=_STOP_GRACE_SECONDS)
    if reader.is_alive():
        raise OutputError("FFmpeg diagnostic reader did not terminate")


def _snapshot_file(path: Path) -> _FileSnapshot:
    """Hash one stable file snapshot and retain identity metadata."""

    digest = hashlib.sha256()
    byte_count = 0
    try:
        with path.open("rb") as stream:
            before = os.fstat(stream.fileno())
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
                byte_count += len(chunk)
            after = os.fstat(stream.fileno())
    except OSError as error:
        raise ScanError(f"cannot hash input video {path}: {error}") from error
    before_identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    after_identity = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if before_identity != after_identity or byte_count != after.st_size:
        raise ScanError(f"input video changed while it was being hashed: {path}")
    return _FileSnapshot(
        f"sha256:{digest.hexdigest()}",
        byte_count,
        after.st_dev,
        after.st_ino,
        after.st_mtime_ns,
    )


def _require_sha256(value: object, label: str) -> None:
    if not isinstance(value, str) or len(value) != 71 or not value.startswith("sha256:"):
        raise ConfigurationError(f"{label} must be sha256 followed by 64 hex digits")
    try:
        int(value[7:], 16)
    except ValueError as error:
        raise ConfigurationError(f"{label} must be sha256 followed by 64 hex digits") from error


def _safe_process_detail(value: str) -> str:
    normalized = " ".join(value.split())[:500]
    return normalized or "no diagnostic output"
