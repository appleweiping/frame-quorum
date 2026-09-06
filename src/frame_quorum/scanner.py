"""Discover and measure ordered image frames."""

from __future__ import annotations

import os
import re
import warnings
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

from .errors import ConfigurationError, ScanError
from .metrics import measure_image
from .models import AnimationConfig, ConcurrencyConfig, Frame, FrameMetrics, ScanConfig
from .timestamps import timestamp_for

_NATURAL_PARTS = re.compile(r"(\d+)")


def scan_frames(
    input_path: str | Path,
    config: ScanConfig | None = None,
    *,
    animation: AnimationConfig | None = None,
    concurrency: ConcurrencyConfig | None = None,
) -> tuple[Frame, ...]:
    """Scan one image or directory into a naturally ordered frame sequence.

    Animated containers stay collapsed to their first internal frame unless an
    ``AnimationConfig`` is supplied, which expands each one under that config's
    frame and decoded-byte limits.

    Decoding is sequential unless a ``ConcurrencyConfig`` asks for more than one
    worker. Extra workers only overlap file decoding; the returned sequence, the
    measurements, and the first reported failure are identical at every worker
    count.
    """

    options = ScanConfig() if config is None else config
    if not isinstance(options, ScanConfig):
        raise ConfigurationError("scan config must be ScanConfig")
    options.validate()
    if animation is not None:
        if not isinstance(animation, AnimationConfig):
            raise ConfigurationError("animation config must be AnimationConfig")
        animation.validate()
    workers = _worker_count(concurrency)
    supplied = Path(input_path).expanduser()
    if supplied.is_symlink():
        raise ScanError(f"refusing to scan a symbolic-link input: {supplied}")
    root = supplied.resolve()
    paths = discover_images(root, options)
    base = root if root.is_dir() else root.parent
    # ``warnings`` filters are process-global state. Entering ``catch_warnings``
    # once per file would let one worker restore the filters while another is
    # still decoding, so the escalation is installed once, here, on the calling
    # thread and stays in force for every decode this scan performs.
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        return tuple(_scan_paths(paths, base, options, animation, workers))


def _worker_count(concurrency: ConcurrencyConfig | None) -> int:
    """Return the validated worker count; absent configuration means sequential."""

    if concurrency is None:
        return 1
    if not isinstance(concurrency, ConcurrencyConfig):
        raise ConfigurationError("concurrency config must be ConcurrencyConfig")
    concurrency.validate()
    return concurrency.workers


def _scan_paths(
    paths: tuple[Path, ...],
    root: Path,
    config: ScanConfig,
    animation: AnimationConfig | None,
    workers: int,
) -> list[Frame]:
    """Measure every discovered path and assemble records in discovery order.

    Measurement is the only part that may overlap. Sequence identity is always
    assigned by walking ``paths`` in order, and a parallel run consumes its
    futures in that same order, so the first failure it reports belongs to the
    earliest failing path rather than to whichever worker finished first.
    """

    frames: list[Frame] = []
    if workers == 1:
        for path in paths:
            measured = _measure_file(path, root, animation)
            frames.extend(_build_frames(measured, len(frames), config))
        return frames
    with ThreadPoolExecutor(
        max_workers=min(workers, len(paths)), thread_name_prefix="frame-quorum-scan"
    ) as pool:
        pending: list[Future[_MeasuredFile]] = [
            pool.submit(_measure_file, path, root, animation) for path in paths
        ]
        try:
            for future in pending:
                frames.extend(_build_frames(future.result(), len(frames), config))
        finally:
            # Sequential scanning never opens a file after the first failure.
            # Cancelling the queue keeps a parallel scan from decoding the rest
            # of the directory before the pool shuts down.
            for future in pending:
                future.cancel()
    return frames


def discover_images(input_path: Path, config: ScanConfig) -> tuple[Path, ...]:
    """Return supported files in deterministic natural-name order."""

    if input_path.is_symlink():
        raise ScanError(f"refusing to scan a symbolic-link input: {input_path}")
    if not input_path.exists():
        raise ScanError(f"input path does not exist: {input_path}")
    allowed = {
        extension.lower() if extension.startswith(".") else f".{extension.lower()}"
        for extension in config.extensions
    }
    if input_path.is_file():
        if input_path.suffix.lower() not in allowed:
            raise ScanError(f"unsupported image extension: {input_path.suffix or '<none>'}")
        return (input_path,)
    iterator = input_path.rglob("*") if config.recursive else input_path.iterdir()
    images = []
    for path in iterator:
        if path.suffix.lower() not in allowed:
            continue
        if path.is_symlink():
            raise ScanError(f"refusing to scan a symbolic-link image: {path}")
        if path.is_file():
            images.append(path)
    images.sort(key=lambda path: _path_sort_key(path.relative_to(input_path).as_posix()))
    if not images:
        raise ScanError(f"no supported images found in {input_path}")
    return tuple(images)


def _natural_key(value: str) -> tuple[tuple[int, int, str, int], ...]:
    parts: list[tuple[int, int, str, int]] = []
    for part in _NATURAL_PARTS.split(value.casefold()):
        if part.isdigit():
            normalized = part.lstrip("0") or "0"
            parts.append((0, len(normalized), normalized, len(part)))
        else:
            parts.append((1, 0, part, 0))
    return tuple(parts)


def _path_sort_key(value: str) -> tuple[tuple[tuple[int, int, str, int], ...], str]:
    return _natural_key(value), value


@dataclass(frozen=True, slots=True)
class _MeasuredFile:
    """One decoded file awaiting its place in the sequence.

    Holds no sequence index, so it can be produced on a worker thread without
    knowing how many frames the files before it contributed.
    """

    path: Path
    relative_path: str
    before: os.stat_result
    measured: tuple[tuple[tuple[int, int], FrameMetrics], ...]


def _measure_file(
    path: Path,
    root: Path,
    animation: AnimationConfig | None,
) -> _MeasuredFile:
    """Decode and measure one file, which may hold several internal frames.

    This is the only part of a scan that may run on a worker thread. It touches
    no shared state, reads a single file, and depends on no other file's result.
    """

    try:
        before = path.stat()
    except OSError as error:
        raise ScanError(f"cannot inspect image {path}: {error}") from error
    try:
        with Image.open(path) as opened:
            count = _expansion_count(opened, path, animation)
            measured = []
            for position in range(count):
                if count > 1:
                    opened.seek(position)
                image = ImageOps.exif_transpose(opened)
                measured.append((image.size, measure_image(image)))
    except (
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
        UnidentifiedImageError,
        EOFError,
        OSError,
        ValueError,
    ) as error:
        raise ScanError(f"cannot decode image {path}: {error}") from error
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError:
        relative = path.name
    return _MeasuredFile(path, relative, before, tuple(measured))


def _build_frames(measured: _MeasuredFile, index: int, config: ScanConfig) -> tuple[Frame, ...]:
    """Give one measured file its indices and timestamps, then re-verify the file.

    Always runs on the calling thread in discovery order, so indices stay
    contiguous across files that expand into different numbers of frames. The
    post-read identity check closes the same window as before: the snapshot was
    taken before decoding and is compared after the timestamp policy has read
    whatever metadata it needs.
    """

    path = measured.path
    relative = measured.relative_path
    count = len(measured.measured)
    frames: list[Frame] = []
    try:
        for position, ((width, height), metrics) in enumerate(measured.measured):
            frames.append(
                Frame(
                    index=index + position,
                    path=path,
                    relative_path=relative if count == 1 else f"{relative}#frame={position}",
                    timestamp=timestamp_for(path, index + position, config),
                    width=width,
                    height=height,
                    byte_size=measured.before.st_size,
                    metrics=metrics,
                    source_frame_index=None if count == 1 else position,
                )
            )
        after = path.stat()
    except ScanError:
        raise
    except OSError as error:
        raise ScanError(f"cannot inspect image {path}: {error}") from error
    if _file_identity(after) != _file_identity(measured.before):
        raise ScanError(f"image changed while it was being scanned: {path}")
    return tuple(frames)


def _expansion_count(image: Image.Image, path: Path, animation: AnimationConfig | None) -> int:
    """Return how many internal frames this file contributes, or refuse to expand."""

    if animation is None:
        return 1
    count: int = getattr(image, "n_frames", 1)
    if count <= 1:
        return 1
    if count > animation.max_frames:
        raise ScanError(
            f"animated image {path} holds {count} frames, above the "
            f"{animation.max_frames}-frame expansion limit"
        )
    width, height = image.size
    decoded_bytes = count * width * height * 3
    if decoded_bytes > animation.max_decoded_bytes:
        raise ScanError(
            f"expanding animated image {path} would decode {decoded_bytes} bytes, above the "
            f"{animation.max_decoded_bytes}-byte expansion limit"
        )
    return count


def _file_identity(status: os.stat_result) -> tuple[int, int, int, int]:
    return (status.st_size, status.st_mtime_ns, status.st_dev, status.st_ino)
