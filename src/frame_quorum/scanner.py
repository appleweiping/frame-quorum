"""Discover and measure ordered image frames."""

from __future__ import annotations

import os
import re
import warnings
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

from .errors import ConfigurationError, ScanError
from .metrics import measure_image
from .models import AnimationConfig, Frame, ScanConfig
from .timestamps import timestamp_for

_NATURAL_PARTS = re.compile(r"(\d+)")


def scan_frames(
    input_path: str | Path,
    config: ScanConfig | None = None,
    *,
    animation: AnimationConfig | None = None,
) -> tuple[Frame, ...]:
    """Scan one image or directory into a naturally ordered frame sequence.

    Animated containers stay collapsed to their first internal frame unless an
    ``AnimationConfig`` is supplied, which expands each one under that config's
    frame and decoded-byte limits.
    """

    options = ScanConfig() if config is None else config
    if not isinstance(options, ScanConfig):
        raise ConfigurationError("scan config must be ScanConfig")
    options.validate()
    if animation is not None:
        if not isinstance(animation, AnimationConfig):
            raise ConfigurationError("animation config must be AnimationConfig")
        animation.validate()
    supplied = Path(input_path).expanduser()
    if supplied.is_symlink():
        raise ScanError(f"refusing to scan a symbolic-link input: {supplied}")
    root = supplied.resolve()
    paths = discover_images(root, options)
    frames: list[Frame] = []
    for path in paths:
        base = root if root.is_dir() else root.parent
        frames.extend(_scan_one(path, base, len(frames), options, animation))
    return tuple(frames)


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


def _scan_one(
    path: Path,
    root: Path,
    index: int,
    config: ScanConfig,
    animation: AnimationConfig | None,
) -> tuple[Frame, ...]:
    """Measure one file, which may contribute several internal animation frames."""

    try:
        before = path.stat()
    except OSError as error:
        raise ScanError(f"cannot inspect image {path}: {error}") from error
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
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
    frames: list[Frame] = []
    try:
        for position, ((width, height), metrics) in enumerate(measured):
            frames.append(
                Frame(
                    index=index + position,
                    path=path,
                    relative_path=relative if count == 1 else f"{relative}#frame={position}",
                    timestamp=timestamp_for(path, index + position, config),
                    width=width,
                    height=height,
                    byte_size=before.st_size,
                    metrics=metrics,
                    source_frame_index=None if count == 1 else position,
                )
            )
        after = path.stat()
    except ScanError:
        raise
    except OSError as error:
        raise ScanError(f"cannot inspect image {path}: {error}") from error
    if _file_identity(after) != _file_identity(before):
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
