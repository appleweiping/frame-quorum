"""Discover and measure ordered image frames."""

from __future__ import annotations

import re
import warnings
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

from .errors import ScanError
from .metrics import measure_image
from .models import Frame, ScanConfig
from .timestamps import timestamp_for

_NATURAL_PARTS = re.compile(r"(\d+)")


def scan_frames(input_path: str | Path, config: ScanConfig | None = None) -> tuple[Frame, ...]:
    """Scan one image or directory into a naturally ordered frame sequence."""

    options = config or ScanConfig()
    options.validate()
    root = Path(input_path).expanduser().resolve()
    paths = discover_images(root, options)
    frames: list[Frame] = []
    for index, path in enumerate(paths):
        frames.append(_scan_one(path, root if root.is_dir() else root.parent, index, options))
    return tuple(frames)


def discover_images(input_path: Path, config: ScanConfig) -> tuple[Path, ...]:
    """Return supported files in deterministic natural-name order."""

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
    images = [path for path in iterator if path.is_file() and path.suffix.lower() in allowed]
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


def _scan_one(path: Path, root: Path, index: int, config: ScanConfig) -> Frame:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(path) as opened:
                image = ImageOps.exif_transpose(opened)
                width, height = image.size
                metrics = measure_image(image)
    except (
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
        UnidentifiedImageError,
        OSError,
        ValueError,
    ) as error:
        raise ScanError(f"cannot decode image {path}: {error}") from error
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError:
        relative = path.name
    try:
        timestamp = timestamp_for(path, index, config)
        byte_size = path.stat().st_size
    except ScanError:
        raise
    except OSError as error:
        raise ScanError(f"cannot inspect image {path}: {error}") from error
    return Frame(
        index=index,
        path=path,
        relative_path=relative,
        timestamp=timestamp,
        width=width,
        height=height,
        byte_size=byte_size,
        metrics=metrics,
    )
