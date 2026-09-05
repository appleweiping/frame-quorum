"""Timestamp extraction policies for ordered image sequences.

Every policy converts one file into a floating-point time coordinate or ``None``.
``exif`` re-opens the file to read metadata only; it never decodes pixels.
"""

from __future__ import annotations

import math
import re
import warnings
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from .errors import ScanError
from .models import ScanConfig

_UNIT_DIVISORS = {"seconds": 1.0, "milliseconds": 1_000.0, "microseconds": 1_000_000.0}
_EXIF_IFD = 0x8769
_EXIF_DATETIME_FORMAT = "%Y:%m:%d %H:%M:%S"
_EXIF_OFFSET_PATTERN = re.compile(r"(?P<sign>[+-])(?P<hours>\d{2}):(?P<minutes>\d{2})")
_EXIF_PLACEHOLDER_CHARACTERS = frozenset("0: ")
_MINUTES_PER_DAY = 24 * 60
# Capture-time precedence, most to least specific, each with its EXIF 2.31 UTC offset tag:
# DateTimeOriginal is when the shutter fired, DateTimeDigitized is when the image was
# digitized, and DateTime is the last time imaging software wrote the file.
_EXIF_CAPTURE_TAGS = (
    ("DateTimeOriginal", 0x9003, 0x9011),
    ("DateTimeDigitized", 0x9004, 0x9012),
    ("DateTime", 0x0132, 0x9010),
)


def timestamp_for(path: Path, index: int, config: ScanConfig) -> float | None:
    """Calculate a timestamp using the explicitly configured policy."""

    if config.timestamp_mode == "none":
        return None
    if config.timestamp_mode == "index":
        try:
            timestamp = index / config.frame_rate
        except (OverflowError, ZeroDivisionError) as error:
            raise ScanError("derived index timestamp must be finite") from error
        if not math.isfinite(timestamp):
            raise ScanError("derived index timestamp must be finite")
        return timestamp
    if config.timestamp_mode == "mtime":
        try:
            timestamp = path.stat().st_mtime
        except OSError as error:
            raise ScanError(f"cannot inspect timestamp for {path}: {error}") from error
        if not math.isfinite(timestamp):
            raise ScanError(f"modification timestamp for {path} must be finite")
        return timestamp
    if config.timestamp_mode == "exif":
        return _timestamp_from_exif(path)
    return _timestamp_from_filename(path, config)


def _timestamp_from_filename(path: Path, config: ScanConfig) -> float:
    assert config.timestamp_regex is not None
    try:
        pattern = re.compile(config.timestamp_regex)
    except re.error as error:
        raise ScanError(f"invalid timestamp regex: {error}") from error
    match = pattern.search(path.stem)
    if match is None:
        raise ScanError(f"timestamp regex did not match {path.name}")
    if "ts" in match.groupdict():
        raw = match.group("ts")
    elif match.lastindex:
        raw = match.group(1)
    else:
        raw = match.group(0)
    try:
        numeric = float(raw)
    except (OverflowError, TypeError, ValueError) as error:
        raise ScanError(f"timestamp {raw!r} in {path.name} is not numeric") from error
    timestamp = numeric / _UNIT_DIVISORS[config.timestamp_unit]
    if not math.isfinite(timestamp):
        raise ScanError(f"timestamp {raw!r} in {path.name} must be finite")
    return timestamp


def _timestamp_from_exif(path: Path) -> float:
    """Convert the most specific populated EXIF capture tag into epoch seconds.

    Tags are consulted in ``_EXIF_CAPTURE_TAGS`` order. A tag that is absent or
    holds the all-zero "not recorded" placeholder is skipped; a tag that is
    populated but unreadable is an error, because falling through to a weaker
    source would silently substitute a different meaning for a corrupt value.
    EXIF capture times carry no zone, so a recorded UTC offset is applied when
    present and the value is otherwise read as UTC.
    """

    values = _exif_values(path)
    for name, tag, offset_tag in _EXIF_CAPTURE_TAGS:
        if tag not in values:
            continue
        text = _exif_text(values[tag], name, path)
        if _is_exif_placeholder(text):
            continue
        try:
            moment = datetime.strptime(text, _EXIF_DATETIME_FORMAT)
        except ValueError as error:
            raise ScanError(
                f"EXIF {name} {text!r} in {path.name} is not a YYYY:MM:DD HH:MM:SS capture time"
            ) from error
        return moment.replace(tzinfo=_exif_offset(values.get(offset_tag), name, path)).timestamp()
    raise ScanError(
        f"{path.name} records no EXIF capture time; DateTimeOriginal, DateTimeDigitized, "
        "and DateTime are absent or unset"
    )


def _exif_offset(value: object, name: str, path: Path) -> timezone:
    """Return the recorded UTC offset for a capture tag, defaulting to UTC."""

    if value is None:
        return UTC
    text = _exif_text(value, f"{name} UTC offset", path)
    if _is_exif_placeholder(text):
        return UTC
    match = _EXIF_OFFSET_PATTERN.fullmatch(text)
    if match is None:
        raise ScanError(f"EXIF {name} UTC offset {text!r} in {path.name} is not written as +HH:MM or -HH:MM")
    minutes = int(match["hours"]) * 60 + int(match["minutes"])
    if int(match["minutes"]) > 59 or minutes >= _MINUTES_PER_DAY:
        raise ScanError(
            f"EXIF {name} UTC offset {text!r} in {path.name} is outside the supported "
            "range of less than one day"
        )
    return timezone(timedelta(minutes=-minutes if match["sign"] == "-" else minutes))


def _exif_values(path: Path) -> dict[int, object]:
    """Read the root and Exif sub-IFD tags without decoding pixels."""

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(path) as opened:
                metadata = opened.getexif()
                values: dict[int, object] = dict(metadata)
                values.update(metadata.get_ifd(_EXIF_IFD))
    except (
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
        UnidentifiedImageError,
        OSError,
        ValueError,
    ) as error:
        raise ScanError(f"cannot read EXIF metadata from {path}: {error}") from error
    return values


def _exif_text(value: object, label: str, path: Path) -> str:
    if not isinstance(value, str):
        raise ScanError(f"EXIF {label} in {path.name} must be text, not {type(value).__name__}")
    return value.replace("\x00", " ").strip()


def _is_exif_placeholder(text: str) -> bool:
    """Treat a value made only of zeros, colons, and spaces as "not recorded"."""

    return set(text) <= _EXIF_PLACEHOLDER_CHARACTERS
