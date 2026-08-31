"""Timestamp extraction policies for ordered image sequences."""

from __future__ import annotations

import math
import re
from pathlib import Path

from .errors import ScanError
from .models import ScanConfig

_UNIT_DIVISORS = {"seconds": 1.0, "milliseconds": 1_000.0, "microseconds": 1_000_000.0}


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
