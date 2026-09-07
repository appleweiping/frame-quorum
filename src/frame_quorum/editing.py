"""CFR scene timing and bounded single-reel, cuts-only EDL interchange."""

from __future__ import annotations

import csv
import io
import re
from collections.abc import Iterable

from .errors import ConfigurationError
from .scenes import Shot
from .timecode import FrameRate, FrameTimecode


def _scenes(scenes: Iterable[Shot], maximum: int) -> tuple[Shot, ...]:
    if not isinstance(scenes, Iterable) or isinstance(scenes, (str, bytes)):
        raise ConfigurationError("scenes must be an iterable of Shot values")
    checked: list[Shot] = []
    for scene in scenes:
        if len(checked) == maximum:
            raise ConfigurationError(f"scene export cannot exceed {maximum} scenes")
        if not isinstance(scene, Shot):
            raise ConfigurationError("scenes must contain Shot values")
        if any(
            type(value) is not int or value < 0
            for value in (scene.ordinal, scene.start_index, scene.end_index)
        ):
            raise ConfigurationError("scene coordinates must be nonnegative integers")
        if scene.end_index <= scene.start_index or scene.ordinal != len(checked):
            raise ConfigurationError("scene ordinals must be contiguous and durations positive")
        if checked and scene.start_index < checked[-1].end_index:
            raise ConfigurationError("source scenes must be ordered and nonoverlapping")
        checked.append(scene)
    if not checked:
        raise ConfigurationError("scene export requires at least one scene")
    return tuple(checked)


def render_scene_timecodes(scenes: Iterable[Shot], rate: FrameRate, *, drop_frame: bool = False) -> str:
    """CSV half-open source bounds with exact seconds and counter/time labels."""
    origin = FrameTimecode(0, rate, drop_frame)
    checked = _scenes(scenes, 100_000)
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(
        (
            "scene",
            "start_frame",
            "end_frame_exclusive",
            "frames",
            "rate",
            "start_seconds",
            "end_seconds",
            "start_timestamp",
            "end_timestamp",
            "start_timecode",
            "end_timecode",
        )
    )
    for scene in checked:
        start, end = origin.shift_frames(scene.start_index), origin.shift_frames(scene.end_index)
        writer.writerow(
            (
                scene.ordinal,
                start.frame_number,
                end.frame_number,
                scene.frame_count,
                str(rate),
                str(start.seconds),
                str(end.seconds),
                start.timestamp(),
                end.timestamp(),
                start.smpte(),
                end.smpte(),
            )
        )
    return output.getvalue()


def render_edl(
    scenes: Iterable[Shot],
    rate: FrameRate,
    *,
    reel: str = "AX",
    title: str = "Frame Quorum",
    drop_frame: bool = False,
    source_start: int = 0,
    record_start: int = 0,
) -> str:
    """Render a CMX-style cuts-only video track with exclusive out-points.

    Supports 24/25/30 nominal FPS, including exact 24000/1001 and 30000/1001.
    No audio, dissolves, speed effects, filenames, or implicit 24-hour wrapping.
    Source gaps are omitted from the assembled contiguous record timeline.
    """
    source = FrameTimecode(source_start, rate, drop_frame)
    record = FrameTimecode(record_start, rate, drop_frame)
    if rate.nominal not in (24, 25, 30):
        raise ConfigurationError("EDL export supports only nominal 24, 25, or 30 FPS")
    if type(reel) is not str or re.fullmatch(r"[A-Za-z0-9_]{1,8}", reel) is None:
        raise ConfigurationError("reel requires 1..8 ASCII letters, digits, or underscores")
    if (
        type(title) is not str
        or not 1 <= len(title) <= 80
        or any(not 32 <= ord(char) < 127 for char in title)
    ):
        raise ConfigurationError("title requires 1..80 printable ASCII characters")
    checked = _scenes(scenes, 999)
    lines = [f"TITLE: {title}", f"FCM: {'DROP FRAME' if drop_frame else 'NON-DROP FRAME'}", ""]
    for scene in checked:
        source_in, source_out = source.shift_frames(scene.start_index), source.shift_frames(scene.end_index)
        record_out = record.shift_frames(scene.frame_count)
        labels = [item.smpte() for item in (source_in, source_out, record, record_out)]
        if any(int(label.split(":", 1)[0]) >= 24 for label in labels):
            raise ConfigurationError("EDL coordinates must remain below 24 timecode hours")
        lines.extend((f"{scene.ordinal + 1:03d}  {reel:<8} V     C        {' '.join(labels)}", ""))
        record = record_out
    return "\n".join(lines)


__all__ = ["render_edl", "render_scene_timecodes"]
