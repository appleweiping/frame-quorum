"""Bounded original OTIO cuts-only serialization with exact integer time pairs.

Editorial references are caller declarations, not source-media certificates.
No optional decoder or OpenTimelineIO package is imported by the pure exporter.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass
from fractions import Fraction
from math import gcd
from pathlib import Path
from typing import Any

from .errors import ConfigurationError, OutputError
from .models import _require_safe_text
from .native_measurements import _bundle
from .native_scenes import NativeSceneResult
from .native_splitting import native_scene_clips
from .native_video import _fraction, _integer, _local_path_text, _pair

_EXACT_INTEGER = 2**53 - 1
_OUTPUT_LIMIT = 64 * 1024 * 1024


def _text(value: str, name: str, maximum: int, *, empty: bool = False) -> None:
    if type(value) is not str or len(value) > maximum or (not empty and not value):
        raise ConfigurationError(f"{name} must be bounded text")
    if value:
        _require_safe_text(value, name)


@dataclass(frozen=True, slots=True)
class OTIOCut:
    start: Fraction
    end: Fraction
    name: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "start", _fraction(self.start, "cut start"))
        object.__setattr__(self, "end", _fraction(self.end, "cut end"))
        if self.end <= self.start:
            raise ConfigurationError("cut end must be strictly after start")
        _text(self.name, "cut name", 128, empty=True)


@dataclass(frozen=True, slots=True)
class OTIOMedia:
    path: Path
    origin: Fraction
    available_start: Fraction | None = None
    available_end: Fraction | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.path, Path) or not self.path.is_absolute():
            raise ConfigurationError("media path must be an absolute local pathlib.Path")
        _text(str(self.path), "media path", 4096)
        _local_path_text(str(self.path))
        object.__setattr__(self, "origin", _fraction(self.origin, "media origin"))
        if (self.available_start is None) != (self.available_end is None):
            raise ConfigurationError("available media bounds must both be supplied or both unknown")
        if self.available_start is not None and self.available_end is not None:
            start = _fraction(self.available_start, "available start")
            end = _fraction(self.available_end, "available end")
            if end <= start:
                raise ConfigurationError("available end must be strictly after start")
            object.__setattr__(self, "available_start", start)
            object.__setattr__(self, "available_end", end)


@dataclass(frozen=True, slots=True)
class OTIOExportConfig:
    title: str = "Frame Quorum"
    include_audio: bool = False
    max_clips: int = 10_000
    max_tick_rate: int = 1_000_000_000
    max_output_bytes: int = _OUTPUT_LIMIT

    def __post_init__(self) -> None:
        _text(self.title, "title", 128)
        if type(self.include_audio) is not bool:
            raise ConfigurationError("include_audio must be an explicit boolean declaration")
        _integer(self.max_clips, "max_clips", 1, 10_000)
        _integer(self.max_tick_rate, "max_tick_rate", 1, 1_000_000_000)
        _integer(self.max_output_bytes, "max_output_bytes", 1, _OUTPUT_LIMIT)


@dataclass(frozen=True, slots=True)
class OTIOExportResult:
    output_dir: Path
    clip_count: int
    track_count: int
    tick_rate: int
    otio_bytes: int
    total_output_bytes: int
    otio_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_dir": str(self.output_dir),
            "clip_count": self.clip_count,
            "track_count": self.track_count,
            "tick_rate": self.tick_rate,
            "otio_bytes": self.otio_bytes,
            "total_output_bytes": self.total_output_bytes,
            "otio_sha256": self.otio_sha256,
            "source_verified": False,
        }


@dataclass(frozen=True, slots=True)
class _Prepared:
    cuts: tuple[OTIOCut, ...]
    media: OTIOMedia
    config: OTIOExportConfig
    rate: int
    # Every pair contains (source start ticks, positive duration ticks).
    ranges: tuple[tuple[int, int], ...]
    available: tuple[int, int] | None
    duration: int
    uri: str


def _prepare(cuts: tuple[OTIOCut, ...], media: OTIOMedia, config: OTIOExportConfig | None) -> _Prepared:
    options = OTIOExportConfig() if config is None else config
    if type(options) is not OTIOExportConfig or type(media) is not OTIOMedia:
        raise ConfigurationError("export requires OTIOExportConfig and OTIOMedia")
    options.__post_init__()
    media.__post_init__()
    if type(cuts) is not tuple or not 1 <= len(cuts) <= options.max_clips:
        raise ConfigurationError("cuts must be a nonempty immutable tuple within max_clips")
    rate = 1

    def include(value: Fraction) -> None:
        nonlocal rate
        factor = value.denominator // gcd(rate, value.denominator)
        if rate > options.max_tick_rate // factor:
            raise ConfigurationError("exact common tick rate exceeds the configured limit")
        rate *= factor

    previous: OTIOCut | None = None
    for cut in cuts:
        if type(cut) is not OTIOCut:
            raise ConfigurationError("cuts must contain only OTIOCut values")
        cut.__post_init__()
        if previous is not None and cut.start < previous.end:
            raise ConfigurationError("cuts must be ordered and nonoverlapping")
        if (
            media.available_start is not None
            and media.available_end is not None
            and (cut.start < media.available_start or cut.end > media.available_end)
        ):
            raise ConfigurationError("cut exceeds declared available media range")
        include(cut.start - media.origin)
        include(cut.end - media.origin)
        previous = cut
    if media.available_start is not None and media.available_end is not None:
        include(media.available_start - media.origin)
        include(media.available_end - media.origin)

    def ticks(value: Fraction) -> int:
        exact = value * rate
        # Rate admission above ensures divisibility; retain a defensive guard.
        if exact.denominator != 1 or abs(exact.numerator) > _EXACT_INTEGER:
            raise ConfigurationError("time ticks exceed the binary64 exact integer range")
        return exact.numerator

    total = 0
    ranges = []
    for cut in cuts:
        start, end = ticks(cut.start - media.origin), ticks(cut.end - media.origin)
        duration = end - start
        total += duration
        if duration > _EXACT_INTEGER or total > _EXACT_INTEGER:
            raise ConfigurationError("duration or accumulated record ticks exceed exact integer range")
        ranges.append((start, duration))
    available = None
    if media.available_start is not None and media.available_end is not None:
        available = (
            ticks(media.available_start - media.origin),
            ticks(media.available_end - media.available_start),
        )
        ticks(media.available_end - media.origin)
    return _Prepared(cuts, media, options, rate, tuple(ranges), available, total, media.path.as_uri())


def _json(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("ascii")


def _time(value: int, rate: int) -> dict[str, Any]:
    return {"OTIO_SCHEMA": "RationalTime.1", "value": value, "rate": rate}


def _range(value: tuple[int, int], rate: int) -> dict[str, Any]:
    return {
        "OTIO_SCHEMA": "TimeRange.1",
        "start_time": _time(value[0], rate),
        "duration": _time(value[1], rate),
    }


def _declarations(prepared: _Prepared) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "timing": "exact_integer_ticks",
        "source_verified": False,
        "media_origin": _pair(prepared.media.origin),
        "audio": "caller_declared_unverified" if prepared.config.include_audio else "not_requested",
        "available_range": "caller_declared_unverified" if prepared.available is not None else "unknown",
    }


def _pieces(prepared: _Prepared) -> Iterator[bytes]:
    """Encode at most one bounded clip at a time, never a complete object graph."""
    root = {
        "OTIO_SCHEMA": "Timeline.1",
        "name": prepared.config.title,
        "global_start_time": _time(0, prepared.rate),
        "metadata": {"frame_quorum": _declarations(prepared)},
    }
    yield _json(root)[:-1] + b',"tracks":{"OTIO_SCHEMA":"Stack.1","children":['
    for track_index, kind in enumerate(("Video", "Audio") if prepared.config.include_audio else ("Video",)):
        if track_index:
            yield b","
        yield (
            _json({"OTIO_SCHEMA": "Track.1", "name": kind + " 1", "kind": kind, "enabled": True})[:-1]
            + b',"children":['
        )
        for index, (cut, interval) in enumerate(zip(prepared.cuts, prepared.ranges, strict=True)):
            if index:
                yield b","
            reference = {
                "OTIO_SCHEMA": "ExternalReference.1",
                "target_url": prepared.uri,
                "available_range": None
                if prepared.available is None
                else _range(prepared.available, prepared.rate),
            }
            yield _json(
                {
                    "OTIO_SCHEMA": "Clip.2",
                    "name": cut.name or f"Scene {index + 1}",
                    "enabled": True,
                    "source_range": _range(interval, prepared.rate),
                    "media_references": {"DEFAULT_MEDIA": reference},
                    "active_media_reference_key": "DEFAULT_MEDIA",
                    "metadata": {
                        "frame_quorum": {
                            "source_start": _pair(cut.start),
                            "source_end_exclusive": _pair(cut.end),
                        }
                    },
                }
            )
        yield b"]}"
    yield b"]}}\n"


def _bounded(prepared: _Prepared) -> Iterator[bytes]:
    total = 0
    for piece in _pieces(prepared):
        total += len(piece)
        if total > prepared.config.max_output_bytes:
            raise OutputError("OTIO output exceeds total byte limit")
        yield piece


def render_otio(cuts: tuple[OTIOCut, ...], media: OTIOMedia, config: OTIOExportConfig | None = None) -> str:
    """Return standard OTIO text, retaining bounded output but no source media.

    All intervals are half-open native coordinates; subtract explicit media.origin
    to obtain editor coordinates. Unknown availability is never manufactured.
    """
    prepared = _prepare(cuts, media, config)
    return b"".join(_bounded(prepared)).decode("ascii")


def write_otio_bundle(
    cuts: tuple[OTIOCut, ...],
    media: OTIOMedia,
    output_dir: str | Path,
    config: OTIOExportConfig | None = None,
) -> OTIOExportResult:
    """Preflight and publish scenes.otio plus audit.json in a new owned directory."""
    prepared = _prepare(cuts, media, config)
    digest = hashlib.sha256()
    size = 0
    for piece in _bounded(prepared):
        size += len(piece)
        digest.update(piece)
    audit = {
        "kind": "frame-quorum-otio-export",
        **_declarations(prepared),
        "clip_count": len(cuts),
        "track_count": 2 if prepared.config.include_audio else 1,
        "tick_rate": prepared.rate,
        "record_duration_ticks": prepared.duration,
        "otio_bytes": size,
        "otio_sha256": digest.hexdigest(),
        "max_output_bytes": prepared.config.max_output_bytes,
    }
    audit_bytes = _json(audit) + b"\n"
    total = size + len(audit_bytes)
    if total > prepared.config.max_output_bytes:
        raise OutputError("OTIO bundle including audit exceeds total byte limit")
    # The second bounded encoding pass is intentional: no staging exists until
    # the complete byte budget is known. Publication/cleanup remain shared.
    target = _bundle(
        output_dir,
        (("scenes.otio", _bounded(prepared)), ("audit.json", iter((audit_bytes,)))),
        prepared.config.max_output_bytes,
    )
    return OTIOExportResult(
        target, len(cuts), audit["track_count"], prepared.rate, size, total, digest.hexdigest()
    )


def otio_cuts_from_native(
    result: NativeSceneResult, *, final_end: Fraction | None = None
) -> tuple[OTIOCut, ...]:
    """Reuse complete unsampled native scene admission; no source authentication."""
    if type(result) is not NativeSceneResult:
        raise ConfigurationError("native OTIO conversion requires NativeSceneResult")
    if len(result.scenes) > 10_000:
        raise ConfigurationError("native OTIO conversion cannot exceed 10000 clips")
    if result.config.video.video_stream != 0:
        raise ConfigurationError("native OTIO conversion supports only video_stream=0")
    return tuple(OTIOCut(clip.start, clip.end) for clip in native_scene_clips(result, final_end=final_end))


__all__ = [
    "OTIOCut",
    "OTIOExportConfig",
    "OTIOExportResult",
    "OTIOMedia",
    "otio_cuts_from_native",
    "render_otio",
    "write_otio_bundle",
]
