"""Bounded cuts-only FCPXML 1.9 serialization for a declared local video asset.

This is an editorial interchange declaration, not a probe of media properties
or a certificate that a particular Final Cut Pro build will import the file.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass
from fractions import Fraction
from html import escape
from pathlib import Path
from typing import Any

from .errors import ConfigurationError, OutputError
from .native_measurements import _bundle
from .native_video import _fraction, _integer, _pair
from .otio_export import OTIOCut, OTIOExportConfig, OTIOMedia, _prepare, _text

_OUTPUT_LIMIT = 64 * 1024 * 1024
_TIME_NUMERATOR_LIMIT = 2**63 - 1
_TIME_DENOMINATOR_LIMIT = 2**31 - 1


def _xml_text(value: str, label: str) -> str:
    _text(value, label, 128)
    if any(ord(char) in (0xFFFE, 0xFFFF) for char in value):
        raise ConfigurationError(f"{label} contains a character forbidden in XML 1.0")
    return value


@dataclass(frozen=True, slots=True)
class FCPXMLExportConfig:
    frame_rate: Fraction
    width: int
    height: int
    title: str = "Frame Quorum"
    max_clips: int = 10_000
    max_output_bytes: int = _OUTPUT_LIMIT

    def __post_init__(self) -> None:
        rate = _fraction(self.frame_rate, "frame_rate", positive=True)
        if rate.numerator > _TIME_DENOMINATOR_LIMIT:
            raise ConfigurationError("frame_duration denominator exceeds FCPXML time range")
        object.__setattr__(self, "frame_rate", rate)
        _integer(self.width, "width", 1, 16_384)
        _integer(self.height, "height", 1, 16_384)
        _xml_text(self.title, "title")
        _integer(self.max_clips, "max_clips", 1, 10_000)
        _integer(self.max_output_bytes, "max_output_bytes", 1, _OUTPUT_LIMIT)


@dataclass(frozen=True, slots=True)
class FCPXMLExportResult:
    output_dir: Path
    clip_count: int
    fcpxml_bytes: int
    total_output_bytes: int
    fcpxml_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_dir": str(self.output_dir),
            "clip_count": self.clip_count,
            "fcpxml_bytes": self.fcpxml_bytes,
            "total_output_bytes": self.total_output_bytes,
            "fcpxml_sha256": self.fcpxml_sha256,
            "source_verified": False,
            "editor_import_verified": False,
        }


@dataclass(frozen=True, slots=True)
class _Prepared:
    cuts: tuple[OTIOCut, ...]
    media: OTIOMedia
    config: FCPXMLExportConfig
    available_start: Fraction
    available_end: Fraction
    sequence_duration: Fraction


def _time(value: Fraction) -> str:
    if abs(value.numerator) > _TIME_NUMERATOR_LIMIT or value.denominator > _TIME_DENOMINATOR_LIMIT:
        raise ConfigurationError("FCPXML rational time exceeds 64-bit numerator or 32-bit denominator")
    if value.denominator == 1:
        return f"{value.numerator}s"
    return f"{value.numerator}/{value.denominator}s"


def _prepare_fcpxml(cuts: tuple[OTIOCut, ...], media: OTIOMedia, config: FCPXMLExportConfig) -> _Prepared:
    if type(config) is not FCPXMLExportConfig:
        raise ConfigurationError("config must be FCPXMLExportConfig")
    config.__post_init__()
    admitted = _prepare(
        cuts,
        media,
        OTIOExportConfig(
            title=config.title, max_clips=config.max_clips, max_output_bytes=config.max_output_bytes
        ),
    )
    if media.available_start is None or media.available_end is None:
        raise ConfigurationError("FCPXML asset requires explicit available media bounds")
    for value in (
        media.available_start - media.origin,
        media.available_end - media.origin,
        *(coordinate - media.origin for cut in cuts for coordinate in (cut.start, cut.end)),
    ):
        if value < 0:
            raise ConfigurationError("FCPXML source times must not precede media origin")
        if (value * config.frame_rate).denominator != 1:
            raise ConfigurationError("FCPXML source time is not on the declared frame lattice")
        _time(value)
    duration = Fraction(admitted.duration, admitted.rate)
    _time(duration)
    _time(Fraction(1, 1) / config.frame_rate)
    return _Prepared(cuts, media, config, media.available_start, media.available_end, duration)


def _attribute(value: str) -> str:
    return escape(value, quote=True)


def _pieces(prepared: _Prepared) -> Iterator[bytes]:
    config, media = prepared.config, prepared.media
    title = _attribute(config.title)
    source = _attribute(media.path.as_uri())
    available_start = _time(prepared.available_start - media.origin)
    available_duration = _time(prepared.available_end - prepared.available_start)
    frame_duration = _time(1 / config.frame_rate)
    yield (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<fcpxml version="1.9">\n'
        "  <resources>\n"
        f'    <format id="r1" frameDuration="{frame_duration}" '
        f'width="{config.width}" height="{config.height}"/>\n'
        f'    <asset id="r2" name="{title}" start="{available_start}" '
        f'duration="{available_duration}" hasVideo="1" format="r1">\n'
        f'      <media-rep kind="original-media" src="{source}"/>\n'
        "    </asset>\n"
        "  </resources>\n"
        "  <library>\n"
        f'    <event name="{title}">\n'
        f'      <project name="{title}">\n'
        f'        <sequence format="r1" duration="{_time(prepared.sequence_duration)}" '
        'tcStart="0s" tcFormat="NDF">\n'
        "          <spine>\n"
    ).encode()
    offset = Fraction(0)
    for index, cut in enumerate(prepared.cuts, 1):
        duration = cut.end - cut.start
        name = _attribute(_xml_text(cut.name, "cut name") if cut.name else f"Scene {index}")
        yield (
            f'            <asset-clip name="{name}" ref="r2" offset="{_time(offset)}" '
            f'start="{_time(cut.start - media.origin)}" duration="{_time(duration)}" '
            'srcEnable="video"/>\n'
        ).encode()
        offset += duration
    yield (
        b"          </spine>\n        </sequence>\n      </project>\n    </event>\n  </library>\n</fcpxml>\n"
    )


def _bounded(prepared: _Prepared) -> Iterator[bytes]:
    total = 0
    for piece in _pieces(prepared):
        total += len(piece)
        if total > prepared.config.max_output_bytes:
            raise OutputError("FCPXML output exceeds total byte limit")
        yield piece


def render_fcpxml(cuts: tuple[OTIOCut, ...], media: OTIOMedia, config: FCPXMLExportConfig) -> str:
    """Render UTF-8 FCPXML for explicit frame-aligned, half-open source cuts."""
    return b"".join(_bounded(_prepare_fcpxml(cuts, media, config))).decode("utf-8")


def write_fcpxml_bundle(
    cuts: tuple[OTIOCut, ...],
    media: OTIOMedia,
    output_dir: str | Path,
    config: FCPXMLExportConfig,
) -> FCPXMLExportResult:
    """Publish scenes.fcpxml and audit.json once in a new owned directory."""
    prepared = _prepare_fcpxml(cuts, media, config)
    digest = hashlib.sha256()
    size = 0
    for piece in _bounded(prepared):
        size += len(piece)
        digest.update(piece)
    audit = {
        "kind": "frame-quorum-fcpxml-export",
        "schema_version": 1,
        "fcpxml_version": "1.9",
        "clip_count": len(cuts),
        "frame_rate": _pair(config.frame_rate),
        "media_origin": _pair(media.origin),
        "available_start": _pair(media.available_start),
        "available_end": _pair(media.available_end),
        "source_verified": False,
        "editor_import_verified": False,
        "fcpxml_bytes": size,
        "fcpxml_sha256": digest.hexdigest(),
    }
    audit_bytes = (json.dumps(audit, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "ascii"
    )
    total = size + len(audit_bytes)
    if total > config.max_output_bytes:
        raise OutputError("FCPXML bundle including audit exceeds total byte limit")
    target = _bundle(
        output_dir,
        (("scenes.fcpxml", _bounded(prepared)), ("audit.json", iter((audit_bytes,)))),
        config.max_output_bytes,
    )
    return FCPXMLExportResult(target, len(cuts), size, total, digest.hexdigest())


__all__ = ["FCPXMLExportConfig", "FCPXMLExportResult", "render_fcpxml", "write_fcpxml_bundle"]
