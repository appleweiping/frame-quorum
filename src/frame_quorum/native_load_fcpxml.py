"""Bounded PySceneDetect scene-list CSV import into an exact CFR FCPXML timeline."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import stat
from dataclasses import dataclass, replace
from fractions import Fraction
from itertools import pairwise
from pathlib import Path
from typing import Any

from .errors import ConfigurationError, OutputError, ScanError
from .fcpxml_export import FCPXMLExportConfig, render_fcpxml
from .native_measurements import _bundle
from .native_splitting import _target_path
from .native_video import (
    NativeVideoConfig,
    NativeVideoStatus,
    NativeVideoStream,
    _fraction,
    _integer,
    _local_path_text,
    _pair,
)
from .otio_export import OTIOCut, OTIOMedia

_MAX_CSV_BYTES = 8 * 1024 * 1024
_MAX_SCENES = 10_000
_MAX_FRAMES = 100_000
_MAX_OUTPUT_BYTES = 64 * 1024 * 1024
_DECIMAL = re.compile(r"[1-9][0-9]*\Z", re.ASCII)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)


@dataclass(frozen=True, slots=True)
class LoadedSceneStarts:
    start_ordinals: tuple[int, ...]
    csv_sha256: str
    row_count: int
    csv_bytes: int

    def __post_init__(self) -> None:
        if (
            type(self.start_ordinals) is not tuple
            or not 1 <= len(self.start_ordinals) <= _MAX_SCENES
            or any(type(value) is not int or not 0 <= value < _MAX_FRAMES for value in self.start_ordinals)
            or self.start_ordinals[0] != 0
            or any(a >= b for a, b in pairwise(self.start_ordinals))
        ):
            raise ConfigurationError("loaded scene starts must be increasing zero-based ordinals")
        if type(self.csv_sha256) is not str or _SHA256.fullmatch(self.csv_sha256) is None:
            raise ConfigurationError("loaded scene CSV digest must be lowercase SHA-256")
        _integer(self.row_count, "row_count", 1, _MAX_SCENES)
        if self.row_count != len(self.start_ordinals):
            raise ConfigurationError("loaded scene row count disagrees with starts")
        _integer(self.csv_bytes, "csv_bytes", 1, _MAX_CSV_BYTES)


@dataclass(frozen=True, slots=True)
class LoadedFCPXMLResult:
    output_dir: Path
    scene_count: int
    frame_count: int
    csv_sha256: str
    fcpxml_bytes: int
    fcpxml_sha256: str
    total_output_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_dir": self.output_dir.as_posix(),
            "scene_count": self.scene_count,
            "frame_count": self.frame_count,
            "csv_sha256": self.csv_sha256,
            "fcpxml_bytes": self.fcpxml_bytes,
            "fcpxml_sha256": self.fcpxml_sha256,
            "total_output_bytes": self.total_output_bytes,
            "source_verified": False,
            "source_content_authenticated": False,
            "editor_import_verified": False,
        }


def _csv_source(path: str | Path, maximum: int) -> bytes:
    if not isinstance(path, (str, Path)):
        raise ConfigurationError("scene CSV must be a local path")
    _local_path_text(str(path))
    try:
        supplied = Path(path).expanduser()
        if supplied.is_symlink():
            raise ScanError("scene CSV must be a regular nonsymlink file")
        resolved = supplied.resolve()
        _local_path_text(str(resolved))
        before = resolved.lstat()
        if not stat.S_ISREG(before.st_mode):
            raise ScanError("scene CSV must be a regular nonsymlink file")
        if before.st_size > maximum:
            raise ScanError("scene CSV exceeds input byte limit")
        with resolved.open("rb") as handle:
            opened = os.fstat(handle.fileno())
            if not stat.S_ISREG(opened.st_mode) or _fingerprint(opened) != _fingerprint(before):
                raise ScanError("scene CSV changed before read")
            payload = handle.read(maximum + 1)
            if len(payload) > maximum:
                raise ScanError("scene CSV exceeds input byte limit")
            if handle.read(1) or _fingerprint(os.fstat(handle.fileno())) != _fingerprint(opened):
                raise ScanError("scene CSV changed while reading")
        if _fingerprint(resolved.lstat()) != _fingerprint(opened):
            raise ScanError("scene CSV changed while reading")
        return payload
    except OSError as error:
        raise ScanError("could not read scene CSV") from error


def _fingerprint(info: os.stat_result) -> tuple[int, int, int, int]:
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns


def _number(text: str, *, maximum: int) -> int:
    if len(text) > len(str(maximum)) or _DECIMAL.fullmatch(text) is None:
        raise ConfigurationError("scene CSV has an invalid positive decimal field")
    value = int(text)
    if value > maximum:
        raise ConfigurationError("scene CSV frame or scene number exceeds limit")
    return value


def _validate_csv_lexical_profile(text: str) -> None:
    """Reject bare quotes and CR-only records before the permissive CSV reader."""
    state = "field_start"
    position = 0
    while position < len(text):
        character = text[position]
        if character == "\r" and (position + 1 >= len(text) or text[position + 1] != "\n"):
            raise ConfigurationError("scene CSV requires every CR to be followed by LF")
        if state == "quoted":
            if character == '"':
                if position + 1 < len(text) and text[position + 1] == '"':
                    position += 2
                    continue
                state = "after_quote"
        elif character == ",":
            state = "field_start"
        elif character in "\r\n":
            if character == "\r":
                position += 1
            state = "field_start"
        elif character == '"' and state == "field_start":
            state = "quoted"
        elif character == '"' or state == "after_quote":
            raise ConfigurationError("scene CSV has invalid field quoting")
        else:
            state = "unquoted"
        position += 1
    if state == "quoted":
        raise ConfigurationError("scene CSV has an unclosed quoted field")


def load_native_scene_csv(
    csv_path: str | Path,
    *,
    max_input_bytes: int = _MAX_CSV_BYTES,
    max_scenes: int = _MAX_SCENES,
) -> LoadedSceneStarts:
    """Parse bounded frozen-style Start Frame CSV without invoking a detector."""
    _integer(max_input_bytes, "max_input_bytes", 1, _MAX_CSV_BYTES)
    _integer(max_scenes, "max_scenes", 1, _MAX_SCENES)
    payload = _csv_source(csv_path, max_input_bytes)
    try:
        decoded = payload.decode("utf-8-sig")
        if "\x00" in decoded:
            raise ConfigurationError("scene CSV contains a NUL character")
        _validate_csv_lexical_profile(decoded)
        reader = iter(csv.reader(io.StringIO(decoded, newline=""), strict=True))
        header = next(reader)
        if not header or header[0] == "Timecode List:":
            header = next(reader)
        if len(set(header)) != len(header) or "Scene Number" not in header or "Start Frame" not in header:
            raise ConfigurationError("scene CSV requires unique Scene Number and Start Frame headers")
        number_column, start_column = header.index("Scene Number"), header.index("Start Frame")
        starts: list[int] = []
        for row in reader:
            if len(row) != len(header) or len(starts) >= max_scenes:
                raise ConfigurationError("scene CSV row width or count is invalid")
            scene_number = _number(row[number_column], maximum=max_scenes)
            start_frame = _number(row[start_column], maximum=_MAX_FRAMES)
            if scene_number != len(starts) + 1 or (starts and start_frame - 1 <= starts[-1]):
                raise ConfigurationError("scene CSV scene numbers or starts are not strictly increasing")
            if not starts and start_frame != 1:
                raise ConfigurationError("scene CSV first start frame must be 1")
            starts.append(start_frame - 1)
        return LoadedSceneStarts(
            tuple(starts), hashlib.sha256(payload).hexdigest(), len(starts), len(payload)
        )
    except (UnicodeError, csv.Error, StopIteration, ValueError) as error:
        raise ConfigurationError("scene CSV is malformed or not UTF-8") from error


def _complete_cfr_video(
    path: str | Path, rate: Fraction, limits: NativeVideoConfig
) -> tuple[Path, Fraction, int, int, int]:
    # One extra returned slot is needed to distinguish an exactly-full file
    # from a frame-limit termination without assuming EOF.
    if limits.max_decoded_frames <= limits.max_frames:
        raise ConfigurationError("max_decoded_frames must exceed max_frames to establish EOF")
    decode = replace(limits, max_frames=limits.max_frames + 1)
    count = 0
    origin: Fraction | None = None
    with NativeVideoStream(path, decode) as stream:
        metadata = stream.metadata
        for frame in stream:
            if count >= limits.max_frames:
                raise ConfigurationError("scene CSV video exceeds configured frame limit")
            if frame.decode_index != count or frame.sample_index != count or frame.generation != 0:
                raise ScanError("native decode ordinals are not contiguous from zero")
            if frame.width != metadata.width or frame.height != metadata.height:
                raise ScanError("video dimensions changed while decoding")
            if origin is None:
                origin = frame.presentation_time
            if frame.presentation_time != origin + Fraction(count, 1) / rate:
                raise ConfigurationError("scene CSV FCPXML export requires exact CFR presentation cadence")
            count += 1
    diagnostics = stream.diagnostics
    if (
        origin is None
        or diagnostics.status is not NativeVideoStatus.EOF
        or not diagnostics.closed
        or diagnostics.cleanup_errors
        or diagnostics.generation != 0
        or diagnostics.decoded_frames != count
        or diagnostics.returned_frames != count
    ):
        raise ConfigurationError("scene CSV FCPXML export requires complete error-free EOF")
    return metadata.path, origin, count, metadata.width, metadata.height


def write_loaded_fcpxml_bundle(
    video_path: str | Path,
    csv_path: str | Path,
    output_dir: str | Path,
    *,
    frame_rate: Fraction,
    final_end: Fraction,
    video_limits: NativeVideoConfig | None = None,
    title: str = "Frame Quorum",
    max_input_bytes: int = _MAX_CSV_BYTES,
    max_scenes: int = _MAX_SCENES,
    max_output_bytes: int = _MAX_OUTPUT_BYTES,
) -> LoadedFCPXMLResult:
    """Compose strict CSV starts and complete observed CFR ordinals into FCPXML."""
    rate = _fraction(frame_rate, "frame_rate", positive=True)
    endpoint = _fraction(final_end, "final_end")
    limits = NativeVideoConfig() if video_limits is None else video_limits
    if type(limits) is not NativeVideoConfig:
        raise ConfigurationError("video_limits must be NativeVideoConfig")
    limits.__post_init__()
    if (
        limits.start is not None
        or limits.end is not None
        or limits.frame_step != 1
        or limits.video_stream != 0
        or limits.max_frames > _MAX_FRAMES
    ):
        raise ConfigurationError("CSV FCPXML requires bounded full-source stream-zero analysis")
    _integer(max_input_bytes, "max_input_bytes", 1, _MAX_CSV_BYTES)
    _integer(max_scenes, "max_scenes", 1, _MAX_SCENES)
    _integer(max_output_bytes, "max_output_bytes", 1, _MAX_OUTPUT_BYTES)
    if limits.max_decoded_frames <= limits.max_frames:
        raise ConfigurationError("max_decoded_frames must exceed max_frames to establish EOF")
    config = FCPXMLExportConfig(
        frame_rate=rate,
        width=1,
        height=1,
        title=title,
        max_clips=max_scenes,
        max_output_bytes=max_output_bytes,
    )
    loaded = load_native_scene_csv(csv_path, max_input_bytes=max_input_bytes, max_scenes=max_scenes)
    _target_path(output_dir)  # Read-only preflight; _bundle rechecks when publishing.
    source, origin, frame_count, width, height = _complete_cfr_video(video_path, rate, limits)
    if loaded.start_ordinals[-1] >= frame_count:
        raise ConfigurationError("scene CSV cut is outside the decoded video")
    if endpoint != origin + Fraction(frame_count, 1) / rate:
        raise ConfigurationError("declared final_end differs from exact CFR frame-count endpoint")
    positions = (*loaded.start_ordinals, frame_count)
    cuts = tuple(
        OTIOCut(origin + Fraction(start, 1) / rate, origin + Fraction(end, 1) / rate)
        for start, end in pairwise(positions)
    )
    config = replace(config, width=width, height=height)
    media = OTIOMedia(source.absolute(), origin, origin, endpoint)
    xml = render_fcpxml(cuts, media, config).encode("utf-8")
    xml_sha256 = hashlib.sha256(xml).hexdigest()
    audit = {
        "kind": "frame-quorum-loaded-fcpxml",
        "schema_version": 1,
        "fcpxml_version": "1.9",
        "scene_count": len(cuts),
        "frame_count": frame_count,
        "csv_bytes": loaded.csv_bytes,
        "csv_sha256": loaded.csv_sha256,
        "frame_rate": _pair(rate),
        "media_origin": _pair(origin),
        "final_end": _pair(endpoint),
        "fcpxml_bytes": len(xml),
        "fcpxml_sha256": xml_sha256,
        "source_verified": False,
        "source_content_authenticated": False,
        "editor_import_verified": False,
    }
    audit_bytes = (json.dumps(audit, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "ascii"
    )
    total = len(xml) + len(audit_bytes)
    if total > max_output_bytes:
        raise OutputError("loaded FCPXML bundle exceeds total byte limit")
    target = _bundle(
        output_dir,
        (("scenes.fcpxml", iter((xml,))), ("audit.json", iter((audit_bytes,)))),
        max_output_bytes,
        expected_files=(
            ("scenes.fcpxml", len(xml), xml_sha256),
            ("audit.json", len(audit_bytes), hashlib.sha256(audit_bytes).hexdigest()),
        ),
    )
    return LoadedFCPXMLResult(target, len(cuts), frame_count, loaded.csv_sha256, len(xml), xml_sha256, total)


__all__ = ["LoadedFCPXMLResult", "LoadedSceneStarts", "load_native_scene_csv", "write_loaded_fcpxml_bundle"]
