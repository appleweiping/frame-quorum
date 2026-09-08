"""Bounded, canonical native measurements and explicitly unverified offline replay.

The cache is data, never executable input. Its digest binds measurements and
historical provenance; neither a digest nor a recorded path authenticates media.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import stat
import tempfile
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, fields
from fractions import Fraction
from pathlib import Path
from typing import Any

from PIL import __version__ as pillow_version

from .errors import ConfigurationError, OutputError, ScanError
from .models import FrameMetrics, _require_safe_text
from .native_scenes import (
    NativeSceneConfig,
    NativeSceneResult,
    NativeSceneSample,
    _analyze_native_samples,
    _collect_native_samples,
)
from .native_splitting import _cleanup, _identity, _managed_file, _publish, _target_path
from .native_video import (
    NativeVideoConfig,
    NativeVideoDiagnostics,
    NativeVideoMetadata,
    NativeVideoStatus,
    _integer,
    _pair,
    _source_path,
)
from .scene_detection import DetectionConfig

_KIND = "frame-quorum-native-measurements"
_METRICS = "frame-quorum-rgb-summary-v1"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True, slots=True)
class NativeMeasurementLimits:
    max_cache_bytes: int = 64 * 1024 * 1024
    max_line_bytes: int = 16 * 1024
    max_samples: int = 100_000
    max_output_bytes: int = 128 * 1024 * 1024

    def __post_init__(self) -> None:
        for name, ceiling in (
            ("max_cache_bytes", 256 * 1024 * 1024),
            ("max_line_bytes", 64 * 1024),
            ("max_samples", 1_000_000),
            ("max_output_bytes", 512 * 1024 * 1024),
        ):
            _integer(getattr(self, name), name, 1, ceiling)


def _limits(value: NativeMeasurementLimits | None) -> NativeMeasurementLimits:
    result = NativeMeasurementLimits() if value is None else value
    if type(result) is not NativeMeasurementLimits:
        raise ConfigurationError("limits must be NativeMeasurementLimits")
    return result


def _video_dict(video: NativeVideoConfig) -> dict[str, Any]:
    result = asdict(video)
    result["start"], result["end"] = _pair(video.start), _pair(video.end)
    return result


@dataclass(frozen=True, slots=True)
class NativeMeasurements:
    video: NativeVideoConfig
    metadata: NativeVideoMetadata
    diagnostics: NativeVideoDiagnostics
    samples: tuple[NativeSceneSample, ...]
    source_sha256: str
    measurement_version: str = _METRICS
    producer_pillow: str = pillow_version
    source_path_label: str | None = None

    def __post_init__(self) -> None:
        if type(self.video) is not NativeVideoConfig or type(self.metadata) is not NativeVideoMetadata:
            raise ConfigurationError("measurements require native configuration and metadata")
        if type(self.diagnostics) is not NativeVideoDiagnostics:
            raise ConfigurationError("measurements require native diagnostics")
        if type(self.source_sha256) is not str or not _SHA256.fullmatch(self.source_sha256):
            raise ConfigurationError("source_sha256 must be a lowercase SHA256 digest")
        if self.measurement_version != _METRICS:
            raise ConfigurationError("unsupported measurement algorithm version")
        if type(self.producer_pillow) is not str or not 1 <= len(self.producer_pillow) <= 64:
            raise ConfigurationError("producer_pillow must be bounded version text")
        _require_safe_text(self.producer_pillow, "producer Pillow version")
        if self.source_path_label is None:
            object.__setattr__(self, "source_path_label", self.metadata.path.as_posix())
        if type(self.source_path_label) is not str or not 1 <= len(self.source_path_label) <= 4096:
            raise ConfigurationError("recorded source path must have 1 to 4096 characters")
        _require_safe_text(self.source_path_label, "recorded source path")
        if len(self.metadata.format_name) > 128 or len(self.metadata.codec_name) > 128:
            raise ConfigurationError("recorded format and codec names exceed 128 characters")
        if type(self.samples) is not tuple or len(self.samples) > self.video.max_frames:
            raise ConfigurationError("samples must be an immutable bounded tuple")
        diag = self.diagnostics
        if (
            not diag.closed
            or diag.cleanup_errors
            or diag.generation != 0
            or diag.status
            not in {
                NativeVideoStatus.EOF,
                NativeVideoStatus.RANGE_END,
                NativeVideoStatus.FRAME_LIMIT,
                NativeVideoStatus.DECODE_LIMIT,
            }
            or diag.returned_frames != len(self.samples)
            or diag.decoded_frames > self.video.max_decoded_frames
            or diag.decoded_pixels_observed < diag.decoded_frames
            or diag.decoded_pixels_observed > self.video.max_total_pixels
            or self.metadata.source_bytes > self.video.max_source_bytes
        ):
            raise ConfigurationError("measurements require consistent successful bounded decode diagnostics")
        if (
            (diag.status is NativeVideoStatus.RANGE_END and self.video.end is None)
            or (diag.status is NativeVideoStatus.FRAME_LIMIT and len(self.samples) != self.video.max_frames)
            or (
                diag.status is NativeVideoStatus.DECODE_LIMIT
                and diag.decoded_frames != self.video.max_decoded_frames
            )
        ):
            raise ConfigurationError("termination disagrees with measurement configuration")
        previous = None
        for position, sample in enumerate(self.samples):
            if type(sample) is not NativeSceneSample:
                raise ConfigurationError("every measurement must be NativeSceneSample")
            if (
                sample.sample_index != position
                or sample.generation != 0
                or sample.decode_index >= diag.decoded_frames
            ):
                raise ConfigurationError("sample position/generation/decode work is inconsistent")
            if previous is not None and (
                sample.decode_index - previous.decode_index != self.video.frame_step
                or sample.presentation_time < previous.presentation_time
            ):
                raise ConfigurationError("measurement stride or presentation order is inconsistent")
            if (self.video.start is not None and sample.presentation_time < self.video.start) or (
                self.video.end is not None and sample.presentation_time >= self.video.end
            ):
                raise ConfigurationError("measurement is outside its original presentation interval")
            previous = sample

    def header(self) -> dict[str, Any]:
        metadata = self.metadata.to_dict()
        metadata["path"] = self.source_path_label
        return {
            "kind": _KIND,
            "schema_version": 1,
            "measurement_version": self.measurement_version,
            "producer_pillow": self.producer_pillow,
            "source_sha256": self.source_sha256,
            "video": _video_dict(self.video),
            "metadata": metadata,
            "diagnostics": self.diagnostics.to_dict(),
            "sample_count": len(self.samples),
        }

    @property
    def digest(self) -> str:
        digest = hashlib.sha256()
        for line in _measurement_lines(self):
            digest.update(line)
        return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class NativeReplayResult:
    """Historical-data analysis, deliberately not a fresh NativeSceneResult subtype."""

    measurement_digest: str
    analysis: NativeSceneResult
    source_path_label: str | None = None

    def __post_init__(self) -> None:
        if type(self.measurement_digest) is not str or not _SHA256.fullmatch(self.measurement_digest):
            raise ConfigurationError("measurement_digest must be a lowercase SHA256 digest")
        if type(self.analysis) is not NativeSceneResult:
            raise ConfigurationError("analysis must be NativeSceneResult")
        if self.source_path_label is None:
            object.__setattr__(self, "source_path_label", self.analysis.metadata.path.as_posix())
        if type(self.source_path_label) is not str or not 1 <= len(self.source_path_label) <= 4096:
            raise ConfigurationError("recorded source path must have 1 to 4096 characters")
        _require_safe_text(self.source_path_label, "recorded source path")

    @property
    def source_verified(self) -> bool:
        return False

    def to_dict(self) -> dict[str, Any]:
        analysis = self.analysis.to_dict()
        analysis["metadata"]["path"] = self.source_path_label
        return {
            "kind": "frame-quorum-native-replay",
            "schema_version": 1,
            "execution": "cached_measurements",
            "source_verified": False,
            "measurement_digest": self.measurement_digest,
            "analysis": analysis,
        }


def analyze_native_measurements(
    measurements: NativeMeasurements,
    *,
    detectors: tuple[DetectionConfig, ...] = (DetectionConfig(),),
    minimum_votes: int = 1,
    min_scene_samples: int = 1,
) -> NativeReplayResult:
    """Recompute all decisions from raw metrics without opening any media path.

    Detector parameters may change. Stream, interval, sampling, limits and
    measurement algorithm are immutable historical identity, not overrides.
    """
    if type(measurements) is not NativeMeasurements:
        raise ConfigurationError("measurements must be NativeMeasurements")
    config = NativeSceneConfig(measurements.video, detectors, minimum_votes, min_scene_samples)
    result = _analyze_native_samples(
        config, measurements.metadata, measurements.diagnostics, measurements.samples
    )
    return NativeReplayResult(measurements.digest, result, measurements.source_path_label)


def _fingerprint(info: os.stat_result) -> tuple[int, int, int, int]:
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns


def _hash_source(path: Path, maximum: int) -> tuple[str, tuple[int, int, int, int]]:
    observed = path.lstat()
    if not stat.S_ISREG(observed.st_mode) or observed.st_size > maximum:
        raise ScanError("source hash requires a regular file within source byte limit")
    with _managed_file(path) as handle:
        info = os.fstat(handle.fileno())
        if not stat.S_ISREG(info.st_mode) or _fingerprint(info) != _fingerprint(observed):
            raise ScanError("source changed before hashing")
        initial = _fingerprint(info)
        digest = hashlib.sha256()
        remaining = info.st_size
        while remaining:
            chunk = handle.read(min(64 * 1024, remaining))
            if not chunk:
                raise ScanError("source changed while hashing")
            digest.update(chunk)
            remaining -= len(chunk)
        if handle.read(1) or _fingerprint(os.fstat(handle.fileno())) != initial:
            raise ScanError("source changed while hashing")
    return digest.hexdigest(), initial


def capture_native_measurements(
    path: str | Path,
    config: NativeVideoConfig | None = None,
    *,
    on_sample: Callable[[NativeSceneSample], None] | None = None,
    limits: NativeMeasurementLimits | None = None,
) -> NativeMeasurements:
    """Hash the whole local source before and after one bounded measurement decode.

    The two bounded hash passes and observed stat identities detect ordinary
    source mutation, not adversarial swap-and-restore or compromised decoders.
    """
    bounds = _limits(limits)
    options = NativeVideoConfig() if config is None else config
    scene_options = NativeSceneConfig(video=options)
    if options.max_frames > bounds.max_samples:
        raise ConfigurationError("video.max_frames exceeds cache sample admission")
    source = _source_path(path)
    try:
        before, identity = _hash_source(source, options.max_source_bytes)
        metadata, diagnostics, samples = _collect_native_samples(source, scene_options, on_sample)
        after, final_identity = _hash_source(source, options.max_source_bytes)
        if before != after or identity != final_identity or _fingerprint(source.stat()) != identity:
            raise ScanError("source changed across measurement capture")
        result = NativeMeasurements(options, metadata, diagnostics, samples, before)
        # Validate the persistable form under the requested limits before success.
        for _ in _cache_lines(result, bounds):
            pass
        return result
    except OSError as error:
        raise ScanError("native measurement source read failed") from error


def _canonical(value: dict[str, Any]) -> bytes:
    try:
        return (
            json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
            + "\n"
        ).encode("ascii")
    except (TypeError, ValueError, RecursionError, OverflowError) as error:
        raise ConfigurationError("measurement data must be finite canonical JSON") from error


def _measurement_lines(data: NativeMeasurements) -> Iterator[bytes]:
    yield _canonical(data.header())
    for sample in data.samples:
        yield _canonical(sample.to_dict())


def _cache_lines(data: NativeMeasurements, limits: NativeMeasurementLimits) -> Iterator[bytes]:
    if len(data.samples) > limits.max_samples:
        raise ConfigurationError("measurement count exceeds cache sample limit")
    digest = hashlib.sha256()
    total = 0
    for line in _measurement_lines(data):
        if len(line) > limits.max_line_bytes:
            raise ConfigurationError("measurement record exceeds cache line limit")
        total += len(line)
        if total > limits.max_cache_bytes:
            raise ConfigurationError("measurements exceed cache byte limit")
        digest.update(line)
        yield line
    footer = _canonical({"kind": "end", "sample_count": len(data.samples), "sha256": digest.hexdigest()})
    if len(footer) > limits.max_line_bytes or total + len(footer) > limits.max_cache_bytes:
        raise ConfigurationError("measurement footer exceeds cache limits")
    yield footer


def _keys(value: Any, names: set[str]) -> dict[str, Any]:
    if type(value) is not dict or value.keys() != names:
        raise ConfigurationError("measurement object has missing, unknown or duplicate fields")
    return value


def _pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in items:
        if key in result:
            raise ConfigurationError("duplicate JSON field")
        result[key] = value
    return result


def _json_integer(token: str) -> int:
    # Derived native presentation numerators may require 127 bits (39 decimal
    # digits); ordinary fields are narrowed further by their schema validators.
    if len(token.lstrip("-")) > 39:
        raise ConfigurationError("JSON integer exceeds 39 digits before conversion")
    return int(token)


def _json_float(token: str) -> float:
    if len(token) > 64:
        raise ConfigurationError("JSON float token exceeds 64 characters before conversion")
    return float(token)


def _reject_constant(token: str) -> None:
    raise ConfigurationError("nonfinite JSON constants are not measurements")


def _preflight_json(line: bytes) -> None:
    if not line.isascii():
        raise ConfigurationError("cache JSON must use canonical ASCII escapes")
    depth = 0
    quoted = escaped = False
    for byte in line:
        if quoted:
            if escaped:
                escaped = False
            elif byte == 92:
                escaped = True
            elif byte == 34:
                quoted = False
        elif byte == 34:
            quoted = True
        elif byte in (91, 123):
            depth += 1
            if depth > 16:
                raise ConfigurationError("cache JSON exceeds 16 structural levels")
        elif byte in (93, 125):
            depth -= 1


def _rational(value: Any, *, optional: bool = False) -> Fraction | None:
    if value is None and optional:
        return None
    parts = _keys(value, {"numerator", "denominator"})
    n = _integer(parts["numerator"], "rational numerator", -(1 << 63) + 1, (1 << 63) - 1)
    d = _integer(parts["denominator"], "rational denominator", 1, (1 << 63) - 1)
    result = Fraction(n, d)
    if _pair(result) != parts:
        raise ConfigurationError("rational coordinates must be reduced")
    return result


def _parse_header(raw: dict[str, Any]) -> dict[str, Any]:
    _keys(
        raw,
        {
            "kind",
            "schema_version",
            "measurement_version",
            "producer_pillow",
            "source_sha256",
            "video",
            "metadata",
            "diagnostics",
            "sample_count",
        },
    )
    if raw["kind"] != _KIND or type(raw["schema_version"]) is not int or raw["schema_version"] != 1:
        raise ConfigurationError("unsupported native measurements schema")
    if raw["measurement_version"] != _METRICS:
        raise ConfigurationError("unsupported measurement algorithm version")
    video = dict(_keys(raw["video"], {item.name for item in fields(NativeVideoConfig)}))
    video["start"], video["end"] = (
        _rational(video["start"], optional=True),
        _rational(video["end"], optional=True),
    )
    metadata = dict(
        _keys(
            raw["metadata"],
            {
                "path",
                "source_bytes",
                "format",
                "codec",
                "stream_index",
                "width",
                "height",
                "time_base",
                "start_pts",
                "duration_pts",
                "average_rate",
                "base_rate",
            },
        )
    )
    if type(metadata["path"]) is not str:
        raise ConfigurationError("recorded source path must be text")
    source_path_label = metadata["path"]
    metadata["path"] = Path(source_path_label)
    metadata["format_name"], metadata["codec_name"] = metadata.pop("format"), metadata.pop("codec")
    metadata["time_base"] = _rational(metadata["time_base"])
    for name in ("average_rate", "base_rate"):
        metadata[name] = _rational(metadata[name], optional=True)
    diagnostics = dict(_keys(raw["diagnostics"], {item.name for item in fields(NativeVideoDiagnostics)}))
    if type(diagnostics["cleanup_errors"]) is not list:
        raise ConfigurationError("cleanup_errors must be an array")
    diagnostics["cleanup_errors"] = tuple(diagnostics["cleanup_errors"])
    diagnostics["status"] = NativeVideoStatus(diagnostics["status"])
    return {
        "video": NativeVideoConfig(**video),
        "metadata": NativeVideoMetadata(**metadata),
        "diagnostics": NativeVideoDiagnostics(**diagnostics),
        "source_sha256": raw["source_sha256"],
        "measurement_version": raw["measurement_version"],
        "producer_pillow": raw["producer_pillow"],
        "source_path_label": source_path_label,
    }


def _parse_sample(raw: dict[str, Any]) -> NativeSceneSample:
    _keys(
        raw,
        {"pts", "time_base", "presentation_time", "decode_index", "sample_index", "generation", "metrics"},
    )
    metrics = dict(_keys(raw["metrics"], {item.name for item in fields(FrameMetrics)}))
    hashed = metrics["perceptual_hash"]
    if type(hashed) is not str or re.fullmatch(r"[0-9a-f]{16}", hashed) is None:
        raise ConfigurationError("perceptual hash must be exactly 16 lowercase hex digits")
    metrics["perceptual_hash"] = int(hashed, 16)
    time_base = _rational(raw["time_base"])
    assert time_base is not None  # _rational without optional=True rejects absence.
    sample = NativeSceneSample(
        raw["pts"],
        time_base,
        raw["decode_index"],
        raw["sample_index"],
        raw["generation"],
        FrameMetrics(**metrics),
    )
    if _canonical(sample.to_dict()) != _canonical(raw):
        raise ConfigurationError("sample derived presentation time contradicts exact PTS")
    return sample


def read_native_measurements(
    path: str | Path,
    *,
    limits: NativeMeasurementLimits | None = None,
) -> NativeMeasurements:
    """Read a complete canonical cache; never open the recorded source-media path."""
    bounds = _limits(limits)
    source = _source_path(path)
    try:
        observed = source.lstat()
        if not stat.S_ISREG(observed.st_mode) or observed.st_size > bounds.max_cache_bytes:
            raise ConfigurationError("cache must be a regular file within byte limit")
        with _managed_file(source) as handle:
            info = os.fstat(handle.fileno())
            if not stat.S_ISREG(info.st_mode) or _fingerprint(info) != _fingerprint(observed):
                raise ConfigurationError("cache changed before reading")
            consumed = 0

            def record() -> tuple[bytes, dict[str, Any]]:
                nonlocal consumed
                line = handle.readline(min(bounds.max_line_bytes, bounds.max_cache_bytes - consumed) + 1)
                consumed += len(line)
                if (
                    not line.endswith(b"\n")
                    or len(line) > bounds.max_line_bytes
                    or consumed > bounds.max_cache_bytes
                ):
                    raise ConfigurationError("cache has a missing, truncated or oversized record")
                _preflight_json(line)
                value = json.loads(
                    line,
                    object_pairs_hook=_pairs,
                    parse_int=_json_integer,
                    parse_float=_json_float,
                    parse_constant=_reject_constant,
                )
                if type(value) is not dict or _canonical(value) != line:
                    raise ConfigurationError("cache record is not canonical JSON")
                return line, value

            first, header = record()
            count = _integer(header.get("sample_count"), "sample_count", 0, bounds.max_samples)
            arguments = _parse_header(header)
            if count > arguments["video"].max_frames or count != arguments["diagnostics"].returned_frames:
                raise ConfigurationError("declared count contradicts decode configuration/diagnostics")
            digest = hashlib.sha256(first)
            samples = []
            for _ in range(count):
                line, raw = record()
                samples.append(_parse_sample(raw))
                digest.update(line)
            _, footer = record()
            _keys(footer, {"kind", "sample_count", "sha256"})
            _integer(footer["sample_count"], "footer sample_count", 0, bounds.max_samples)
            if footer != {"kind": "end", "sample_count": count, "sha256": digest.hexdigest()}:
                raise ConfigurationError("cache completion count or checksum disagrees")
            if (
                handle.read(1)
                or consumed != info.st_size
                or _fingerprint(os.fstat(handle.fileno())) != _fingerprint(info)
            ):
                raise ConfigurationError("cache has trailing bytes or changed while reading")
            result = NativeMeasurements(samples=tuple(samples), **arguments)
            if _canonical(result.header()) != _canonical(header):
                raise ConfigurationError("cache header is not in normalized form")
            return result
    except (ValueError, TypeError, RecursionError, OverflowError, UnicodeError) as error:
        raise ConfigurationError("malformed native measurements cache") from error
    except OSError as error:
        raise ScanError("native measurement cache read failed") from error


def _bundle(
    output_dir: str | Path,
    files: tuple[tuple[str, Iterator[bytes]], ...],
    maximum: int,
) -> Path:
    target = _target_path(output_dir)
    stage = Path(tempfile.mkdtemp(prefix=".frame-quorum-measurements-", dir=target.parent))
    try:
        identity = _identity(stage.lstat())
    except BaseException as error:
        detail = f"could not establish staging identity; inspect unremoved directory {stage}"
        if not isinstance(error, Exception):
            error.add_note(detail)
            raise
        raise OutputError(detail) from error
    owned: dict[Path, tuple[int, int]] = {}
    attempted = False
    total = 0
    try:
        for name, chunks in files:
            with _managed_file(stage / name, owned) as handle:
                for chunk in chunks:
                    total += len(chunk)
                    if total > maximum:
                        raise OutputError("measurement output exceeds total byte limit")
                    if handle.write(chunk) != len(chunk):
                        raise OutputError("measurement output encountered a short write")
        attempted = True
        _publish(stage, target)
        return target
    except BaseException as error:
        if attempted:
            error.add_note(f"publication was attempted; inspect {target} before retrying")
        _cleanup(stage, error, owned, identity)
        if isinstance(error, OSError):
            raise OutputError("measurement output publication failed") from error
        raise


def write_native_measurements(
    measurements: NativeMeasurements,
    output_dir: str | Path,
    *,
    limits: NativeMeasurementLimits | None = None,
) -> Path:
    """Publish a new directory containing measurements.fqm.jsonl, without replacement."""
    if type(measurements) is not NativeMeasurements:
        raise ConfigurationError("measurements must be NativeMeasurements")
    bounds = _limits(limits)
    return (
        _bundle(
            output_dir,
            (("measurements.fqm.jsonl", _cache_lines(measurements, bounds)),),
            bounds.max_output_bytes,
        )
        / "measurements.fqm.jsonl"
    )


def _statistics(result: NativeReplayResult) -> Iterator[bytes]:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")

    def row(values: tuple[Any, ...]) -> bytes:
        output.seek(0)
        output.truncate()
        writer.writerow(values)
        return output.getvalue().encode("ascii")

    yield row(
        (
            "sample_index",
            "decode_index",
            "pts",
            "time_base_numerator",
            "time_base_denominator",
            "presentation_numerator",
            "presentation_denominator",
            "luminance",
            "content_score",
            "detector",
            "detector_score",
            "candidate",
            "qualified",
            "detector_reason",
            "votes",
            "accepted",
            "reason",
        )
    )
    for statistic in result.analysis.statistics:
        sample = statistic.sample
        for detector in statistic.detectors:
            yield row(
                (
                    sample.sample_index,
                    sample.decode_index,
                    sample.pts,
                    sample.time_base.numerator,
                    sample.time_base.denominator,
                    sample.presentation_time.numerator,
                    sample.presentation_time.denominator,
                    sample.metrics.luminance,
                    statistic.content_score,
                    detector.detector,
                    detector.score,
                    str(detector.candidate).lower(),
                    str(detector.qualified).lower(),
                    detector.reason,
                    statistic.votes,
                    str(statistic.accepted).lower(),
                    statistic.reason,
                )
            )


def render_native_detection_csv(result: NativeReplayResult, *, max_bytes: int = 128 * 1024 * 1024) -> str:
    if type(result) is not NativeReplayResult:
        raise ConfigurationError("result must be NativeReplayResult")
    _integer(max_bytes, "max_bytes", 1, 512 * 1024 * 1024)
    output = bytearray()
    for chunk in _statistics(result):
        if len(output) + len(chunk) > max_bytes:
            raise OutputError("native statistics exceed byte limit")
        output.extend(chunk)
    return output.decode("ascii")


def write_native_replay(
    result: NativeReplayResult,
    output_dir: str | Path,
    *,
    limits: NativeMeasurementLimits | None = None,
) -> Path:
    if type(result) is not NativeReplayResult:
        raise ConfigurationError("result must be NativeReplayResult")
    bounds = _limits(limits)

    if len(result.analysis.statistics) > bounds.max_samples:
        raise ConfigurationError("replay count exceeds sample limit")

    def report() -> Iterator[bytes]:
        # iterencode bounds output growth; the analysis.to_dict allocation remains
        # bounded by the admitted sample x detector ceiling, not by max_output_bytes.
        encoder = json.JSONEncoder(sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
        for text in encoder.iterencode(result.to_dict()):
            yield text.encode("ascii")
        yield b"\n"

    return _bundle(
        output_dir,
        (("replay.json", report()), ("statistics.csv", _statistics(result))),
        bounds.max_output_bytes,
    )


__all__ = [
    "NativeMeasurementLimits",
    "NativeMeasurements",
    "NativeReplayResult",
    "analyze_native_measurements",
    "capture_native_measurements",
    "read_native_measurements",
    "render_native_detection_csv",
    "write_native_measurements",
    "write_native_replay",
]
