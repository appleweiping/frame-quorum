"""Exact native pixel-pair evidence and explicitly unverified offline replay."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, Literal, cast

from .errors import ConfigurationError, ScanError
from .metrics import measure_image
from .native_measurements import (
    NativeMeasurementLimits,
    NativeMeasurements,
    _bounded_cache_lines,
    _bundle,
    _canonical,
    _capture_source_records,
    _keys,
    _limits,
    _parse_header,
    _parse_sample,
    _read_cache,
)
from .native_scenes import (
    NativeDetectorStatistic,
    NativeScene,
    NativeSceneSample,
    _collect_native_records,
    _partition_native_samples,
)
from .native_video import NativeVideoConfig, NativeVideoFrame, _integer, _pair, _source_path
from .pixel_changes import (
    PixelChange,
    PixelChangeConfig,
    PixelChangeLimits,
    PixelChangeWeights,
    _change_config,
    _change_limits,
    _measure_rgb_change,
    _pixel_count,
)
from .pixel_histograms import _owned_image
from .scene_detection import DetectionConfig, _adaptive_scores, _decisions, _number

_KIND = "frame-quorum-native-pixel-changes"
_MEASUREMENT = "frame-quorum-circular-hsv-gradient-v1"
_ZERO = (0.0, 0.0, 0.0, 0.0)


@dataclass(frozen=True, slots=True)
class NativePixelChangeSample:
    sample: NativeSceneSample
    width: int
    height: int
    change: PixelChange | None

    def __post_init__(self) -> None:
        if type(self.sample) is not NativeSceneSample:
            raise ConfigurationError("pixel sample requires native coordinates")
        _pixel_count(self.width, self.height)
        if (self.change is None) != (self.sample.sample_index == 0):
            raise ConfigurationError("only the first sample has absent pair evidence")
        if self.change is not None and (
            type(self.change) is not PixelChange
            or (self.change.width, self.change.height) != (self.width, self.height)
        ):
            raise ConfigurationError("pixel evidence dimensions disagree with the sample")


def _area(width: int | None, height: int | None, count: int) -> int:
    if count == 0:
        if width is not None or height is not None:
            raise ConfigurationError("empty pixel measurements must have absent dimensions")
        return 0
    if type(width) is not int or type(height) is not int:
        raise ConfigurationError("nonempty pixel measurements require actual dimensions")
    return _pixel_count(width, height)


def _work_pixels(count: int, area: int) -> int:
    # Initial snapshot admission, then two full images for every compared pair.
    return 0 if count == 0 else (2 * count - 1) * area


@dataclass(frozen=True, slots=True)
class NativePixelChangeMeasurements:
    base: NativeMeasurements
    config: PixelChangeConfig
    width: int | None
    height: int | None
    changes: tuple[PixelChange | None, ...]

    def __post_init__(self) -> None:
        if type(self.base) is not NativeMeasurements or type(self.config) is not PixelChangeConfig:
            raise ConfigurationError("pixel measurements require base measurements and PixelChangeConfig")
        if type(self.changes) is not tuple or len(self.changes) != len(self.base.samples):
            raise ConfigurationError("every native sample requires one immutable pair record")
        count = len(self.changes)
        area = _area(self.width, self.height, count)
        if area > self.base.video.max_frame_pixels or self.measurement_pixels > 1_000_000_000:
            raise ConfigurationError("pixel dimensions or work exceed the capture/compiled ceiling")
        unselected = self.base.diagnostics.decoded_frames - count
        if area * count + unselected > self.base.diagnostics.decoded_pixels_observed:
            raise ConfigurationError("observed decode pixels cannot cover selected and unselected frames")
        for index, change in enumerate(self.changes):
            if index == 0:
                if change is not None:
                    raise ConfigurationError("first sample must not invent prior pixel evidence")
            elif (
                type(change) is not PixelChange
                or change.config != self.config
                or (change.width, change.height) != (self.width, self.height)
            ):
                raise ConfigurationError("every later pair must have matching configuration and dimensions")

    @property
    def measurement_pixels(self) -> int:
        return _work_pixels(len(self.changes), _area(self.width, self.height, len(self.changes)))

    def header(self) -> dict[str, Any]:
        return {
            "kind": _KIND,
            "schema_version": 1,
            "measurement_version": _MEASUREMENT,
            "pixel_config": asdict(self.config),
            "width": self.width,
            "height": self.height,
            "sample_count": len(self.changes),
            "base": self.base.header(),
        }

    @property
    def digest(self) -> str:
        digest = hashlib.sha256()
        for line in _lines(self):
            digest.update(line)
        return digest.hexdigest()


def _admit_capacity(
    video: NativeVideoConfig, bounds: NativeMeasurementLimits, pixels: PixelChangeLimits
) -> None:
    if video.max_frames > bounds.max_samples:
        raise ConfigurationError("video.max_frames exceeds pixel sample admission")
    if 6 * video.max_frame_pixels > pixels.max_pair_rgb_bytes:
        raise ConfigurationError("declared maximum two-frame RGB capacity exceeds pair byte limit")


def _admit(
    data: NativePixelChangeMeasurements, bounds: NativeMeasurementLimits, pixels: PixelChangeLimits
) -> None:
    if type(data) is not NativePixelChangeMeasurements:
        raise ConfigurationError("measurements must be NativePixelChangeMeasurements, not older summaries")
    _admit_capacity(data.base.video, bounds, pixels)
    if data.measurement_pixels > pixels.max_measurement_pixels:
        raise ConfigurationError("pixel work exceeds measurement limit")


def capture_native_pixel_changes(
    path: str | Path,
    video: NativeVideoConfig | None = None,
    *,
    config: PixelChangeConfig | None = None,
    on_sample: Callable[[NativePixelChangeSample], None] | None = None,
    limits: NativeMeasurementLimits | None = None,
    pixel_limits: PixelChangeLimits | None = None,
) -> NativePixelChangeMeasurements:
    """One decode retains at most previous/current RGB, not per-frame pixel maps."""
    bounds, pixels = _limits(limits), _change_limits(pixel_limits)
    options = NativeVideoConfig() if video is None else video
    if type(options) is not NativeVideoConfig:
        raise ConfigurationError("video must be NativeVideoConfig")
    acquisition = _change_config(config)
    _admit_capacity(options, bounds, pixels)
    previous: tuple[int, int, bytes] | None = None
    spent = 0

    def measure(frame: NativeVideoFrame) -> NativePixelChangeSample:
        nonlocal previous, spent
        area = _pixel_count(frame.width, frame.height)
        if previous is not None and previous[:2] != (frame.width, frame.height):
            raise ConfigurationError("pixel change capture rejects changing frame dimensions")
        charge = area if previous is None else 2 * area
        if spent + charge > pixels.max_measurement_pixels:
            raise ConfigurationError("pixel work exceeds measurement limit")
        spent += charge
        with _owned_image(frame.image()) as image:
            metrics = measure_image(image)
        change = (
            None
            if previous is None
            else _measure_rgb_change(previous[2], frame.rgb, frame.width, frame.height, acquisition, pixels)
        )
        sample = NativeSceneSample(
            frame.pts, frame.time_base, frame.decode_index, frame.sample_index, frame.generation, metrics
        )
        previous = (frame.width, frame.height, frame.rgb)
        return NativePixelChangeSample(sample, frame.width, frame.height, change)

    try:
        digest, metadata, diagnostics, rows = _capture_source_records(
            _source_path(path),
            options,
            lambda source: _collect_native_records(source, options, measure, on_sample),
        )
        base = NativeMeasurements(options, metadata, diagnostics, tuple(row.sample for row in rows), digest)
        result = NativePixelChangeMeasurements(
            base,
            acquisition,
            rows[0].width if rows else None,
            rows[0].height if rows else None,
            tuple(row.change for row in rows),
        )
        _admit(result, bounds, pixels)
        for _ in _bounded_cache_lines(_lines(result), len(rows), bounds):
            pass
        return result
    except OSError as error:
        raise ScanError("native pixel change capture failed") from error
    finally:
        previous = None


def _lines(data: NativePixelChangeMeasurements) -> Iterator[bytes]:
    yield _canonical(data.header())
    for sample, change in zip(data.base.samples, data.changes, strict=True):
        yield _canonical({"sample": sample.to_dict(), "change": None if change is None else change.to_dict()})


def write_native_pixel_changes(
    data: NativePixelChangeMeasurements,
    output_dir: str | Path,
    *,
    limits: NativeMeasurementLimits | None = None,
    pixel_limits: PixelChangeLimits | None = None,
) -> Path:
    bounds = _limits(limits)
    _admit(data, bounds, _change_limits(pixel_limits))
    return (
        _bundle(
            output_dir,
            (("pixel-changes.fqm.jsonl", _bounded_cache_lines(_lines(data), len(data.changes), bounds)),),
            bounds.max_output_bytes,
        )
        / "pixel-changes.fqm.jsonl"
    )


def read_native_pixel_changes(
    path: str | Path,
    *,
    limits: NativeMeasurementLimits | None = None,
    pixel_limits: PixelChangeLimits | None = None,
) -> NativePixelChangeMeasurements:
    """Read only sufficient pair evidence; never upgrade summary or histogram data."""
    bounds, pixels = _limits(limits), _change_limits(pixel_limits)

    def prepare(
        header: dict[str, Any], count: int
    ) -> tuple[
        Callable[[dict[str, Any]], NativePixelChangeSample],
        Callable[[tuple[NativePixelChangeSample, ...]], NativePixelChangeMeasurements],
    ]:
        _keys(
            header,
            {
                "kind",
                "schema_version",
                "measurement_version",
                "pixel_config",
                "width",
                "height",
                "sample_count",
                "base",
            },
        )
        if (
            header["kind"] != _KIND
            or type(header["schema_version"]) is not int
            or header["schema_version"] != 1
            or header["measurement_version"] != _MEASUREMENT
        ):
            raise ConfigurationError(
                "unsupported pixel change schema; older measurements lack pixel-pair evidence"
            )
        config = PixelChangeConfig(**_keys(header["pixel_config"], {"edge_radius"}))
        arguments = _parse_header(header["base"])
        base_count = _integer(header["base"]["sample_count"], "base sample count", 0, bounds.max_samples)
        _admit_capacity(arguments["video"], bounds, pixels)
        if (
            count != base_count
            or count > arguments["video"].max_frames
            or count != arguments["diagnostics"].returned_frames
        ):
            raise ConfigurationError("pixel count contradicts base configuration/diagnostics")
        width, height = header["width"], header["height"]
        area = _area(width, height, count)
        if _work_pixels(count, area) > pixels.max_measurement_pixels:
            raise ConfigurationError("pixel work exceeds measurement limit before record parsing")
        if area > arguments["video"].max_frame_pixels:
            raise ConfigurationError("pixel dimensions exceed original frame admission")
        if (
            count * area + arguments["diagnostics"].decoded_frames - count
            > arguments["diagnostics"].decoded_pixels_observed
        ):
            raise ConfigurationError("pixel dimensions contradict observed decode work")

        def parse(raw: dict[str, Any]) -> NativePixelChangeSample:
            _keys(raw, {"sample", "change"})
            change = None
            if raw["change"] is not None:
                values = _keys(
                    raw["change"], {"width", "height", "hue_sum", "saturation_sum", "value_sum", "edge_sum"}
                )
                change = PixelChange(config, **values)
            row = NativePixelChangeSample(
                _parse_sample(raw["sample"]), cast(int, width), cast(int, height), change
            )
            return row

        def finish(rows: tuple[NativePixelChangeSample, ...]) -> NativePixelChangeMeasurements:
            base = NativeMeasurements(samples=tuple(row.sample for row in rows), **arguments)
            result = NativePixelChangeMeasurements(
                base, config, width, height, tuple(row.change for row in rows)
            )
            _admit(result, bounds, pixels)
            return result

        return parse, finish

    return _read_cache(path, bounds, prepare)


@dataclass(frozen=True, slots=True)
class PixelChangeDetectionConfig:
    detector: Literal["content", "adaptive"] = "content"
    weights: PixelChangeWeights = field(default_factory=PixelChangeWeights)
    value_only: bool = False
    threshold: float = 0.30
    min_scene_samples: int = 1
    window_radius: int = 2
    adaptive_ratio: float = 3.0
    min_content: float = 0.15

    def __post_init__(self) -> None:
        if type(self.detector) is not str or self.detector not in {"content", "adaptive"}:
            raise ConfigurationError("pixel detector must be content or adaptive")
        if type(self.weights) is not PixelChangeWeights or type(self.value_only) is not bool:
            raise ConfigurationError("pixel replay requires PixelChangeWeights and a boolean value_only")
        _policy(self).validate()


def _policy(config: PixelChangeDetectionConfig) -> DetectionConfig:
    return DetectionConfig(
        detector=config.detector,
        threshold=config.threshold,
        min_scene_frames=config.min_scene_samples,
        window_radius=config.window_radius,
        adaptive_ratio=config.adaptive_ratio,
        min_content=config.min_content,
    )


@dataclass(frozen=True, slots=True)
class PixelChangeStatistic:
    weighted_score: float
    score: float | None
    candidate: bool
    accepted: bool
    reason: str

    def __post_init__(self) -> None:
        _number(self.weighted_score, "weighted pixel score", minimum=0, maximum=1)
        # Reuse the shared scored-candidate/reason invariant checker. The complete
        # result additionally recomputes the exact selected policy and evidence.
        NativeDetectorStatistic("adaptive", self.score, self.candidate, self.accepted, self.reason)


def _statistics(
    data: NativePixelChangeMeasurements, config: PixelChangeDetectionConfig
) -> tuple[PixelChangeStatistic, ...]:
    weighted = [
        0.0 if change is None else change.components[2] if config.value_only else change.score(config.weights)
        for change in data.changes
    ]
    scores: list[float | None]
    if config.detector == "adaptive":
        scores = _adaptive_scores(weighted, config.window_radius)
        candidates = {
            index: "adaptive_peak"
            for index, score in enumerate(scores)
            if index > 0
            and score is not None
            and score >= config.adaptive_ratio
            and weighted[index] >= config.min_content
        }
    else:
        scores = list(weighted)
        candidates = {
            index: "distance_threshold"
            for index, score in enumerate(weighted)
            if index > 0 and score >= config.threshold
        }
    return tuple(
        PixelChangeStatistic(weighted[index], scores[index], index in candidates, accepted, reason)
        for index, (accepted, reason) in enumerate(_decisions(scores, candidates, _policy(config)))
    )


@dataclass(frozen=True, slots=True)
class NativePixelChangeReplayResult:
    measurements: NativePixelChangeMeasurements
    config: PixelChangeDetectionConfig
    statistics: tuple[PixelChangeStatistic, ...]

    def __post_init__(self) -> None:
        if (
            type(self.measurements) is not NativePixelChangeMeasurements
            or type(self.config) is not PixelChangeDetectionConfig
        ):
            raise ConfigurationError("pixel replay requires typed measurements and detection configuration")
        if (
            type(self.statistics) is not tuple
            or len(self.statistics) != len(self.measurements.changes)
            or any(type(row) is not PixelChangeStatistic for row in self.statistics)
        ):
            raise ConfigurationError("pixel replay statistics must match every stored sample")
        if self.statistics != _statistics(self.measurements, self.config):
            raise ConfigurationError("pixel replay scores or decisions contradict measured integer evidence")

    @property
    def source_verified(self) -> bool:
        return False

    @property
    def cut_positions(self) -> tuple[int, ...]:
        return tuple(index for index, row in enumerate(self.statistics) if row.accepted)

    @property
    def cut_times(self) -> tuple[Fraction, ...]:
        return tuple(self.measurements.base.samples[index].presentation_time for index in self.cut_positions)

    @property
    def scenes(self) -> tuple[NativeScene, ...]:
        base = self.measurements.base
        return _partition_native_samples(base.samples, list(self.cut_positions), base.video, base.diagnostics)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "frame-quorum-native-pixel-change-replay",
            "schema_version": 1,
            "execution": "cached_pixel_changes",
            "source_verified": False,
            "measurement_digest": self.measurements.digest,
            "capture": self.measurements.header(),
            "detector": asdict(self.config),
            "cut_positions": list(self.cut_positions),
            "cut_times": [_pair(value) for value in self.cut_times],
            "scenes": [scene.to_dict() for scene in self.scenes],
            "statistics": [asdict(row) for row in self.statistics],
        }


def analyze_native_pixel_changes(
    data: NativePixelChangeMeasurements,
    config: PixelChangeDetectionConfig | None = None,
    *,
    limits: NativeMeasurementLimits | None = None,
    pixel_limits: PixelChangeLimits | None = None,
) -> NativePixelChangeReplayResult:
    options = PixelChangeDetectionConfig() if config is None else config
    if type(options) is not PixelChangeDetectionConfig:
        raise ConfigurationError("config must be PixelChangeDetectionConfig")
    _admit(data, _limits(limits), _change_limits(pixel_limits))
    return NativePixelChangeReplayResult(data, options, _statistics(data, options))


def _csv(result: NativePixelChangeReplayResult) -> Iterator[bytes]:
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
            "hue_change",
            "saturation_change",
            "value_change",
            "edge_change",
            "weighted_score",
            "detector_score",
            "candidate",
            "accepted",
            "reason",
        )
    )
    for sample, change, item in zip(
        result.measurements.base.samples, result.measurements.changes, result.statistics, strict=True
    ):
        yield row(
            (
                sample.sample_index,
                sample.decode_index,
                sample.pts,
                sample.time_base.numerator,
                sample.time_base.denominator,
                sample.presentation_time.numerator,
                sample.presentation_time.denominator,
                *(change.components if change is not None else _ZERO),
                item.weighted_score,
                item.score,
                str(item.candidate).lower(),
                str(item.accepted).lower(),
                item.reason,
            )
        )


def write_native_pixel_change_replay(
    result: NativePixelChangeReplayResult,
    output_dir: str | Path,
    *,
    limits: NativeMeasurementLimits | None = None,
    pixel_limits: PixelChangeLimits | None = None,
) -> Path:
    if type(result) is not NativePixelChangeReplayResult:
        raise ConfigurationError("result must be NativePixelChangeReplayResult")
    bounds = _limits(limits)
    _admit(result.measurements, bounds, _change_limits(pixel_limits))

    def report() -> Iterator[bytes]:
        encoder = json.JSONEncoder(sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
        for text in encoder.iterencode(result.to_dict()):
            yield text.encode("ascii")
        yield b"\n"

    return _bundle(
        output_dir, (("replay.json", report()), ("statistics.csv", _csv(result))), bounds.max_output_bytes
    )


__all__ = [
    "NativePixelChangeMeasurements",
    "NativePixelChangeReplayResult",
    "NativePixelChangeSample",
    "PixelChangeDetectionConfig",
    "PixelChangeStatistic",
    "analyze_native_pixel_changes",
    "capture_native_pixel_changes",
    "read_native_pixel_changes",
    "write_native_pixel_change_replay",
    "write_native_pixel_changes",
]
