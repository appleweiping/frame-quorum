"""Bounded native RGB cell histograms, separate wire identity and offline replay."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

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
from .native_scenes import NativeScene, NativeSceneSample, _collect_native_records, _partition_native_samples
from .native_video import NativeVideoConfig, NativeVideoFrame, _integer, _pair, _source_path
from .pixel_histograms import (
    HistogramDetectionConfig,
    PixelHistogram,
    PixelHistogramConfig,
    PixelHistogramLimits,
    _dimensions,
    _histogram_limits,
    _owned_image,
    histogram_distance,
    measure_pixel_histogram,
)
from .scene_detection import DetectionConfig, _decisions, _number

_KIND = "frame-quorum-native-histograms"
_MEASUREMENT = "frame-quorum-rgb-cell-histogram-v1"


@dataclass(frozen=True, slots=True)
class NativeHistogramSample:
    sample: NativeSceneSample
    histogram: PixelHistogram

    def __post_init__(self) -> None:
        if type(self.sample) is not NativeSceneSample or type(self.histogram) is not PixelHistogram:
            raise ConfigurationError("histogram sample requires native coordinates and pixel histogram")

    def to_dict(self) -> dict[str, Any]:
        return {"sample": self.sample.to_dict(), "histogram": self.histogram.to_dict()}


@dataclass(frozen=True, slots=True)
class NativeHistogramMeasurements:
    base: NativeMeasurements
    config: PixelHistogramConfig
    histograms: tuple[PixelHistogram, ...]

    def __post_init__(self) -> None:
        if type(self.base) is not NativeMeasurements or type(self.config) is not PixelHistogramConfig:
            raise ConfigurationError("histogram capture requires fixed base measurements and configuration")
        if type(self.histograms) is not tuple or len(self.histograms) != len(self.base.samples):
            raise ConfigurationError("every native sample requires one immutable histogram")
        if self.base.video.max_frames * self.config.values_per_frame > 16_000_000:
            raise ConfigurationError("declared histogram count slots exceed compiled ceiling")
        pixels = 0
        for item in self.histograms:
            if type(item) is not PixelHistogram or item.config != self.config:
                raise ConfigurationError("every histogram must use the declared measurement configuration")
            area = item.width * item.height
            if area > self.base.video.max_frame_pixels:
                raise ConfigurationError("histogram dimensions exceed original per-frame budget")
            pixels += area
            if pixels > 1_000_000_000 or pixels > self.base.diagnostics.decoded_pixels_observed:
                raise ConfigurationError("selected histogram pixels exceed observed decode work or ceiling")
        unselected = self.base.diagnostics.decoded_frames - len(self.histograms)
        if pixels + unselected > self.base.diagnostics.decoded_pixels_observed:
            raise ConfigurationError("decoded pixel total cannot cover selected and unselected frames")

    def header(self) -> dict[str, Any]:
        return {
            "kind": _KIND,
            "schema_version": 1,
            "measurement_version": _MEASUREMENT,
            "histogram_config": asdict(self.config),
            "sample_count": len(self.histograms),
            "base": self.base.header(),
        }

    @property
    def digest(self) -> str:
        digest = hashlib.sha256()
        for line in _histogram_lines(self):
            digest.update(line)
        return digest.hexdigest()


def _admit_capacity(
    video: NativeVideoConfig,
    config: PixelHistogramConfig,
    bounds: NativeMeasurementLimits,
    pixel_bounds: PixelHistogramLimits,
) -> None:
    if video.max_frames > bounds.max_samples:
        raise ConfigurationError("video.max_frames exceeds histogram sample admission")
    if video.max_frames * config.values_per_frame > pixel_bounds.max_histogram_values:
        raise ConfigurationError("declared histogram count slots exceed limit")


def _admit(
    data: NativeHistogramMeasurements,
    bounds: NativeMeasurementLimits,
    pixel_bounds: PixelHistogramLimits,
) -> None:
    if type(data) is not NativeHistogramMeasurements:
        raise ConfigurationError("measurements must be NativeHistogramMeasurements, not summary-only cache")
    _admit_capacity(data.base.video, data.config, bounds, pixel_bounds)
    if sum(item.width * item.height for item in data.histograms) > pixel_bounds.max_measurement_pixels:
        raise ConfigurationError("histogram measurement pixels exceed limit")


def capture_native_histograms(
    path: str | Path,
    video: NativeVideoConfig | None = None,
    *,
    histogram: PixelHistogramConfig | None = None,
    on_sample: Callable[[NativeHistogramSample], None] | None = None,
    limits: NativeMeasurementLimits | None = None,
    pixel_limits: PixelHistogramLimits | None = None,
) -> NativeHistogramMeasurements:
    """One native decode measures old summaries and full-pixel counts together."""
    bounds, pixel_bounds = _limits(limits), _histogram_limits(pixel_limits)
    options = NativeVideoConfig() if video is None else video
    config = PixelHistogramConfig() if histogram is None else histogram
    if type(options) is not NativeVideoConfig or type(config) is not PixelHistogramConfig:
        raise ConfigurationError("capture requires NativeVideoConfig and PixelHistogramConfig")
    _admit_capacity(options, config, bounds, pixel_bounds)
    measured_pixels = 0

    def measure(frame: NativeVideoFrame) -> NativeHistogramSample:
        nonlocal measured_pixels
        pixels = _dimensions(frame.width, frame.height, config)
        if measured_pixels + pixels > pixel_bounds.max_measurement_pixels:
            raise ConfigurationError("histogram measurement pixels exceed limit")
        measured_pixels += pixels
        with _owned_image(frame.image()) as image:
            metrics = measure_image(image)
            counts = measure_pixel_histogram(image, config, limits=pixel_bounds)
        sample = NativeSceneSample(
            frame.pts, frame.time_base, frame.decode_index, frame.sample_index, frame.generation, metrics
        )
        return NativeHistogramSample(sample, counts)

    try:
        digest, metadata, diagnostics, records = _capture_source_records(
            _source_path(path),
            options,
            lambda source: _collect_native_records(source, options, measure, on_sample),
        )
        base = NativeMeasurements(
            options, metadata, diagnostics, tuple(row.sample for row in records), digest
        )
        result = NativeHistogramMeasurements(base, config, tuple(row.histogram for row in records))
        _admit(result, bounds, pixel_bounds)
        for _ in _histogram_cache_lines(result, bounds):
            pass
        return result
    except OSError as error:
        raise ScanError("native histogram source read failed") from error


def _histogram_lines(data: NativeHistogramMeasurements) -> Iterator[bytes]:
    yield _canonical(data.header())
    for sample, histogram in zip(data.base.samples, data.histograms, strict=True):
        yield _canonical(NativeHistogramSample(sample, histogram).to_dict())


def _histogram_cache_lines(
    data: NativeHistogramMeasurements, bounds: NativeMeasurementLimits
) -> Iterator[bytes]:
    yield from _bounded_cache_lines(_histogram_lines(data), len(data.histograms), bounds)


def write_native_histograms(
    measurements: NativeHistogramMeasurements,
    output_dir: str | Path,
    *,
    limits: NativeMeasurementLimits | None = None,
    pixel_limits: PixelHistogramLimits | None = None,
) -> Path:
    bounds, pixel_bounds = _limits(limits), _histogram_limits(pixel_limits)
    _admit(measurements, bounds, pixel_bounds)
    return (
        _bundle(
            output_dir,
            (("histograms.fqm.jsonl", _histogram_cache_lines(measurements, bounds)),),
            bounds.max_output_bytes,
        )
        / "histograms.fqm.jsonl"
    )


def read_native_histograms(
    path: str | Path,
    *,
    limits: NativeMeasurementLimits | None = None,
    pixel_limits: PixelHistogramLimits | None = None,
) -> NativeHistogramMeasurements:
    """Read bounded data only; summary-only caches cannot supply missing counts."""
    bounds, pixel_bounds = _limits(limits), _histogram_limits(pixel_limits)

    def prepare(
        header: dict[str, Any], count: int
    ) -> tuple[
        Callable[[dict[str, Any]], NativeHistogramSample],
        Callable[[tuple[NativeHistogramSample, ...]], NativeHistogramMeasurements],
    ]:
        _keys(
            header,
            {"kind", "schema_version", "measurement_version", "histogram_config", "sample_count", "base"},
        )
        if (
            header["kind"] != _KIND
            or type(header["schema_version"]) is not int
            or header["schema_version"] != 1
            or header["measurement_version"] != _MEASUREMENT
        ):
            raise ConfigurationError("unsupported histogram schema; summary-only metrics cannot be upgraded")
        config = PixelHistogramConfig(**_keys(header["histogram_config"], {"bins", "rows", "columns"}))
        arguments = _parse_header(header["base"])
        base_count = _integer(header["base"]["sample_count"], "base sample count", 0, bounds.max_samples)
        _admit_capacity(arguments["video"], config, bounds, pixel_bounds)
        if (
            count != base_count
            or count > arguments["video"].max_frames
            or count != arguments["diagnostics"].returned_frames
        ):
            raise ConfigurationError("histogram count contradicts base configuration/diagnostics")
        measured_pixels = 0

        def parse(raw: dict[str, Any]) -> NativeHistogramSample:
            nonlocal measured_pixels
            _keys(raw, {"sample", "histogram"})
            values = _keys(raw["histogram"], {"width", "height", "counts"})
            area = _dimensions(values["width"], values["height"], config)
            if measured_pixels + area > pixel_bounds.max_measurement_pixels:
                raise ConfigurationError("histogram measurement pixels exceed limit")
            measured_pixels += area
            if type(values["counts"]) is not list or len(values["counts"]) != config.values_per_frame:
                raise ConfigurationError("histogram count slot length disagrees with schema")
            histogram = PixelHistogram(config, values["width"], values["height"], tuple(values["counts"]))
            return NativeHistogramSample(_parse_sample(raw["sample"]), histogram)

        def finish(records: tuple[NativeHistogramSample, ...]) -> NativeHistogramMeasurements:
            base = NativeMeasurements(samples=tuple(row.sample for row in records), **arguments)
            result = NativeHistogramMeasurements(base, config, tuple(row.histogram for row in records))
            _admit(result, bounds, pixel_bounds)
            return result

        return parse, finish

    return _read_cache(path, bounds, prepare)


@dataclass(frozen=True, slots=True)
class HistogramStatistic:
    score: float
    candidate: bool
    accepted: bool
    reason: str

    def __post_init__(self) -> None:
        _number(self.score, "histogram score", minimum=0, maximum=1)
        if type(self.candidate) is not bool or type(self.accepted) is not bool:
            raise ConfigurationError("histogram decisions must be booleans")
        if type(self.reason) is not str or self.reason not in {
            "sequence_start",
            "below_threshold",
            "short_previous_scene",
            "short_final_scene",
            "distance_threshold",
        }:
            raise ConfigurationError("unsupported histogram decision reason")


def _statistics(scores: list[float], config: HistogramDetectionConfig) -> tuple[HistogramStatistic, ...]:
    candidates = {
        index: "distance_threshold" for index in range(1, len(scores)) if scores[index] >= config.threshold
    }
    policy = DetectionConfig(
        detector="content", threshold=config.threshold, min_scene_frames=config.min_scene_samples
    )
    decisions = _decisions(scores, candidates, policy)
    return tuple(
        HistogramStatistic(score, index in candidates, accepted, reason)
        for index, (score, (accepted, reason)) in enumerate(zip(scores, decisions, strict=True))
    )


def _histogram_scores(data: NativeHistogramMeasurements, config: HistogramDetectionConfig) -> list[float]:
    values = data.histograms
    return ([0.0] if values else []) + [
        histogram_distance(values[index - 1], values[index], mode=config.mode)
        for index in range(1, len(values))
    ]


@dataclass(frozen=True, slots=True)
class NativeHistogramReplayResult:
    """Historical histogram analysis; deliberately not a NativeSceneResult subtype."""

    measurements: NativeHistogramMeasurements
    config: HistogramDetectionConfig
    statistics: tuple[HistogramStatistic, ...]

    def __post_init__(self) -> None:
        if (
            type(self.measurements) is not NativeHistogramMeasurements
            or type(self.config) is not HistogramDetectionConfig
        ):
            raise ConfigurationError(
                "histogram replay requires its own measurements and detector configuration"
            )
        if type(self.statistics) is not tuple or len(self.statistics) != len(self.measurements.histograms):
            raise ConfigurationError("every histogram requires one immutable statistic")
        if any(type(row) is not HistogramStatistic for row in self.statistics):
            raise ConfigurationError("every statistic must be HistogramStatistic")
        scores = [row.score for row in self.statistics]
        if scores != _histogram_scores(self.measurements, self.config):
            raise ConfigurationError("histogram scores disagree with actual stored pixel counts")
        if self.statistics != _statistics(scores, self.config):
            raise ConfigurationError("histogram decisions contradict threshold or sample minimum")

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
            "kind": "frame-quorum-native-histogram-replay",
            "schema_version": 1,
            "execution": "cached_histograms",
            "source_verified": False,
            "measurement_digest": self.measurements.digest,
            "capture": self.measurements.header(),
            "detector": asdict(self.config),
            "cut_positions": list(self.cut_positions),
            "cut_times": [_pair(value) for value in self.cut_times],
            "scenes": [scene.to_dict() for scene in self.scenes],
            "statistics": [
                {"sample": sample.to_dict(), **asdict(statistic)}
                for sample, statistic in zip(self.measurements.base.samples, self.statistics, strict=True)
            ],
        }


def analyze_native_histograms(
    measurements: NativeHistogramMeasurements,
    config: HistogramDetectionConfig | None = None,
    *,
    limits: NativeMeasurementLimits | None = None,
    pixel_limits: PixelHistogramLimits | None = None,
) -> NativeHistogramReplayResult:
    _admit(measurements, _limits(limits), _histogram_limits(pixel_limits))
    options = HistogramDetectionConfig() if config is None else config
    if type(options) is not HistogramDetectionConfig:
        raise ConfigurationError("config must be HistogramDetectionConfig")
    scores = _histogram_scores(measurements, options)
    return NativeHistogramReplayResult(measurements, options, _statistics(scores, options))


def _csv(result: NativeHistogramReplayResult) -> Iterator[bytes]:
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
            "histogram_score",
            "candidate",
            "accepted",
            "reason",
        )
    )
    for sample, item in zip(result.measurements.base.samples, result.statistics, strict=True):
        yield row(
            (
                sample.sample_index,
                sample.decode_index,
                sample.pts,
                sample.time_base.numerator,
                sample.time_base.denominator,
                sample.presentation_time.numerator,
                sample.presentation_time.denominator,
                item.score,
                str(item.candidate).lower(),
                str(item.accepted).lower(),
                item.reason,
            )
        )


def write_native_histogram_replay(
    result: NativeHistogramReplayResult,
    output_dir: str | Path,
    *,
    limits: NativeMeasurementLimits | None = None,
    pixel_limits: PixelHistogramLimits | None = None,
) -> Path:
    if type(result) is not NativeHistogramReplayResult:
        raise ConfigurationError("result must be NativeHistogramReplayResult")
    bounds = _limits(limits)
    _admit(result.measurements, bounds, _histogram_limits(pixel_limits))

    def report() -> Iterator[bytes]:
        encoder = json.JSONEncoder(sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
        for text in encoder.iterencode(result.to_dict()):
            yield text.encode("ascii")
        yield b"\n"

    return _bundle(
        output_dir, (("replay.json", report()), ("statistics.csv", _csv(result))), bounds.max_output_bytes
    )


__all__ = [
    "HistogramStatistic",
    "NativeHistogramMeasurements",
    "NativeHistogramReplayResult",
    "NativeHistogramSample",
    "analyze_native_histograms",
    "capture_native_histograms",
    "read_native_histograms",
    "write_native_histogram_replay",
    "write_native_histograms",
]
