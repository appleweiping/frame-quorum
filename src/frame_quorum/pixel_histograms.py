"""Original full-pixel RGB cell counts and normalized distribution distances.

No thumbnail, learned features, joint-channel histogram, HSV or color-profile
transform is applied. Counts describe Pillow's eight-bit RGB conversion.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Literal

from PIL import Image

from .errors import ConfigurationError
from .native_video import _integer
from .scene_detection import _number

_PIXELS = 67_108_864
HistogramMode = Literal["global", "spatial"]


@dataclass(frozen=True, slots=True)
class PixelHistogramConfig:
    bins: int = 32
    rows: int = 2
    columns: int = 2

    def __post_init__(self) -> None:
        if type(self.bins) is not int or self.bins not in {8, 16, 32, 64, 128, 256}:
            raise ConfigurationError("bins must be one of 8, 16, 32, 64, 128, 256")
        _integer(self.rows, "rows", 1, 2)
        _integer(self.columns, "columns", 1, 2)

    @property
    def values_per_frame(self) -> int:
        return self.rows * self.columns * 3 * self.bins


@dataclass(frozen=True, slots=True)
class PixelHistogramLimits:
    max_histogram_values: int = 8_000_000
    max_measurement_pixels: int = 100_000_000

    def __post_init__(self) -> None:
        _integer(self.max_histogram_values, "max_histogram_values", 1, 16_000_000)
        _integer(self.max_measurement_pixels, "max_measurement_pixels", 1, 1_000_000_000)


@dataclass(frozen=True, slots=True)
class HistogramDetectionConfig:
    mode: HistogramMode = "spatial"
    threshold: float = 0.5
    min_scene_samples: int = 1

    def __post_init__(self) -> None:
        _mode(self.mode)
        _number(self.threshold, "threshold", minimum=0, maximum=1)
        _integer(self.min_scene_samples, "min_scene_samples", 1, 1_000_000)


def _mode(value: object) -> None:
    if type(value) is not str or value not in {"global", "spatial"}:
        raise ConfigurationError("histogram mode must be global or spatial")


def _histogram_limits(value: PixelHistogramLimits | None) -> PixelHistogramLimits:
    bounds = PixelHistogramLimits() if value is None else value
    if type(bounds) is not PixelHistogramLimits:
        raise ConfigurationError("pixel limits must be PixelHistogramLimits")
    return bounds


def _boxes(width: int, height: int, config: PixelHistogramConfig) -> Iterator[tuple[int, int, int, int]]:
    for row in range(config.rows):
        for column in range(config.columns):
            yield (
                column * width // config.columns,
                row * height // config.rows,
                (column + 1) * width // config.columns,
                (row + 1) * height // config.rows,
            )


def _dimensions(width: int, height: int, config: PixelHistogramConfig, maximum: int = _PIXELS) -> int:
    _integer(width, "width", config.columns, _PIXELS)
    _integer(height, "height", config.rows, _PIXELS)
    pixels = width * height
    if pixels > min(_PIXELS, maximum):
        raise ConfigurationError("histogram pixels exceed measurement limit")
    return pixels


@dataclass(frozen=True, slots=True)
class PixelHistogram:
    """Cell-major, then R/G/B-major, then increasing-bin integer counts."""

    config: PixelHistogramConfig
    width: int
    height: int
    counts: tuple[int, ...]

    def __post_init__(self) -> None:
        if type(self.config) is not PixelHistogramConfig:
            raise ConfigurationError("histogram requires PixelHistogramConfig")
        _dimensions(self.width, self.height, self.config)
        if type(self.counts) is not tuple or len(self.counts) != self.config.values_per_frame:
            raise ConfigurationError("histogram requires exactly the configured immutable count slots")
        bins = self.config.bins
        for cell, (x0, y0, x1, y1) in enumerate(_boxes(self.width, self.height, self.config)):
            area = (x1 - x0) * (y1 - y0)
            for channel in range(3):
                offset = (cell * 3 + channel) * bins
                counts = self.counts[offset : offset + bins]
                for value in counts:
                    _integer(value, "histogram count", 0, area)
                if sum(counts) != area:
                    raise ConfigurationError("each histogram channel count sum must equal its cell area")

    def to_dict(self) -> dict[str, Any]:
        return {"width": self.width, "height": self.height, "counts": list(self.counts)}


@contextmanager
def _owned_image(image: Image.Image, *, borrowed: tuple[Image.Image, ...] = ()) -> Iterator[Image.Image]:
    """Close only a newly owned image without masking primary control exceptions."""
    if any(image is original for original in borrowed):
        raise ConfigurationError("image conversion/crop must return a distinct owned image")
    try:
        yield image
    except BaseException as primary:
        try:
            image.close()
        except BaseException as cleanup:
            if isinstance(primary, Exception) and not isinstance(cleanup, Exception):
                raise cleanup from primary
            primary.add_note(f"owned image close failed: {type(cleanup).__name__}")
        raise
    else:
        image.close()


def measure_pixel_histogram(
    image: Image.Image,
    config: PixelHistogramConfig | None = None,
    *,
    limits: PixelHistogramLimits | None = None,
) -> PixelHistogram:
    """Measure every RGB pixel; never close or modify the caller-owned image.

    Alpha is discarded by RGB conversion, not composited. Embedded color profiles
    and EXIF orientation are not applied here. Native input is already RGB.
    """
    options = PixelHistogramConfig() if config is None else config
    if type(options) is not PixelHistogramConfig:
        raise ConfigurationError("config must be PixelHistogramConfig")
    bounds = _histogram_limits(limits)
    if not isinstance(image, Image.Image):
        raise ConfigurationError("image must be a Pillow image")
    _dimensions(image.width, image.height, options, bounds.max_measurement_pixels)
    if options.values_per_frame > bounds.max_histogram_values:
        raise ConfigurationError("histogram count slots exceed limit")
    counts: list[int] = []
    with _owned_image(image.convert("RGB"), borrowed=(image,)) as rgb:
        for box in _boxes(image.width, image.height, options):
            with _owned_image(rgb.crop(box), borrowed=(image, rgb)) as cell:
                raw = cell.histogram()
            # Pillow's fixed 3 x 256 raw bins are accumulated exactly, not rounded.
            width = 256 // options.bins
            counts.extend(sum(raw[start : start + width]) for start in range(0, 768, width))
    return PixelHistogram(options, image.width, image.height, tuple(counts))


def histogram_distance(
    left: PixelHistogram, right: PixelHistogram, *, mode: HistogramMode = "spatial"
) -> float:
    """Mean total variation in [0, 1], with per-channel/cell mass normalization.

    Global mode merges cells before normalization. Spatial mode weights relative
    cells equally, including unequal areas in odd-sized images. Different image
    dimensions are supported; corresponding cells need not have equal areas.
    """
    _mode(mode)
    if type(left) is not PixelHistogram or type(right) is not PixelHistogram:
        raise ConfigurationError("distance requires two PixelHistogram values")
    if left.config != right.config:
        raise ConfigurationError("histogram configurations must match; remeasure to change bins/grid")
    bins = left.config.bins
    if mode == "global":
        groups = 3
        a = [sum(left.counts[offset :: 3 * bins]) for offset in range(3 * bins)]
        b = [sum(right.counts[offset :: 3 * bins]) for offset in range(3 * bins)]
        areas_a = [left.width * left.height] * 3
        areas_b = [right.width * right.height] * 3
    else:
        groups = left.config.rows * left.config.columns * 3
        a, b = list(left.counts), list(right.counts)
        areas_a = [
            (x1 - x0) * (y1 - y0)
            for x0, y0, x1, y1 in _boxes(left.width, left.height, left.config)
            for _ in range(3)
        ]
        areas_b = [
            (x1 - x0) * (y1 - y0)
            for x0, y0, x1, y1 in _boxes(right.width, right.height, right.config)
            for _ in range(3)
        ]
    differences = (
        abs(a[group * bins + index] / areas_a[group] - b[group * bins + index] / areas_b[group])
        for group in range(groups)
        for index in range(bins)
    )
    return min(1.0, math.fsum(differences) / (2 * groups))


__all__ = [
    "HistogramDetectionConfig",
    "HistogramMode",
    "PixelHistogram",
    "PixelHistogramConfig",
    "PixelHistogramLimits",
    "histogram_distance",
    "measure_pixel_histogram",
]
