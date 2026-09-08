"""Original circular-HSV and forward-gradient evidence from every RGB pixel.

Only two packed RGB snapshots are needed. Pixel maps and Python-object-per-pixel
lists are not retained. The four integer sums are sufficient for weight replay,
not for changing the acquisition's color or gradient definition afterward.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from PIL import Image

from .errors import ConfigurationError
from .native_video import _integer
from .pixel_histograms import _owned_image

_PIXELS = 67_108_864
_HUE_PERIOD = 1536
_HUE_SCALE = 768 * 255


@dataclass(frozen=True, slots=True)
class PixelChangeConfig:
    edge_radius: int = 1

    def __post_init__(self) -> None:
        _integer(self.edge_radius, "edge_radius", 1, 4)


@dataclass(frozen=True, slots=True)
class PixelChangeLimits:
    max_measurement_pixels: int = 50_000_000
    max_pair_rgb_bytes: int = 128 * 1024 * 1024

    def __post_init__(self) -> None:
        _integer(self.max_measurement_pixels, "max_measurement_pixels", 1, 1_000_000_000)
        _integer(self.max_pair_rgb_bytes, "max_pair_rgb_bytes", 6, 512 * 1024 * 1024)


@dataclass(frozen=True, slots=True)
class PixelChangeWeights:
    hue: float = 1.0
    saturation: float = 1.0
    value: float = 1.0
    edges: float = 0.0

    def __post_init__(self) -> None:
        for weight in self.values:
            if type(weight) not in (int, float) or not 0 <= weight <= 1_000_000:
                raise ConfigurationError("pixel weights must be finite built-in numbers in [0, 1000000]")
        if not any(self.values):
            raise ConfigurationError("at least one pixel weight must be positive")

    @property
    def values(self) -> tuple[float, float, float, float]:
        return self.hue, self.saturation, self.value, self.edges


def _change_config(value: PixelChangeConfig | None) -> PixelChangeConfig:
    result = PixelChangeConfig() if value is None else value
    if type(result) is not PixelChangeConfig:
        raise ConfigurationError("config must be PixelChangeConfig")
    return result


def _change_limits(value: PixelChangeLimits | None) -> PixelChangeLimits:
    result = PixelChangeLimits() if value is None else value
    if type(result) is not PixelChangeLimits:
        raise ConfigurationError("pixel limits must be PixelChangeLimits")
    return result


def _pixel_count(width: int, height: int) -> int:
    _integer(width, "pixel width", 1, _PIXELS)
    _integer(height, "pixel height", 1, _PIXELS)
    pixels = width * height
    if pixels > _PIXELS:
        raise ConfigurationError("pixel change dimensions exceed the compiled pixel ceiling")
    return pixels


def _admit_pair(width: int, height: int, limits: PixelChangeLimits) -> int:
    pixels = _pixel_count(width, height)
    if 2 * pixels > limits.max_measurement_pixels:
        raise ConfigurationError("two measured frames exceed the pixel work limit")
    if 6 * pixels > limits.max_pair_rgb_bytes:
        raise ConfigurationError("two packed RGB frames exceed the pair byte limit")
    return pixels


@dataclass(frozen=True, slots=True)
class PixelChange:
    config: PixelChangeConfig
    width: int
    height: int
    hue_sum: int
    saturation_sum: int
    value_sum: int
    edge_sum: int

    def __post_init__(self) -> None:
        if type(self.config) is not PixelChangeConfig:
            raise ConfigurationError("pixel change requires PixelChangeConfig")
        pixels = _pixel_count(self.width, self.height)
        for name, scale in (
            ("hue_sum", _HUE_SCALE),
            ("saturation_sum", 255),
            ("value_sum", 255),
            ("edge_sum", 510),
        ):
            _integer(getattr(self, name), name, 0, pixels * scale)

    @property
    def components(self) -> tuple[float, float, float, float]:
        pixels = self.width * self.height
        return (
            self.hue_sum / (pixels * _HUE_SCALE),
            self.saturation_sum / (pixels * 255),
            self.value_sum / (pixels * 255),
            self.edge_sum / (pixels * 510),
        )

    def score(self, weights: PixelChangeWeights | None = None) -> float:
        options = PixelChangeWeights() if weights is None else weights
        if type(options) is not PixelChangeWeights:
            raise ConfigurationError("weights must be PixelChangeWeights")
        # Scaling before summation is stable even when every admitted weight is
        # subnormal. The explicit weight cap also prevents oversized arithmetic.
        maximum = max(options.values)
        scaled = tuple(value / maximum for value in options.values)
        score = math.fsum(value * weight for value, weight in zip(self.components, scaled, strict=True))
        return min(1.0, max(0.0, score / math.fsum(scaled)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "hue_sum": self.hue_sum,
            "saturation_sum": self.saturation_sum,
            "value_sum": self.value_sum,
            "edge_sum": self.edge_sum,
        }


def _hsv(red: int, green: int, blue: int) -> tuple[int, int, int]:
    value = max(red, green, blue)
    chroma = value - min(red, green, blue)
    if chroma == 0:
        return 0, 0, value
    if value == red:
        sector = (green - blue) % (6 * chroma)
    elif value == green:
        sector = blue - red + 2 * chroma
    else:
        sector = red - green + 4 * chroma
    return sector * 256 // chroma, 255 * chroma // value, value


def _measure_rgb_change(
    left: bytes,
    right: bytes,
    width: int,
    height: int,
    config: PixelChangeConfig,
    limits: PixelChangeLimits,
) -> PixelChange:
    """Shared packed-byte kernel: exact arithmetic, O(pixels), O(1) extra state."""
    if type(config) is not PixelChangeConfig or type(limits) is not PixelChangeLimits:
        raise ConfigurationError("packed pixel measurement requires typed configuration and limits")
    pixels = _admit_pair(width, height, limits)
    if (
        type(left) is not bytes
        or type(right) is not bytes
        or len(left) != 3 * pixels
        or len(right) != 3 * pixels
    ):
        raise ConfigurationError("pixel change requires exact immutable RGB byte snapshots")
    hue_sum = saturation_sum = value_sum = edge_sum = 0
    stride = width * 3
    radius = config.edge_radius
    for y in range(height):
        row = y * stride
        down = min(y + radius, height - 1) * stride
        for x in range(width):
            offset = row + 3 * x
            next_x = row + 3 * min(x + radius, width - 1)
            next_y = down + 3 * x
            ha, sa, va = _hsv(left[offset], left[offset + 1], left[offset + 2])
            hb, sb, vb = _hsv(right[offset], right[offset + 1], right[offset + 2])
            difference = abs(ha - hb)
            hue_sum += min(difference, _HUE_PERIOD - difference) * min(sa, sb)
            saturation_sum += abs(sa - sb)
            value_sum += abs(va - vb)
            ga = abs(max(left[next_x], left[next_x + 1], left[next_x + 2]) - va) + abs(
                max(left[next_y], left[next_y + 1], left[next_y + 2]) - va
            )
            gb = abs(max(right[next_x], right[next_x + 1], right[next_x + 2]) - vb) + abs(
                max(right[next_y], right[next_y + 1], right[next_y + 2]) - vb
            )
            edge_sum += abs(ga - gb)
    return PixelChange(config, width, height, hue_sum, saturation_sum, value_sum, edge_sum)


def _image_rgb(image: Image.Image, borrowed: tuple[Image.Image, ...]) -> bytes:
    with _owned_image(image.convert("RGB"), borrowed=borrowed) as converted:
        if converted.mode != "RGB" or converted.size != image.size:
            raise ConfigurationError("RGB conversion must preserve the admitted image dimensions")
        return converted.tobytes()


def measure_pixel_change(
    left: Image.Image,
    right: Image.Image,
    config: PixelChangeConfig | None = None,
    *,
    limits: PixelChangeLimits | None = None,
) -> PixelChange:
    """Compare every corresponding pixel without mutating or closing borrowed images.

    Sizes must match. Alpha is discarded, not composited; ICC profiles and EXIF
    orientation are not applied. Pair byte limits cover packed RGB snapshots,
    not caller images, conversion scratch space or hard process RSS.
    """
    options, bounds = _change_config(config), _change_limits(limits)
    if not isinstance(left, Image.Image) or not isinstance(right, Image.Image):
        raise ConfigurationError("pixel change inputs must be Pillow images")
    if left.size != right.size:
        raise ConfigurationError("pixel change requires identical image dimensions")
    _admit_pair(left.width, left.height, bounds)
    # Conversion images are closed sequentially; only their packed snapshots
    # coexist afterward. Never retain an image or pixel object per sample.
    a = _image_rgb(left, (left, right))
    b = _image_rgb(right, (left, right))
    return _measure_rgb_change(a, b, left.width, left.height, options, bounds)


__all__ = [
    "PixelChange",
    "PixelChangeConfig",
    "PixelChangeLimits",
    "PixelChangeWeights",
    "measure_pixel_change",
]
