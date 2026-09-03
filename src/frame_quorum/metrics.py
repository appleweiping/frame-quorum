"""Small, deterministic visual measurements built on Pillow."""

from __future__ import annotations

import math
from collections.abc import Iterable

from PIL import Image, ImageFilter, ImageStat

from .models import FrameMetrics

_HASH_WIDTH = 9
_HASH_HEIGHT = 8


def measure_image(image: Image.Image) -> FrameMetrics:
    """Measure an image after applying EXIF orientation in the scanner."""

    rgb = image.convert("RGB")
    sample = _bounded_copy(rgb, 128)
    gray = sample.convert("L")
    mean = ImageStat.Stat(sample).mean
    return FrameMetrics(
        perceptual_hash=difference_hash(gray),
        luminance=round(ImageStat.Stat(gray).mean[0] / 255.0, 8),
        entropy=round(grayscale_entropy(gray), 8),
        sharpness=round(edge_energy(gray), 8),
        colorfulness=round(colorfulness(sample), 8),
        mean_red=round(mean[0] / 255.0, 8),
        mean_green=round(mean[1] / 255.0, 8),
        mean_blue=round(mean[2] / 255.0, 8),
    )


def _bounded_copy(image: Image.Image, maximum: int) -> Image.Image:
    copy = image.copy()
    copy.thumbnail((maximum, maximum), Image.Resampling.LANCZOS)
    return copy


def difference_hash(gray: Image.Image) -> int:
    """Return a 64-bit horizontal difference hash."""

    resized = gray.resize((_HASH_WIDTH, _HASH_HEIGHT), Image.Resampling.LANCZOS).convert("L")
    pixels = resized.tobytes()
    value = 0
    for y in range(_HASH_HEIGHT):
        row = y * _HASH_WIDTH
        for x in range(_HASH_WIDTH - 1):
            value = (value << 1) | int(pixels[row + x] > pixels[row + x + 1])
    return value


def grayscale_entropy(gray: Image.Image) -> float:
    """Return Shannon entropy normalized to the interval [0, 1]."""

    histogram = gray.histogram()
    total = sum(histogram)
    if total == 0:
        return 0.0
    entropy = -sum((count / total) * math.log2(count / total) for count in histogram if count)
    return min(1.0, entropy / 8.0)


def edge_energy(gray: Image.Image) -> float:
    """Estimate sharpness from the mean strength of interior edges."""

    if gray.width < 3 or gray.height < 3:
        return 0.0
    edges = gray.filter(ImageFilter.FIND_EDGES).crop((1, 1, gray.width - 1, gray.height - 1))
    return min(1.0, ImageStat.Stat(edges).mean[0] / 96.0)


def colorfulness(rgb: Image.Image) -> float:
    """Estimate chroma spread and mean chroma without NumPy."""

    resized = rgb.resize((32, 32), Image.Resampling.BILINEAR).convert("RGB")
    bands = resized.tobytes()
    pixels = [(bands[base], bands[base + 1], bands[base + 2]) for base in range(0, len(bands), 3)]
    if not pixels:
        return 0.0
    rg = [float(red) - green for red, green, _ in pixels]
    yb = [0.5 * (red + green) - blue for red, green, blue in pixels]
    rg_mean, rg_std = _mean_std(rg)
    yb_mean, yb_std = _mean_std(yb)
    raw = math.sqrt(rg_std**2 + yb_std**2) + 0.3 * math.sqrt(rg_mean**2 + yb_mean**2)
    return min(1.0, raw / 180.0)


def _mean_std(values: Iterable[float]) -> tuple[float, float]:
    sequence = list(values)
    mean = sum(sequence) / len(sequence)
    variance = sum((value - mean) ** 2 for value in sequence) / len(sequence)
    return mean, math.sqrt(variance)


def content_distance(left: FrameMetrics, right: FrameMetrics) -> float:
    """Combine structural and mean-color distance on a normalized scale."""

    hash_distance = (left.perceptual_hash ^ right.perceptual_hash).bit_count() / 64.0
    color_distance = math.sqrt(
        (left.mean_red - right.mean_red) ** 2
        + (left.mean_green - right.mean_green) ** 2
        + (left.mean_blue - right.mean_blue) ** 2
    ) / math.sqrt(3.0)
    luminance_distance = abs(left.luminance - right.luminance)
    return min(1.0, 0.65 * hash_distance + 0.25 * color_distance + 0.10 * luminance_distance)
