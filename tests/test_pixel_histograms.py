from dataclasses import FrozenInstanceError, replace
from fractions import Fraction

import pytest
from PIL import Image

from frame_quorum.errors import ConfigurationError
from frame_quorum.metrics import content_distance, measure_image
from frame_quorum.pixel_histograms import (
    HistogramDetectionConfig,
    PixelHistogramConfig,
    PixelHistogramLimits,
    _owned_image,
    histogram_distance,
    measure_pixel_histogram,
)


def test_real_pixels_reveal_change_invisible_to_existing_content_distance():
    with Image.new("RGB", (16, 16)) as split, Image.new("RGB", (16, 16), (127, 127, 127)) as flat:
        split.paste((254, 254, 254), (0, 8, 16, 16))
        assert content_distance(measure_image(split), measure_image(flat)) == 0
        left, right = measure_pixel_histogram(split), measure_pixel_histogram(flat)
    assert histogram_distance(left, right, mode="global") == 1
    assert histogram_distance(left, right, mode="spatial") == 1


def test_global_distribution_and_relative_spatial_cells_have_distinct_semantics():
    with Image.new("RGB", (4, 4)) as first, Image.new("RGB", (4, 4)) as second:
        first.paste((255, 255, 255), (0, 2, 4, 4))
        second.paste((255, 255, 255), (0, 0, 4, 2))
        left, right = measure_pixel_histogram(first), measure_pixel_histogram(second)
    assert histogram_distance(left, right, mode="global") == 0
    assert histogram_distance(left, right, mode="spatial") == 1


def test_direct_integer_count_oracle_for_odd_cells_and_bin_edges():
    config = PixelHistogramConfig(bins=8, rows=2, columns=2)
    pixels = [(i * 31 % 256, i * 47 % 256, i * 79 % 256) for i in range(15)]
    expected = [0] * (4 * 3 * 8)
    # Independently enumerate each supplied pixel, not Pillow.histogram or a production helper.
    for y in range(3):
        for x in range(5):
            cell = (0 if y < 1 else 1) * 2 + (0 if x < 2 else 1)
            for channel, value in enumerate(pixels[y * 5 + x]):
                expected[(cell * 3 + channel) * 8 + value // 32] += 1
    with Image.new("RGB", (5, 3)) as image:
        image.putdata(pixels)
        measured = measure_pixel_histogram(image, config)
    assert measured.counts == tuple(expected)
    assert (measured.width, measured.height) == (5, 3)


def test_manual_half_distance_and_normalization_across_image_sizes():
    config = PixelHistogramConfig(bins=8, rows=1, columns=1)
    with Image.new("RGB", (2, 2)) as half, Image.new("RGB", (3, 3)) as black:
        half.paste((255, 255, 255), (0, 1, 2, 2))
        left = measure_pixel_histogram(half, config)
        right = measure_pixel_histogram(black, config)
    for mode in ("global", "spatial"):
        assert histogram_distance(left, right, mode=mode) == float(Fraction(1, 2))
        assert histogram_distance(right, left, mode=mode) == 0.5


@pytest.mark.parametrize("bins", [8, 16, 32, 64, 128, 256])
@pytest.mark.parametrize("rows,columns", [(1, 1), (1, 2), (2, 1), (2, 2)])
def test_each_supported_shape_exact_channels_and_total(bins, rows, columns):
    config = PixelHistogramConfig(bins, rows, columns)
    with Image.new("RGB", (3, 5), (0, 127, 255)) as image:
        data = measure_pixel_histogram(image, config)
    assert len(data.counts) == rows * columns * 3 * bins
    assert sum(data.counts) == 45
    for row in range(rows):
        for col in range(columns):
            area = ((row + 1) * 5 // rows - row * 5 // rows) * ((col + 1) * 3 // columns - col * 3 // columns)
            offset = (row * columns + col) * 3 * bins
            assert data.counts[offset] == area
            assert data.counts[offset + bins + 127 * bins // 256] == area
            assert data.counts[offset + 3 * bins - 1] == area
    assert histogram_distance(data, data) == 0
    assert histogram_distance(data, data, mode="global") == 0


@pytest.mark.parametrize("field", ["bins", "rows", "columns"])
@pytest.mark.parametrize("value", [True, 0, -1, 3, 10**1000, float("nan"), float("inf"), "2"])
def test_strict_configuration(field, value):
    with pytest.raises(ConfigurationError):
        PixelHistogramConfig(**{field: value})


@pytest.mark.parametrize("field", ["max_histogram_values", "max_measurement_pixels"])
@pytest.mark.parametrize("value", [True, 0, -1, 10**1000, float("nan"), float("inf"), "2"])
def test_strict_work_limits(field, value):
    with pytest.raises(ConfigurationError):
        PixelHistogramLimits(**{field: value})


@pytest.mark.parametrize(
    "field,value",
    [
        ("mode", None),
        ("mode", "hsv"),
        ("mode", True),
        ("threshold", True),
        ("threshold", -0.1),
        ("threshold", 1.1),
        ("threshold", float("nan")),
        ("threshold", float("inf")),
        ("threshold", 10**1000),
        ("min_scene_samples", False),
        ("min_scene_samples", 0),
        ("min_scene_samples", 1_000_001),
    ],
)
def test_strict_detection_configuration(field, value):
    with pytest.raises(ConfigurationError):
        HistogramDetectionConfig(**{field: value})


def test_images_are_unchanged_and_rgb_alpha_conversion_is_explicit():
    with Image.new("RGBA", (2, 2), (255, 0, 0, 0)) as image:
        before = image.tobytes()
        data = measure_pixel_histogram(image)
        assert image.mode == "RGBA" and image.tobytes() == before
    with Image.new("RGB", (2, 2), (255, 0, 0)) as rgb, Image.new("RGB", (2, 2)) as black:
        assert data == measure_pixel_histogram(rgb)
        assert histogram_distance(data, measure_pixel_histogram(black)) == pytest.approx(1 / 3)
    detached = data.to_dict()
    detached["counts"][0] = 99
    assert data.counts[0] == 0
    with pytest.raises(FrozenInstanceError):
        data.width = 99


@pytest.mark.parametrize(
    "kwargs",
    [
        {"config": object()},
        {"limits": object()},
        {"limits": PixelHistogramLimits(max_histogram_values=1)},
        {"limits": PixelHistogramLimits(max_measurement_pixels=3)},
    ],
)
def test_measurement_admission_before_conversion(kwargs, monkeypatch):
    with Image.new("RGB", (2, 2)) as image:
        monkeypatch.setattr(image, "convert", lambda *a: pytest.fail("converted before admission"))
        with pytest.raises(ConfigurationError):
            measure_pixel_histogram(image, **kwargs)


def test_invalid_histogram_objects_and_empty_cells():
    with pytest.raises(ConfigurationError):
        measure_pixel_histogram(None)
    with Image.new("RGB", (1, 1)) as small:
        with pytest.raises(ConfigurationError):
            measure_pixel_histogram(small)
        valid = measure_pixel_histogram(small, PixelHistogramConfig(rows=1, columns=1))
    for changes in (
        {"width": True},
        {"height": 0},
        {"width": 67_108_864, "height": 2},
        {"config": object()},
        {"counts": []},
        {"counts": valid.counts[:-1]},
    ):
        with pytest.raises(ConfigurationError):
            replace(valid, **changes)
    for value in (True, -1, 2, 10**1000, float("nan")):
        with pytest.raises(ConfigurationError):
            replace(valid, counts=(value, *valid.counts[1:]))
    with pytest.raises(ConfigurationError, match="sum"):
        replace(valid, counts=(0, *valid.counts[1:]))
    for left, right, mode in ((None, valid, "global"), (valid, None, "global"), (valid, valid, "unknown")):
        with pytest.raises(ConfigurationError):
            histogram_distance(left, right, mode=mode)
    with Image.new("RGB", (2, 2)) as image:
        different = measure_pixel_histogram(image)
    with pytest.raises(ConfigurationError, match="configurations"):
        histogram_distance(valid, different)


@pytest.mark.parametrize(
    "primary,cleanup,expected",
    [
        (KeyboardInterrupt, OSError, KeyboardInterrupt),
        (SystemExit, KeyboardInterrupt, SystemExit),
        (ValueError, KeyboardInterrupt, KeyboardInterrupt),
        (ValueError, OSError, ValueError),
    ],
)
def test_owned_image_cleanup_preserves_primary_control(primary, cleanup, expected):
    class Resource:
        def close(self):
            raise cleanup("close")

    with pytest.raises(expected), _owned_image(Resource()):
        raise primary("body")


def test_owned_image_cleanup_failure_after_success_propagates():
    class Resource:
        def close(self):
            raise OSError("close")

    with pytest.raises(OSError, match="close"), _owned_image(Resource()):
        pass


def test_convert_returning_borrowed_input_is_rejected_without_closing_it(monkeypatch):
    with Image.new("RGB", (2, 2)) as image:
        monkeypatch.setattr(image, "convert", lambda *args: image)
        with pytest.raises(ConfigurationError, match="distinct owned"):
            measure_pixel_histogram(image)
        assert image.tobytes() == bytes(12)


@pytest.mark.parametrize("return_input", [True, False])
def test_crop_returning_borrowed_image_is_rejected_and_only_owned_copy_closed(monkeypatch, return_input):
    with Image.new("RGB", (2, 2)) as image:
        converted = image.copy()
        monkeypatch.setattr(image, "convert", lambda *args: converted)
        monkeypatch.setattr(converted, "crop", lambda *args: image if return_input else converted)
        with pytest.raises(ConfigurationError, match="distinct owned"):
            measure_pixel_histogram(image)
        assert image.tobytes() == bytes(12)
        with pytest.raises(ValueError, match="closed"):
            converted.tobytes()
