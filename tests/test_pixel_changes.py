"""Hand-computed pixel evidence, independent of the measurement implementation."""

from fractions import Fraction

import pytest
from PIL import Image

import frame_quorum.pixel_changes as module
from frame_quorum.errors import ConfigurationError
from frame_quorum.pixel_changes import (
    PixelChange,
    PixelChangeConfig,
    PixelChangeLimits,
    PixelChangeWeights,
    measure_pixel_change,
)


def test_primary_complement_has_maximum_circular_hue_change():
    with Image.new("RGB", (1, 1), (255, 0, 0)) as left, Image.new("RGB", (1, 1), (0, 255, 255)) as right:
        result = measure_pixel_change(left, right)
    assert result.hue_sum == 768 * 255
    assert (result.saturation_sum, result.value_sum, result.edge_sum) == (0, 0, 0)
    assert result.components == (1.0, 0.0, 0.0, 0.0)
    assert result.score(PixelChangeWeights(hue=1, saturation=0, value=0)) == 1


def test_hue_wrap_and_gray_suppression_are_explicit():
    with Image.new("RGB", (1, 1), (255, 0, 0)) as red:
        with Image.new("RGB", (1, 1), (255, 0, 1)) as near_wrap:
            result = measure_pixel_change(red, near_wrap)
        with Image.new("RGB", (1, 1), (255, 255, 255)) as gray:
            achromatic = measure_pixel_change(red, gray)
    # Quantized H values are 0 and 1534, not a nearly full-circle change.
    assert result.hue_sum == 2 * 255
    assert result.components[0] == float(Fraction(1, 384))
    assert achromatic.components == (0.0, 1.0, 0.0, 0.0)


def test_equal_rgb_marginals_do_not_determine_hsv_change():
    with (
        Image.frombytes("RGB", (2, 1), bytes((255, 0, 0, 0, 255, 255))) as left,
        Image.frombytes("RGB", (2, 1), bytes((255, 255, 0, 0, 0, 255))) as right,
    ):
        assert left.histogram() == right.histogram()
        result = measure_pixel_change(left, right)
    assert result.hue_sum == 2 * 256 * 255
    assert result.components == (float(Fraction(1, 3)), 0.0, 0.0, 0.0)


def test_checkerboard_and_stripes_have_independent_value_and_edge_changes():
    checker = bytes(value for y in range(4) for x in range(4) for value in [(x + y) % 2 * 255] * 3)
    stripes = bytes(value for _y in range(4) for x in range(4) for value in [x % 2 * 255] * 3)
    with Image.frombytes("RGB", (4, 4), checker) as left, Image.frombytes("RGB", (4, 4), stripes) as right:
        assert left.histogram() == right.histogram()
        for box in ((0, 0, 2, 2), (2, 0, 4, 2), (0, 2, 2, 4), (2, 2, 4, 4)):
            with left.crop(box) as a, right.crop(box) as b:
                assert a.histogram() == b.histogram()
        result = measure_pixel_change(left, right)
    # Exactly eight value positions change. Edge maps differ by 255 at twelve
    # positions: the first three rows, including their clamped final columns.
    assert result.value_sum == 8 * 255
    assert result.edge_sum == 12 * 255
    assert result.components == (0.0, 0.0, 0.5, 0.375)
    assert result.score(PixelChangeWeights(hue=0, saturation=0, value=0, edges=1)) == 0.375


def test_constant_black_white_changes_brightness_without_inventing_edges():
    with Image.new("RGB", (3, 2), "black") as left, Image.new("RGB", (3, 2), "white") as right:
        result = measure_pixel_change(left, right)
    assert result.components == (0.0, 0.0, 1.0, 0.0)
    assert result.score() == pytest.approx(1 / 3)


@pytest.mark.parametrize(
    ("color", "distance"),
    [
        ((255, 0, 0), 0),
        ((255, 255, 0), 256),
        ((0, 255, 0), 512),
        ((0, 255, 255), 768),
        ((0, 0, 255), 512),
        ((255, 0, 255), 256),
    ],
)
def test_all_primary_secondary_hues_have_known_circle_positions(color, distance):
    with Image.new("RGB", (1, 1), (255, 0, 0)) as red, Image.new("RGB", (1, 1), color) as other:
        assert measure_pixel_change(red, other).hue_sum == distance * 255


def test_nonprimary_integer_quantization_is_hand_calculated():
    with Image.new("RGB", (1, 1), (255, 0, 0)) as red, Image.new("RGB", (1, 1), (80, 40, 20)) as other:
        result = measure_pixel_change(red, other)
    # chroma=60, V=80, H=floor(20*256/60)=85, S=floor(60*255/80)=191.
    assert (result.hue_sum, result.saturation_sum, result.value_sum) == (85 * 191, 64, 175)


@pytest.mark.parametrize(("radius", "edge_sum"), [(1, 510), (2, 1020), (3, 765), (4, 510)])
def test_forward_gradient_radius_and_clamped_single_row(radius, edge_sum):
    raw = bytes(channel for value in (0, 0, 255, 255, 0) for channel in (value, value, value))
    with Image.new("RGB", (5, 1)) as left, Image.frombytes("RGB", (5, 1), raw) as right:
        result = measure_pixel_change(left, right, PixelChangeConfig(radius))
    assert result.edge_sum == edge_sum and result.value_sum == 510


@pytest.mark.parametrize("bad", [True, False, 0, 5, -1, 1.0, 10**1000])
def test_invalid_radius_is_rejected(bad):
    with pytest.raises(ConfigurationError):
        PixelChangeConfig(bad)


@pytest.mark.parametrize(
    "options",
    [
        {"max_measurement_pixels": 0},
        {"max_measurement_pixels": True},
        {"max_measurement_pixels": 1_000_000_001},
        {"max_pair_rgb_bytes": 5},
        {"max_pair_rgb_bytes": False},
        {"max_pair_rgb_bytes": 512 * 1024 * 1024 + 1},
    ],
)
def test_invalid_resource_limits_are_rejected(options):
    with pytest.raises(ConfigurationError):
        PixelChangeLimits(**options)


@pytest.mark.parametrize("bad", [True, -1, float("nan"), float("inf"), 1_000_001, 10**1000, "1"])
def test_invalid_weights_are_rejected(bad):
    with pytest.raises(ConfigurationError):
        PixelChangeWeights(hue=bad)


def test_weight_normalization_handles_subnormal_and_maximum_values():
    result = PixelChange(PixelChangeConfig(), 1, 1, 0, 0, 255, 0)
    for scale in (5e-324, 1, 1_000_000):
        assert result.score(PixelChangeWeights(scale, scale, scale, scale)) == 0.25
    with pytest.raises(ConfigurationError):
        PixelChangeWeights(0, 0, 0, 0)
    with pytest.raises(ConfigurationError):
        result.score(object())


@pytest.mark.parametrize(
    "options",
    [
        {"config": None},
        {"width": 0},
        {"height": True},
        {"width": 67_108_864, "height": 2},
        {"hue_sum": 195841},
        {"saturation_sum": 256},
        {"value_sum": -1},
        {"edge_sum": 511},
        {"hue_sum": 0.0},
    ],
)
def test_record_contradictions_are_rejected(options):
    values = {
        "config": PixelChangeConfig(),
        "width": 1,
        "height": 1,
        "hue_sum": 0,
        "saturation_sum": 0,
        "value_sum": 0,
        "edge_sum": 0,
    }
    values.update(options)
    with pytest.raises(ConfigurationError):
        PixelChange(**values)


@pytest.mark.parametrize("options", [{"config": object()}, {"limits": object()}])
def test_wrong_configuration_types_reject_before_image_access(options):
    with pytest.raises(ConfigurationError):
        measure_pixel_change(None, None, **options)


def test_size_and_work_admission_precede_conversion(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("conversion began before admission")

    with Image.new("RGB", (2, 2)) as left, Image.new("RGB", (3, 2)) as different:
        monkeypatch.setattr(Image.Image, "convert", forbidden)
        for right, limits in (
            (different, PixelChangeLimits()),
            (left, PixelChangeLimits(max_measurement_pixels=7)),
            (left, PixelChangeLimits(max_pair_rgb_bytes=23)),
        ):
            with pytest.raises(ConfigurationError):
                measure_pixel_change(left, right, limits=limits)
        with pytest.raises(ConfigurationError):
            measure_pixel_change(left, object())


def test_alpha_is_discarded_and_caller_images_remain_unchanged():
    with Image.new("RGBA", (2, 2), (255, 0, 0, 0)) as left, Image.new("RGB", (2, 2), "red") as right:
        before = left.tobytes(), right.tobytes()
        assert measure_pixel_change(left, right).components == (0, 0, 0, 0)
        assert (left.tobytes(), right.tobytes()) == before
        assert measure_pixel_change(left, left).components == (0, 0, 0, 0)


def test_borrowed_conversion_alias_is_rejected_without_closing_it(monkeypatch):
    with Image.new("RGB", (2, 2)) as left, Image.new("RGB", (2, 2)) as right:
        monkeypatch.setattr(Image.Image, "convert", lambda *args, **kwargs: right)
        with pytest.raises(ConfigurationError, match="distinct owned"):
            measure_pixel_change(left, right)
        assert len(left.tobytes()) == len(right.tobytes()) == 12


@pytest.mark.parametrize("bad", ["mode", "dimensions"])
def test_wrong_converted_representation_is_closed_before_rejection(monkeypatch, bad):
    converted = Image.new("L" if bad == "mode" else "RGB", (2, 2) if bad == "mode" else (3, 2))
    with Image.new("RGB", (2, 2)) as image:
        monkeypatch.setattr(Image.Image, "convert", lambda *args, **kwargs: converted)
        with pytest.raises(ConfigurationError, match="preserve"):
            measure_pixel_change(image, image)
    with pytest.raises(ValueError):
        converted.tobytes()


@pytest.mark.parametrize("bad", [bytearray(3), memoryview(b"000"), b"", None])
def test_packed_kernel_rejects_nonexact_byte_records(bad):
    with pytest.raises(ConfigurationError):
        module._measure_rgb_change(bad, b"000", 1, 1, PixelChangeConfig(), PixelChangeLimits())


def test_packed_kernel_requires_typed_options_and_serializes_integer_evidence():
    with pytest.raises(ConfigurationError):
        module._measure_rgb_change(b"000", b"000", 1, 1, None, PixelChangeLimits())
    result = module._measure_rgb_change(b"000", b"000", 1, 1, PixelChangeConfig(), PixelChangeLimits())
    assert result.to_dict() == {
        "width": 1,
        "height": 1,
        "hue_sum": 0,
        "saturation_sum": 0,
        "value_sum": 0,
        "edge_sum": 0,
    }


@pytest.mark.parametrize(
    "primary,cleanup,expected",
    [
        (KeyboardInterrupt, OSError, KeyboardInterrupt),
        (SystemExit, OSError, SystemExit),
        (OSError, KeyboardInterrupt, KeyboardInterrupt),
    ],
)
def test_conversion_cleanup_preserves_control_exception_precedence(monkeypatch, primary, cleanup, expected):
    converted = Image.new("RGB", (1, 1))
    original_close = converted.close

    def read():
        raise primary("read failed")

    def close():
        original_close()
        raise cleanup("close failed")

    with Image.new("RGB", (1, 1)) as borrowed:
        monkeypatch.setattr(borrowed, "convert", lambda *args: converted)
        monkeypatch.setattr(converted, "tobytes", read)
        monkeypatch.setattr(converted, "close", close)
        with pytest.raises(expected):
            measure_pixel_change(borrowed, borrowed)
        assert borrowed.tobytes() == b"\x00\x00\x00"
