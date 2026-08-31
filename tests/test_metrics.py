from __future__ import annotations

from PIL import Image, ImageDraw, ImageFilter

from frame_quorum.metrics import (
    colorfulness,
    content_distance,
    difference_hash,
    edge_energy,
    grayscale_entropy,
    measure_image,
)


def test_difference_hash_is_stable() -> None:
    image = Image.linear_gradient("L")
    assert difference_hash(image) == difference_hash(image.copy())


def test_difference_hash_not_equal_for_opposite_gradients() -> None:
    forward = Image.linear_gradient("L").rotate(90)
    reverse = forward.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    assert difference_hash(forward) != difference_hash(reverse)


def test_solid_image_has_zero_entropy() -> None:
    assert grayscale_entropy(Image.new("L", (64, 64), 127)) == 0.0


def test_gradient_has_substantial_entropy() -> None:
    assert grayscale_entropy(Image.linear_gradient("L")) > 0.9


def test_edges_score_above_blurred_edges() -> None:
    image = Image.new("L", (96, 96), 0)
    ImageDraw.Draw(image).rectangle((25, 25, 70, 70), fill=255)
    blurred = image.filter(ImageFilter.GaussianBlur(8))
    assert edge_energy(image) > edge_energy(blurred)


def test_neutral_image_has_no_colorfulness() -> None:
    assert colorfulness(Image.new("RGB", (32, 32), (80, 80, 80))) == 0.0


def test_color_blocks_are_colorful() -> None:
    image = Image.new("RGB", (64, 64), "red")
    ImageDraw.Draw(image).rectangle((32, 0, 63, 63), fill="blue")
    assert colorfulness(image) > 0.5


def test_content_distance_is_zero_for_same_metrics() -> None:
    metrics = measure_image(Image.new("RGB", (80, 80), (10, 20, 30)))
    assert content_distance(metrics, metrics) == 0.0


def test_content_distance_not_blind_to_solid_color() -> None:
    red = measure_image(Image.new("RGB", (80, 80), "red"))
    blue = measure_image(Image.new("RGB", (80, 80), "blue"))
    assert red.perceptual_hash == blue.perceptual_hash
    assert content_distance(red, blue) > 0.1


def test_measured_values_are_normalized() -> None:
    metrics = measure_image(Image.effect_noise((128, 128), 80).convert("RGB"))
    values = [
        metrics.luminance,
        metrics.entropy,
        metrics.sharpness,
        metrics.colorfulness,
        metrics.mean_red,
        metrics.mean_green,
        metrics.mean_blue,
    ]
    assert all(0.0 <= value <= 1.0 for value in values)
