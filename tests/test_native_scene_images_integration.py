from __future__ import annotations

import hashlib
import json
from fractions import Fraction

import pytest
from PIL import Image

from frame_quorum import (
    DetectionConfig,
    NativeSceneConfig,
    NativeSceneImageConfig,
    NativeVideoConfig,
    detect_native_scenes,
    export_native_scene_images,
)

av = pytest.importorskip("av", reason="install frame-quorum[video] for genuine native scene image tests")
PTS = (5000, 5040, 5110, 5180, 5270, 5310, 5470, 5680)
TIME_BASE = Fraction(1, 1000)
SIZE = (16, 12)


def gray(value):
    return bytes((value, value, value)) * (SIZE[0] * SIZE[1])


def textured():
    return bytes((offset * 37 + (offset // 11) * 53) % 256 for offset in range(SIZE[0] * SIZE[1] * 3))


def make_video(path, frames, pts=PTS):
    with av.open(str(path), "w", format="matroska") as container:
        stream = container.add_stream("ffv1", rate=25)
        stream.width, stream.height, stream.pix_fmt = *SIZE, "bgr0"
        stream.time_base = stream.codec_context.time_base = TIME_BASE
        for pixels, timestamp in zip(frames, pts, strict=True):
            with Image.frombytes("RGB", SIZE, pixels) as image:
                frame = av.VideoFrame.from_image(image)
            frame.pts, frame.time_base = timestamp, TIME_BASE
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def independently_decode(path):
    # This oracle never calls the product decoder, image encoder, or selector.
    with av.open(str(path)) as container:
        snapshots = []
        for frame in container.decode(video=0):
            with frame.to_image() as image, image.convert("RGB") as rgb:
                snapshots.append((frame.pts, frame.time_base, rgb.size, rgb.tobytes()))
    return snapshots


def pair(value):
    return {"numerator": value.numerator, "denominator": value.denominator}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def manifest(target):
    return json.loads((target / "manifest.json").read_text(encoding="utf-8"))


def scene_config(video=None):
    return NativeSceneConfig(
        video=video or NativeVideoConfig(),
        detectors=(DetectionConfig(detector="luminance", threshold=0.5),),
    )


def assert_image_row(
    target,
    row,
    snapshot,
    *,
    sample_index,
    decode_index,
    expected_rgb=None,
    expected_size=SIZE,
    image_format="PNG",
):
    pts, time_base, source_size, source_rgb = snapshot
    transformed = source_rgb if expected_rgb is None else expected_rgb
    assert row["source_sample_index"] == sample_index
    assert row["source_decode_index"] == decode_index
    assert row["source_generation"] == 0
    assert row["source_pts"] == pts
    assert row["source_time_base"] == pair(time_base)
    assert row["source_time"] == pair(pts * time_base)
    assert (row["source_width"], row["source_height"]) == source_size
    assert (row["width"], row["height"]) == expected_size
    assert row["source_rgb_sha256"] == sha256(source_rgb)
    assert row["transformed_rgb_sha256"] == sha256(transformed)
    path = target / row["file"]
    encoded = path.read_bytes()
    assert row["bytes"] == len(encoded)
    assert row["sha256"] == sha256(encoded)
    with Image.open(path) as image:
        assert image.format == image_format
        assert image.size == expected_size
        image.load()
        assert not image.info.get("exif") and not image.info.get("icc_profile")
        with image.convert("RGB") as rgb:
            decoded = rgb.tobytes()
    assert row["decoded_rgb_sha256"] == sha256(decoded)
    if image_format == "PNG":
        assert decoded == transformed
    return decoded


def test_real_vfr_png_repeated_slots_preserve_exact_pts_pixels_and_scene_endpoints(tmp_path):
    source, target = tmp_path / "cut.mkv", tmp_path / "images"
    original = [gray(value) for value in (0, 8, 16, 24, 220, 228, 236, 244)]
    make_video(source, original)
    independent = independently_decode(source)
    assert [item[0] * item[1] for item in independent] == [pts * TIME_BASE for pts in PTS]
    assert [item[3] for item in independent] == original
    options = scene_config()
    scenes = detect_native_scenes(source, options)
    assert scenes.cut_positions == (4,)
    result = export_native_scene_images(source, target, options)
    document = manifest(target)
    assert document["scenes"] == [scene.to_dict() for scene in scenes.scenes]
    assert document["scenes"][-1]["end_time"] is None
    assert document["scenes"][-1]["end_reason"] == "unknown"
    assert document["scenes"][0]["end_time"] == pair(PTS[4] * TIME_BASE)
    assert document["detection_diagnostics"] == document["replay_diagnostics"]
    assert document["detection_diagnostics"]["status"] == "eof"
    assert document["scene_count"] == result.scene_count == 2
    assert document["image_count"] == result.image_count == 6
    assert document["unique_sample_count"] == result.unique_sample_count == 4
    expected_indices = (1, 1, 2, 5, 5, 6)
    assert len(document["images"]) == len(expected_indices)
    for ordinal, (row, index) in enumerate(zip(document["images"], expected_indices, strict=True)):
        assert row["scene_ordinal"] == ordinal // 3
        assert row["image_index"] == ordinal % 3
        assert row["file"] == f"scene-{ordinal // 3 + 1:06d}-image-{ordinal % 3 + 1:06d}.png"
        assert_image_row(target, row, independent[index], sample_index=index, decode_index=index)
    assert result.output_dir == target
    assert result.total_output_bytes == sum(path.stat().st_size for path in target.iterdir())
    assert result.manifest_sha256 == sha256((target / "manifest.json").read_bytes())
    assert {path.name for path in target.iterdir()} == {
        "manifest.json",
        *(row["file"] for row in document["images"]),
    }
    assert {path.name for path in tmp_path.iterdir()} == {source.name, target.name}
    source.unlink()  # Both native passes have released the source without relying on GC.


@pytest.mark.parametrize(
    "count,images,margin,indices",
    [
        (7, 3, 1, (1, 3, 5)),
        (4, 3, 1, (1, 1, 2)),
        (2, 3, 1, (0, 0, 1)),
        (1, 3, 100, (0, 0, 0)),
        (6, 1, 0, (2,)),
    ],
)
def test_real_short_scenes_use_observed_positions_and_keep_requested_slot_count(
    tmp_path, count, images, margin, indices
):
    source, target = tmp_path / "short.mkv", tmp_path / "images"
    make_video(source, [gray(index * 8) for index in range(count)], PTS[:count])
    independent = independently_decode(source)
    export_native_scene_images(
        source,
        target,
        scene_config(),
        NativeSceneImageConfig(images_per_scene=images, sample_margin=margin),
    )
    document = manifest(target)
    assert document["scene_count"] == 1
    assert document["image_count"] == images
    assert document["unique_sample_count"] == len(set(indices))
    assert [row["source_sample_index"] for row in document["images"]] == list(indices)
    for row, index in zip(document["images"], indices, strict=True):
        assert_image_row(target, row, independent[index], sample_index=index, decode_index=index)
    assert document["scenes"][-1]["end_time"] is None


def test_real_stride_keeps_sample_and_decode_indices_distinct_and_known_requested_end(tmp_path):
    source, target = tmp_path / "stride.mkv", tmp_path / "images"
    make_video(source, [gray(value) for value in (0, 8, 16, 24, 220, 228, 236, 244)])
    independent = independently_decode(source)
    options = scene_config(NativeVideoConfig(start=Fraction(126, 25), end=Fraction(28, 5), frame_step=2))
    scenes = detect_native_scenes(source, options)
    assert [row.sample.pts for row in scenes.statistics] == [5040, 5180, 5310]
    assert scenes.cut_positions == (2,)
    export_native_scene_images(source, target, options)
    document = manifest(target)
    sample_indices = (0, 0, 1, 2, 2, 2)
    source_indices = (1, 1, 3, 5, 5, 5)
    assert document["image_count"] == 6 and document["unique_sample_count"] == 3
    assert document["scenes"] == [scene.to_dict() for scene in scenes.scenes]
    assert document["scenes"][-1]["end_time"] == pair(Fraction(28, 5))
    assert document["scenes"][-1]["end_reason"] == "requested_end"
    assert document["detection_diagnostics"]["status"] == "range_end"
    assert document["detection_diagnostics"] == document["replay_diagnostics"]
    for row, sample_index, source_index in zip(
        document["images"], sample_indices, source_indices, strict=True
    ):
        assert_image_row(
            target,
            row,
            independent[source_index],
            sample_index=sample_index,
            decode_index=scenes.statistics[sample_index].sample.decode_index,
        )
    assert scenes.statistics[1].sample.decode_index - scenes.statistics[0].sample.decode_index == 2


@pytest.mark.parametrize("count_kind", ["frame", "decode"])
def test_real_count_limit_exports_only_observed_samples_and_never_invents_a_terminal_end(
    tmp_path, count_kind
):
    source, target = tmp_path / "limited.mkv", tmp_path / "images"
    make_video(source, [gray(value) for value in (0, 8, 16, 24, 220, 228, 236, 244)])
    independent = independently_decode(source)
    options = scene_config(
        NativeVideoConfig(
            max_frames=4 if count_kind == "frame" else 100,
            max_decoded_frames=4 if count_kind == "decode" else 100,
            end=Fraction(6),
        )
    )
    export_native_scene_images(source, target, options)
    document = manifest(target)
    assert document["scene_count"] == 1 and document["image_count"] == 3
    assert document["unique_sample_count"] == 2
    assert document["detection_diagnostics"] == document["replay_diagnostics"]
    assert document["detection_diagnostics"]["status"] == f"{count_kind}_limit"
    assert document["detection_diagnostics"]["returned_frames"] == 4
    assert document["detection_diagnostics"]["decoded_frames"] == 4
    (scene,) = document["scenes"]
    assert scene["last_sample_time"] == pair(PTS[3] * TIME_BASE)
    assert scene["end_time"] is None and scene["end_reason"] == "unknown"
    for row, index in zip(document["images"], (1, 1, 2), strict=True):
        assert_image_row(target, row, independent[index], sample_index=index, decode_index=index)
    source.unlink()


@pytest.mark.parametrize("interpolation", ["nearest", "bilinear", "bicubic", "lanczos"])
@pytest.mark.parametrize(
    "resize,expected_size",
    [
        ({"width": 7}, (7, 5)),
        ({"height": 5}, (6, 5)),
        ({"width": 7, "height": 9}, (7, 9)),
        ({"scale": Fraction(3, 2)}, (24, 18)),
        ({"scale": Fraction(2, 3)}, (10, 8)),
    ],
)
def test_real_png_resize_matches_independent_pillow_pixels_and_floor_dimensions(
    tmp_path, interpolation, resize, expected_size
):
    source, target = tmp_path / "resize.mkv", tmp_path / "images"
    make_video(source, [textured()], PTS[:1])
    (snapshot,) = independently_decode(source)
    options = NativeSceneImageConfig(images_per_scene=1, interpolation=interpolation, **resize)
    export_native_scene_images(source, target, scene_config(), options)
    document = manifest(target)
    (row,) = document["images"]
    resample = getattr(Image.Resampling, interpolation.upper())
    with (
        Image.frombytes("RGB", snapshot[2], snapshot[3]) as image,
        image.resize(expected_size, resample=resample) as resized,
    ):
        expected = resized.tobytes()
    assert_image_row(
        target,
        row,
        snapshot,
        sample_index=0,
        decode_index=0,
        expected_rgb=expected,
        expected_size=expected_size,
    )


@pytest.mark.parametrize("compression", [0, 9])
def test_real_png_compression_extremes_remain_exact_lossless_rgb(tmp_path, compression):
    source, target = tmp_path / "png.mkv", tmp_path / "images"
    make_video(source, [textured()], PTS[:1])
    (snapshot,) = independently_decode(source)
    export_native_scene_images(
        source,
        target,
        scene_config(),
        NativeSceneImageConfig(images_per_scene=1, png_compression=compression),
    )
    (row,) = manifest(target)["images"]
    assert_image_row(target, row, snapshot, sample_index=0, decode_index=0)


@pytest.mark.parametrize("quality", [0, 95, 100])
def test_real_jpeg_records_lossy_decoded_digest_separately_from_source_and_transform(tmp_path, quality):
    source, target = tmp_path / "jpeg.mkv", tmp_path / "images"
    make_video(source, [textured()], PTS[:1])
    (snapshot,) = independently_decode(source)
    export_native_scene_images(
        source,
        target,
        scene_config(),
        NativeSceneImageConfig(images_per_scene=1, image_format="jpeg", jpeg_quality=quality),
    )
    document = manifest(target)
    (row,) = document["images"]
    decoded = assert_image_row(target, row, snapshot, sample_index=0, decode_index=0, image_format="JPEG")
    assert decoded != snapshot[3]
    assert row["source_rgb_sha256"] == row["transformed_rgb_sha256"]
    assert row["decoded_rgb_sha256"] != row["transformed_rgb_sha256"]
    assert document["scenes"][-1]["end_time"] is None
    source.unlink()
