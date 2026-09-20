"""Independent real-codec acceptance for declared-composite scene stills."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import pytest
from PIL import Image

from frame_quorum import (
    DetectionConfig,
    NativeConcatClip,
    NativeConcatConfig,
    NativeConcatSceneConfig,
    NativeSceneImageConfig,
    export_native_concat_scene_images,
)

av = pytest.importorskip("av", reason="install frame-quorum[video] for composite still tests")


def _write_gray(path, values, stamps, clock, rate, format_name):
    with av.open(str(path), "w", format=format_name) as container:
        video = container.add_stream("ffv1", rate=rate)
        video.width, video.height, video.pix_fmt = 16, 12, "bgr0"
        video.time_base = video.codec_context.time_base = clock
        for gray, stamp in zip(values, stamps, strict=True):
            with Image.new("RGB", (16, 12), (gray, gray, gray)) as image:
                frame = av.VideoFrame.from_image(image)
            frame.pts, frame.time_base = stamp, clock
            for packet in video.encode(frame):
                container.mux(packet)
        for packet in video.encode():
            container.mux(packet)


def _media(tmp_path):
    first, second = tmp_path / "first.mkv", tmp_path / "second.nut"
    _write_gray(first, (0, 0, 0, 0), (5000, 5040, 5150, 5300), Fraction(1, 1000), 25, "matroska")
    _write_gray(second, (255, 255, 255, 255), (48000, 49200, 51600, 57600), Fraction(1, 48000), 30, "nut")
    return (
        NativeConcatClip(first, start=5, end=Fraction(27, 5)),
        NativeConcatClip(second, start=1, end=Fraction(13, 10)),
    )


def _direct(clips, *, step=1):
    """Only direct PyAV, declared arithmetic and RGB bytes; no product decoder."""
    all_rows = []
    offset = Fraction(0)
    for occurrence, clip in enumerate(clips):
        native_ordinal = 0
        with av.open(str(clip.path)) as container:
            for decode_index, frame in enumerate(container.decode(video=clip.video_stream)):
                time = frame.pts * frame.time_base
                if clip.start <= time < clip.end:
                    with frame.to_image() as image, image.convert("RGB") as rgb:
                        data = rgb.tobytes()
                        size = rgb.size
                    all_rows.append(
                        (
                            occurrence,
                            offset + time - clip.start,
                            frame.pts,
                            frame.time_base,
                            decode_index,
                            native_ordinal,
                            size,
                            data,
                        )
                    )
                    native_ordinal += 1
        offset += clip.end - clip.start
    return all_rows[::step]


def _pair(value):
    return {"numerator": value.numerator, "denominator": value.denominator}


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _manifest(target):
    data = (target / "manifest.json").read_bytes()
    document = json.loads(data)
    assert data == (
        json.dumps(document, ensure_ascii=True, sort_keys=True, allow_nan=False, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    return document


def test_real_composite_still_api_is_available_and_produces_manifest(tmp_path):
    clips = _media(tmp_path)
    result = export_native_concat_scene_images(clips, tmp_path / "images")
    assert result.output_dir.is_dir()
    assert result.scene_count == 2


@pytest.mark.parametrize("optimized", [False, True])
def test_generated_offline_example_runs_isolated_and_under_optimization(tmp_path, optimized):
    example = Path(__file__).resolve().parents[1] / "examples" / "native_concat_scene_images.py"
    process = subprocess.run(
        [sys.executable, "-I", "-B", *(["-O"] if optimized else []), str(example)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert process.returncode == 0, process.stderr
    assert not process.stderr
    assert json.loads(process.stdout) == {
        "scene_count": 2,
        "image_count": 6,
        "sample_count": 8,
        "cut_time": "1/5",
        "native_clocks": ["1/1000", "1/48000"],
        "coverage": "declared_only",
        "source_content_authenticated": False,
        "decode_passes": 2,
        "verified_pixels": True,
    }
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("image_format", ["png", "jpeg"])
@pytest.mark.parametrize("step", [1, 2])
def test_direct_pyav_oracle_covers_every_selected_pixel_and_native_coordinate(tmp_path, image_format, step):
    clips = _media(tmp_path)
    target = tmp_path / "images"
    direct = _direct(clips, step=step)
    assert len(direct) == 8 // step
    config = NativeConcatSceneConfig(
        video=NativeConcatConfig(frame_step=step),
        detectors=(DetectionConfig(detector="luminance", threshold=0.5),),
    )
    result = export_native_concat_scene_images(
        clips, target, config, NativeSceneImageConfig(image_format=image_format)
    )
    document = _manifest(target)
    assert set(document) == {
        "kind",
        "schema_version",
        "selection",
        "coverage",
        "source_content_authenticated",
        "timeline_digest",
        "scene_config",
        "image_config",
        "detection_diagnostics",
        "replay_diagnostics",
        "sample_count",
        "scene_count",
        "image_count",
        "unique_sample_count",
        "scenes",
        "images",
        "transformed_pixels",
        "verification_pixels",
        "decode_passes",
        "pillow_version",
        "pyav_version",
        "native_libraries",
        "jpeg_encoding",
    }
    assert document["kind"] == "frame-quorum-native-concat-scene-images"
    assert document["selection"] == "observed-composite-samples-v1"
    assert document["coverage"] == "declared_only"
    assert document["source_content_authenticated"] is False
    assert document["decode_passes"] == 2
    assert document["sample_count"] == len(direct)
    assert document["detection_diagnostics"] == document["replay_diagnostics"]
    assert document["detection_diagnostics"]["status"] == "intervals_exhausted"
    assert document["scene_config"] == config.to_dict()
    assert document["scene_count"] == result.scene_count == 2
    cut = 4 // step
    assert [(scene["start_position"], scene["end_position"]) for scene in document["scenes"]] == [
        (0, cut),
        (cut, len(direct)),
    ]
    assert document["scenes"][0]["end_time"] == _pair(Fraction(2, 5))
    assert document["scenes"][1]["end_time"] == _pair(Fraction(7, 10))
    expected_slots = [1, 1, 2, 5, 5, 6] if step == 1 else [0, 0, 1, 2, 2, 3]
    assert document["image_count"] == result.image_count == len(expected_slots)
    assert document["unique_sample_count"] == result.unique_sample_count == len(set(expected_slots))
    for ordinal, (row, index) in enumerate(zip(document["images"], expected_slots, strict=True)):
        occurrence, time, pts, clock, decode_index, native_ordinal, size, rgb = direct[index]
        assert set(row) == {
            "scene_ordinal",
            "image_index",
            "sample_index",
            "presentation_time",
            "clip_index",
            "activation",
            "generation",
            "native_decode_index",
            "native_sample_index",
            "native_generation",
            "native_pts",
            "native_time_base",
            "source_width",
            "source_height",
            "width",
            "height",
            "source_rgb_sha256",
            "transformed_rgb_sha256",
            "decoded_rgb_sha256",
            "file",
            "bytes",
            "sha256",
        }
        assert row["scene_ordinal"] == ordinal // 3
        assert row["image_index"] == ordinal % 3
        assert row["sample_index"] == index
        assert row["presentation_time"] == _pair(time)
        assert row["clip_index"] == occurrence
        assert row["activation"] == 1 and row["generation"] == row["native_generation"] == 0
        assert (row["native_pts"], row["native_time_base"]) == (pts, _pair(clock))
        assert (row["native_decode_index"], row["native_sample_index"]) == (decode_index, native_ordinal)
        assert (row["source_width"], row["source_height"]) == size
        assert (row["width"], row["height"]) == size
        assert row["source_rgb_sha256"] == row["transformed_rgb_sha256"] == _sha(rgb)
        extension = "png" if image_format == "png" else "jpg"
        assert row["file"] == f"scene-{ordinal // 3 + 1:06d}-image-{ordinal % 3 + 1:06d}.{extension}"
        encoded = (target / row["file"]).read_bytes()
        assert row["bytes"] == len(encoded) and row["sha256"] == _sha(encoded)
        with Image.open(target / row["file"]) as image:
            assert image.format == image_format.upper()
            assert image.size == size and image.mode == "RGB"
            assert not image.info.get("exif") and not image.info.get("icc_profile")
            image.load()
            decoded = image.tobytes()
        assert row["decoded_rgb_sha256"] == _sha(decoded)
        if image_format == "png":
            assert decoded == rgb
    assert result.manifest_sha256 == _sha((target / "manifest.json").read_bytes())
    assert result.timeline_digest == document["timeline_digest"]
    assert result.total_output_bytes == sum(path.stat().st_size for path in target.iterdir())
    assert {path.name for path in target.iterdir()} == {
        "manifest.json",
        *(row["file"] for row in document["images"]),
    }
    for clip in clips:
        clip.path.unlink()


def test_repeated_path_and_unobserved_declared_occurrence_remain_distinct(tmp_path):
    first, second = _media(tmp_path)
    clips = (
        first,
        second,
        NativeConcatClip(first.path, start=9, end=Fraction(91, 10)),
        first,
    )
    direct = _direct(clips)
    assert [row[0] for row in direct] == [0] * 4 + [1] * 4 + [3] * 4
    target = tmp_path / "images"
    export_native_concat_scene_images(clips, target)
    document = _manifest(target)
    assert document["sample_count"] == 12
    assert document["scene_count"] == 3
    assert document["detection_diagnostics"]["source_activations"] == [2, 2, 2, 2]
    assert document["scenes"][-1]["end_time"] == _pair(Fraction(6, 5))
    assert any(span["clip_index"] == 2 for scene in document["scenes"] for span in scene["spans"])


def test_composite_png_resize_matches_independent_direct_decode_and_pillow_pixels(tmp_path):
    clips = _media(tmp_path)
    target = tmp_path / "images"
    direct = _direct(clips)
    export_native_concat_scene_images(
        clips,
        target,
        image_config=NativeSceneImageConfig(width=8, height=6, interpolation="nearest", images_per_scene=1),
    )
    document = _manifest(target)
    assert document["image_count"] == 2 and document["unique_sample_count"] == 2
    for row, index in zip(document["images"], (1, 5), strict=True):
        source = direct[index]
        with (
            Image.frombytes("RGB", source[6], source[7]) as original,
            original.resize((8, 6), Image.Resampling.NEAREST) as expected,
        ):
            pixels = expected.tobytes()
        assert row["sample_index"] == index
        assert row["transformed_rgb_sha256"] == _sha(pixels)
        with Image.open(target / row["file"]) as actual:
            assert actual.size == (8, 6)
            assert actual.tobytes() == pixels


def test_explicit_composite_range_end_is_not_inferred_from_eof(tmp_path):
    clips = _media(tmp_path)
    target = tmp_path / "images"
    end = Fraction(3, 5)
    config = NativeConcatSceneConfig(video=NativeConcatConfig(end=end))
    export_native_concat_scene_images(clips, target, config)
    document = _manifest(target)
    assert document["detection_diagnostics"]["status"] == "range_end"
    assert document["sample_count"] == 7
    assert document["scenes"][-1]["end_time"] == _pair(end)
    assert document["scenes"][-1]["end_reason"] == "declared_end"
