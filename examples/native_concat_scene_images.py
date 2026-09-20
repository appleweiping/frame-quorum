"""Generate, export and verify mixed-clock composite scene stills offline."""

from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory

import av
from PIL import Image

from frame_quorum import (
    DetectionConfig,
    NativeConcatClip,
    NativeConcatSceneConfig,
    NativeSceneImageConfig,
    export_native_concat_scene_images,
)


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def write_video(path: Path, level: int, clock: Fraction, origin: int, step: int, rate: int) -> None:
    with av.open(str(path), "w", format="matroska" if path.suffix == ".mkv" else "nut") as container:
        stream = container.add_stream("ffv1", rate=rate)
        stream.width, stream.height, stream.pix_fmt = 16, 12, "bgr0"
        stream.time_base = stream.codec_context.time_base = clock
        for index in range(4):
            with Image.new("RGB", (16, 12), (level, level, level)) as image:
                frame = av.VideoFrame.from_image(image)
            frame.pts, frame.time_base = origin + step * index, clock
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def main() -> None:
    with TemporaryDirectory(prefix="frame-quorum-concat-stills-") as directory:
        root = Path(directory)
        first, second = root / "first.mkv", root / "second.nut"
        write_video(first, 0, Fraction(1, 1000), 5000, 50, 20)
        write_video(second, 255, Fraction(1, 48000), 48000, 2400, 30)
        clips = (
            NativeConcatClip(first, start=5, end=Fraction(1, 5) + 5),
            NativeConcatClip(second, start=1, end=Fraction(6, 5)),
        )
        scenes = NativeConcatSceneConfig(detectors=(DetectionConfig(detector="luminance", threshold=0.5),))
        result = export_native_concat_scene_images(
            clips, root / "images", scenes, NativeSceneImageConfig(images_per_scene=3)
        )
        manifest_bytes = (result.output_dir / "manifest.json").read_bytes()
        document = json.loads(manifest_bytes)
        expect(result.manifest_sha256 == hashlib.sha256(manifest_bytes).hexdigest(), "manifest hash mismatch")
        expect(result.scene_count == 2 and result.image_count == 6, "cross-source scene partition mismatch")
        expect(document["sample_count"] == 8, "native sample count mismatch")
        expect(document["scenes"][0]["end_time"] == {"numerator": 1, "denominator": 5}, "cut mismatch")
        expect(document["coverage"] == "declared_only", "declaration coverage must be explicit")
        expect(document["source_content_authenticated"] is False, "source cannot be called authenticated")
        expect(document["detection_diagnostics"] == document["replay_diagnostics"], "replay mismatch")
        expected_rgb = [bytes([0]) * 16 * 12 * 3] * 3 + [bytes([255]) * 16 * 12 * 3] * 3
        for row, rgb in zip(document["images"], expected_rgb, strict=True):
            with Image.open(result.output_dir / row["file"]) as image:
                image.load()
                expect(image.format == "PNG" and image.tobytes() == rgb, "published PNG pixel mismatch")
        print(
            json.dumps(
                {
                    "scene_count": result.scene_count,
                    "image_count": result.image_count,
                    "sample_count": document["sample_count"],
                    "cut_time": "1/5",
                    "native_clocks": ["1/1000", "1/48000"],
                    "coverage": document["coverage"],
                    "source_content_authenticated": document["source_content_authenticated"],
                    "decode_passes": document["decode_passes"],
                    "verified_pixels": True,
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
