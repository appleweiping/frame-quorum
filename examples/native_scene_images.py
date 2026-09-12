"""Generate VFR media, export observed scene stills and verify their exact pixels.

Run: python examples/native_scene_images.py (requires frame-quorum[video]).
All media is generated in an owned temporary directory and removed on exit.
"""

from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

from frame_quorum import (
    DetectionConfig,
    NativeSceneConfig,
    NativeSceneImageConfig,
    export_native_scene_images,
)


def main() -> None:
    import av

    times = (5000, 5040, 5110, 5180, 5270, 5310, 5470, 5680)
    width, height = 16, 12
    originals = tuple(
        bytes(
            (180 if index >= 4 else 0) + index + (x + y) % 16
            for y in range(height)
            for x in range(width)
            for _channel in range(3)
        )
        for index in range(len(times))
    )
    with TemporaryDirectory(prefix="frame-quorum-scene-images-") as directory:
        source = Path(directory) / "generated.nut"
        with av.open(str(source), "w", format="nut") as container:
            stream = container.add_stream("ffv1", rate=25)
            stream.width, stream.height, stream.pix_fmt = width, height, "bgr0"
            stream.time_base = stream.codec_context.time_base = Fraction(1, 1000)
            for pts, pixels in zip(times, originals, strict=True):
                with Image.frombytes("RGB", (width, height), pixels) as image:
                    frame = av.VideoFrame.from_image(image)
                frame.pts, frame.time_base = pts, Fraction(1, 1000)
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)

        result = export_native_scene_images(
            source,
            Path(directory) / "stills",
            NativeSceneConfig(detectors=(DetectionConfig(detector="luminance", threshold=0.5),)),
            NativeSceneImageConfig(images_per_scene=2, sample_margin=0, width=8, interpolation="nearest"),
        )
        manifest_bytes = (result.output_dir / "manifest.json").read_bytes()
        document = json.loads(manifest_bytes)
        positions = [0, 3, 4, 7]
        assert (result.scene_count, result.image_count, result.unique_sample_count) == (2, 4, 4)
        assert [row["source_sample_index"] for row in document["images"]] == positions
        assert document["scenes"][-1]["end_time"] is None
        assert hashlib.sha256(manifest_bytes).hexdigest() == result.manifest_sha256
        for row, position in zip(document["images"], positions, strict=True):
            assert row["source_rgb_sha256"] == hashlib.sha256(originals[position]).hexdigest()
            assert Fraction(**row["source_time"]) == Fraction(times[position], 1000)
            path = result.output_dir / row["file"]
            assert row["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
            # Independent 2:1 nearest-neighbor pixel-center oracle, not a resize call.
            expected = bytes(
                originals[position][((2 * y + 1) * width + (2 * x + 1)) * 3 + channel]
                for y in range(6)
                for x in range(8)
                for channel in range(3)
            )
            with Image.open(path) as image:
                assert image.format == "PNG" and image.size == (8, 6)
                assert image.tobytes() == expected
        print(
            json.dumps(
                {
                    "scenes": result.scene_count,
                    "images": result.image_count,
                    "observed_samples": positions,
                    "exact_pixels_verified": True,
                    "last_scene_end_known": False,
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
