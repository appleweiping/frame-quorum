"""Generated offline scene-to-QP example with a direct decoder ordinal oracle."""

from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

from frame_quorum import DetectionConfig, NativeSceneConfig, detect_native_scenes, write_native_qp_bundle


def main() -> None:
    import av

    values = (0, 0, 255, 255, 255, 0, 0)
    pts = (5000, 5040, 5110, 5180, 5270, 5310, 5470)
    with TemporaryDirectory(prefix="frame-quorum-qp-example-") as directory:
        root = Path(directory)
        source = root / "sample.mkv"
        with av.open(str(source), "w", format="matroska") as container:
            stream = container.add_stream("ffv1", rate=25)
            stream.width, stream.height, stream.pix_fmt = 8, 6, "bgr0"
            stream.time_base = stream.codec_context.time_base = Fraction(1, 1000)
            for value, timestamp in zip(values, pts, strict=True):
                with Image.new("RGB", (8, 6), (value, value, value)) as image:
                    frame = av.VideoFrame.from_image(image)
                frame.pts, frame.time_base = timestamp, Fraction(1, 1000)
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        with av.open(str(source)) as container:
            observed = []
            for ordinal, frame in enumerate(container.decode(video=0)):
                with frame.to_rgb().to_image() as image:
                    observed.append((ordinal, frame.pts, image.getpixel((0, 0))[0]))
        if observed != [(index, pts[index], values[index]) for index in range(len(values))]:
            raise RuntimeError("direct decoder ordinal/PTS/pixel oracle disagrees with source")
        scenes = detect_native_scenes(
            source,
            NativeSceneConfig(detectors=(DetectionConfig(detector="luminance", threshold=0.5),)),
        )
        if scenes.cut_positions != (2, 5):
            raise RuntimeError("expected scene boundaries differ from generated pixel sequence")
        result = write_native_qp_bundle(scenes, root / "qp")
        actual = result.output_path.read_bytes()
        expected = b"0 I -1\n2 I -1\n5 I -1\n"
        if actual != expected or result.qp_sha256 != hashlib.sha256(expected).hexdigest():
            raise RuntimeError("QP file differs from direct decoded-frame ordinal oracle")
        audit = json.loads((result.output_path.parent / "audit.json").read_bytes())
        if audit["qp_sha256"] != result.qp_sha256 or audit["encoder_input_verified"] is not False:
            raise RuntimeError("QP audit disagrees with published bytes or provenance")
        print(
            json.dumps(
                {
                    "frame_count": result.frame_count,
                    "cut_count": result.cut_count,
                    "encoder_input_verified": False,
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
