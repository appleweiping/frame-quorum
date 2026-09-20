"""Offline edited-cut CSV to QP example with independent decoded-frame oracle."""

from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

from frame_quorum import write_loaded_qp_bundle


def main() -> None:
    import av

    pts = (5000, 5040, 5110, 5180, 5270, 5310, 5470)
    with TemporaryDirectory(prefix="frame-quorum-loaded-qp-example-") as directory:
        root = Path(directory)
        source = root / "sample.mkv"
        with av.open(str(source), "w", format="matroska") as container:
            stream = container.add_stream("ffv1", rate=25)
            stream.width, stream.height, stream.pix_fmt = 8, 6, "bgr0"
            stream.time_base = stream.codec_context.time_base = Fraction(1, 1000)
            for ordinal, timestamp in enumerate(pts):
                with Image.new("RGB", (8, 6), (ordinal * 29,) * 3) as image:
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
        if observed != [(i, pts[i], i * 29) for i in range(len(pts))]:
            raise RuntimeError("direct decoder ordinal/PTS/pixel oracle disagrees with source")
        csv_path = root / "edited.csv"
        csv_bytes = b"Scene Number,Start Frame,End Frame\n1,1,999\n2,3,999\n3,6,999\n"
        csv_path.write_bytes(csv_bytes)
        result = write_loaded_qp_bundle(source, csv_path, root / "qp")
        expected = b"0 I -1\n2 I -1\n5 I -1\n"
        audit = json.loads((root / "qp" / "audit.json").read_bytes())
        if (
            result.output_path.read_bytes() != expected
            or result.qp_sha256 != hashlib.sha256(expected).hexdigest()
            or audit["csv_sha256"] != hashlib.sha256(csv_bytes).hexdigest()
            or audit["detector_performed"] is not False
            or audit["encoder_input_verified"] is not False
        ):
            raise RuntimeError("loaded QP output or provenance disagrees with independent input")
        print(json.dumps({"frame_count": result.frame_count, "cut_count": result.cut_count}, sort_keys=True))


if __name__ == "__main__":
    main()
