"""Generate a lossless VFR source and export exact cuts, entirely offline."""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

from frame_quorum import (
    DetectionConfig,
    NativeSceneConfig,
    OTIOMedia,
    detect_native_scenes,
    otio_cuts_from_native,
    write_otio_bundle,
)


def main() -> None:
    import av

    with TemporaryDirectory(prefix="frame-quorum-otio-example-") as directory:
        source = Path(directory) / "source.nut"
        pts = (5000, 5040, 5110, 5180, 5270, 5310, 5470, 5680)
        with av.open(str(source), "w", format="nut") as container:
            stream = container.add_stream("ffv1", rate=25)
            stream.width, stream.height, stream.pix_fmt = 8, 6, "bgr0"
            stream.time_base = stream.codec_context.time_base = Fraction(1, 1000)
            for index, timestamp in enumerate(pts):
                with Image.new("RGB", (8, 6), (0, 0, 0) if index < 4 else (255, 255, 255)) as image:
                    frame = av.VideoFrame.from_image(image)
                frame.pts, frame.time_base = timestamp, Fraction(1, 1000)
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        with av.open(str(source)) as direct:
            assert [frame.pts * frame.time_base for frame in direct.decode(video=0)] == [
                Fraction(value, 1000) for value in pts
            ]
        analysis = detect_native_scenes(
            source, NativeSceneConfig(detectors=(DetectionConfig(detector="luminance"),))
        )
        cuts = otio_cuts_from_native(analysis, final_end=Fraction(57, 10))
        result = write_otio_bundle(cuts, OTIOMedia(source, Fraction(5)), Path(directory) / "editor")
        document = json.loads((result.output_dir / "scenes.otio").read_text())
        ranges = [clip["source_range"] for clip in document["tracks"]["children"][0]["children"]]
        durations = [Fraction(item["duration"]["value"], item["duration"]["rate"]) for item in ranges]
        assert durations == [Fraction(27, 100), Fraction(43, 100)]
        print(
            json.dumps(
                {
                    "cuts": [[str(c.start), str(c.end)] for c in cuts],
                    "duration": str(sum(durations)),
                    "source_verified": False,
                }
            )
        )


if __name__ == "__main__":
    main()
