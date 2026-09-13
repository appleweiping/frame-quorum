"""Generate two local lossless sources and audit a real exact-declared composite."""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

from frame_quorum import NativeConcatClip, NativeConcatConfig, NativeConcatStream


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    try:
        import av
    except ImportError as error:
        raise SystemExit("Install frame-quorum[video] for this offline concat example.") from error

    with TemporaryDirectory(prefix="frame-quorum-concat-") as directory:
        paths = (Path(directory) / "first.mkv", Path(directory) / "second.nut")
        clocks = (Fraction(1, 1000), Fraction(1, 48000))
        stamps = ((5000, 5040, 5150, 5300), (48000, 49200, 51600, 57600))
        for source, path in enumerate(paths):
            with av.open(str(path), "w", format="matroska" if source == 0 else "nut") as output:
                video = output.add_stream("ffv1", rate=25 if source == 0 else 30)
                video.width, video.height, video.pix_fmt = 16, 12, "bgr0"
                video.time_base = video.codec_context.time_base = clocks[source]
                for index, timestamp in enumerate(stamps[source]):
                    with Image.new("RGB", (16, 12)) as image:
                        image.putdata(
                            [
                                (
                                    (x * 13 + index * 41) % 256,
                                    (y * 19 + source * 73) % 256,
                                    (x + y + index * 23) % 256,
                                )
                                for y in range(12)
                                for x in range(16)
                            ]
                        )
                        frame = av.VideoFrame.from_image(image)
                    frame.pts, frame.time_base = timestamp, clocks[source]
                    for packet in video.encode(frame):
                        output.mux(packet)
                for packet in video.encode():
                    output.mux(packet)
        clips = (
            NativeConcatClip(paths[0], start=5, end=Fraction(53, 10)),
            NativeConcatClip(paths[1], start=1, end=Fraction(6, 5)),
        )
        expected = []
        for source, (path, origin, endpoint, offset) in enumerate(
            zip(
                paths,
                (Fraction(5), Fraction(1)),
                (Fraction(53, 10), Fraction(6, 5)),
                (Fraction(0), Fraction(3, 10)),
                strict=True,
            )
        ):
            with av.open(str(path)) as decoder:
                for frame in decoder.decode(video=0):
                    instant = frame.pts * frame.time_base
                    if origin <= instant < endpoint:
                        with frame.to_image() as raw, raw.convert("RGB") as image:
                            expected.append(
                                (
                                    source,
                                    frame.pts,
                                    frame.time_base,
                                    image.tobytes(),
                                    offset + instant - origin,
                                )
                            )
        with NativeConcatStream(clips, NativeConcatConfig(frame_step=2)) as stream:
            records = list(stream)
            observed = [
                (f.clip_index, f.native.pts, f.native.time_base, f.rgb, f.presentation_time) for f in records
            ]
            expect(observed == expected[::2], "composite RGB or exact timing differs from direct decode")
            first_pass = stream.diagnostics.to_dict()
            stream.seek(Fraction(3, 10))
            selected = next(stream)
            expect(selected.presentation_time == Fraction(3, 10), "seek missed the right-hand seam")
            spans = [span.to_dict() for span in stream.map_span(Fraction(1, 10), Fraction(2, 5))]
        print(
            json.dumps(
                {
                    "profile": "caller_declared",
                    "complete_rgb_and_native_timing_equal": True,
                    "source_time_bases": [str(value) for value in clocks],
                    "global_positions": [str(frame.presentation_time) for frame in records],
                    "first_pass": first_pass,
                    "after_seek_and_close": stream.diagnostics.to_dict(),
                    "spans": spans,
                    "declared_endpoints_are_not_measured_duration": True,
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
