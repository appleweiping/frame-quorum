"""Capture an original VFR fixture, remove its source, then inspect two thresholds."""

from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory

import av
from PIL import Image

from frame_quorum import (
    DetectionConfig,
    analyze_native_measurements,
    capture_native_measurements,
    read_native_measurements,
    write_native_measurements,
    write_native_replay,
)


def main() -> None:
    times = (5000, 5040, 5110, 5180, 5270, 5310)
    colors = (0, 0, 128, 128, 255, 255)
    with TemporaryDirectory(prefix="frame-quorum-measurement-demo-") as temporary:
        root = Path(temporary)
        source = root / "source.mkv"
        with av.open(str(source), "w", format="matroska") as container:
            stream = container.add_stream("ffv1", rate=25)
            stream.width, stream.height, stream.pix_fmt = 16, 12, "bgr0"
            stream.time_base = stream.codec_context.time_base = Fraction(1, 1000)
            for pts, color in zip(times, colors, strict=True):
                with Image.new("RGB", (16, 12), (color, color, color)) as image:
                    frame = av.VideoFrame.from_image(image)
                frame.pts, frame.time_base = pts, Fraction(1, 1000)
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        data = capture_native_measurements(source)
        cache = write_native_measurements(data, root / "cache")
        source.unlink()
        for threshold, expected in ((0.4, (Fraction(511, 100), Fraction(527, 100))), (0.6, ())):
            replay = analyze_native_measurements(
                read_native_measurements(cache),
                detectors=(DetectionConfig(detector="luminance", threshold=threshold),),
            )
            assert replay.analysis.cut_times == expected
            assert replay.source_verified is False
            write_native_replay(replay, root / str(threshold))
            print(f"threshold={threshold}: cuts={list(map(str, expected))}; source_verified=False")


if __name__ == "__main__":
    main()
