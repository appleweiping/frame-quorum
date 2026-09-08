"""Generate a layout change: global histograms ignore it; spatial histograms detect it."""

from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory

import av
from PIL import Image

from frame_quorum import (
    HistogramDetectionConfig,
    analyze_native_histograms,
    capture_native_histograms,
    read_native_histograms,
    write_native_histogram_replay,
    write_native_histograms,
)


def main() -> None:
    with TemporaryDirectory(prefix="frame-quorum-pixel-histogram-demo-") as temporary:
        root = Path(temporary)
        source = root / "source.mkv"
        with av.open(str(source), "w", format="matroska") as container:
            stream = container.add_stream("ffv1", rate=25)
            stream.width, stream.height, stream.pix_fmt = 16, 12, "bgr0"
            stream.time_base = stream.codec_context.time_base = Fraction(1, 1000)
            for index, pts in enumerate((5000, 5040, 5110, 5180, 5270, 5310)):
                changed = index in (2, 3)
                with Image.new("RGB", (16, 12)) as image:
                    image.paste((254, 254, 254), (0, 0 if changed else 6, 16, 6 if changed else 12))
                    frame = av.VideoFrame.from_image(image)
                frame.pts, frame.time_base = pts, Fraction(1, 1000)
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        data = capture_native_histograms(source)
        cache = write_native_histograms(data, root / "cache")
        source.unlink()
        for mode, expected in (("global", ()), ("spatial", (Fraction(511, 100), Fraction(527, 100)))):
            result = analyze_native_histograms(
                read_native_histograms(cache), HistogramDetectionConfig(mode=mode)
            )
            assert result.cut_times == expected
            assert result.source_verified is False
            write_native_histogram_replay(result, root / mode)
            print(f"{mode}: cuts={list(map(str, expected))}; source_verified=False")


if __name__ == "__main__":
    main()
