"""Generate a lossless VFR checkerboard/stripe change; replay after deleting video."""

from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory

import av
from PIL import Image

from frame_quorum import (
    PixelChangeDetectionConfig,
    PixelChangeWeights,
    analyze_native_pixel_changes,
    capture_native_pixel_changes,
    read_native_pixel_changes,
    write_native_pixel_change_replay,
    write_native_pixel_changes,
)


def main() -> None:
    with TemporaryDirectory(prefix="frame-quorum-pixel-change-demo-") as temporary:
        root = Path(temporary)
        source = root / "source.mkv"
        with av.open(str(source), "w", format="matroska") as container:
            stream = container.add_stream("ffv1", rate=25)
            stream.width, stream.height, stream.pix_fmt = 16, 12, "bgr0"
            stream.time_base = stream.codec_context.time_base = Fraction(1, 1000)
            for index, pts in enumerate((5000, 5040, 5110, 5180, 5270, 5310)):
                with Image.new("RGB", (16, 12)) as image:
                    for y in range(12):
                        for x in range(16):
                            value = 255 * (x % 2 if index in (2, 3) else (x + y) % 2)
                            image.putpixel((x, y), (value, value, value))
                    frame = av.VideoFrame.from_image(image)
                frame.pts, frame.time_base = pts, Fraction(1, 1000)
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        data = capture_native_pixel_changes(source)
        cache = write_native_pixel_changes(data, root / "cache")
        source.unlink()
        restored = read_native_pixel_changes(cache)
        for name, weights in (
            ("value", PixelChangeWeights(0, 0, 1, 0)),
            ("edge", PixelChangeWeights(0, 0, 0, 1)),
        ):
            result = analyze_native_pixel_changes(restored, PixelChangeDetectionConfig(weights=weights))
            assert result.cut_times == (Fraction(511, 100), Fraction(527, 100))
            assert result.source_verified is False
            write_native_pixel_change_replay(result, root / name)
            print(f"{name}: cuts={list(map(str, result.cut_times))}; source_verified=False")


if __name__ == "__main__":
    main()
