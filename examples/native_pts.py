"""Generate four local lossless VFR frames and measure their exact native PTS."""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

from frame_quorum import NativeVideoConfig, NativeVideoStream


def main() -> None:
    try:
        import av
    except ImportError as exc:
        raise SystemExit("Install frame-quorum[video] to run this offline native-codec example.") from exc
    pts = (5000, 5040, 5150, 5300)
    colors = ((255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 255))
    with TemporaryDirectory(prefix="frame-quorum-native-") as directory:
        path = Path(directory) / "vfr.mkv"
        with av.open(str(path), "w", format="matroska") as container:
            video = container.add_stream("ffv1", rate=25)
            video.width, video.height, video.pix_fmt = 16, 12, "bgr0"
            video.time_base = video.codec_context.time_base = Fraction(1, 1000)
            for timestamp, color in zip(pts, colors, strict=True):
                with Image.new("RGB", (16, 12), color) as image:
                    frame = av.VideoFrame.from_image(image)
                frame.pts, frame.time_base = timestamp, Fraction(1, 1000)
                for packet in video.encode(frame):
                    container.mux(packet)
            for packet in video.encode():
                container.mux(packet)
        with NativeVideoStream(path, NativeVideoConfig(max_frames=10)) as stream:
            measured = [{**frame.to_dict(), "metrics": frame.measure().serializable()} for frame in stream]
        assert [frame["pts"] for frame in measured] == list(pts)
        print(
            json.dumps(
                {
                    "pyav_version": av.__version__,
                    "frames": measured,
                    "diagnostics": stream.diagnostics.to_dict(),
                    "rate_is_not_timestamp_source": True,
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
