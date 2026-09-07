"""Generate a tiny lossless VFR video locally and report its exact-PTS fade cut.

Run: python examples/native_scenes.py (requires frame-quorum[video]).
No media download, paid model, image directory or external FFmpeg process.
"""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory

import av
from PIL import Image

from frame_quorum import DetectionConfig, NativeSceneConfig, detect_native_scenes


def main() -> None:
    with TemporaryDirectory(prefix="frame-quorum-native-scenes-") as temporary:
        path = Path(temporary) / "generated-fade.mkv"
        with av.open(str(path), "w", format="matroska") as container:
            stream = container.add_stream("ffv1", rate=25)
            stream.width, stream.height, stream.pix_fmt = 32, 24, "bgr0"
            stream.time_base = stream.codec_context.time_base = Fraction(1, 1000)
            pts = (5000, 5040, 5110, 5180, 5270, 5310, 5470, 5680)
            colors = (255, 255, 0, 0, 0, 0, 255, 255)
            for timestamp, value in zip(pts, colors, strict=True):
                with Image.new("RGB", (32, 24), (value, value, value)) as image:
                    frame = av.VideoFrame.from_image(image)
                frame.pts, frame.time_base = timestamp, Fraction(1, 1000)
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        result = detect_native_scenes(
            path,
            NativeSceneConfig(
                detectors=(DetectionConfig(detector="threshold", min_dark_frames=2, fade_bias=0),),
            ),
        )
        assert result.cut_times == (Fraction(527, 100),)
        assert result.scenes[-1].end_time is None  # PTS does not tell the last sample's duration.
        print(json.dumps(result.to_dict(), indent=2, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
