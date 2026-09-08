"""Offline exact-PTS splitting and independent output inspection; install [video]."""

from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory

import av
from PIL import Image

from frame_quorum import NativeClip, split_native_video


def main() -> None:
    times = (0, 1001, 2002, 4004, 5005, 8008)
    tick = Fraction(1, 30000)
    with TemporaryDirectory(prefix="frame-quorum-split-demo-") as temporary:
        source = Path(temporary) / "source.nut"
        target = Path(temporary) / "clips"
        with av.open(str(source), "w", format="nut") as container:
            stream = container.add_stream("ffv1")
            stream.width, stream.height, stream.pix_fmt = 32, 24, "bgr0"
            stream.time_base = stream.codec_context.time_base = tick
            for index, pts in enumerate(times):
                with Image.new("RGB", (32, 24), (index * 40, 10, 255 - index * 40)) as image:
                    frame = av.VideoFrame.from_image(image)
                frame.pts, frame.time_base = pts, tick
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        result = split_native_video(source, target, (NativeClip(0, 4004 * tick), NativeClip(4004 * tick, 1)))
        for ordinal, indices in enumerate(((0, 1, 2), (3, 4, 5))):
            with av.open(str(target / f"clip-{ordinal:06d}.nut")) as independent:
                frames = list(independent.decode(video=0))
                assert [frame.pts * frame.time_base for frame in frames] == [
                    (times[index] - times[indices[0]]) * tick for index in indices
                ]
                for frame, index in zip(frames, indices, strict=True):
                    with frame.to_image() as image:
                        assert image.getpixel((0, 0)) == (index * 40, 10, 255 - index * 40)
        print(
            f"verified {result.clip_count} clips / {result.frame_count} frames; "
            f"manifest {result.manifest_sha256}"
        )


if __name__ == "__main__":
    main()
