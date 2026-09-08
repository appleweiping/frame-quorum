"""Generate local lossless VFR video, then consume decisions without a full result table."""

from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

from frame_quorum import NativePixelChangeStream, NativePixelChangeUpdate, PixelChangeDetectionConfig


def main() -> None:
    import av

    with TemporaryDirectory(prefix="frame-quorum-online-demo-") as temporary:
        path = Path(temporary) / "input.mkv"
        pts_values = (5000, 5040, 5110, 5180, 5270, 5310, 5410, 5530, 5590)
        with av.open(str(path), "w", format="matroska") as container:
            encoder = container.add_stream("ffv1", rate=25)
            encoder.width, encoder.height, encoder.pix_fmt = 16, 12, "bgr0"
            encoder.time_base = encoder.codec_context.time_base = Fraction(1, 1000)
            for index, pts in enumerate(pts_values):
                with Image.new("RGB", (16, 12), "white" if 3 <= index < 6 else "black") as image:
                    frame = av.VideoFrame.from_image(image)
                frame.pts, frame.time_base = pts, Fraction(1, 1000)
                for packet in encoder.encode(frame):
                    container.mux(packet)
            for packet in encoder.encode():
                container.mux(packet)
        config = PixelChangeDetectionConfig(
            detector="adaptive", value_only=True, window_radius=1, min_scene_samples=2
        )
        cuts = 0
        with NativePixelChangeStream(path, detection=config) as stream:
            for event in stream:
                if isinstance(event, NativePixelChangeUpdate):
                    if event.statistic.accepted:
                        cuts += 1
                        print(
                            f"cut={event.sample.sample.presentation_time}, "
                            f"confirmed={event.observed_through_time}"
                        )
                else:
                    assert event.diagnostics.observed_samples == 9
                    assert event.final_scene.end_time is None
                    print(
                        f"completed={event.diagnostics.video.status.value}, "
                        f"source_verified={event.source_verified}"
                    )
        assert cuts == 2


if __name__ == "__main__":
    main()
