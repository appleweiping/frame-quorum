"""Generate local VFR/stereo media, split and independently verify every output value.

Run: python examples/native_av_splitting.py
Requires the existing optional video extra; no external media or executable.
"""

from __future__ import annotations

import struct
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory

import av
from PIL import Image

from frame_quorum import NativeClip, split_native_av

PTS = (150000, 151001, 152002, 153503, 154004, 155005)
TB = Fraction(1, 30000)
RATE = 48000


def pixels(index: int) -> bytes:
    return bytes((index * 37 + offset * 11) % 256 for offset in range(8 * 6 * 3))


def samples(start: int, end: int) -> bytes:
    values = [value for index in range(start, end) for value in (index - 4500, 4500 - index)]
    return struct.pack(f"<{len(values)}h", *values)


def main() -> None:
    with TemporaryDirectory(prefix="frame-quorum-av-demo-") as folder:
        source, target = Path(folder) / "generated.nut", Path(folder) / "verified-clips"
        with av.open(str(source), "w", format="nut") as container:
            video = container.add_stream("ffv1", rate=30)
            video.width, video.height, video.pix_fmt = 8, 6, "bgr0"
            video.time_base = video.codec_context.time_base = TB
            audio = container.add_stream("pcm_s16le", rate=RATE)
            audio.layout = "stereo"
            audio.time_base = audio.codec_context.time_base = Fraction(1, RATE)
            for index, pts in enumerate(PTS):
                with Image.frombytes("RGB", (8, 6), pixels(index)) as image:
                    frame = av.VideoFrame.from_image(image)
                frame.pts, frame.time_base = pts, TB
                for packet in video.encode(frame):
                    container.mux(packet)
            for offset in range(0, 9000, 997):
                count = min(997, 9000 - offset)
                sound = av.AudioFrame(format="s16", layout="stereo", samples=count)
                sound.sample_rate, sound.pts, sound.time_base = RATE, 5 * RATE + offset, Fraction(1, RATE)
                sound.planes[0].update(samples(offset, offset + count))
                for packet in audio.encode(sound):
                    container.mux(packet)
            for stream in (video, audio):
                for packet in stream.encode():
                    container.mux(packet)
        intervals = (NativeClip(PTS[1] * TB, PTS[2] * TB), NativeClip(PTS[2] * TB, PTS[5] * TB))
        result = split_native_av(source, target, intervals)
        expected = (
            ((1,), 1602, 3204, 5 + Fraction(1601, RATE)),
            ((2, 3, 4), 3204, 8008, 5 + Fraction(3203, RATE)),
        )
        for ordinal, (indices, start, end, epoch) in enumerate(expected):
            with av.open(str(target / f"clip-{ordinal:06d}.nut"), "r", format="nut") as reopened:
                frames = list(reopened.decode())
            images = [frame for frame in frames if isinstance(frame, av.VideoFrame)]
            audio_frames = [frame for frame in frames if isinstance(frame, av.AudioFrame)]
            assert len(images) == len(indices)
            for image_frame, index in zip(images, indices, strict=True):
                assert image_frame.pts * image_frame.time_base == PTS[index] * TB - epoch
                with image_frame.to_image() as image:
                    assert image.tobytes() == pixels(index)
            cursor, pcm = start, bytearray()
            for audio_frame in audio_frames:
                assert audio_frame.sample_rate == RATE
                assert audio_frame.pts * audio_frame.time_base == 5 + Fraction(cursor, RATE) - epoch
                pcm.extend(bytes(audio_frame.planes[0])[: audio_frame.samples * 4])
                cursor += audio_frame.samples
            assert cursor == end and pcm == samples(start, end)
        assert result.frame_count == 4 and result.audio_sample_count == 6406
        print("verified 2 clips / 4 video frames / 6406 stereo sample positions")


if __name__ == "__main__":
    main()
