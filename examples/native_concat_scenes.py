"""Offline, real mixed-clock video scene analysis. Requires frame-quorum[video]."""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory

import av
from PIL import Image

from frame_quorum import (
    DetectionConfig,
    NativeConcatClip,
    NativeConcatSceneConfig,
    detect_native_concat_scenes,
)


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def write_video(path: Path, level: int, clock: Fraction, origin: int, step: int, rate: int) -> None:
    with av.open(str(path), "w", format="matroska" if path.suffix == ".mkv" else "nut") as container:
        stream = container.add_stream("ffv1", rate=rate)
        stream.width, stream.height, stream.pix_fmt = 16, 12, "bgr0"
        stream.time_base = stream.codec_context.time_base = clock
        for index in range(4):
            with Image.new("RGB", (16, 12), (level, level, level)) as image:
                frame = av.VideoFrame.from_image(image)
            frame.pts, frame.time_base = origin + step * index, clock
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def main() -> None:
    with TemporaryDirectory(prefix="frame-quorum-concat-scenes-") as directory:
        first, second = Path(directory) / "first.mkv", Path(directory) / "second.nut"
        write_video(first, 0, Fraction(1, 1000), 5000, 50, 20)
        write_video(second, 255, Fraction(1, 48000), 48000, 2400, 30)
        clips = (
            NativeConcatClip(first, start=5, end=Fraction(26, 5)),
            NativeConcatClip(second, start=1, end=Fraction(6, 5)),
        )
        config = NativeConcatSceneConfig(
            detectors=tuple(
                DetectionConfig(detector=name, window_radius=1)
                for name in ("content", "color", "luminance", "adaptive", "threshold")
            ),
            minimum_votes=3,
        )
        result = detect_native_concat_scenes(clips, config)
        expect(result.cut_positions == (4,), "the visible seam transition must be detected")
        expect(result.cut_times == (Fraction(1, 5),), "the cut must use the declared composite clock")
        expect(result.diagnostics.closed, "decoder cleanup must be positively acknowledged")
        expect(
            [row.sample.native.pts for row in result.statistics]
            == [5000, 5050, 5100, 5150, 48000, 50400, 52800, 55200],
            "native PTS must remain unchanged",
        )
        expect(
            [row.sample.native.time_base for row in result.statistics]
            == [Fraction(1, 1000)] * 4 + [Fraction(1, 48000)] * 4,
            "native clocks must remain distinct",
        )
        expect(
            result.scenes[-1].end_time == Fraction(2, 5), "the tail must use the explicit declared endpoint"
        )
        document = result.to_dict()
        first.unlink()
        second.unlink()
        print(
            json.dumps(
                {
                    "sample_count": len(result.statistics),
                    "cut_positions": list(result.cut_positions),
                    "cut_times": [str(t) for t in result.cut_times],
                    "native_clocks": [str(m.time_base) for m in result.metadata],
                    "declared_end": str(result.scenes[-1].end_time),
                    "source_verified": document["source_verified"],
                    "coverage": document["coverage"],
                    "closed": result.diagnostics.closed,
                    "complete_native_identity_equal": True,
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
