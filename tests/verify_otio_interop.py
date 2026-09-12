"""Explicit independent-reader acceptance, not an optional-skipped pytest gate.

Run in an environment containing OpenTimelineIO, or supply an existing trusted
reader package root. This script never installs dependencies or fetches media.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

from frame_quorum import (
    DetectionConfig,
    NativeSceneConfig,
    OTIOCut,
    OTIOExportConfig,
    OTIOMedia,
    detect_native_scenes,
    otio_cuts_from_native,
    render_otio,
)


def exact(time):
    # Do not use exporter arithmetic or OTIO float seconds as an oracle.
    assert time.value.is_integer() and time.rate.is_integer()
    return Fraction(int(time.value), int(time.rate))


def verify(otio, cuts, media, *, audio=False, known_negative_decimal_roundtrip=False):
    wire = render_otio(cuts, media, OTIOExportConfig(include_audio=audio))
    timeline = otio.core.deserialize_json_from_string(wire)
    assert isinstance(timeline, otio.schema.Timeline)
    assert [track.kind for track in timeline.tracks] == (["Video", "Audio"] if audio else ["Video"])
    expected_duration = sum((cut.end - cut.start for cut in cuts), Fraction(0))
    for track in timeline.tracks:
        assert len(track) == len(cuts)
        assert exact(track.duration()) == expected_duration
        for clip, cut in zip(track, cuts, strict=True):
            assert exact(clip.source_range.start_time) == cut.start - media.origin
            assert exact(clip.source_range.duration) == cut.end - cut.start
            assert exact(clip.source_range.end_time_exclusive()) == cut.end - media.origin
            reference = clip.media_reference
            assert reference.target_url == media.path.as_uri()
            if media.available_start is None:
                assert reference.available_range is None
            else:
                assert exact(reference.available_range.start_time) == media.available_start - media.origin
                assert (
                    exact(reference.available_range.end_time_exclusive())
                    == media.available_end - media.origin
                )
    # Independent core writer/reader roundtrip must retain all range fields.
    rewritten = otio.core.serialize_json_to_string(timeline)
    roundtrip = otio.core.deserialize_json_from_string(rewritten)
    limitation = None
    for original, restored in zip(timeline.tracks, roundtrip.tracks, strict=True):
        assert original.kind == restored.kind
        for left, right in zip(original, restored, strict=True):
            if left.source_range != right.source_range:
                # Retain the exact independently observed 0.18.1 defect, rather
                # than dropping this vector or claiming complete roundtrip PASS.
                assert known_negative_decimal_roundtrip and otio.__version__ == "0.18.1"
                assert left.source_range.start_time.value == -9007199254740991
                assert right.source_range.start_time.value == -9007199254740990
                assert left.source_range.duration == right.source_range.duration
                assert left.source_range.start_time.rate == right.source_range.start_time.rate == 1
                assert '"value":-9007199254740991' in wire
                assert '"value": -9007199254740991.0' in rewritten
                limitation = {
                    "kind": "otio-0.18.1-negative-decimal-reparse",
                    "initial_integer_value": -9007199254740991,
                    "initial_read_value": int(left.source_range.start_time.value),
                    "writer_decimal": "-9007199254740991.0",
                    "second_read_value": int(right.source_range.start_time.value),
                    "original_wire": wire,
                    "writer_wire": rewritten,
                    "original_wire_sha256": hashlib.sha256(wire.encode()).hexdigest(),
                    "writer_wire_sha256": hashlib.sha256(rewritten.encode()).hexdigest(),
                }
            assert left.media_reference.available_range == right.media_reference.available_range
    return timeline, limitation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reader-root", type=Path)
    args = parser.parse_args()
    if args.reader_root is not None:
        sys.path.insert(0, str(args.reader_root.resolve()))
    otio = importlib.import_module("opentimelineio")
    import av

    with TemporaryDirectory(prefix="frame-quorum-otio-interop-") as directory:
        root = Path(directory)
        media_path = root / "source #é.nut"
        limit = 2**53 - 1
        cases = (
            (((4, 6), (10, 11)), Fraction(0), None, False),
            (((4, 6), (10, 11)), Fraction(0), None, True),
            (((-5, -4), (-3, -1)), Fraction(-5), None, True),
            (((-5, -4),), Fraction(0), None, False),
            (((-5, -4), (-3, -1)), Fraction(-5), (Fraction(-6), Fraction(0)), False),
            (((limit - 1, limit),), Fraction(0), None, False),
            (((-limit, -limit + 1),), Fraction(0), None, False),
            (
                (
                    (Fraction(0), Fraction(1001, 30000)),
                    (Fraction(1, 10), Fraction(1, 10) + Fraction(1, 48000)),
                ),
                Fraction(0),
                None,
                True,
            ),
        )
        limitations = []
        for index, (intervals, origin, available, audio) in enumerate(cases):
            cuts = tuple(OTIOCut(Fraction(start), Fraction(end)) for start, end in intervals)
            media = OTIOMedia(media_path, origin, *(available or (None, None)))
            _, limitation = verify(
                otio, cuts, media, audio=audio, known_negative_decimal_roundtrip=index == 6
            )
            if limitation is not None:
                limitations.append(limitation)

        # Actual lossless VFR media, independently decoded RGB/PTS before analysis.
        pts = (5000, 5040, 5110, 5180, 5270, 5310, 5470, 5680)
        with av.open(str(media_path), "w", format="nut") as container:
            stream = container.add_stream("ffv1", rate=25)
            stream.width, stream.height, stream.pix_fmt = 8, 6, "bgr0"
            stream.time_base = stream.codec_context.time_base = Fraction(1, 1000)
            for index, timestamp in enumerate(pts):
                with Image.new("RGB", (8, 6), (0, 0, 0) if index < 4 else (255, 255, 255)) as image:
                    frame = av.VideoFrame.from_image(image)
                frame.pts, frame.time_base = timestamp, Fraction(1, 1000)
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        with av.open(str(media_path)) as direct:
            observed = []
            for frame in direct.decode(video=0):
                with frame.to_image() as image:
                    observed.append((frame.pts * frame.time_base, image.tobytes()))
        assert observed == [
            (Fraction(timestamp, 1000), bytes([0 if i < 4 else 255]) * 144) for i, timestamp in enumerate(pts)
        ]
        analysis = detect_native_scenes(
            media_path, NativeSceneConfig(detectors=(DetectionConfig(detector="luminance"),))
        )
        cuts = otio_cuts_from_native(analysis, final_end=Fraction(57, 10))
        assert [(cut.start, cut.end) for cut in cuts] == [
            (Fraction(5), Fraction(527, 100)),
            (Fraction(527, 100), Fraction(57, 10)),
        ]
        timeline, limitation = verify(otio, cuts, OTIOMedia(media_path, Fraction(5)))
        assert limitation is None
        assert exact(timeline.duration()) == Fraction(7, 10)
        print(
            json.dumps(
                {
                    "reader": otio.__version__,
                    "reader_path": otio.__file__,
                    "python": sys.executable,
                    "synthetic_vectors": len(cases),
                    "native_frames": len(observed),
                    "native_duration": "7/10",
                    "initial_read_vectors": len(cases) + 1,
                    "roundtrip_matched_vectors": len(cases) + 1 - len(limitations),
                    "known_third_party_roundtrip_limitations": limitations,
                }
            )
        )


if __name__ == "__main__":
    main()
