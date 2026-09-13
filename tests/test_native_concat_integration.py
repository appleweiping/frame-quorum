from __future__ import annotations

import json
import runpy
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import pytest
from PIL import Image

from frame_quorum import NativeConcatClip, NativeConcatConfig, NativeConcatLimits, NativeConcatStream

av = pytest.importorskip("av", reason="install frame-quorum[video] for genuine concat codec tests")


@pytest.mark.parametrize("optimized", [False, True])
def test_isolated_offline_example_preserves_checks_and_seek_under_optimization(tmp_path, optimized):
    script = Path(__file__).resolve().parents[1] / "examples" / "native_concat.py"
    process = subprocess.run(
        [sys.executable, "-I", "-B", *(["-O"] if optimized else []), str(script)],
        cwd=tmp_path,
        capture_output=True,
        check=False,
        timeout=30,
    )
    assert process.returncode == 0, process.stderr
    assert not process.stderr
    payload = json.loads(process.stdout)
    assert payload["complete_rgb_and_native_timing_equal"] is True
    assert payload["global_positions"] == ["0", "3/20", "13/40"]
    assert payload["after_seek_and_close"]["generation"] == 1
    assert payload["after_seek_and_close"]["returned_frames"] == 4
    assert payload["after_seek_and_close"]["closed"] is True
    assert list(tmp_path.iterdir()) == []


@pytest.fixture
def media(tmp_path):
    paths = (tmp_path / "first.mkv", tmp_path / "second.nut")
    for source, (path, pts, clock, rate) in enumerate(
        zip(
            paths,
            ((5000, 5040, 5150, 5300), (48000, 49200, 51600, 57600)),
            (Fraction(1, 1000), Fraction(1, 48000)),
            (25, 30),
            strict=True,
        )
    ):
        with av.open(str(path), "w", format="matroska" if source == 0 else "nut") as container:
            video = container.add_stream("ffv1", rate=rate)
            video.width, video.height, video.pix_fmt = 16, 12, "bgr0"
            video.time_base = video.codec_context.time_base = clock
            for index, stamp in enumerate(pts):
                with Image.new("RGB", (16, 12)) as image:
                    image.putdata(
                        [
                            (
                                (x * 17 + index * 29) % 256,
                                (y * 19 + source * 67) % 256,
                                (x * y + index * 23) % 256,
                            )
                            for y in range(12)
                            for x in range(16)
                        ]
                    )
                    frame = av.VideoFrame.from_image(image)
                frame.pts, frame.time_base = stamp, clock
                for packet in video.encode(frame):
                    container.mux(packet)
            for packet in video.encode():
                container.mux(packet)
    return paths


def clips_for(paths, *, first_end=Fraction(53, 10), second_end=Fraction(6, 5)):
    return (
        NativeConcatClip(paths[0], start=5, end=first_end),
        NativeConcatClip(paths[1], start=1, end=second_end),
    )


def independent(clips):
    """No concat/native helpers: independent decode of every full RGB byte and exact original clock."""
    expected, offset = [], Fraction(0)
    for index, clip in enumerate(clips):
        with av.open(str(clip.path)) as decoder:
            for frame in decoder.decode(video=clip.video_stream):
                native_time = frame.pts * frame.time_base
                if clip.start <= native_time < clip.end:
                    with frame.to_image() as source, source.convert("RGB") as rgb:
                        expected.append(
                            (
                                index,
                                frame.pts,
                                frame.time_base,
                                rgb.tobytes(),
                                offset + native_time - clip.start,
                            )
                        )
        offset += clip.end - clip.start
    return expected


def actual(frame):
    return frame.clip_index, frame.native.pts, frame.native.time_base, frame.rgb, frame.presentation_time


@pytest.mark.parametrize("step", [1, 2, 3, 4, 7])
def test_real_mixed_timebases_complete_rgb_provenance_global_stride_and_lifetime_accounting(media, step):
    clips = clips_for(media)
    expected = independent(clips)
    with NativeConcatStream(clips, NativeConcatConfig(frame_step=step)) as stream:
        records = list(stream)
        assert [actual(frame) for frame in records] == expected[::step]
        assert [m.time_base for m in stream.metadata] == [Fraction(1, 1000), Fraction(1, 48000)]
        assert [m.average_rate for m in stream.metadata] == [25, 30]
        assert [m.duration_pts for m in stream.metadata] == [None, None]
        diagnostic = stream.diagnostics
        assert diagnostic.status == "intervals_exhausted" and diagnostic.closed
        assert diagnostic.activations == 4 and diagnostic.source_activations == (2, 2)
        assert diagnostic.opened_source_bytes == 2 * sum(path.stat().st_size for path in media)
        assert diagnostic.decoded_frames == 8 and diagnostic.source_decoded_frames == (4, 4)
        assert diagnostic.owned_rgb_frames == 6 and diagnostic.source_rgb_frames == (3, 3)
        assert diagnostic.decoded_pixels_observed == 8 * 16 * 12
        assert diagnostic.returned_frames == len(expected[::step])
        assert [frame.sample_index for frame in records] == list(range(len(records)))
        assert {frame.activation for frame in records} == {1}  # Probe is activation zero.
        assert all(frame.generation == 0 and frame.native.generation == 0 for frame in records)
        assert [row[-1] for row in expected] == [
            0,
            Fraction(1, 25),
            Fraction(3, 20),
            Fraction(3, 10),
            Fraction(13, 40),
            Fraction(3, 8),
        ]
        assert json.loads(json.dumps(records[0].to_dict()))["timeline_digest"] == stream.timeline.digest
    for frame in records:
        with frame.image() as image:
            assert image.tobytes() == frame.rgb
    for path in media:
        moved = path.with_suffix(".closed")
        path.rename(moved)
        moved.unlink()


@pytest.mark.parametrize(
    "start,end",
    [
        (Fraction(1, 25), Fraction(13, 40)),
        (Fraction(3, 10), None),  # Internal seam selects the right source.
        (Fraction(31, 100), Fraction(2, 5)),
        (Fraction(1, 2), None),  # Declared EOF acquires no source.
    ],
)
def test_real_seek_reopens_exact_native_target_and_preserves_lifetime(media, start, end):
    clips = clips_for(media)
    expected = independent(clips)
    with NativeConcatStream(clips) as stream:
        assert actual(next(stream)) == expected[0]
        before = stream.diagnostics
        stream.seek(start, end=end)
        after = stream.diagnostics
        assert after.seeks == after.generation == 1
        assert after.activations == before.activations + (start != stream.timeline.duration)
        result = list(stream)
        assert [actual(frame) for frame in result] == [
            row for row in expected if start <= row[-1] < (Fraction(1, 2) if end is None else end)
        ]
        assert all(frame.generation == 1 for frame in result)
        assert [frame.sample_index for frame in result] == list(range(1, 1 + len(result)))
        assert stream.diagnostics.decoded_frames >= before.decoded_frames
        stream.reset()
        repeated = list(stream)
        assert [actual(frame) for frame in repeated] == expected
        assert all(frame.generation == 2 for frame in repeated)
        assert repeated[0].sample_index == 1 + len(result)


def test_real_declared_gap_and_empty_clip_do_not_rewrite_seam_or_claim_measured_duration(media):
    clips = (
        NativeConcatClip(media[0], start=5, end=6),  # Last sample is 5.3, but seam stays at declared 1.
        NativeConcatClip(media[1], start=Fraction(6, 5), end=Fraction(3, 2)),
        NativeConcatClip(media[0], start=7, end=8),  # No eligible samples, positive declared length.
        NativeConcatClip(media[1], start=1, end=Fraction(6, 5)),
    )
    with NativeConcatStream(clips) as stream:
        frames = list(stream)
        assert [actual(frame) for frame in frames] == independent(clips)
        assert stream.timeline.offsets == (0, 1, Fraction(13, 10), Fraction(23, 10), Fraction(5, 2))
        assert stream.timeline.to_dict()["policy"] == "caller_declared"
        assert {frame.clip_index for frame in frames} == {0, 1, 3}
        assert next(frame.presentation_time for frame in frames if frame.clip_index == 3) == Fraction(23, 10)
        spans = stream.map_span(Fraction(1, 2), Fraction(12, 5))
        assert [span.clip_index for span in spans] == [0, 1, 2, 3]
        assert all(span.to_dict()["coverage"] == "declared_only" for span in spans)
        stream.seek(Fraction(7, 10))
        assert next(stream).presentation_time == 1


@pytest.mark.parametrize(
    "limit,value,status",
    [
        ("max_decoded_frames", 5, "decode_limit"),
        ("max_rgb_frames", 4, "rgb_limit"),
        ("max_frames", 2, "frame_limit"),
    ],
)
def test_real_native_budget_is_not_promoted_to_source_exhaustion(media, limit, value, status):
    with NativeConcatStream(
        clips_for(media), NativeConcatConfig(limits=NativeConcatLimits(**{limit: value}))
    ) as stream:
        frames = list(stream)
        assert stream.diagnostics.status == status and stream.diagnostics.closed
        assert stream.diagnostics.returned_frames == len(frames)
        assert stream.diagnostics.limit_scope == "total"
        assert stream.diagnostics.status != "intervals_exhausted"


def test_offline_example_checks_complete_rgb_and_native_clock_without_repository_writes(
    tmp_path, monkeypatch, capsys
):
    script = Path(__file__).resolve().parents[1] / "examples" / "native_concat.py"
    monkeypatch.chdir(tmp_path)
    runpy.run_path(str(script), run_name="__main__")
    payload = json.loads(capsys.readouterr().out)
    assert payload["complete_rgb_and_native_timing_equal"] is True
    assert payload["source_time_bases"] == ["1/1000", "1/48000"]
    assert payload["global_positions"] == ["0", "3/20", "13/40"]
    assert payload["first_pass"]["owned_rgb_frames"] == 6
    assert payload["first_pass"]["returned_frames"] == 3
    assert payload["after_seek_and_close"]["closed"] is True
    assert list(tmp_path.iterdir()) == []
