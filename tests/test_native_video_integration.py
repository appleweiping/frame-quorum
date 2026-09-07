from __future__ import annotations

import json
import runpy
from fractions import Fraction
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from frame_quorum.cli import main
from frame_quorum.errors import ScanError
from frame_quorum.native_video import NativeVideoConfig, NativeVideoStatus, NativeVideoStream, _deny_secondary

av = pytest.importorskip("av", reason="install frame-quorum[video] for genuine native codec tests")
COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 255)]


def make_video(path, pts=(0, 40, 80, 120), *, format="matroska", codec="ffv1", time_base=Fraction(1, 1000)):
    with av.open(str(path), "w", format=format) as container:
        stream = container.add_stream(codec, rate=25)
        stream.width, stream.height, stream.pix_fmt = 16, 12, "bgr0"
        stream.time_base = time_base
        stream.codec_context.time_base = time_base
        for timestamp, color in zip(pts, COLORS, strict=True):
            with Image.new("RGB", (16, 12), color) as image:
                frame = av.VideoFrame.from_image(image)
            frame.pts, frame.time_base = timestamp, time_base
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


@pytest.mark.parametrize("pts", [(0, 40, 80, 120), (5000, 5040, 5150, 5300)])
def test_real_cfr_vfr_and_nonzero_pts_match_hand_authored_colors_and_direct_decode(tmp_path, pts):
    path = tmp_path / "fixture.mkv"
    make_video(path, pts)
    with NativeVideoStream(path) as stream:
        actual = list(stream)
        metadata = stream.metadata
        assert stream.diagnostics.status is NativeVideoStatus.EOF
        assert stream.diagnostics.closed
    assert [record.pts for record in actual] == list(pts)
    assert [record.time_base for record in actual] == [Fraction(1, 1000)] * 4
    assert [record.presentation_time for record in actual] == [Fraction(t, 1000) for t in pts]
    assert [tuple(record.rgb[:3]) for record in actual] == COLORS
    assert metadata.average_rate == 25 and metadata.start_pts == pts[0]
    with av.open(str(path)) as independent:
        direct = [
            (frame.pts, frame.time_base, list(bytes(frame.to_rgb().planes[0])[:3]))
            for frame in independent.decode(video=0)
        ]
    assert direct == [(pts[i], Fraction(1, 1000), list(COLORS[i])) for i in range(4)]


def test_real_seek_window_half_open_and_replay_keep_exact_timestamps(tmp_path):
    path = tmp_path / "fixture.mkv"
    make_video(path, (5000, 5040, 5150, 5300))
    with NativeVideoStream(path, NativeVideoConfig(start=Fraction(126, 25), end=Fraction(53, 10))) as stream:
        records = list(stream)
        assert [frame.pts for frame in records] == [5040, 5150]
        assert stream.diagnostics.status is NativeVideoStatus.RANGE_END
        stream.seek(Fraction(10081, 2000), end=Fraction(53, 10))
        assert [frame.pts for frame in stream] == [5150]
        stream.seek(None)
        assert [frame.pts for frame in stream] == [5000, 5040, 5150, 5300]


def test_real_avi_lossless_color_and_early_close_releases_file(tmp_path):
    path = tmp_path / "fixture.avi"
    make_video(path, (0, 1, 2, 3), format="avi", time_base=Fraction(1, 25))
    with NativeVideoStream(path) as stream:
        assert next(stream).rgb[:3] == bytes(COLORS[0])
        assert stream.metadata.format_name == "avi"
    renamed = path.with_suffix(".renamed")
    path.rename(renamed)
    renamed.unlink()
    assert stream.diagnostics.status is NativeVideoStatus.CLOSED


def test_real_corrupt_supported_signature_and_no_video_cleanup(tmp_path):
    corrupt = tmp_path / "corrupt.mkv"
    corrupt.write_bytes(b"\x1aE\xdf\xa3" + b"broken" * 8)
    with pytest.raises(ScanError), NativeVideoStream(corrupt):
        pass
    corrupt.unlink()
    audio = tmp_path / "audio.mka"
    with av.open(str(audio), "w", format="matroska") as container:
        stream = container.add_stream("pcm_s16le", rate=8000)
        frame = av.AudioFrame(format="s16", layout="mono", samples=80)
        frame.sample_rate = 8000
        frame.planes[0].update(bytes(frame.planes[0].buffer_size))
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    with pytest.raises(ScanError, match="video stream"), NativeVideoStream(audio):
        pass
    audio.unlink()


def test_real_pyav_secondary_open_callback_denies_local_hls_segment(tmp_path):
    """The backend rejects HLS entirely; separately exercise PyAV's real I/O hook."""
    attempted = []
    segment = (tmp_path / "not-opened.ts").as_posix()
    playlist = f"#EXTM3U\n#EXT-X-TARGETDURATION:1\n#EXTINF:1,\n{segment}\n#EXT-X-ENDLIST\n".encode()

    def deny(url, flags, options):
        attempted.append(url)
        return _deny_secondary(url, flags, options)

    with (
        pytest.raises((PermissionError, av.error.FFmpegError)),
        av.open(
            BytesIO(playlist), format="hls", io_open=deny, options={"protocol_whitelist": "file"}
        ) as container,
    ):
        list(container.decode())
    assert attempted and attempted[0].endswith("not-opened.ts")
    assert not (tmp_path / "not-opened.ts").exists()


def test_real_interframe_mp4_seek_decodes_forward_from_preceding_keyframe(tmp_path):
    path = tmp_path / "interframe.mp4"
    with av.open(str(path), "w", format="mp4") as container:
        stream = container.add_stream("mpeg4", rate=25)
        stream.width, stream.height, stream.pix_fmt = 16, 16, "yuv420p"
        stream.time_base = stream.codec_context.time_base = Fraction(1, 25)
        stream.codec_context.gop_size = 4
        stream.codec_context.max_b_frames = 0
        for index in range(8):
            with Image.new("RGB", (16, 16), (100, 20, 0)) as image:
                frame = av.VideoFrame.from_image(image)
            frame.pts, frame.time_base = index, Fraction(1, 25)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    with av.open(str(path)) as independent:
        decoded = list(independent.decode(video=0))
        assert [i for i, frame in enumerate(decoded) if frame.key_frame] == [0, 4]
        direct_times = [frame.pts * frame.time_base for frame in decoded]
    assert direct_times == [Fraction(i, 25) for i in range(8)]
    with NativeVideoStream(path, NativeVideoConfig(start=Fraction(3, 25), end=Fraction(7, 25))) as stream:
        actual = list(stream)
        assert [frame.presentation_time for frame in actual] == [Fraction(i, 25) for i in range(3, 7)]
        assert actual[0].decode_index == 3
        assert actual[0].sample_index == 0
        assert stream.diagnostics.status is NativeVideoStatus.RANGE_END


def test_real_native_scan_cli_emits_exact_jsonl_and_distinguishes_limit_from_eof(tmp_path, capsys):
    path = tmp_path / "fixture.mkv"
    make_video(path, (5000, 5040, 5150, 5300))
    assert main(["native-scan", str(path), "--start", "5.04", "--end", "53/10", "--max-frames", "1"]) == 0
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [row["record_type"] for row in records] == ["source", "frame", "summary"]
    assert records[1]["pts"] == 5040
    assert records[1]["presentation_time"] == {"numerator": 126, "denominator": 25}
    assert records[1]["metrics"]["mean_green"] == 1
    assert records[2]["diagnostics"]["status"] == "frame_limit"
    assert records[2]["diagnostics"]["closed"] is True


@pytest.mark.parametrize("text", ["nan", "1/0", "x" * 129])
def test_native_cli_rejects_invalid_exact_time_before_open(text):
    with pytest.raises(SystemExit) as error:
        main(["native-scan", "unused.mkv", "--start", text])
    assert error.value.code == 2


def test_native_offline_example_uses_generated_vfr_and_real_metrics(capsys):
    runpy.run_path(str(Path(__file__).parents[1] / "examples" / "native_pts.py"), run_name="__main__")
    report = json.loads(capsys.readouterr().out)
    assert [frame["pts"] for frame in report["frames"]] == [5000, 5040, 5150, 5300]
    assert report["frames"][2]["presentation_time"] == {"numerator": 103, "denominator": 20}
    assert report["diagnostics"]["status"] == "eof"
    assert report["rate_is_not_timestamp_source"] is True
