from __future__ import annotations

import hashlib
import json
import runpy
import struct
from fractions import Fraction
from pathlib import Path

import pytest
from PIL import Image

import frame_quorum.native_av_splitting as module
from frame_quorum import NativeAVSplitConfig, NativeClip, split_native_av
from frame_quorum.cli import main
from frame_quorum.errors import ConfigurationError, OutputError, ScanError

av = pytest.importorskip("av", reason="install frame-quorum[video] for genuine AV splitting")
PTS = (5000, 5021, 5040, 5075, 5100, 5124)


def rgb(index):
    return bytes((index * 37 + offset * 11) % 256 for offset in range(8 * 6 * 3))


def pcm(left, right, channels=2):
    values = [
        value for i in range(left, right) for value in ((i - 500, 500 - i) if channels == 2 else (i - 500,))
    ]
    return struct.pack(f"<{len(values)}h", *values)


def make_source(
    path,
    *,
    channels=2,
    rate=8000,
    count=1000,
    block=137,
    anchor=Fraction(5),
    gap=0,
    gap_after=None,
    extra_audio_first=False,
    with_audio=True,
    audio_codec="pcm_s16le",
    pts=PTS,
    time_base=Fraction(1, 1000),
):
    with av.open(str(path), "w", format="nut") as container:
        video = container.add_stream("ffv1", rate=25)
        video.width, video.height, video.pix_fmt = 8, 6, "bgr0"
        video.time_base = video.codec_context.time_base = time_base
        other_audio = None
        if extra_audio_first:
            other_audio = container.add_stream("pcm_f32le", rate=8000)
            other_audio.layout = "mono"
        audio = None
        if with_audio:
            audio = container.add_stream(audio_codec, rate=rate)
            audio.layout = "mono" if channels == 1 else "stereo"
            audio.time_base = audio.codec_context.time_base = Fraction(1, rate)
        for index, timestamp in enumerate(pts):
            with Image.frombytes("RGB", (8, 6), rgb(index)) as image:
                frame = av.VideoFrame.from_image(image)
            frame.pts, frame.time_base = timestamp, time_base
            for packet in video.encode(frame):
                container.mux(packet)
        for offset in range(0, count if with_audio else 0, block):
            samples = min(block, count - offset)
            is_float = audio_codec == "pcm_f32le"
            frame = av.AudioFrame(
                format="flt" if is_float else "s16", layout=audio.layout.name, samples=samples
            )
            timestamp = (
                int(anchor * rate)
                + offset
                + (gap if offset >= (block if gap_after is None else gap_after) else 0)
            )
            frame.sample_rate, frame.pts, frame.time_base = rate, timestamp, Fraction(1, rate)
            frame.planes[0].update(
                bytes(samples * channels * 4) if is_float else pcm(offset, offset + samples, channels)
            )
            for packet in audio.encode(frame):
                container.mux(packet)
        for stream in (video,) if audio is None else (video, audio):
            for packet in stream.encode():
                container.mux(packet)
        if other_audio is not None:
            frame = av.AudioFrame(format="flt", layout="mono", samples=8)
            frame.sample_rate, frame.pts, frame.time_base = 8000, 40000, Fraction(1, 8000)
            frame.planes[0].update(bytes(32))
            for packet in other_audio.encode(frame):
                container.mux(packet)
            for packet in other_audio.encode():
                container.mux(packet)


@pytest.mark.parametrize("channels", [1, 2])
def test_real_half_open_audio_chunks_and_video_timestamps_independent_reopen(tmp_path, channels):
    source, target = tmp_path / "source.nut", tmp_path / "clips"
    make_source(source, channels=channels)
    clips = (
        NativeClip(5 + Fraction(1, 30000), Fraction(126, 25)),
        NativeClip(Fraction(126, 25), Fraction(51, 10)),
    )
    result = split_native_av(source, target, clips)
    assert result.frame_count == 3
    assert result.audio_sample_count == 319 + 480
    for ordinal, (indices, first, last, epoch) in enumerate(
        (((1,), 1, 320, Fraction(5)), ((2, 3), 320, 800, Fraction(126, 25)))
    ):
        with av.open(str(target / f"clip-{ordinal:06d}.nut"), "r", format="nut") as reopened:
            frames = list(reopened.decode())
        video = [frame for frame in frames if isinstance(frame, av.VideoFrame)]
        audio = [frame for frame in frames if isinstance(frame, av.AudioFrame)]
        assert [frame.pts * frame.time_base for frame in video] == [
            Fraction(PTS[i], 1000) - epoch for i in indices
        ]
        for frame, index in zip(video, indices, strict=True):
            with frame.to_image() as image:
                assert image.tobytes() == rgb(index)
        assert sum(frame.samples for frame in audio) == last - first
        assert b"".join(bytes(frame.planes[0])[: frame.samples * channels * 2] for frame in audio) == pcm(
            first, last, channels
        )
        cursor = first
        for frame in audio:
            assert frame.pts * frame.time_base == 5 + Fraction(cursor, 8000) - epoch
            cursor += frame.samples
    manifest = json.loads((target / "manifest.json").read_text())
    assert manifest["kind"] == "frame-quorum-native-av-split"
    assert manifest["rebasing"] == "source-audio-grid-floor"
    assert result.manifest_sha256 == hashlib.sha256((target / "manifest.json").read_bytes()).hexdigest()
    assert result.total_output_bytes == sum(path.stat().st_size for path in target.iterdir())
    assert not list(tmp_path.glob(".frame-quorum-av-split-*"))
    source.unlink()


@pytest.mark.parametrize(
    "name,value",
    [
        ("max_frames", 2),
        ("max_decoded_frames", 2),
        ("max_source_bytes", 10),
        ("max_frame_pixels", 47),
        ("max_source_pixels", 95),
        ("max_verification_pixels", 95),
        ("max_output_bytes", 100),
        ("max_manifest_bytes", 100),
        ("max_audio_blocks", 1),
        ("max_audio_block_samples", 136),
        ("max_source_audio_samples", 150),
        ("max_audio_samples", 500),
        ("max_verification_audio_samples", 500),
        ("max_verification_audio_blocks", 1),
        ("max_pending_packet_bytes", 1),
        ("video_stream", 1),
        ("audio_stream", 1),
    ],
)
def test_each_resource_limit_aborts_without_success_manifest(tmp_path, name, value):
    source, target = tmp_path / "source.nut", tmp_path / "clips"
    make_source(source)
    with pytest.raises((ConfigurationError, OutputError, ScanError)):
        split_native_av(
            source, target, (NativeClip(5, Fraction(51, 10)),), NativeAVSplitConfig(**{name: value})
        )
    assert not target.exists() and not list(tmp_path.glob(".frame-quorum-av-split-*"))
    source.unlink()


@pytest.mark.parametrize("options", [{"with_audio": False}, {"gap": 1}, {"audio_codec": "pcm_f32le"}])
def test_missing_discontinuous_or_float_audio_is_not_synthesized(tmp_path, options):
    source, target = tmp_path / "source.nut", tmp_path / "clips"
    make_source(source, **options)
    with pytest.raises(ScanError):
        split_native_av(source, target, (NativeClip(5, 6),))
    assert not target.exists() and not list(tmp_path.glob(".frame-quorum-av-split-*"))
    source.unlink()


def test_audio_ordinal_selects_second_real_track_and_discards_unselected_float_track(tmp_path):
    source, target = tmp_path / "source.nut", tmp_path / "clips"
    make_source(source, extra_audio_first=True)
    with av.open(str(source)) as original:
        assert [stream.codec_context.name for stream in original.streams.audio] == ["pcm_f32le", "pcm_s16le"]
    result = split_native_av(
        source, target, (NativeClip(5, Fraction(51, 10)),), NativeAVSplitConfig(audio_stream=1)
    )
    with av.open(str(target / "clip-000000.nut")) as independent:
        assert len(independent.streams) == 2
        audio = list(independent.decode(audio=0))
    assert result.audio_sample_count == 800
    assert b"".join(bytes(item.planes[0])[: item.samples * 4] for item in audio) == pcm(0, 800)
    manifest = json.loads((target / "manifest.json").read_text())
    assert manifest["source_audio"]["stream_index"] == 2
    with pytest.raises(ScanError, match="stream limit"):
        split_native_av(
            source, tmp_path / "too-many", (NativeClip(5, 6),), NativeAVSplitConfig(max_streams=2)
        )


def test_unused_trailing_audio_is_not_claimed_continuous(tmp_path):
    source, target = tmp_path / "source.nut", tmp_path / "clips"
    make_source(source, gap=13, gap_after=822)
    result = split_native_av(source, target, (NativeClip(5, Fraction(126, 25)),))
    assert result.audio_sample_count == 320
    manifest = json.loads((target / "manifest.json").read_text())
    assert manifest["source_audio"]["status"] == "range_end"
    assert manifest["source_audio"]["decoded_samples"] < 822
    with pytest.raises(ScanError, match="grid"):
        split_native_av(source, tmp_path / "longer", (NativeClip(5, 6),))


@pytest.mark.parametrize(
    "clips",
    [
        (NativeClip(0, 1),),
        (NativeClip(5, Fraction(51, 10)), NativeClip(6, 7)),
        (NativeClip(5, Fraction(5001, 1000)), NativeClip(Fraction(5001, 1000), Fraction(5002, 1000))),
    ],
)
def test_empty_first_or_later_track_selection_rolls_back_earlier_clips(tmp_path, clips):
    source, target = tmp_path / "source.nut", tmp_path / "clips"
    make_source(source)
    with pytest.raises(ScanError, match=r"no video frames or audio|requires nonempty"):
        split_native_av(source, target, clips)
    assert not target.exists() and not list(tmp_path.glob(".frame-quorum-av-split-*"))


@pytest.mark.parametrize("anchor", [Fraction(49, 10), Fraction(501, 100)])
def test_unequal_track_starts_keep_original_sync_without_padding(tmp_path, anchor):
    source, target = tmp_path / "source.nut", tmp_path / "clips"
    make_source(source, anchor=anchor, count=2000)
    result = split_native_av(source, target, (NativeClip(5, Fraction(51, 10)),))
    with av.open(str(target / "clip-000000.nut")) as independent:
        frames = list(independent.decode())
    video = [item for item in frames if isinstance(item, av.VideoFrame)]
    audio = [item for item in frames if isinstance(item, av.AudioFrame)]
    assert video[0].pts * video[0].time_base == 0
    first = 800 if anchor < 5 else 0
    last = 1600 if anchor < 5 else 720
    assert audio[0].pts * audio[0].time_base == anchor + Fraction(first, 8000) - 5
    assert b"".join(bytes(item.planes[0])[: item.samples * 4] for item in audio) == pcm(first, last)
    assert result.audio_sample_count == last - first


@pytest.mark.parametrize("rate", [44100, 48000, 192000])
def test_non_sample_aligned_ntsc_clip_common_epoch_real_reopen(tmp_path, rate):
    source, target = tmp_path / "source.nut", tmp_path / "clips"
    # 4 frames whose clock cannot generally represent the audio sample lattice.
    points = (0, 1001, 2002, 3003)
    make_source(
        source, anchor=Fraction(0), rate=rate, count=rate // 10, pts=points, time_base=Fraction(1, 30000)
    )
    start, end = Fraction(1001, 30000), Fraction(2002, 30000)
    result = split_native_av(source, target, (NativeClip(start, end),))
    # Independent sample membership, rather than invoking the implementation's slice helper.
    selected = [index for index in range(rate // 10) if start <= Fraction(index, rate) < end]
    epoch = Fraction((1001 * rate) // 30000, rate)
    with av.open(str(target / "clip-000000.nut")) as independent:
        frames = list(independent.decode())
    video = [item for item in frames if isinstance(item, av.VideoFrame)]
    audio = [item for item in frames if isinstance(item, av.AudioFrame)]
    assert len(video) == 1 and video[0].pts * video[0].time_base == start - epoch
    assert audio[0].pts * audio[0].time_base == Fraction(selected[0], rate) - epoch
    assert result.audio_sample_count == len(selected)
    assert b"".join(bytes(item.planes[0])[: item.samples * 4] for item in audio) == pcm(
        selected[0], selected[-1] + 1
    )


@pytest.mark.parametrize("phase", ["write", "write_audio", "finish"])
@pytest.mark.parametrize("failure", [ValueError("codec"), KeyboardInterrupt("cancel"), SystemExit(9)])
def test_encoder_failures_or_interrupts_close_before_cleanup(tmp_path, monkeypatch, phase, failure):
    source, target = tmp_path / "source.nut", tmp_path / "clips"
    make_source(source)

    def fail(*args):
        raise failure

    monkeypatch.setattr(module._AVEncoder, phase, fail)
    with pytest.raises(OutputError if isinstance(failure, Exception) else type(failure)):
        split_native_av(source, target, (NativeClip(5, Fraction(51, 10)),))
    assert not target.exists() and not list(tmp_path.glob(".frame-quorum-av-split-*"))
    source.unlink()


def test_no_replace_and_lost_publication_acknowledgement(tmp_path, monkeypatch):
    source = tmp_path / "source.nut"
    make_source(source)
    publish = module._publish
    for mode in ("compete", "lost_ack"):
        target = tmp_path / mode

        def replacement(stage, target, mode=mode):
            if mode == "compete":
                target.mkdir()
                (target / "foreign").write_bytes(b"preserve")
            publish(stage, target)
            raise OSError("lost acknowledgement")

        monkeypatch.setattr(module, "_publish", replacement)
        with pytest.raises(OutputError, match="inspect"):
            split_native_av(source, target, (NativeClip(5, Fraction(51, 10)),))
        assert target.exists() and not list(tmp_path.glob(".frame-quorum-av-split-*"))
        assert (target / ("foreign" if mode == "compete" else "manifest.json")).is_file()


def test_new_cli_and_actual_offline_example(tmp_path, capsys):
    source, target = tmp_path / "source.nut", tmp_path / "clips"
    make_source(source)
    argv = ["native-av-split", str(source), "--output-dir", str(target), "--clip", "5", "51/10"]
    assert main(argv) == 0
    assert json.loads(capsys.readouterr().out)["audio_sample_count"] == 800
    assert main(argv) == 2 and "new output" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main([*argv, "--resample-rate", "16000"])
    runpy.run_path(
        str(Path(__file__).parents[1] / "examples" / "native_av_splitting.py"), run_name="__main__"
    )
    assert "verified 2 clips / 4 video frames / 6406 stereo sample positions" in capsys.readouterr().out


def test_selected_audio_budget_aborts_all_outputs(tmp_path):
    source, target = tmp_path / "source.nut", tmp_path / "clips"
    make_source(source)
    from frame_quorum.errors import ScanError

    with pytest.raises(ScanError, match="selected audio"):
        split_native_av(
            source, target, (NativeClip(5, Fraction(51, 10)),), NativeAVSplitConfig(max_audio_samples=1)
        )
    assert not target.exists()
    assert not list(tmp_path.glob(".frame-quorum-av-split-*"))
    source.unlink()


def test_buffered_audio_tail_source_mutation_after_video_range_end_is_rejected(tmp_path, monkeypatch):
    source, target = tmp_path / "source.nut", tmp_path / "clips"
    make_source(source)
    finish, read = module.NativeVideoStream._finish, module._AudioReader.next
    video_ended = []
    mutated = []

    def record_finish(self, status, **kwargs):
        result = finish(self, status, **kwargs)
        if status is module.NativeVideoStatus.RANGE_END:
            video_ended.append(True)
        return result

    def mutate_last_buffered_audio(self):
        block = read(self)
        if self.path == source and block is not None and block.end >= Fraction(128, 25) and not mutated:
            assert video_ended  # The video cursor has already reached its terminal range status.
            with source.open("ab") as handle:
                handle.write(b"observed mutation after the native read buffer was filled")
            mutated.append(True)
        return block

    monkeypatch.setattr(module.NativeVideoStream, "_finish", record_finish)
    monkeypatch.setattr(module._AudioReader, "next", mutate_last_buffered_audio)
    with pytest.raises(ScanError, match="changed"):
        split_native_av(source, target, (NativeClip(5, Fraction(128, 25)),))
    assert mutated and not target.exists() and not list(tmp_path.glob(".frame-quorum-av-split-*"))
    source.unlink()


def test_differing_open_fingerprints_fail_before_staging(tmp_path, monkeypatch):
    source, target = tmp_path / "source.nut", tmp_path / "clips"
    make_source(source)
    enter = module._AudioReader.__enter__

    def changed_reader(self):
        result = enter(self)
        self.fingerprint = (*self.fingerprint[:3], self.fingerprint[3] + 1)
        return result

    monkeypatch.setattr(module._AudioReader, "__enter__", changed_reader)
    with pytest.raises(ScanError, match="different source identities"):
        split_native_av(source, target, (NativeClip(5, Fraction(51, 10)),))
    assert not target.exists() and not list(tmp_path.glob(".frame-quorum-av-split-*"))
    source.unlink()


def test_abandoned_audio_prefix_is_not_natural_eof(tmp_path, monkeypatch):
    source, target = tmp_path / "source.nut", tmp_path / "clips"
    make_source(source)
    read = module._AudioReader.next

    def abandon(self):
        return None if self.blocks == 2 else read(self)

    monkeypatch.setattr(module._AudioReader, "next", abandon)
    with pytest.raises(ScanError, match="truncated source audio"):
        split_native_av(source, target, (NativeClip(5, Fraction(51, 10)),))
    assert not target.exists() and not list(tmp_path.glob(".frame-quorum-av-split-*"))


def test_unknown_staging_identity_reports_exact_unremoved_residue(tmp_path, monkeypatch):
    source, target = tmp_path / "source.nut", tmp_path / "clips"
    make_source(source)
    lstat = Path.lstat

    def fail_stage(self):
        if self.name.startswith(".frame-quorum-av-split-"):
            raise OSError("stage identity unavailable")
        return lstat(self)

    with monkeypatch.context() as scoped:
        scoped.setattr(Path, "lstat", fail_stage)
        with pytest.raises(OutputError) as caught:
            split_native_av(source, target, (NativeClip(5, Fraction(51, 10)),))
    stages = list(tmp_path.glob(".frame-quorum-av-split-*"))
    assert len(stages) == 1 and not target.exists()
    assert str(stages[0]) in caught.value.__cause__.__notes__[0]
    assert not list(stages[0].iterdir())
    stages[0].rmdir()


def test_cleanup_does_not_remove_foreign_entries(tmp_path, monkeypatch):
    source, target = tmp_path / "source.nut", tmp_path / "clips"
    make_source(source)

    def fail(self, block):
        (self.path.parent / "foreign").write_bytes(b"preserve")
        raise ValueError("codec")

    monkeypatch.setattr(module._AVEncoder, "write_audio", fail)
    with pytest.raises(OutputError, match="cleanup incomplete"):
        split_native_av(source, target, (NativeClip(5, Fraction(51, 10)),))
    (stage,) = tmp_path.glob(".frame-quorum-av-split-*")
    assert not target.exists() and [path.name for path in stage.iterdir()] == ["foreign"]
    assert (stage / "foreign").read_bytes() == b"preserve"


def test_cleanup_interrupt_survives_later_unlink_failure(tmp_path, monkeypatch):
    source, target = tmp_path / "source.nut", tmp_path / "clips"
    make_source(source)
    close = module._AVEncoder.close

    def fail(self, block):
        raise ValueError("codec")

    def close_then_interrupt(self, primary=None):
        close(self, primary)
        raise KeyboardInterrupt("closing")

    with monkeypatch.context() as scoped:
        scoped.setattr(module._AVEncoder, "write_audio", fail)
        scoped.setattr(module._AVEncoder, "close", close_then_interrupt)
        scoped.setattr(Path, "unlink", lambda self: (_ for _ in ()).throw(OSError("unlink")))
        with pytest.raises(KeyboardInterrupt, match="closing") as caught:
            split_native_av(source, target, (NativeClip(5, Fraction(51, 10)),))
    assert not target.exists() and "cleanup incomplete" in caught.value.__notes__[0]
    source.unlink()
