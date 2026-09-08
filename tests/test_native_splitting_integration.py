from __future__ import annotations

import hashlib
import json
import runpy
from fractions import Fraction
from pathlib import Path

import pytest
from PIL import Image

import frame_quorum.native_splitting as module
from frame_quorum import (
    NativeClip,
    NativeSceneConfig,
    NativeSplitConfig,
    NativeVideoConfig,
    NativeVideoStream,
    detect_native_scenes,
    native_scene_clips,
    split_native_video,
)
from frame_quorum.cli import main
from frame_quorum.errors import ConfigurationError, OutputError, ScanError

av = pytest.importorskip("av", reason="install frame-quorum[video] for genuine native splitting tests")
PTS = (150000, 151001, 152002, 154004, 157007)
TB = Fraction(1, 30000)


def pixels(index):
    return bytes((index * 31 + offset * 17) % 256 for offset in range(16 * 12 * 3))


def make_source(path, pts=PTS, time_base=TB, *, audio=False):
    with av.open(str(path), "w", format="nut") as container:
        stream = container.add_stream("ffv1", rate=25)
        stream.width, stream.height, stream.pix_fmt = 16, 12, "bgr0"
        stream.time_base = stream.codec_context.time_base = time_base
        if audio:
            audio_stream = container.add_stream("pcm_s16le", rate=48000)
            audio_stream.layout = "mono"
        for index, timestamp in enumerate(pts):
            with Image.frombytes("RGB", (16, 12), pixels(index)) as image:
                frame = av.VideoFrame.from_image(image)
            frame.pts, frame.time_base = timestamp, time_base
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
        if audio:
            sound = av.AudioFrame(format="s16", layout="mono", samples=480)
            sound.sample_rate, sound.pts, sound.time_base = 48000, 240000, Fraction(1, 48000)
            sound.planes[0].update(bytes(sound.planes[0].buffer_size))
            for packet in audio_stream.encode(sound):
                container.mux(packet)
            for packet in audio_stream.encode():
                container.mux(packet)


def test_real_ntsc_vfr_precise_selection_and_independent_rgb_pts_reopen(tmp_path):
    source, target = tmp_path / "original.nut", tmp_path / "clips"
    make_source(source)
    intervals = (
        NativeClip(PTS[0] * TB, PTS[2] * TB),
        NativeClip(PTS[2] * TB + Fraction(1, 60000), Fraction(158008, 30000)),
    )
    result = split_native_video(source, target, intervals)
    assert result.output_dir == target
    expected_indices = ((0, 1), (3, 4))
    for ordinal, indices in enumerate(expected_indices):
        path = target / f"clip-{ordinal:06d}.nut"
        with av.open(str(path), "r", format="nut") as independent:
            assert len(independent.streams.video) == 1 and len(independent.streams.audio) == 0
            frames = list(independent.decode(video=0))
            assert [frame.pts * frame.time_base for frame in frames] == [
                (PTS[index] - PTS[indices[0]]) * TB for index in indices
            ]
            for frame, index in zip(frames, indices, strict=True):
                with frame.to_image() as image, image.convert("RGB") as rgb:
                    assert rgb.tobytes() == pixels(index)
        with NativeVideoStream(path) as stream:
            assert [frame.rgb for frame in stream] == [pixels(index) for index in indices]
    document = json.loads((target / "manifest.json").read_text())
    assert document["kind"] == "frame-quorum-native-split"
    assert document["frame_count"] == 4
    assert result.total_output_bytes == sum(path.stat().st_size for path in target.iterdir())
    assert result.manifest_sha256 == hashlib.sha256((target / "manifest.json").read_bytes()).hexdigest()
    assert sorted(path.name for path in target.iterdir()) == [
        "clip-000000.nut",
        "clip-000001.nut",
        "manifest.json",
    ]
    assert not list(tmp_path.glob(".frame-quorum-split-*"))
    source.unlink()  # The operation owns and closes all source handles.


def test_source_audio_is_explicitly_excluded_from_video_only_output(tmp_path):
    source, target = tmp_path / "source.nut", tmp_path / "clips"
    make_source(source, audio=True)
    with av.open(str(source)) as original:
        assert len(original.streams.audio) == 1
    split_native_video(source, target, (NativeClip(5, 6),))
    with av.open(str(target / "clip-000000.nut")) as independent:
        assert len(independent.streams) == len(independent.streams.video) == 1
        assert len(independent.streams.audio) == 0
        assert len(list(independent.decode(video=0))) == 5


@pytest.mark.parametrize(
    "start,end,indices",
    [
        (Fraction(0), PTS[1] * TB, (0,)),
        (PTS[1] * TB, PTS[2] * TB, (1,)),
        (PTS[1] * TB + Fraction(1, 60000), PTS[3] * TB, (2,)),
        (PTS[-1] * TB, Fraction(6), (4,)),
    ],
)
def test_half_open_selection_and_single_frame_clips(tmp_path, start, end, indices):
    source, target = tmp_path / "source.nut", tmp_path / "clips"
    make_source(source)
    result = split_native_video(source, target, (NativeClip(start, end),))
    with av.open(str(target / "clip-000000.nut")) as independent:
        frames = list(independent.decode(video=0))
        assert len(frames) == result.frame_count == len(indices)
        assert [item.pts * item.time_base for item in frames] == [
            (PTS[i] - PTS[indices[0]]) * TB for i in indices
        ]


@pytest.mark.parametrize(
    "clips",
    [
        (NativeClip(0, 1),),
        (NativeClip(0, 1), NativeClip(5, 6)),
        (NativeClip(5, Fraction(51, 10)), NativeClip(8, 9)),
        (
            NativeClip(5, PTS[1] * TB),
            NativeClip(PTS[1] * TB + TB, PTS[2] * TB - TB),
            NativeClip(PTS[3] * TB, 6),
        ),
    ],
)
def test_empty_first_middle_last_intervals_abort_entire_directory(tmp_path, clips):
    source, target = tmp_path / "source.nut", tmp_path / "clips"
    make_source(source)
    with pytest.raises(ScanError, match="no frames"):
        split_native_video(source, target, clips)
    assert not target.exists() and not list(tmp_path.glob(".frame-quorum-split-*"))
    source.unlink()


@pytest.mark.parametrize(
    "name,value",
    [
        ("max_frames", 2),
        ("max_decoded_frames", 2),
        ("max_source_bytes", 10),
        ("max_frame_pixels", 191),
        ("max_source_pixels", 192 * 2),
        ("max_verification_pixels", 192 * 2),
        ("max_output_bytes", 100),
        ("max_manifest_bytes", 100),
    ],
)
def test_all_resource_caps_abort_before_publication_and_close_source(tmp_path, name, value):
    source, target = tmp_path / "source.nut", tmp_path / "clips"
    make_source(source)
    with pytest.raises((ConfigurationError, ScanError, OutputError)):
        split_native_video(
            source,
            target,
            (NativeClip(5, PTS[2] * TB), NativeClip(PTS[2] * TB, 6)),
            NativeSplitConfig(**{name: value}),
        )
    assert not target.exists() and not list(tmp_path.glob(".frame-quorum-split-*"))
    source.unlink()


def test_byte_cap_includes_manifest_not_only_clip_bytes(tmp_path):
    source = tmp_path / "source.nut"
    make_source(source)
    baseline = split_native_video(source, tmp_path / "baseline", (NativeClip(5, 6),))
    with pytest.raises(OutputError, match="total output"):
        split_native_video(
            source,
            tmp_path / "limited",
            (NativeClip(5, 6),),
            NativeSplitConfig(max_output_bytes=baseline.total_output_bytes - 100),
        )
    assert not (tmp_path / "limited").exists()


@pytest.mark.parametrize("failure", [ValueError("encode"), KeyboardInterrupt("cancel"), SystemExit(9)])
def test_encode_failure_or_control_closes_and_removes_stage(tmp_path, monkeypatch, failure):
    source, target = tmp_path / "source.nut", tmp_path / "clips"
    make_source(source)

    def fail(self, frame):
        raise failure

    monkeypatch.setattr(module._Encoder, "write", fail)
    expected = OutputError if isinstance(failure, Exception) else type(failure)
    with pytest.raises(expected):
        split_native_video(source, target, (NativeClip(5, 6),))
    assert not target.exists() and not list(tmp_path.glob(".frame-quorum-split-*"))
    source.unlink()


def test_close_control_survives_subsequent_ordinary_unlink_error(tmp_path, monkeypatch):
    source, target = tmp_path / "source.nut", tmp_path / "clips"
    make_source(source)
    close = module._Encoder.close

    def fail(self, frame):
        raise ValueError("encode")

    def close_then_interrupt(self, primary=None):
        close(self, primary)
        raise KeyboardInterrupt("closing")

    with monkeypatch.context() as scoped:
        scoped.setattr(module._Encoder, "write", fail)
        scoped.setattr(module._Encoder, "close", close_then_interrupt)
        scoped.setattr(Path, "unlink", lambda self: (_ for _ in ()).throw(OSError("unlink")))
        with pytest.raises(KeyboardInterrupt, match="closing") as caught:
            split_native_video(source, target, (NativeClip(5, 6),))
        assert "cleanup incomplete" in caught.value.__notes__[0]
    assert not target.exists()
    stages = list(tmp_path.glob(".frame-quorum-split-*"))
    assert len(stages) == 1
    assert str(stages[0]) in caught.value.__notes__[0]
    for path in stages[0].iterdir():
        path.unlink()
    stages[0].rmdir()
    source.unlink()


def test_independent_verification_detects_actual_file_tamper(tmp_path, monkeypatch):
    source, target = tmp_path / "source.nut", tmp_path / "clips"
    make_source(source)
    verify = module._verify

    def tamper(encoder, *args):
        encoder.path.write_bytes(b"not a video")
        return verify(encoder, *args)

    monkeypatch.setattr(module, "_verify", tamper)
    with pytest.raises(ScanError):
        split_native_video(source, target, (NativeClip(5, 6),))
    assert not target.exists() and not list(tmp_path.glob(".frame-quorum-split-*"))


def test_competing_publication_is_not_overwritten_or_cleaned(tmp_path, monkeypatch):
    source, target = tmp_path / "source.nut", tmp_path / "clips"
    make_source(source)
    publish = module._publish

    def compete(stage, destination):
        destination.mkdir()
        (destination / "theirs").write_bytes(b"preserve")
        publish(stage, destination)

    monkeypatch.setattr(module, "_publish", compete)
    with pytest.raises(OutputError, match="publication did not acknowledge"):
        split_native_video(source, target, (NativeClip(5, 6),))
    assert (target / "theirs").read_bytes() == b"preserve"
    assert not list(tmp_path.glob(".frame-quorum-split-*"))


def test_late_publication_failure_is_truthful_and_never_deletes_destination(tmp_path, monkeypatch):
    source, target = tmp_path / "source.nut", tmp_path / "clips"
    make_source(source)
    publish = module._publish

    def moved_then_failed(stage, destination):
        publish(stage, destination)
        raise OSError("late acknowledgement failure")

    monkeypatch.setattr(module, "_publish", moved_then_failed)
    with pytest.raises(OutputError, match="inspect"):
        split_native_video(source, target, (NativeClip(5, 6),))
    assert (target / "manifest.json").is_file()
    assert not list(tmp_path.glob(".frame-quorum-split-*"))


def test_optional_codec_failure_has_no_staging_side_effect(tmp_path, monkeypatch):
    class AV:
        class codec:
            @staticmethod
            def Codec(*args):
                raise RuntimeError("no encoder")

    monkeypatch.setattr(module, "_load_av", lambda: AV)
    with pytest.raises(OutputError):
        split_native_video("missing", tmp_path / "clips", (NativeClip(0, 1),))
    assert list(tmp_path.iterdir()) == []


def test_scene_bridge_unknown_endpoint_requires_exact_caller_bound(tmp_path):
    source = tmp_path / "source.nut"
    make_source(source)
    result = detect_native_scenes(source)
    with pytest.raises(ConfigurationError, match="unknown final"):
        native_scene_clips(result)
    clips = native_scene_clips(result, final_end=Fraction(6))
    assert clips[0].start == 5 and clips[-1].end == 6
    assert split_native_video(source, tmp_path / "clips", clips).frame_count == 5
    for invalid in (PTS[-1] * TB, Fraction(4), 6.0):
        with pytest.raises(ConfigurationError):
            native_scene_clips(result, final_end=invalid)


@pytest.mark.parametrize(
    "video",
    [
        NativeVideoConfig(frame_step=2),
        NativeVideoConfig(max_frames=2),
        NativeVideoConfig(max_decoded_frames=2),
    ],
)
def test_scene_bridge_rejects_sampling_or_partial_analysis(tmp_path, video):
    source = tmp_path / "source.nut"
    make_source(source)
    result = detect_native_scenes(source, NativeSceneConfig(video=video))
    with pytest.raises(ConfigurationError, match="complete, unsampled"):
        native_scene_clips(result, final_end=Fraction(6))


def test_scene_bridge_known_range_and_empty_result(tmp_path):
    source = tmp_path / "source.nut"
    make_source(source)
    end = PTS[3] * TB
    result = detect_native_scenes(source, NativeSceneConfig(video=NativeVideoConfig(end=end)))
    assert native_scene_clips(result)[-1].end == end
    assert native_scene_clips(result, final_end=end)[-1].end == end
    for wrong in (end + TB, end - TB):
        with pytest.raises(ConfigurationError):
            native_scene_clips(result, final_end=wrong)
    empty = detect_native_scenes(source, NativeSceneConfig(video=NativeVideoConfig(end=Fraction(1))))
    with pytest.raises(ConfigurationError, match="no clips"):
        native_scene_clips(empty)
    with pytest.raises(ConfigurationError, match="NativeSceneResult"):
        native_scene_clips(object())


def test_cli_real_split_strict_parser_and_error_return(tmp_path, capsys):
    source, target = tmp_path / "source.nut", tmp_path / "clips"
    make_source(source)
    args = ["native-split", str(source), "--output-dir", str(target), "--clip", "5", "6"]
    assert main(args) == 0
    assert json.loads(capsys.readouterr().out)["frame_count"] == 5
    assert main(args) == 2
    assert "new output directory" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main([*args, "--frame-step", "2"])


def test_offline_generated_split_example(capsys):
    runpy.run_path(str(Path(__file__).parents[1] / "examples" / "native_splitting.py"), run_name="__main__")
    assert "verified 2 clips / 6 frames" in capsys.readouterr().out
