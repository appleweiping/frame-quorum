import hashlib
import os
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import pytest

import frame_quorum.native_av as audio_module
import frame_quorum.native_av_splitting as module
from frame_quorum import (
    NativeAVSplitConfig,
    NativeAVSplitResult,
    NativeClip,
    NativeSplitConfig,
    split_native_av,
)
from frame_quorum.errors import ConfigurationError, OutputError, ScanError


@pytest.mark.parametrize("value", [True, 0, -1, 10**1000, float("nan"), float("inf")])
def test_av_limits_reject_invalid_before_source_open(value, tmp_path):
    with pytest.raises(ConfigurationError):
        config = NativeAVSplitConfig(max_audio_samples=value)
        split_native_av(tmp_path / "missing.nut", tmp_path / "out", (NativeClip(0, 1),), config)
    assert not (tmp_path / "out").exists()


def test_audio_grid_epoch_independent_integer_oracle():
    from frame_quorum.native_av import _audio_epoch, _sample_slice

    assert _audio_epoch(Fraction(1001, 30000), Fraction(0), 48000) == Fraction(1601, 48000)
    assert _sample_slice(Fraction(0), 48000, 2000, Fraction(1001, 30000), Fraction(1, 25)) == (
        1602,
        1920,
    )
    assert _sample_slice(Fraction(5), 8000, 3, Fraction(5), Fraction(40001, 8000)) == (0, 1)


@pytest.mark.parametrize("name", NativeAVSplitConfig.__dataclass_fields__)
@pytest.mark.parametrize("value", [True, -1, 10**1000, float("nan"), "1"])
def test_every_inherited_and_audio_option_is_strict(name, value):
    with pytest.raises(ConfigurationError):
        NativeAVSplitConfig(**{name: value})


@pytest.mark.parametrize(
    "clips", [[], (), (object(),), (NativeClip(1, 2), NativeClip(0, 1)), (NativeClip(0, 2), NativeClip(1, 3))]
)
def test_clip_admission_precedes_native_side_effects(tmp_path, monkeypatch, clips):
    monkeypatch.setattr(module, "_load_av", lambda: pytest.fail("must validate before codec loading"))
    with pytest.raises(ConfigurationError):
        split_native_av("missing", tmp_path / "clips", clips)
    assert not list(tmp_path.iterdir())


def test_wrong_config_clip_cap_endian_and_tampered_result(tmp_path, monkeypatch):
    for options in (object(), False, NativeSplitConfig()):
        with pytest.raises(ConfigurationError, match="config"):
            split_native_av("missing", tmp_path / "clips", (NativeClip(0, 1),), options)
    with pytest.raises(ConfigurationError, match="bounded"):
        split_native_av(
            "missing", tmp_path / "clips", (NativeClip(0, 1),) * 2, NativeAVSplitConfig(max_clips=1)
        )
    monkeypatch.setattr(module.sys, "byteorder", "big")
    with pytest.raises(ConfigurationError, match="little-endian"):
        split_native_av("missing", tmp_path / "clips", (NativeClip(0, 1),))
    result = NativeAVSplitResult(tmp_path, 1, 1, 1, "a" * 64, 3)
    assert result.to_dict()["audio_sample_count"] == 3
    for value in (0, True, 1 << 37):
        with pytest.raises(ConfigurationError, match="audio_sample_count"):
            replace(result, audio_sample_count=value)


def test_sample_selection_exhaustive_independent_membership():
    for rate in (8000, 44100, 48000, 192000):
        anchor = Fraction(-7, 3)
        for left in range(-2, 7):
            for right in range(left + 1, 9):
                start, end = anchor + Fraction(left, 2 * rate), anchor + Fraction(right, 2 * rate)
                expected = [index for index in range(4) if start <= anchor + Fraction(index, rate) < end]
                first, last = audio_module._sample_slice(anchor, rate, 4, start, end)
                assert list(range(first, last)) == expected
                epoch = audio_module._audio_epoch(start, anchor, rate)
                assert epoch <= start < epoch + Fraction(1, rate)
                assert ((epoch - anchor) * rate).denominator == 1


def sound_frame(*, samples=3, rate=8000, pts=0, channels=2, layout="stereo", format="s16", planes=None):
    return SimpleNamespace(
        samples=samples,
        sample_rate=rate,
        pts=pts,
        time_base=Fraction(1, rate),
        layout=SimpleNamespace(name=layout, channels=(None,) * channels),
        format=SimpleNamespace(name=format),
        planes=[bytes(samples * channels * 2)] if planes is None else planes,
    )


@pytest.mark.parametrize(
    "change",
    [
        {"samples": 0},
        {"samples": True},
        {"samples": 4},
        {"rate": 7999},
        {"pts": None},
        {"pts": 1 << 63},
        {"channels": 3},
        {"layout": "downmix"},
        {"format": "flt"},
    ],
)
def test_audio_shape_rejects_unsupported_before_pcm_copy(change):
    with pytest.raises((ConfigurationError, ScanError)):
        audio_module._audio_shape(sound_frame(**change), 3)


@pytest.mark.parametrize(
    "layout,channels", [("mono", 1), ("stereo", 2), ("1 channels", 1), ("2 channels", 2)]
)
def test_explicit_or_unspecified_mono_stereo_layout_admission(layout, channels):
    assert audio_module._audio_shape(sound_frame(layout=layout, channels=channels), 3) == (
        3,
        8000,
        channels,
        Fraction(0),
    )


def test_pcm_packed_planar_padding_and_channel_order():
    left, right = b"\x01\x00\xfe\xff\x03\x00", b"\xff\xff\x02\x00\xfd\xff"
    expected = b"\x01\x00\xff\xff\xfe\xff\x02\x00\x03\x00\xfd\xff"
    stereo = sound_frame(format="s16p", planes=[left + b"padding", right + b"ignored"])
    assert audio_module._pcm_bytes(stereo, 3, 2) == expected
    packed = sound_frame(planes=[expected + b"nativepadding"])
    assert audio_module._pcm_bytes(packed, 3, 2) == expected
    mono = sound_frame(format="s16p", layout="mono", channels=1, planes=[left + b"ignored"])
    assert audio_module._pcm_bytes(mono, 3, 1) == left
    block = audio_module._AudioBlock(Fraction(0), 8000, 2, 3, expected)
    selected = block.trim(Fraction(1, 8000), Fraction(3, 8000), maximum=2)
    assert selected.pcm == expected[4:] and selected.start == Fraction(1, 8000) and selected.samples == 2
    assert block.trim(Fraction(3, 8000), Fraction(4, 8000), maximum=0) is None
    with pytest.raises(ScanError, match="selected audio"):
        block.trim(Fraction(0), Fraction(3, 8000), maximum=2)
    for planes in ([], [b"short"], [expected, expected]):
        with pytest.raises(ScanError, match="plane"):
            audio_module._pcm_bytes(sound_frame(planes=planes), 3, 2)


@pytest.mark.parametrize("mode", ["success", "short", "bad-second", "copy-interrupt"])
def test_every_pcm_export_is_released_even_on_partial_failure(monkeypatch, mode):
    exported = []
    view = memoryview

    def acquire(value):
        result = view(value)
        exported.append(result)
        return result

    monkeypatch.setattr(audio_module, "memoryview", acquire, raising=False)
    if mode == "copy-interrupt":

        def interrupt(*args):
            raise KeyboardInterrupt("copy")

        monkeypatch.setattr(audio_module, "bytearray", interrupt, raising=False)
    planes = [b"\1\0" * 3, object() if mode == "bad-second" else b"" if mode == "short" else b"\2\0" * 3]
    frame = sound_frame(format="s16p", planes=planes)
    if mode == "success":
        assert len(audio_module._pcm_bytes(frame, 3, 2)) == 12
    else:
        with pytest.raises(
            {"short": ScanError, "bad-second": TypeError, "copy-interrupt": KeyboardInterrupt}[mode]
        ):
            audio_module._pcm_bytes(frame, 3, 2)
    assert exported
    for item in exported:
        with pytest.raises(ValueError, match="released memoryview"):
            _ = item.obj


def opened_reader(frames, **limits):
    reader = audio_module._AudioReader("source.nut", NativeAVSplitConfig(**limits), None)
    reader.status, reader.frames = "open", iter(frames)
    return reader


@pytest.mark.parametrize(
    "change", [{"pts": 4}, {"pts": 2}, {"rate": 16000}, {"channels": 1, "layout": "mono"}]
)
def test_reader_rejects_gap_overlap_rate_or_channel_change(change):
    reader = opened_reader([sound_frame(), sound_frame(**change)])
    assert reader.next().samples == 3
    with pytest.raises(ScanError, match="grid"):
        reader.next()


def test_reader_sample_budget_precedes_copy_and_block_cap_precedes_next(monkeypatch):
    reader = opened_reader([sound_frame()], max_source_audio_samples=2)
    monkeypatch.setattr(audio_module, "_pcm_bytes", lambda *args: pytest.fail("copy must not run"))
    with pytest.raises(ScanError, match="source audio sample"):
        reader.next()
    reader = opened_reader([], max_audio_blocks=1)
    reader.blocks = 1
    with pytest.raises(ScanError, match="before EOF"):
        reader.next()
    reader.blocks = 0
    assert reader.next() is None and reader.next() is None and reader.status == "eof"
    reader.status = "closed"
    with pytest.raises(ConfigurationError, match="not open"):
        reader.next()


def test_reader_close_detaches_all_resources_and_preserves_control():
    closed = []

    class Resource:
        def close(self):
            closed.append(True)
            raise OSError("close")

    reader = opened_reader([])
    reader.frames = reader.container = reader.handle = Resource()
    primary = KeyboardInterrupt("cancel")
    reader.close(primary)
    assert closed == [True] * 3 and reader.status == "closed"
    assert reader.frames is reader.container is reader.handle is None
    assert "audio.file.close" in primary.__notes__[0]
    reader.close()


def test_source_reader_file_admission_and_native_inventory_caps(tmp_path, monkeypatch):
    reader = audio_module._AudioReader(tmp_path / "missing.nut", NativeAVSplitConfig(), None)
    with pytest.raises(ScanError, match="regular nonsymlink"):
        reader.__enter__()
    path = tmp_path / "source.nut"
    path.write_bytes(b"nut/multimedia container\0" + b"fixture")

    class Inventory:
        def __len__(self):
            return 3

    closed = []
    container = SimpleNamespace(streams=Inventory(), close=lambda: closed.append(True))
    monkeypatch.setattr(audio_module, "_open_native_container", lambda *args: container)
    reader = audio_module._AudioReader(path, NativeAVSplitConfig(max_streams=2), None)
    with pytest.raises(ScanError, match="stream limit"):
        reader.__enter__()
    assert closed == [True] and reader.handle is None
    path.unlink()


@pytest.mark.parametrize("primary", [None, ValueError("original"), KeyboardInterrupt("original")])
def test_final_source_check_control_and_cleanup_always_attempted(tmp_path, monkeypatch, primary):
    path = tmp_path / "source.nut"
    path.write_bytes(b"data")
    reader = opened_reader([])
    reader.path, reader.handle = path, path.open("rb")
    handle = reader.handle
    info = os.fstat(handle.fileno())
    reader.fingerprint = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
    closed = []

    class Resource:
        def close(self):
            closed.append(True)
            raise SystemExit("cleanup control")

    reader.frames = reader.container = Resource()

    def changed(*args):
        raise KeyboardInterrupt("identity control")

    monkeypatch.setattr(Path, "lstat", changed)
    if isinstance(primary, KeyboardInterrupt):
        reader.close(primary)
        assert "first control" in primary.__notes__[-1]
    else:
        with pytest.raises(KeyboardInterrupt, match="identity control"):
            reader.close(primary)
    assert closed == [True, True] and handle.closed


def test_ordinary_final_check_does_not_mask_original_failure(tmp_path, monkeypatch):
    path = tmp_path / "source.nut"
    path.write_bytes(b"data")
    reader = opened_reader([])
    reader.frames = None
    reader.path, reader.handle = path, path.open("rb")
    info = os.fstat(reader.handle.fileno())
    reader.fingerprint = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
    primary = ValueError("operation")
    monkeypatch.setattr(Path, "lstat", lambda *args: (_ for _ in ()).throw(OSError("unavailable")))
    reader.close(primary)
    assert primary.__notes__ == ["native audio final source-identity check failed"]


def test_unaccompanied_cleanup_control_propagates_after_other_resources_close():
    reader = opened_reader([])
    reader.frames = SimpleNamespace(close=lambda: (_ for _ in ()).throw(KeyboardInterrupt("close")))
    closed = []
    reader.container = SimpleNamespace(close=lambda: closed.append(True))
    with pytest.raises(KeyboardInterrupt, match="close"):
        reader.close()
    assert closed == [True]


def packet(*, pts=0, dts=0, size=10, time_base=Fraction(1, 8000)):
    return SimpleNamespace(pts=pts, dts=dts, size=size, time_base=time_base)


def bare_encoder():
    encoder = object.__new__(module._AVEncoder)
    encoder.config = NativeAVSplitConfig(max_pending_packet_bytes=10)
    encoder._packet_time, encoder._last_packet_time = Fraction(0), None
    encoder.peak_packet_bytes = 0
    encoder.writer = SimpleNamespace(failed=False)
    encoder.container = SimpleNamespace(mux=lambda packet: None)
    return encoder


@pytest.mark.parametrize(
    "packets",
    [
        [],
        [packet(), packet()],
        [packet(size=11)],
        [packet(pts=1)],
        [packet(dts=1)],
        [packet(pts=None)],
        [packet(time_base=Fraction(0))],
    ],
)
def test_immediate_packet_contract_rejects_missing_delayed_oversize_or_changed_time(packets):
    encoder = bare_encoder()
    with pytest.raises((ConfigurationError, ScanError)):
        encoder._mux_packets(iter(packets))


def test_mux_chronology_flush_and_swallowed_writer_failure():
    encoder = bare_encoder()
    encoder._mux_packets([packet()])
    assert encoder.peak_packet_bytes == 10
    encoder._last_packet_time = Fraction(1)
    with pytest.raises(ScanError, match="reordered"):
        encoder._mux_packets([packet()])
    encoder._packet_time = None
    encoder._mux_packets([])
    with pytest.raises(ScanError, match="delayed"):
        encoder._mux_packets([packet()])
    encoder._packet_time, encoder._last_packet_time = Fraction(0), None
    encoder.writer.failed = True
    with pytest.raises(OutputError, match="writer failed"):
        encoder._mux_packets([packet()])


@pytest.mark.parametrize("change", ["format", "gap", "offgrid"])
def test_audio_encoder_rejects_format_gap_or_nonrepresentable_pts_before_allocation(change):
    encoder = bare_encoder()
    encoder.rate, encoder.channels = 8000, 1
    encoder.origin, encoder.audio_first, encoder.audio_samples = Fraction(0), None, 0
    block = audio_module._AudioBlock(Fraction(0), 8000, 1, 1, b"\0\0")
    if change == "format":
        block = replace(block, channels=2)
    elif change == "gap":
        encoder.audio_first, encoder.audio_samples = Fraction(0), 1
    else:
        block = replace(block, start=Fraction(1, 16000))
    with pytest.raises(ScanError):
        encoder.write_audio(block)


def test_audio_clock_negotiation_failure_closes_created_resources(tmp_path, monkeypatch):
    closed = []
    audio = SimpleNamespace(codec_context=SimpleNamespace())
    container = SimpleNamespace(
        add_stream=lambda *args, **kwargs: audio,
        start_encoding=lambda: setattr(audio, "time_base", Fraction(1, 1000)),
    )

    def setup(self, *args, **kwargs):
        self.container = container

    monkeypatch.setattr(module._Encoder, "__init__", setup)
    monkeypatch.setattr(module._Encoder, "close", lambda *args: closed.append(True))
    frame = SimpleNamespace(time_base=Fraction(1, 30000))
    sound = audio_module._AudioBlock(Fraction(0), 8000, 1, 1, b"\0\0")
    with pytest.raises(ScanError, match="NUT audio clock"):
        module._AVEncoder(
            tmp_path / "output", frame, sound, Fraction(0), NativeAVSplitConfig(), None, None, {}
        )
    assert closed == [True]


@pytest.mark.parametrize("failure", [False, True])
def test_finish_closes_with_flush_failure_or_swallowed_writer_error(failure):
    encoder = bare_encoder()
    closed = []
    encoder.close = lambda primary=None: closed.append(primary)

    def flush():
        if failure:
            raise KeyboardInterrupt("flush")
        return []

    encoder.stream = encoder.audio_stream = SimpleNamespace(encode=flush)
    encoder.writer.failed = True
    with pytest.raises(KeyboardInterrupt if failure else OutputError):
        encoder.finish()
    assert len(closed) == 1
    assert isinstance(closed[0], KeyboardInterrupt) if failure else closed[0] is None


@pytest.mark.parametrize(
    "change",
    [
        "inventory",
        "codec",
        "extra",
        "missing",
        "rate",
        "channels",
        "pts",
        "pcm",
        "samples_budget",
        "blocks_budget",
    ],
)
def test_independent_audio_verifier_rejects_corruption(monkeypatch, change):
    block = audio_module._AudioBlock(Fraction(0), 8000, 1, 3, b"\1\0\2\0\3\0")
    blocks = [block]
    if change == "extra":
        blocks.append(block)
    elif change == "missing":
        blocks = []
    elif change in ("rate", "channels", "pts", "pcm"):
        blocks = [
            replace(
                block,
                **{
                    "rate": {"rate": 16000},
                    "channels": {"channels": 2},
                    "pts": {"start": Fraction(1, 8000)},
                    "pcm": {"pcm": b"\0" * 6},
                }[change],
            )
        ]
    streams = SimpleNamespace(
        audio=[
            SimpleNamespace(codec_context=SimpleNamespace(name="bad" if change == "codec" else "pcm_s16le"))
        ],
        video=[object()],
    )

    class Streams:
        audio, video = streams.audio, streams.video

        def __len__(self):
            return 3 if change == "inventory" else 2

    class Reader:
        container = SimpleNamespace(streams=Streams())

        def __init__(self, *args):
            self.items, self.blocks = iter(blocks), len(blocks)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def next(self):
            return next(self.items, None)

    monkeypatch.setattr(module, "_AudioReader", Reader)
    encoder = SimpleNamespace(
        config=NativeAVSplitConfig(),
        path="output",
        av=None,
        rate=8000,
        channels=1,
        audio_samples=3,
        audio_first=Fraction(5),
        origin=Fraction(5),
        audio_hash=hashlib.sha256(block.pcm),
    )
    with pytest.raises(OutputError):
        module._verify_audio(
            encoder, 2 if change == "samples_budget" else 4, 0 if change == "blocks_budget" else 4
        )
