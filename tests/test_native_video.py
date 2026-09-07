from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, replace
from fractions import Fraction
from types import SimpleNamespace

import pytest
from PIL import Image

import frame_quorum.native_video as native
from frame_quorum.cli import main
from frame_quorum.errors import ConfigurationError, ScanError
from frame_quorum.native_video import (
    NativeVideoConfig,
    NativeVideoDiagnostics,
    NativeVideoFrame,
    NativeVideoStatus,
    NativeVideoStream,
)


class FakeFrame:
    def __init__(self, pts=5000, *, time_base=Fraction(1, 1000), width=2, height=2):
        self.pts, self.time_base, self.width, self.height = pts, time_base, width, height
        self.converted = False

    def to_image(self):
        self.converted = True
        return Image.new("RGB", (self.width, self.height), (255, 0, 0))


class FakeContainer:
    def __init__(self, frames):
        self.frames = frames
        self.decoded = 0
        self.closed = False
        self.generator_closed = False
        self.seek_calls = []
        self.streams = SimpleNamespace(
            video=[
                SimpleNamespace(
                    codec_context=SimpleNamespace(width=2, height=2, name="ffv1", thread_count=0),
                    index=2,
                    time_base=Fraction(1, 1000),
                    start_time=5000,
                    duration=200,
                    average_rate=Fraction(25),
                    base_rate=Fraction(25),
                )
            ]
        )

    def seek(self, offset, **kwargs):
        self.seek_calls.append((offset, kwargs))

    def decode(self, **kwargs):
        assert kwargs == {"video": 0}
        try:
            for frame in self.frames:
                self.decoded += 1
                if isinstance(frame, BaseException):
                    raise frame
                yield frame
        finally:
            self.generator_closed = True

    def close(self):
        self.closed = True


@pytest.fixture
def decoder(tmp_path, monkeypatch):
    source = tmp_path / "source.mkv"
    source.write_bytes(b"\x1aE\xdf\xa3" + b"0" * 12)
    containers, readers, calls = [], [], []
    supplied = [FakeFrame(5000), FakeFrame(5040), FakeFrame(5100)]

    def open_container(reader, **kwargs):
        container = FakeContainer(list(supplied))
        readers.append(reader)
        containers.append(container)
        calls.append(kwargs)
        return container

    monkeypatch.setattr(native, "_load_av", lambda: SimpleNamespace(open=open_container))
    return SimpleNamespace(
        source=source, frames=supplied, containers=containers, readers=readers, calls=calls
    )


def test_lazy_open_exact_rational_range_and_stride_preserve_native_pts(decoder):
    config = NativeVideoConfig(start=Fraction(126, 25), end=Fraction(51, 10))
    stream = NativeVideoStream(decoder.source, config)
    assert not decoder.containers
    with stream:
        container = decoder.containers[-1]
        assert container.decoded == 0
        assert stream.metadata.average_rate == 25
        assert stream.metadata.time_base == Fraction(1, 1000)
        assert stream.metadata.start_pts == 5000 and stream.metadata.stream_index == 2
        records = list(stream)
        assert [record.pts for record in records] == [5040]
        assert records[0].presentation_time == Fraction(126, 25)
        assert records[0].decode_index == 1 and records[0].sample_index == 0
        assert container.seek_calls[0][0] == 5040
        assert container.seek_calls[0][1] == {
            "stream": container.streams.video[0],
            "backward": True,
            "any_frame": False,
        }
        assert stream.diagnostics.status is NativeVideoStatus.RANGE_END
        assert stream.diagnostics.decoded_frames == 3
        assert stream.diagnostics.closed
    assert container.closed and container.generator_closed and decoder.readers[0].handle.closed


def test_fractional_tick_seek_rounds_backward_and_filters_forward(decoder):
    with NativeVideoStream(decoder.source, NativeVideoConfig(start=Fraction(10081, 2000))) as stream:
        assert decoder.containers[0].seek_calls[0][0] == 5040
        assert [frame.pts for frame in stream] == [5100]
        assert stream.diagnostics.status is NativeVideoStatus.EOF


def test_stride_indexes_decode_work_not_interpolated_frame_positions(decoder):
    with NativeVideoStream(decoder.source, NativeVideoConfig(frame_step=2)) as stream:
        records = list(stream)
        assert [frame.pts for frame in records] == [5000, 5100]
        assert [frame.decode_index for frame in records] == [0, 2]
        assert [frame.sample_index for frame in records] == [0, 1]
        assert stream.diagnostics.returned_frames == 2 and stream.diagnostics.decoded_frames == 3


def test_seek_reopens_and_resets_only_decode_generation_not_lifetime_budgets(decoder):
    with NativeVideoStream(decoder.source, NativeVideoConfig(max_frames=4)) as stream:
        first = next(stream)
        stream.seek(Fraction(126, 25))
        assert decoder.containers[0].closed and decoder.readers[0].handle.closed
        records = list(stream)
        assert [record.pts for record in records] == [5040, 5100]
        assert records[0].decode_index == 1 and records[0].generation == 1
        assert records[0].sample_index == 1
        stream.seek(None)
        final = next(stream)
        assert final.pts == first.pts and final.generation == 2 and final.sample_index == 3
        assert stream.diagnostics.status is NativeVideoStatus.FRAME_LIMIT
        assert stream.diagnostics.closed
        with pytest.raises(ScanError, match="budget"):
            stream.seek(None)


def test_images_and_metrics_are_detached_from_native_resources(decoder):
    with NativeVideoStream(decoder.source) as stream:
        frame = next(stream)
    assert decoder.containers[0].closed
    with frame.image() as image:
        image.putpixel((0, 0), (0, 0, 0))
    with frame.image() as image:
        assert image.getpixel((0, 0)) == (255, 0, 0)
    assert frame.measure().mean_red == 1 and frame.measure().mean_green == 0
    assert frame.to_dict()["presentation_time"] == {"numerator": 5, "denominator": 1}
    with pytest.raises(FrozenInstanceError):
        frame.pts = 1


@pytest.mark.parametrize(
    "frames,match",
    [
        ([FakeFrame(None)], "lacks PTS"),
        ([FakeFrame(time_base=None)], "lacks PTS"),
        ([FakeFrame(5000), FakeFrame(4999)], "decreased"),
        ([FakeFrame(1.5)], "integer"),
        ([FakeFrame(time_base=Fraction(0))], "rational"),
    ],
)
def test_missing_invalid_or_decreasing_timestamps_fail_without_interpolation(decoder, frames, match):
    decoder.frames[:] = frames
    stream = NativeVideoStream(decoder.source)
    with pytest.raises((ScanError, ConfigurationError), match=match), stream:
        list(stream)
    assert stream.diagnostics.status is NativeVideoStatus.ERROR
    assert decoder.containers[0].closed and decoder.readers[0].handle.closed


def test_equal_pts_and_negative_absolute_pts_are_not_retimed(decoder):
    decoder.frames[:] = [FakeFrame(-10), FakeFrame(-10), FakeFrame(0)]
    with NativeVideoStream(decoder.source) as stream:
        assert [f.presentation_time for f in stream] == [Fraction(-1, 100), Fraction(-1, 100), 0]


@pytest.mark.parametrize(
    "config,expected,decoded",
    [
        (NativeVideoConfig(max_frames=1), NativeVideoStatus.FRAME_LIMIT, 1),
        (NativeVideoConfig(max_decoded_frames=1), NativeVideoStatus.DECODE_LIMIT, 1),
        (NativeVideoConfig(frame_step=10, max_decoded_frames=2), NativeVideoStatus.DECODE_LIMIT, 2),
    ],
)
def test_limits_close_without_reading_an_extra_frame(decoder, config, expected, decoded):
    with NativeVideoStream(decoder.source, config) as stream:
        list(stream)
        assert stream.diagnostics.status is expected and stream.diagnostics.closed
        assert decoder.containers[0].decoded == decoded


def test_pixel_budget_checks_before_rgb_conversion(decoder):
    stream = NativeVideoStream(decoder.source, NativeVideoConfig(max_total_pixels=4))
    with pytest.raises(ScanError, match="total limit"), stream:
        list(stream)
    assert stream.diagnostics.status is NativeVideoStatus.PIXEL_LIMIT
    assert stream.diagnostics.decoded_pixels_observed == 8
    assert not decoder.frames[1].converted
    decoder.frames[:] = [FakeFrame(width=100, height=100)]
    stream = NativeVideoStream(decoder.source, NativeVideoConfig(max_frame_pixels=4))
    with pytest.raises(ScanError, match="pixel limit"), stream:
        next(stream)
    assert not decoder.frames[0].converted and stream.diagnostics.closed


@pytest.mark.parametrize(
    "failure,status",
    [
        (ValueError("codec failure"), NativeVideoStatus.ERROR),
        (KeyboardInterrupt(), NativeVideoStatus.INTERRUPTED),
    ],
)
def test_decoder_failure_or_interruption_closes_every_owned_resource(decoder, failure, status):
    decoder.frames[:] = [failure]
    stream = NativeVideoStream(decoder.source)
    with pytest.raises(ScanError if isinstance(failure, Exception) else KeyboardInterrupt), stream:
        next(stream)
    assert stream.diagnostics.status is status
    assert decoder.containers[0].closed and decoder.readers[0].handle.closed


def test_context_body_interruption_and_explicit_early_close(decoder):
    stream = NativeVideoStream(decoder.source)
    with pytest.raises(RuntimeError), stream:
        next(stream)
        raise RuntimeError("consumer interrupted")
    assert stream.diagnostics.status is NativeVideoStatus.INTERRUPTED
    with NativeVideoStream(decoder.source) as stream:
        stream.close()
        assert stream.diagnostics.status is NativeVideoStatus.CLOSED
        assert list(stream) == []
        with pytest.raises(ConfigurationError, match="cannot seek"):
            stream.seek(None)


def test_context_required_single_entry_and_single_thread_owner(decoder):
    stream = NativeVideoStream(decoder.source)
    with pytest.raises(ConfigurationError, match="metadata"):
        _ = stream.metadata
    with pytest.raises(ConfigurationError, match="context"):
        next(stream)
    with pytest.raises(ConfigurationError, match="context"):
        iter(stream)
    with stream:
        with ThreadPoolExecutor(1) as executor, pytest.raises(ConfigurationError, match="single-owner"):
            executor.submit(lambda: next(stream)).result()
        with pytest.raises(ConfigurationError, match="twice"):
            stream.__enter__()
        assert stream._owner == threading.get_ident()
    with pytest.raises(ConfigurationError, match="cannot seek"):
        stream.seek(None)


def test_secondary_opens_are_denied_and_container_format_is_forced(decoder):
    with NativeVideoStream(decoder.source):
        arguments = decoder.calls[0]
        assert arguments["format"] == "matroska"
        assert arguments["options"] == {
            "protocol_whitelist": "file",
            "enable_drefs": "0",
            "use_absolute_path": "0",
        }
        for target in ("http://example.invalid/a", "file:/secret", "/other/file"):
            with pytest.raises(PermissionError, match="secondary"):
                arguments["io_open"](target, 0, {})


@pytest.mark.parametrize(
    "value",
    [
        "https://example.invalid/video",
        "file:/tmp/video",
        "pipe:0",
        "concat:a|b",
        "//server/share",
        "\\\\server\\share",
        None,
    ],
)
def test_primary_protocol_and_network_paths_are_refused_without_native_open(value):
    with pytest.raises(ConfigurationError):
        NativeVideoStream(value)


@pytest.mark.parametrize(
    "payload", [b"#EXTM3U\nhttps://example.invalid/a.ts", b"ffconcat version 1.0\nfile secret", b"garbage"]
)
def test_real_playlist_and_unknown_file_signatures_never_reach_pyav(decoder, payload):
    decoder.source.write_bytes(payload)
    stream = NativeVideoStream(decoder.source)
    with pytest.raises(ScanError, match="supports only"), stream:
        pass
    assert not decoder.containers and stream.diagnostics.closed


def test_source_and_metadata_limits_release_setup_resources(decoder):
    stream = NativeVideoStream(decoder.source, NativeVideoConfig(max_source_bytes=4))
    with pytest.raises(ScanError, match="byte limit"), stream:
        pass
    assert stream.diagnostics.status is NativeVideoStatus.SOURCE_LIMIT and stream._file is None
    stream = NativeVideoStream(decoder.source, NativeVideoConfig(max_frame_pixels=3))
    with pytest.raises(ScanError, match="pixel limit"), stream:
        pass
    assert decoder.containers[0].closed and decoder.readers[0].handle.closed
    stream = NativeVideoStream(decoder.source, NativeVideoConfig(video_stream=1))
    with pytest.raises(ScanError, match="does not exist"), stream:
        pass
    assert decoder.containers[-1].closed


def test_changed_source_replay_and_bounded_reader_fail_closed(decoder):
    with NativeVideoStream(decoder.source) as stream:
        reader = decoder.readers[0]
        assert reader.readable() and reader.seekable()
        assert reader.seek(0) == reader.tell() == 0
        assert len(reader.read()) == 16 and reader.read() == b""
        decoder.source.write_bytes(b"\x1aE\xdf\xa3" + b"1" * 20)
        with pytest.raises(OSError, match="changed"):
            reader.read(1)
        with pytest.raises(ScanError, match="changed"):
            stream.seek(None)
        assert stream.diagnostics.status is NativeVideoStatus.ERROR


def test_optional_import_failure_and_setup_native_failure_are_clear(decoder, monkeypatch):
    monkeypatch.setattr(native, "_load_av", lambda: (_ for _ in ()).throw(ConfigurationError("video extra")))
    with pytest.raises(ConfigurationError, match="extra"), NativeVideoStream(decoder.source):
        pass
    monkeypatch.setattr(
        native,
        "_load_av",
        lambda: SimpleNamespace(open=lambda *a, **k: (_ for _ in ()).throw(ValueError("bad codec"))),
    )
    stream = NativeVideoStream(decoder.source)
    with pytest.raises(ScanError, match="setup failed"), stream:
        pass
    assert stream._file is None and stream.diagnostics.closed


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_frames": True},
        {"max_frame_pixels": 0},
        {"max_decoded_frames": 10_000_001},
        {"start": 1.5},
        {"start": Fraction(1), "end": Fraction(1)},
        {"end": Fraction(1 << 80)},
        {"start": Fraction(1, 1 << 80)},
    ],
)
def test_configuration_bounds_are_strict(kwargs):
    with pytest.raises(ConfigurationError):
        NativeVideoConfig(**kwargs)


def test_frame_snapshots_reject_mutable_or_inconsistent_data():
    frame = NativeVideoFrame(1, Fraction(1, 1000), 0, 0, 0, 1, 1, b"abc")
    for changes in (
        {"rgb": bytearray(b"abc")},
        {"rgb": b"a"},
        {"width": 67_108_864, "height": 2},
        {"pts": True},
    ):
        with pytest.raises(ConfigurationError):
            replace(frame, **changes)


def test_missing_optional_av_dependency_is_reported_without_import_side_effects(monkeypatch):
    def unavailable(name):
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(native.importlib, "import_module", unavailable)
    with pytest.raises(ConfigurationError, match=r"frame-quorum\[video\]"):
        native._load_av()


def test_missing_file_and_symlink_refusal_precede_native_setup(decoder, monkeypatch):
    with pytest.raises(ScanError, match="regular"), NativeVideoStream(decoder.source.parent / "missing"):
        pass
    original = native.Path.is_symlink
    monkeypatch.setattr(native.Path, "is_symlink", lambda path: path == decoder.source or original(path))
    with pytest.raises(ScanError, match="symbolic-link"):
        NativeVideoStream(decoder.source)
    assert not decoder.containers


def test_empty_stream_native_metadata_and_invalid_dimensions(decoder, monkeypatch):
    decoder.frames.clear()
    with NativeVideoStream(decoder.source) as stream:
        assert list(stream) == [] and stream.diagnostics.status is NativeVideoStatus.EOF
        assert stream.metadata.to_dict()["duration_pts"] == 200
    original = native._load_av().open

    def unknown_metadata(reader, **kwargs):
        container = original(reader, **kwargs)
        container.streams.video[0].average_rate = None
        container.streams.video[0].base_rate = None
        container.streams.video[0].start_time = None
        container.streams.video[0].duration = None
        return container

    monkeypatch.setattr(native, "_load_av", lambda: SimpleNamespace(open=unknown_metadata))
    with NativeVideoStream(decoder.source) as stream:
        document = stream.metadata.to_dict()
        assert document["average_rate"] is document["base_rate"] is None

    def invalid_dimensions(reader, **kwargs):
        container = original(reader, **kwargs)
        container.streams.video[0].codec_context.width = 0
        return container

    monkeypatch.setattr(native, "_load_av", lambda: SimpleNamespace(open=invalid_dimensions))
    stream = NativeVideoStream(decoder.source)
    with pytest.raises(ScanError, match="dimensions"), stream:
        pass
    assert stream.diagnostics.closed


def test_reentrant_decode_and_nonconfig_input_are_refused(decoder):
    with pytest.raises(ConfigurationError, match="config"):
        NativeVideoStream(decoder.source, {})
    stream = NativeVideoStream(decoder.source)

    def reentrant():
        return next(stream)

    decoder.frames[0].to_image = reentrant
    with pytest.raises(ConfigurationError, match="reentrant"), stream:
        next(stream)
    assert stream.diagnostics.closed


def test_closed_before_enter_and_seek_config_validation(decoder):
    stream = NativeVideoStream(decoder.source)
    stream.close()
    with pytest.raises(ConfigurationError, match="twice"):
        stream.__enter__()
    with NativeVideoStream(decoder.source) as stream:
        with pytest.raises(ConfigurationError, match="greater"):
            stream.seek(Fraction(1), end=Fraction(0))
        assert next(stream).pts == 5000


class FailingClose:
    def __init__(self, resource, failure=None):
        self.resource = resource
        self.failure = RuntimeError("close failed") if failure is None else failure
        self.fail = True
        self.calls = 0

    def __next__(self):
        return next(self.resource)

    def close(self):
        self.calls += 1
        if self.fail:
            raise self.failure
        self.resource.close()


@pytest.mark.parametrize(
    "attributes", [("_frames",), ("_container",), ("_file",), ("_frames", "_container", "_file")]
)
def test_cleanup_failure_attempts_every_resource_and_retains_unknown_handles_for_retry(decoder, attributes):
    with NativeVideoStream(decoder.source) as stream:
        next(stream)
        failed = {}
        for attribute in attributes:
            wrapped = FailingClose(getattr(stream, attribute))
            setattr(stream, attribute, wrapped)
            failed[attribute] = wrapped
        with pytest.raises(ScanError, match="cleanup failed"):
            stream.close()
        assert stream.diagnostics.status is NativeVideoStatus.ERROR
        assert not stream.diagnostics.closed
        assert stream.diagnostics.cleanup_errors == tuple(
            attribute[1:] + ".close" for attribute in attributes
        )
        for attribute in ("_frames", "_container", "_file"):
            if attribute not in attributes:
                assert getattr(stream, attribute) is None
        for attribute, wrapped in failed.items():
            assert getattr(stream, attribute) is wrapped and wrapped.calls == 1
            wrapped.fail = False
        stream.close()
        assert stream.diagnostics.closed
        assert stream.diagnostics.cleanup_errors
        assert all(wrapped.calls == 2 for wrapped in failed.values())
    assert decoder.readers[0].handle.closed


def test_cleanup_failure_does_not_replace_active_decode_error(decoder):
    decoder.frames[:] = [ValueError("primary decode failure")]
    with NativeVideoStream(decoder.source) as stream:
        wrapped = FailingClose(stream._container)
        stream._container = wrapped
        with pytest.raises(ScanError, match="decode failed"):
            next(stream)
        assert stream.diagnostics.cleanup_errors == ("container.close",)
        assert not stream.diagnostics.closed and decoder.readers[0].handle.closed
        wrapped.fail = False
        stream.close()


def test_eof_cleanup_failure_is_error_not_successful_eof(decoder):
    decoder.frames.clear()
    with NativeVideoStream(decoder.source) as stream:
        wrapped = FailingClose(stream._frames)
        stream._frames = wrapped
        with pytest.raises(ScanError, match="cleanup failed"):
            next(stream)
        assert stream.diagnostics.status is NativeVideoStatus.ERROR
        assert not stream.diagnostics.closed and stream._container is None and stream._file is None
        wrapped.fail = False
        stream.close()


def test_genuine_interrupt_from_cleanup_is_not_swallowed(decoder):
    with NativeVideoStream(decoder.source) as stream:
        wrapped = FailingClose(stream._container, KeyboardInterrupt())
        stream._container = wrapped
        with pytest.raises(KeyboardInterrupt):
            stream.close()
        assert stream.diagnostics.status is NativeVideoStatus.INTERRUPTED
        assert not stream.diagnostics.closed and decoder.readers[0].handle.closed
        wrapped.fail = False
        stream.close()


@pytest.mark.parametrize(
    "changes",
    [
        {"path": []},
        {"format_name": []},
        {"codec_name": None},
        {"width": 67_108_864, "height": 2},
        {"time_base": []},
        {"duration_pts": -1},
    ],
)
def test_public_metadata_rejects_mutable_or_invalid_fields(decoder, changes):
    with NativeVideoStream(decoder.source) as stream, pytest.raises(ConfigurationError):
        replace(stream.metadata, **changes)


@pytest.mark.parametrize(
    "changes",
    [
        {"status": "eof"},
        {"closed": 1},
        {"returned_frames": 1},
        {"decoded_frames": True},
        {"cleanup_errors": []},
        {"cleanup_errors": ("unknown",)},
        {"cleanup_errors": ("file.close", "file.close")},
    ],
)
def test_public_diagnostics_are_strict_immutable_snapshots(changes):
    base = NativeVideoDiagnostics(NativeVideoStatus.NEW, 0, 0, 0, 0, True)
    with pytest.raises(ConfigurationError):
        replace(base, **changes)


def test_cli_decode_failure_leaves_prefix_without_success_summary(decoder, capsys):
    decoder.frames[:] = [FakeFrame(5000), ValueError("decode failed")]
    assert main(["native-scan", str(decoder.source)]) == 2
    output = capsys.readouterr()
    records = [json.loads(line) for line in output.out.splitlines()]
    assert [record["record_type"] for record in records] == ["source", "frame"]
    assert records[1]["pts"] == 5000
    assert "decode failed" in output.err
    assert decoder.readers[0].handle.closed and decoder.containers[0].closed


@pytest.mark.parametrize(
    "value",
    [
        "1e1000000000",
        "1e-1000000000",
        "0e99999999999999999999",
        "1e1_000",
        "1e+1_000",
        "1e-1_000",
        "1E1000  ",
        "1e1000\n",
    ],
)
def test_cli_rejects_enormous_decimal_exponent_before_fraction_materialization(
    decoder, capsys, monkeypatch, value
):
    def forbidden_fraction(text):
        raise AssertionError("oversized exponent reached Fraction construction")

    monkeypatch.setattr("frame_quorum.cli.Fraction", forbidden_fraction)
    with pytest.raises(SystemExit) as error:
        main(["native-scan", str(decoder.source), "--start", value])
    assert error.value.code == 2
    assert "bounded exact" in capsys.readouterr().err
    assert not decoder.containers


def test_local_spelling_resolving_to_unc_is_rejected_before_file_setup(decoder, monkeypatch):
    monkeypatch.setattr(native.Path, "resolve", lambda path: native.Path("//server/share/video.mkv"))
    with pytest.raises(ConfigurationError, match="UNC"):
        NativeVideoStream(decoder.source)
    assert not decoder.readers and not decoder.containers
