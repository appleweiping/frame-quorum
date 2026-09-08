from __future__ import annotations

import errno
import hashlib
import io
import os
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import pytest

import frame_quorum.native_splitting as module
from frame_quorum import NativeClip, NativeSplitConfig, NativeSplitResult, NativeVideoFrame
from frame_quorum.errors import ConfigurationError, OutputError, ScanError
from frame_quorum.native_video import NativeVideoStatus, _format


def frame(pts=10, *, tb=Fraction(1, 30), width=2, height=2):
    return NativeVideoFrame(pts, tb, 0, 0, 0, width, height, bytes(width * height * 3))


def ownership(stage):
    return {
        path: module._identity(path.lstat()) for path in stage.iterdir() if path.is_file()
    }, module._identity(stage.lstat())


@pytest.mark.parametrize("start,end", [(0.0, 1), (True, 1), (0, False), (0, 0), (1, 0), (0, 10**1000)])
def test_clip_rejects_inexact_unbounded_or_empty_times(start, end):
    with pytest.raises(ConfigurationError):
        NativeClip(start, end)


def test_clip_normalizes_exact_ints_and_negative_native_times():
    assert NativeClip(-1, 2).start == Fraction(-1)
    assert type(NativeClip(0, 1).end) is Fraction


@pytest.mark.parametrize("name", NativeSplitConfig.__dataclass_fields__)
@pytest.mark.parametrize("value", [True, -1, 10**1000, float("nan"), "1"])
def test_config_strict_bounds(name, value):
    with pytest.raises(ConfigurationError):
        NativeSplitConfig(**{name: value})


@pytest.mark.parametrize(
    "field,value",
    [
        ("output_dir", "bad"),
        ("clip_count", 0),
        ("frame_count", 0),
        ("total_output_bytes", 0),
        ("manifest_sha256", "f" * 63),
        ("manifest_sha256", "G" * 64),
        ("manifest_sha256", None),
    ],
)
def test_result_validation(tmp_path, field, value):
    with pytest.raises(ConfigurationError):
        replace(NativeSplitResult(tmp_path, 1, 1, 1, "a" * 64), **{field: value})


@pytest.mark.parametrize(
    "clips", [[], (), (object(),), (NativeClip(1, 2), NativeClip(0, 1)), (NativeClip(0, 2), NativeClip(1, 3))]
)
def test_invalid_clips_fail_before_native_loading(tmp_path, monkeypatch, clips):
    monkeypatch.setattr(module, "_load_av", lambda: pytest.fail("must validate first"))
    with pytest.raises(ConfigurationError):
        module.split_native_video("missing", tmp_path / "clips", clips)
    assert list(tmp_path.iterdir()) == []


def test_wrong_config_and_clip_limit_preflight(tmp_path):
    for config in (object(), False):
        with pytest.raises(ConfigurationError, match="config"):
            module.split_native_video("missing", tmp_path / "clips", (NativeClip(0, 1),), config)
    with pytest.raises(ConfigurationError, match="bounded"):
        module.split_native_video(
            "missing", tmp_path / "clips", (NativeClip(0, 1),) * 2, NativeSplitConfig(max_clips=1)
        )


@pytest.mark.parametrize("target", [True, "https://example.test/clips", "//remote/clips", "bad\nname"])
def test_target_rejects_nonlocal_or_unsafe_paths(target):
    with pytest.raises(ConfigurationError):
        module._target_path(target)


def test_target_requires_new_leaf_existing_parent(tmp_path, monkeypatch):
    with pytest.raises(OutputError, match="existing output parent"):
        module._target_path(tmp_path / "missing" / "clips")
    with pytest.raises(OutputError, match="new output"):
        module._target_path(tmp_path)
    monkeypatch.setattr(Path, "is_symlink", lambda self: True)
    with pytest.raises(OutputError, match="symbolic-link"):
        module._target_path(tmp_path / "clips")


def test_writer_growth_overwrite_sparse_seek_and_aggregate_budget():
    budget = module._ByteBudget(10)
    handle = io.BytesIO()
    writer = module._Writer(handle, budget, 8)
    assert writer.writable() and writer.seekable()
    writer.write(b"abcd")
    assert writer.seek(-2, os.SEEK_CUR) == 2
    writer.write(b"xy")
    assert writer.tell() == 4 and budget.total == 4
    assert writer.seek(2, os.SEEK_END) == 6
    writer.write(b"zz")
    assert handle.getvalue() == b"abxy\0\0zz" and budget.total == 8
    second = module._Writer(io.BytesIO(), budget, 8)
    second.write(b"ok")
    with pytest.raises(OutputError, match="total output"):
        second.write(b"!")
    assert second.failed and budget.total == 10
    with pytest.raises(OutputError, match="file exceeds"):
        writer.write(b"!")
    assert writer.failed


@pytest.mark.parametrize("offset,whence", [(-1, os.SEEK_SET), (9, os.SEEK_SET), (0, 999)])
def test_writer_rejects_bad_seek_and_marks_failure(offset, whence):
    writer = module._Writer(io.BytesIO(), module._ByteBudget(8), 8)
    with pytest.raises((OSError, OutputError)):
        writer.seek(offset, whence)
    assert writer.failed


def test_writer_short_write_marks_failure():
    class Short(io.BytesIO):
        def write(self, data):
            return super().write(data[:1])

    writer = module._Writer(Short(), module._ByteBudget(8), 8)
    with pytest.raises(OutputError, match="short"):
        writer.write(b"abc")
    assert writer.failed


@pytest.mark.parametrize("primary", [None, ValueError("operation"), KeyboardInterrupt("operation")])
def test_close_attempts_every_resource_and_preserves_primary(primary):
    seen = []

    class Resource:
        def close(self):
            seen.append(1)
            raise OSError("close")

    if primary is None:
        with pytest.raises(OutputError, match="cleanup"):
            module._close_all((("one", Resource()), ("two", Resource()), ("none", None)))
    else:
        module._close_all((("one", Resource()), ("two", Resource())), primary)
        assert "one, two" in primary.__notes__[0]
    assert seen == [1, 1]


def test_cleanup_control_is_not_swallowed_and_later_resources_close():
    seen = []

    class Stop:
        def close(self):
            raise SystemExit(12)

    with pytest.raises(SystemExit) as caught:
        module._close_all(
            (("stop", Stop()), ("later", SimpleNamespace(close=lambda: seen.append(1)))), ValueError()
        )
    assert caught.value.code == 12 and seen == [1]


def test_hash_stream_is_bounded(tmp_path):
    path = tmp_path / "clip.nut"
    path.write_bytes(b"abc")
    assert module._hash_file(path, 3) == hashlib.sha256(b"abc").hexdigest()
    with pytest.raises(OutputError, match="byte limit"):
        module._hash_file(path, 2)


def test_nut_magic_requires_complete_signature():
    assert _format(b"nut/multimedia container\0" + b"tail") == "nut"
    for partial in (b"nut", b"nut/multimedia container", b"nut/multimedia containerX"):
        with pytest.raises(ScanError):
            _format(partial)


def test_publish_actual_platform_never_replaces_competing_target(tmp_path):
    stage, target = tmp_path / "stage", tmp_path / "target"
    stage.mkdir()
    (stage / "ours").write_bytes(b"ours")
    target.mkdir()
    with pytest.raises(OSError):
        module._publish(stage, target)
    assert list(target.iterdir()) == [] and (stage / "ours").read_bytes() == b"ours"


def test_publish_linux_wrapper_checked_arguments_and_errno(tmp_path, monkeypatch):
    stage, target = tmp_path / "stage", tmp_path / "target"
    calls = []

    class Rename:
        def __call__(self, *args):
            calls.append(args)
            return -1

    monkeypatch.setattr(module.sys, "platform", "linux")
    monkeypatch.setattr(module.ctypes, "CDLL", lambda *args, **kwargs: SimpleNamespace(renameat2=Rename()))
    monkeypatch.setattr(module.ctypes, "get_errno", lambda: errno.EEXIST)
    with pytest.raises(FileExistsError):
        module._publish(stage, target)
    assert calls == [(-100, os.fsencode(stage), -100, os.fsencode(target), 1)]
    monkeypatch.setattr(module.ctypes, "CDLL", lambda *args, **kwargs: SimpleNamespace())
    with pytest.raises(OutputError, match="unavailable"):
        module._publish(stage, target)
    monkeypatch.setattr(module.sys, "platform", "darwin")
    with pytest.raises(OutputError, match="only Windows and Linux"):
        module._publish(stage, target)


def test_cleanup_refuses_nested_foreign_directory_and_reports_residue(tmp_path):
    stage = tmp_path / "owned"
    stage.mkdir()
    (stage / "clip.nut").write_bytes(b"owned")
    owned, identity = ownership(stage)
    foreign = stage / "foreign"
    foreign.mkdir()
    (foreign / "retain").write_bytes(b"never remove")
    with pytest.raises(OutputError, match="cleanup incomplete") as caught:
        module._cleanup(stage, ValueError("operation"), owned, identity)
    assert str(stage) in str(caught.value)
    assert not (stage / "clip.nut").exists()
    assert (foreign / "retain").read_bytes() == b"never remove"


def test_cleanup_ordinary_error_preserves_interruption_with_residue_note(tmp_path, monkeypatch):
    stage = tmp_path / "owned"
    stage.mkdir()
    (stage / "clip.nut").write_bytes(b"owned")
    owned, identity = ownership(stage)
    primary = KeyboardInterrupt("operation")
    monkeypatch.setattr(Path, "unlink", lambda self: (_ for _ in ()).throw(OSError("denied")))
    module._cleanup(stage, primary, owned, identity)
    assert str(stage) in primary.__notes__[0]


@pytest.mark.parametrize("control", [KeyboardInterrupt, SystemExit])
def test_cleanup_propagates_control_and_still_attempts_directory_removal(tmp_path, monkeypatch, control):
    stage = tmp_path / "owned"
    stage.mkdir()
    (stage / "clip.nut").write_bytes(b"owned")
    owned, identity = ownership(stage)
    calls = []
    monkeypatch.setattr(Path, "unlink", lambda self: (_ for _ in ()).throw(control()))
    monkeypatch.setattr(Path, "rmdir", lambda self: calls.append(self))
    with pytest.raises(control):
        module._cleanup(stage, ValueError(), owned, identity)
    assert calls == [stage]


def test_cleanup_unvalidated_identity_is_not_removed(tmp_path, monkeypatch):
    stage = tmp_path / "owned"
    stage.mkdir()
    owned, identity = ownership(stage)
    monkeypatch.setattr(Path, "lstat", lambda self: (_ for _ in ()).throw(OSError("stat")))
    with pytest.raises(OutputError, match="validate staging directory identity"):
        module._cleanup(stage, ValueError(), owned, identity)
    assert stage.exists()


def test_cleanup_preserves_unknown_regular_file_and_replaced_owned_file(tmp_path):
    stage = tmp_path / "owned"
    stage.mkdir()
    path = stage / "clip.nut"
    path.write_bytes(b"ours")
    owned, identity = ownership(stage)
    path.rename(stage / "original-moved")
    path.write_bytes(b"replacement")
    unknown = stage / "not-created-by-us"
    unknown.write_bytes(b"preserve")
    with pytest.raises(OutputError, match="cleanup incomplete"):
        module._cleanup(stage, ValueError(), owned, identity)
    assert path.read_bytes() == b"replacement" and unknown.read_bytes() == b"preserve"
    assert (stage / "original-moved").read_bytes() == b"ours"


def test_cleanup_preserves_replacement_stage(tmp_path):
    stage = tmp_path / "owned"
    stage.mkdir()
    path = stage / "clip.nut"
    path.write_bytes(b"ours")
    owned, identity = ownership(stage)
    stage.rename(tmp_path / "moved")
    stage.mkdir()
    path.write_bytes(b"replacement")
    with pytest.raises(OutputError, match="identity"):
        module._cleanup(stage, ValueError(), owned, identity)
    assert path.read_bytes() == b"replacement"


def test_cleanup_missing_owned_file_is_harmless(tmp_path):
    stage = tmp_path / "owned"
    stage.mkdir()
    path = stage / "clip.nut"
    path.write_bytes(b"ours")
    owned, identity = ownership(stage)
    path.unlink()
    module._cleanup(stage, ValueError(), owned, identity)
    assert not stage.exists()


@pytest.mark.parametrize("control", [KeyboardInterrupt, SystemExit])
def test_hash_read_control_survives_file_close_failure(tmp_path, monkeypatch, control):
    class Handle(io.BytesIO):
        def read(self, *args):
            raise control("read")

        def close(self):
            super().close()
            raise OSError("close")

    handle = Handle(b"value")
    monkeypatch.setattr(Path, "open", lambda *args: handle)
    with pytest.raises(control, match="read") as caught:
        module._hash_file(tmp_path / "value", 10)
    assert handle.closed and "file.close" in caught.value.__notes__[0]


def test_managed_manifest_control_survives_file_close_failure(tmp_path, monkeypatch):
    class Handle(io.BytesIO):
        def close(self):
            super().close()
            raise OSError("close")

    handle = Handle()
    monkeypatch.setattr(module, "_open_new", lambda *args: handle)
    with (
        pytest.raises(KeyboardInterrupt, match="serialize") as caught,
        module._managed_file(tmp_path / "manifest", {}),
    ):
        raise KeyboardInterrupt("serialize")
    assert handle.closed and "file.close" in caught.value.__notes__[0]


def test_created_file_identity_failure_closes_handle_reports_unowned_residue(tmp_path, monkeypatch):
    path = tmp_path / "clip.nut"
    monkeypatch.setattr(module.os, "fstat", lambda *args: (_ for _ in ()).throw(KeyboardInterrupt()))
    owned = {}
    with pytest.raises(KeyboardInterrupt):
        module._open_new(path, owned)
    assert not owned
    path.unlink()


@pytest.fixture
def encoder(tmp_path):
    calls = []

    class Container:
        def add_stream(self, name):
            assert name == "ffv1"
            self.stream = SimpleNamespace(codec_context=SimpleNamespace(), encode=self.encode)
            return self.stream

        def encode(self, encoded=None):
            calls.append(encoded)
            return [b"packet"]

        def mux(self, packet):
            self.writer.write(packet)

        def close(self):
            calls.append("closed")

    container = Container()

    def open_container(writer, *args, **kwargs):
        assert kwargs["format"] == "nut"
        container.writer = writer
        return container

    av = SimpleNamespace(
        open=open_container, VideoFrame=SimpleNamespace(from_image=lambda image: SimpleNamespace())
    )
    instance = module._Encoder(tmp_path / "clip.nut", frame(), module._ByteBudget(100), av)
    yield instance, calls, container
    instance.close()


def test_encoder_exact_ticks_state_and_footer(encoder):
    instance, calls, container = encoder
    instance.write(frame())
    instance.write(frame(13))
    assert [(item.pts, item.time_base) for item in calls] == [(0, Fraction(1, 30)), (3, Fraction(1, 30))]
    instance.finish()
    assert instance.handle.closed and calls[-2:] == [None, "closed"]
    assert container.stream.pix_fmt == "bgr0" and container.stream.codec_context.thread_count == 1


@pytest.mark.parametrize(
    "changed,match",
    [
        (frame(width=3), "dimensions"),
        (frame(10), "strictly increasing"),
        (frame(21, tb=Fraction(1, 60)), "representable"),
        (frame(2**63 - 1, tb=Fraction(1)), "rebased output pts"),
    ],
)
def test_encoder_rejects_unsupported_frame_before_encoding(encoder, changed, match):
    instance, calls, _ = encoder
    instance.write(frame())
    before = len(calls)
    with pytest.raises((ScanError, ConfigurationError), match=match):
        instance.write(changed)
    assert len(calls) == before


def test_encoder_tick_ceiling_before_file_create(tmp_path):
    path = tmp_path / "clip.nut"
    with pytest.raises(ConfigurationError, match="tick denominator"):
        module._Encoder(path, frame(tb=Fraction(1, 1_000_000_001)), module._ByteBudget(10), None)
    assert not path.exists()


def test_encoder_constructor_failure_closes_file(tmp_path):
    path = tmp_path / "clip.nut"
    av = SimpleNamespace(open=lambda *args, **kwargs: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        module._Encoder(path, frame(), module._ByteBudget(10), av)
    path.unlink()


@pytest.mark.parametrize("phase", ["frame", "footer"])
def test_swallowed_writer_error_still_prevents_success(encoder, phase):
    instance, _, _ = encoder
    instance.writer.failed = True
    with pytest.raises(OutputError, match="writer failed"):
        instance.write(frame()) if phase == "frame" else instance.finish()


def test_footer_primary_is_preserved_and_file_is_closed(encoder, monkeypatch):
    instance, _, container = encoder
    monkeypatch.setattr(container.stream, "encode", lambda: (_ for _ in ()).throw(KeyboardInterrupt("flush")))
    with pytest.raises(KeyboardInterrupt, match="flush"):
        instance.finish()
    assert instance.handle.closed


@pytest.mark.parametrize(
    "change", ["codec", "container", "extra", "missing", "pts", "rgb", "dimensions", "truncated", "no_budget"]
)
def test_verification_independent_expected_records_reject_each_corruption(monkeypatch, change):
    original = frame()
    output = frame(0)
    emitted = [output]
    if change == "extra":
        emitted.append(frame(1))
    elif change == "missing":
        emitted = []
    elif change == "pts":
        emitted = [frame(1)]
    elif change == "rgb":
        emitted = [replace(output, rgb=b"x" * 12)]
    elif change == "dimensions":
        emitted = [frame(0, width=1)]

    class Stream:
        metadata = SimpleNamespace(
            codec_name="wrong" if change == "codec" else "ffv1",
            format_name="wrong" if change == "container" else "nut",
        )
        diagnostics = SimpleNamespace(
            status=NativeVideoStatus.FRAME_LIMIT if change == "truncated" else NativeVideoStatus.EOF,
            decoded_pixels_observed=4,
        )

        def __init__(self, *args):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def __iter__(self):
            return iter(emitted)

    monkeypatch.setattr(module, "NativeVideoStream", Stream)
    encoder = SimpleNamespace(
        path=Path("clip.nut"),
        origin=original.presentation_time,
        expected=[
            module._Expected(
                original.pts, original.time_base, 0, 2, 2, hashlib.sha256(original.rgb).hexdigest()
            )
        ],
    )
    with pytest.raises(OutputError):
        module._verify(encoder, NativeSplitConfig(), 0 if change == "no_budget" else 10)
