from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, replace
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import pytest

import frame_quorum
import frame_quorum.native_concat as concat
import frame_quorum.native_video as native
from frame_quorum import (
    NativeConcatClip,
    NativeConcatConfig,
    NativeConcatLimits,
    NativeConcatSpan,
    NativeConcatStream,
    NativeConcatTimeline,
)
from frame_quorum.errors import ConfigurationError, ScanError
from tests.test_native_video import FakeContainer, FakeFrame


@pytest.mark.parametrize(
    "name",
    [
        "NativeConcatClip",
        "NativeConcatLimits",
        "NativeConcatConfig",
        "NativeConcatTimeline",
        "NativeConcatFrame",
        "NativeConcatSpan",
        "NativeConcatDiagnostics",
        "NativeConcatStream",
    ],
)
def test_public_native_concat_api_exists(name):
    assert hasattr(frame_quorum, name)
    assert name in frame_quorum.__all__


@pytest.fixture
def decoder(tmp_path, monkeypatch):
    paths = (tmp_path / "a.mkv", tmp_path / "b.mkv")
    for path in paths:
        path.write_bytes(b"\x1aE\xdf\xa3" + b"0" * 12)
    state = SimpleNamespace(
        paths=paths,
        frames=[[FakeFrame(5000), FakeFrame(5040), FakeFrame(5100), FakeFrame(5200)] for _ in paths],
        containers=[],
        readers=[],
        peak=0,
        hook=None,
    )

    def open_container(reader, **kwargs):
        active = sum(not item.closed for item in state.containers)
        assert active == 0, "a second native container was acquired before previous close acknowledgement"
        assert all(item.handle.closed for item in state.readers), "previous source file remains owned"
        index = paths.index(Path(reader.handle.name))
        container = FakeContainer(list(state.frames[index]))
        state.containers.append(container)
        state.readers.append(reader)
        state.peak = max(state.peak, active + 1)
        if state.hook:
            state.hook(container, index)
        return container

    monkeypatch.setattr(native, "_load_av", lambda: SimpleNamespace(open=open_container))
    state.clips = tuple(NativeConcatClip(path, start=5, end=Fraction(26, 5)) for path in paths)
    return state


@pytest.mark.parametrize("mode", ["close", "exit", "body-error", "body-control"])
@pytest.mark.parametrize("nested", ["close", "next", "seek", "enter", "exit"])
def test_public_cleanup_rejects_callback_reentrancy_and_releases_guard(decoder, mode, nested):
    stream = NativeConcatStream(decoder.clips)
    stream.__enter__()
    next(stream)
    container = decoder.containers[-1]
    original = container.close
    observations = []

    def callback_close():
        if not observations:
            observations.append("entered")
            operations = {
                "close": stream.close,
                "next": lambda: next(stream),
                "seek": lambda: stream.seek(0),
                "enter": stream.__enter__,
                "exit": lambda: stream.__exit__(None, None, None),
            }
            with pytest.raises(ConfigurationError, match="reentrant"):
                operations[nested]()
            observations.append("blocked")
        original()

    container.close = callback_close
    try:
        if mode == "close":
            stream.close()
        else:
            primary = (
                ValueError("body")
                if mode == "body-error"
                else KeyboardInterrupt("body")
                if mode == "body-control"
                else None
            )
            stream.__exit__(None if primary is None else type(primary), primary, None)
        assert observations == ["entered", "blocked"]
        assert stream.diagnostics.closed and not stream._busy
        assert stream.diagnostics.status == (
            "error" if mode == "body-error" else "interrupted" if mode == "body-control" else "closed"
        )
    finally:
        container.close = original
        stream.close()


def test_actual_children_are_serially_probed_and_rgb_accounted_across_global_stride(decoder):
    stream = NativeConcatStream(decoder.clips, NativeConcatConfig(frame_step=2))
    assert stream.diagnostics.status == "new" and stream.diagnostics.closed
    with pytest.raises(ConfigurationError, match="metadata"):
        _ = stream.metadata
    with stream:
        assert len(decoder.containers) == 2 and all(c.closed for c in decoder.containers)
        assert all(c.decoded == 0 for c in decoder.containers)
        assert stream.diagnostics.opened_source_bytes == 32
        frames = list(stream)
        assert [(frame.clip_index, frame.native.pts) for frame in frames] == [(0, 5000), (0, 5100), (1, 5040)]
        assert [frame.presentation_time for frame in frames] == [0, Fraction(1, 10), Fraction(6, 25)]
        assert [frame.sample_index for frame in frames] == [0, 1, 2]
        assert stream.diagnostics.source_decoded_frames == (4, 4)
        assert stream.diagnostics.source_rgb_frames == (3, 3)
        assert stream.diagnostics.source_pixels == (16, 16)
        assert stream.diagnostics.status == "intervals_exhausted"
        assert stream.diagnostics.opened_source_bytes == 64
        assert json.loads(json.dumps(stream.diagnostics.to_dict()))["source_activations"] == [2, 2]
        assert stream.map_span(0, Fraction(2, 5)) == stream.timeline.map_span(0, Fraction(2, 5))
    assert decoder.peak == 1
    assert all(c.closed for c in decoder.containers) and all(r.handle.closed for r in decoder.readers)


@pytest.mark.parametrize("value", [None, True, 1.0, "1", Fraction(1, 1 << 64), 1 << 128])
def test_composite_times_reject_inexact_or_unbounded_values(value):
    with pytest.raises(ConfigurationError):
        NativeConcatConfig(start=value)


@pytest.mark.parametrize("start,end", [(0, 0), (2, 1), (-1, None), (0, False)])
def test_config_ranges_are_positive(start, end):
    with pytest.raises(ConfigurationError):
        NativeConcatConfig(start=start, end=end)


@pytest.mark.parametrize(
    "changes",
    [{"frame_step": True}, {"frame_step": 0}, {"frame_step": 1_000_001}, {"limits": {}}, {"limits": None}],
)
def test_config_exact_types_and_step_bounds(changes):
    with pytest.raises(ConfigurationError):
        NativeConcatConfig(**changes)


@pytest.mark.parametrize("field,maximum", concat._BUDGET_FIELDS)
@pytest.mark.parametrize("bad", [0, -1, True, 1.5, "above"])
def test_every_limit_is_positive_exact_integer_and_cannot_raise_hard_cap(field, maximum, bad):
    with pytest.raises(ConfigurationError):
        NativeConcatLimits(**{field: maximum + 1 if bad == "above" else bad})


@pytest.mark.parametrize(
    "changes",
    [
        {"path": None},
        {"path": "x" * 4097},
        {"path": "\ud800"},
        {"path": "https://example.org/video.mkv"},
        {"start": False},
        {"start": 1.5},
        {"start": Fraction(1, 1 << 64)},
        {"start": 1 << 64},
        {"end": 5},
        {"video_stream": True},
        {"video_stream": 1024},
    ],
)
def test_clip_boundaries_and_paths_rejected_before_any_open(decoder, changes):
    options = {"path": decoder.paths[0], "start": 5, "end": 6} | changes
    with pytest.raises(ConfigurationError):
        NativeConcatClip(**options)
    assert not decoder.containers


@pytest.mark.parametrize("clips", [(), [], (None,), ("path",)])
def test_exact_nonempty_manifest_admission(clips):
    with pytest.raises(ConfigurationError):
        NativeConcatTimeline(clips)


def test_manifest_snapshots_exact_arithmetic_digest_and_boundaries(decoder):
    timeline = NativeConcatTimeline(decoder.clips)
    assert timeline.clips == decoder.clips and timeline.clips[0] is not decoder.clips[0]
    assert timeline.offsets == (0, Fraction(1, 5), Fraction(2, 5))
    expected = hashlib.sha256(
        json.dumps(
            timeline.to_dict(), ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(",", ":")
        ).encode()
    ).hexdigest()
    assert timeline.digest == expected
    assert NativeConcatTimeline(tuple(reversed(decoder.clips))).digest != expected
    assert timeline.map_span(Fraction(1, 5), Fraction(1, 5)) == ()
    pieces = timeline.map_span(Fraction(1, 10), Fraction(3, 10))
    assert [(s.clip_index, s.local_start, s.local_end) for s in pieces] == [
        (0, Fraction(51, 10), Fraction(26, 5)),
        (1, 5, Fraction(51, 10)),
    ]
    assert pieces[1].path == decoder.paths[1]
    assert pieces[0].to_dict()["coverage"] == "declared_only"
    with pytest.raises(FrozenInstanceError):
        timeline.digest = "changed"
    with pytest.raises(ConfigurationError, match="piece"):
        timeline.map_span(0, timeline.duration, max_parts=1)
    with pytest.raises(ConfigurationError):
        NativeConcatTimeline(decoder.clips * 65)
    with pytest.raises(ConfigurationError, match="source occurrence"):
        NativeConcatStream(decoder.clips, NativeConcatConfig(limits=NativeConcatLimits(max_sources=1)))
    with pytest.raises(ConfigurationError, match="config"):
        NativeConcatStream(decoder.clips, {})


@pytest.mark.parametrize("start,end", [(-1, 0), (1, 2), (Fraction(3, 10), Fraction(1, 10)), (0.0, 0)])
def test_span_outside_or_inexact_coordinates_rejected(decoder, start, end):
    with pytest.raises(ConfigurationError):
        NativeConcatTimeline(decoder.clips).map_span(start, end)


@pytest.mark.parametrize(
    "changes",
    [
        {"timeline": None},
        {"clip_index": True},
        {"clip_index": 2},
        {"global_start": 0, "global_end": 0},
        {"global_end": 1},
    ],
)
def test_span_cannot_cross_or_escape_one_source(decoder, changes):
    values = {
        "timeline": NativeConcatTimeline(decoder.clips),
        "clip_index": 0,
        "global_start": 0,
        "global_end": Fraction(1, 10),
    } | changes
    with pytest.raises(ConfigurationError):
        NativeConcatSpan(**values)


def test_public_map_and_configuration_are_readonly_and_detached(decoder):
    config = NativeConcatConfig(limits=NativeConcatLimits(max_span_parts=1))
    stream = NativeConcatStream(decoder.clips, config)
    assert stream.config == config and stream.config is not config
    assert stream.config.limits is not config.limits
    for key, value in (("timeline", None), ("config", None)):
        with pytest.raises(AttributeError):
            setattr(stream, key, value)
    with pytest.raises(ConfigurationError, match="piece"):
        stream.map_span(0, Fraction(2, 5))


@pytest.mark.parametrize(
    "start,end", [(1, None), (Fraction(1, 10), 1), (0, 0), (True, None), (Fraction(1, 5), Fraction(1, 10))]
)
def test_invalid_seek_preflight_does_not_close_decode_or_change_generation(decoder, start, end):
    with NativeConcatStream(decoder.clips) as stream:
        assert next(stream).native.pts == 5000
        before = stream.diagnostics
        with pytest.raises(ConfigurationError):
            stream.seek(start, end=end)
        assert stream.diagnostics == before
        assert next(stream).native.pts == 5040


def test_seek_seam_reopens_one_child_and_resets_stride_but_not_lifetime(decoder):
    with NativeConcatStream(decoder.clips, NativeConcatConfig(frame_step=2)) as stream:
        first = next(stream)
        active = decoder.containers[-1]
        stream.seek(Fraction(1, 5))
        assert active.closed
        assert len(decoder.containers) == 4 and not decoder.containers[-1].closed
        frames = list(stream)
        assert [(f.clip_index, f.native.pts, f.generation, f.sample_index) for f in frames] == [
            (1, 5000, 1, 1),
            (1, 5100, 1, 2),
        ]
        assert frames[0].native.generation == 0
        stream.seek(Fraction(2, 5))
        assert stream.diagnostics.closed and stream.diagnostics.activations == 4
        assert list(stream) == []
        stream.reset()
        repeated = next(stream)
        assert repeated.rgb == first.rgb and repeated.generation == 3 and repeated.sample_index == 3
        assert stream.diagnostics.source_activations == (3, 2)


@pytest.mark.parametrize(
    "limit,value,status,scope",
    [
        ("max_frames", 2, "frame_limit", "total"),
        ("max_rgb_frames", 2, "rgb_limit", "total"),
        ("max_source_rgb_frames", 2, "rgb_limit", "source"),
        ("max_decoded_frames", 2, "decode_limit", "total"),
        ("max_source_decoded_frames", 2, "decode_limit", "source"),
        ("max_activations", 2, "activation_limit", "total"),
        ("max_opened_source_bytes", 32, "source_limit", "total"),
    ],
)
def test_each_runtime_budget_is_terminal_not_a_fabricated_source_eof(decoder, limit, value, status, scope):
    with NativeConcatStream(
        decoder.clips, NativeConcatConfig(limits=NativeConcatLimits(**{limit: value}))
    ) as s:
        records = list(s)
        assert s.diagnostics.status == status and s.diagnostics.closed
        assert s.diagnostics.limit_scope == scope
        assert s.diagnostics.returned_frames == len(records)
        with pytest.raises(ConfigurationError):
            s.reset()
    assert len(decoder.containers) <= 3


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_source_bytes", 15),
        ("max_manifest_source_bytes", 31),
        ("max_activations", 1),
        ("max_opened_source_bytes", 16),
    ],
)
def test_setup_limit_closes_every_acquired_resource(decoder, field, value):
    stream = NativeConcatStream(
        decoder.clips, NativeConcatConfig(limits=NativeConcatLimits(**{field: value}))
    )
    with pytest.raises(ScanError), stream:
        pytest.fail("setup must not succeed")
    assert stream.diagnostics.status in ("source_limit", "activation_limit")
    assert stream.diagnostics.closed and all(r.handle.closed for r in decoder.readers)


@pytest.mark.parametrize(
    "field,scope", [("max_total_pixels", "total"), ("max_source_total_pixels", "source")]
)
def test_actual_excess_pixel_observation_is_charged_even_when_native_raises(decoder, field, scope):
    with NativeConcatStream(decoder.clips, NativeConcatConfig(limits=NativeConcatLimits(**{field: 5}))) as s:
        assert next(s).native.pts == 5000
        with pytest.raises(ScanError, match="pixel"):
            next(s)
        assert s.diagnostics.status == "pixel_limit" and s.diagnostics.limit_scope == scope
        assert s.diagnostics.decoded_pixels_observed == 8 and s.diagnostics.source_pixels == (8, 0)
        assert s.diagnostics.decoded_frames == 2 and s.diagnostics.owned_rgb_frames == 1


@pytest.mark.parametrize(
    "field,scope", [("max_total_pixels", "total"), ("max_source_total_pixels", "source")]
)
def test_propagated_pixel_limit_retains_terminal_reason_after_context_exit(decoder, field, scope):
    stream = NativeConcatStream(decoder.clips, NativeConcatConfig(limits=NativeConcatLimits(**{field: 5})))
    with pytest.raises(ScanError, match="pixel"), stream:
        next(stream)
        next(stream)
    diagnostic = stream.diagnostics
    assert diagnostic.status == "pixel_limit" and diagnostic.limit_scope == scope
    assert diagnostic.last_child_status == "pixel_limit" and diagnostic.closed
    assert diagnostic.decoded_pixels_observed == 8 and diagnostic.source_pixels == (8, 0)
    assert diagnostic.decoded_frames == 2 and diagnostic.owned_rgb_frames == 1
    assert not stream._active and not stream._busy and stream._child is None
    assert all(container.closed for container in decoder.containers)
    assert all(reader.handle.closed for reader in decoder.readers)


@pytest.mark.parametrize("budget", ["frame", "pixel"])
@pytest.mark.parametrize("primary_kind", [ValueError, KeyboardInterrupt, SystemExit])
def test_terminal_budget_body_priority_releases_exit_guard(decoder, budget, primary_kind):
    limits = NativeConcatLimits(max_frames=1) if budget == "frame" else NativeConcatLimits(max_total_pixels=5)
    stream = NativeConcatStream(decoder.clips, NativeConcatConfig(limits=limits))
    primary = primary_kind("new body failure")
    with pytest.raises(primary_kind) as caught, stream:
        next(stream)
        if budget == "pixel":
            with pytest.raises(ScanError, match="pixel"):
                next(stream)
        assert stream.diagnostics.status == budget + "_limit"
        raise primary
    assert caught.value is primary
    assert stream.diagnostics.status == (
        budget + "_limit" if isinstance(primary, Exception) else "interrupted"
    )
    assert stream.diagnostics.closed and stream.diagnostics.limit_scope == "total"
    assert not stream._active and not stream._busy and stream._child is None
    stream.close()
    assert stream.diagnostics.closed and not stream._busy


def test_frame_pixel_limit_rejects_probe_before_rgb_and_closes(decoder):
    with (
        pytest.raises(ScanError),
        NativeConcatStream(decoder.clips, NativeConcatConfig(limits=NativeConcatLimits(max_frame_pixels=3))),
    ):
        pass
    assert all(c.closed for c in decoder.containers) and all(r.handle.closed for r in decoder.readers)


def test_seek_budget_preflight_preserves_active_child(decoder):
    with NativeConcatStream(decoder.clips, NativeConcatConfig(limits=NativeConcatLimits(max_seeks=1))) as s:
        s.seek(0)
        before = s.diagnostics
        with pytest.raises(ScanError, match="seek lifetime"):
            s.seek(0)
        assert s.diagnostics == before
        assert next(s).native.pts == 5000


@pytest.mark.parametrize("field,value", [("max_activations", 3), ("max_opened_source_bytes", 48)])
def test_activation_budget_seek_rejection_is_nonmutating(decoder, field, value):
    with NativeConcatStream(
        decoder.clips, NativeConcatConfig(limits=NativeConcatLimits(**{field: value}))
    ) as s:
        next(s)
        before = s.diagnostics
        with pytest.raises(ScanError, match="budget"):
            s.seek(0)
        assert s.diagnostics == before
        assert next(s).native.pts == 5040


def test_repeated_path_is_separate_occurrence_and_charged_twice(decoder):
    with NativeConcatStream((decoder.clips[0], decoder.clips[0])) as stream:
        frames = list(stream)
        assert [f.clip_index for f in frames] == [0, 0, 0, 1, 1, 1]
        assert stream.diagnostics.source_decoded_frames == (4, 4)
        assert stream.diagnostics.opened_source_bytes == 64


def test_dimension_mismatch_rejects_complete_manifest_before_playback(decoder):
    def dimensions(container, index):
        if index == 1:
            container.streams.video[0].codec_context.width = 3

    decoder.hook = dimensions
    with pytest.raises(ScanError, match="dimensions"), NativeConcatStream(decoder.clips):
        pass
    assert all(c.closed for c in decoder.containers)
    assert all(c.decoded == 0 for c in decoder.containers)


def test_source_replacement_between_probe_and_playback_is_rejected_without_reopen(decoder):
    with NativeConcatStream(decoder.clips) as stream:
        decoder.paths[0].write_bytes(b"\x1aE\xdf\xa3" + b"new-size" * 10)
        with pytest.raises(ScanError, match="changed between"):
            next(stream)
        assert stream.diagnostics.status == "error" and stream.diagnostics.closed
        assert stream.diagnostics.activations == 2


def test_metadata_changed_between_activations_is_rejected_and_new_child_closed(decoder):
    with NativeConcatStream(decoder.clips) as stream:
        decoder.hook = lambda c, _: setattr(c.streams.video[0], "average_rate", Fraction(30))
        with pytest.raises(ScanError, match="metadata changed"):
            next(stream)
        assert stream.diagnostics.activations == 3 and stream.diagnostics.closed


@pytest.mark.parametrize("bad", [FakeFrame(None), FakeFrame(4990), RuntimeError("decode failed")])
def test_child_timestamp_and_decode_failures_never_advance_to_later_source(decoder, bad):
    decoder.frames[0][:] = [FakeFrame(5000), bad]
    with NativeConcatStream(decoder.clips) as stream:
        next(stream)
        with pytest.raises(ScanError):
            next(stream)
        assert stream.diagnostics.status == "error" and stream.diagnostics.closed
        assert stream.diagnostics.source_activations == (2, 1)
        assert stream.diagnostics.source_decoded_frames[1] == 0


def test_dimension_change_on_stride_discarded_frame_is_rejected(decoder):
    decoder.frames[0][1] = FakeFrame(5040, width=3)
    with NativeConcatStream(decoder.clips, NativeConcatConfig(frame_step=2)) as stream:
        next(stream)
        with pytest.raises(ScanError, match="dimensions changed"):
            next(stream)
        assert stream.diagnostics.owned_rgb_frames == 2
        assert stream.diagnostics.decoded_pixels_observed == 10


def test_close_and_context_owner_preconditions_without_resource_leaks(decoder):
    stream = NativeConcatStream(decoder.clips)
    for action in (lambda: next(stream), lambda: iter(stream), lambda: stream.seek(0)):
        with pytest.raises(ConfigurationError):
            action()
    with stream:
        with pytest.raises(ConfigurationError):
            stream.__enter__()
        next(stream)
        with ThreadPoolExecutor(max_workers=1) as pool:
            for action in (stream.close, lambda: next(stream), lambda: stream.seek(0), lambda: iter(stream)):
                with pytest.raises(ConfigurationError, match="single-owner"):
                    pool.submit(action).result()
        stream._busy = True
        with pytest.raises(ConfigurationError, match="reentrant"):
            next(stream)
        stream._busy = False
        stream.close()
        assert list(stream) == [] and stream.diagnostics.status == "closed"
        with pytest.raises(ConfigurationError):
            stream.seek(0)
    assert all(c.closed for c in decoder.containers)
    unopened = NativeConcatStream(decoder.clips)
    unopened.close()
    with pytest.raises(ConfigurationError):
        unopened.__enter__()


@pytest.mark.parametrize("primary_kind", [KeyboardInterrupt, SystemExit, RuntimeError])
@pytest.mark.parametrize("cleanup_kind", [KeyboardInterrupt, SystemExit, RuntimeError])
@pytest.mark.parametrize("stage", ["decode", "body", "setup", "output"])
def test_original_control_identity_and_cleanup_resource_retry(
    decoder, monkeypatch, primary_kind, cleanup_kind, stage
):
    primary, cleanup = primary_kind("original"), cleanup_kind("cleanup")
    fail = [True]

    def hook(container, _):
        def close():
            if fail[0]:
                raise cleanup
            container.closed = True

        if stage == "setup" or len(decoder.containers) > 2:
            container.close = close
        if stage == "setup":
            # Probe has no seek. Fail native metadata admission through its property.
            class Codec:
                height, name, thread_count = 2, "ffv1", 0

                @property
                def width(self):
                    raise primary

            container.streams.video[0].codec_context = Codec()

    decoder.hook = hook
    if stage == "decode":
        decoder.frames[0][:] = [primary]
    if stage == "output":

        def allocate(*_):
            raise primary

        monkeypatch.setattr(concat, "NativeConcatFrame", allocate)
    stream = NativeConcatStream(decoder.clips)
    expected = (
        primary
        if not isinstance(primary, Exception)
        else cleanup
        if not isinstance(cleanup, Exception)
        else primary
        if stage == "body"
        else ScanError
    )
    with pytest.raises(type(expected) if isinstance(expected, BaseException) else expected) as raised, stream:
        if stage == "body":
            next(stream)
            raise primary
        next(stream)
    if not isinstance(expected, type):
        assert raised.value is expected
    assert not stream.diagnostics.closed
    assert stream.diagnostics.cleanup_errors == ("container.close",)
    assert all(r.handle.closed for r in decoder.readers)
    before = len(decoder.containers)
    with pytest.raises(ConfigurationError):
        stream.seek(0)
    assert len(decoder.containers) == before
    fail[0] = False
    stream.close()
    assert stream.diagnostics.closed


def test_close_failure_blocks_next_child_and_retains_reference_for_explicit_retry(decoder):
    stream = NativeConcatStream(decoder.clips)
    with stream:
        next(stream)
        child = decoder.containers[-1]
        original_close = child.close

        def broken():
            raise OSError("close acknowledgement unavailable")

        child.close = broken
        with pytest.raises(ScanError):
            list(stream)
        assert not stream.diagnostics.closed and stream.diagnostics.status == "error"
        assert len(decoder.containers) == 3
        child.close = original_close
        stream.close()
    assert stream.diagnostics.closed


def test_concat_frame_validation_and_detached_image(decoder):
    with NativeConcatStream(decoder.clips) as stream:
        frame = next(stream)
    assert frame.to_dict()["native"] == frame.native.to_dict()
    for changes in (
        {"timeline": None},
        {"native": None},
        {"clip_index": 5},
        {"activation": True},
        {"generation": -1},
        {"sample_index": 1 << 64},
        {"native": replace(frame.native, pts=6000)},
    ):
        with pytest.raises(ConfigurationError):
            replace(frame, **changes)
    with frame.image() as image:
        image.putpixel((0, 0), (1, 2, 3))
    assert frame.rgb[:3] == bytes((255, 0, 0))
    with pytest.raises(FrozenInstanceError):
        frame.generation = 9


@pytest.mark.parametrize(
    "changes",
    [
        {"status": "fake"},
        {"status": []},
        {"closed": 1},
        {"decoded_frames": -1},
        {"returned_frames": 1},
        {"generation": 1},
        {"source_decoded_frames": None},
        {"source_decoded_frames": ()},
        {"source_rgb_frames": (0,)},
        {"source_pixels": (0, True)},
        {"source_activations": (0, 1)},
        {"current_clip": 2},
        {"limit_scope": "unknown"},
        {"cleanup_errors": []},
        {"cleanup_errors": ([],)},
        {"cleanup_errors": ("bad",)},
        {"cleanup_errors": ("file.close", "file.close")},
    ],
)
def test_diagnostics_reject_malformed_types_and_inconsistent_aggregates(decoder, changes):
    with pytest.raises(ConfigurationError):
        replace(NativeConcatStream(decoder.clips).diagnostics, **changes)


def test_composite_fraction_lcm_and_derived_numerator_bounds(decoder):
    with pytest.raises(ConfigurationError, match="composite rational"):
        NativeConcatConfig(start=Fraction(1 << 127))
    clips = (
        NativeConcatClip(decoder.paths[0], start=0, end=Fraction(1, (1 << 62) - 1)),
        NativeConcatClip(decoder.paths[1], start=0, end=Fraction(1, (1 << 62) - 3)),
    )
    with pytest.raises(ConfigurationError, match="composite rational"):
        NativeConcatTimeline(clips)
    with pytest.raises(ConfigurationError, match="canonical"):
        concat._canonical({"bounded": "x" * (1 << 20)})


def test_manifest_negative_origins_duplicate_pts_and_declared_metadata_not_timeline(decoder):
    decoder.frames[0][:] = [FakeFrame(-200), FakeFrame(-200), FakeFrame(-100), FakeFrame(0)]
    decoder.frames[1][:] = [FakeFrame(5000), FakeFrame(5040)]
    clips = (NativeConcatClip(decoder.paths[0], start=Fraction(-1, 5), end=0), decoder.clips[1])

    def metadata(container, index):
        container.streams.video[0].start_time = None
        container.streams.video[0].average_rate = None
        container.streams.video[0].base_rate = None
        # Both wildly under/overestimated declarations are ignored as timeline inputs.
        container.streams.video[0].duration = 1 if index == 0 else 999_999

    decoder.hook = metadata
    with NativeConcatStream(clips) as stream:
        frames = list(stream)
        assert [f.presentation_time for f in frames] == [
            0,
            0,
            Fraction(1, 10),
            Fraction(1, 5),
            Fraction(6, 25),
        ]
        assert [f.native.pts for f in frames[:3]] == [-200, -200, -100]
        assert [m.start_pts for m in stream.metadata] == [None, None]
        assert [m.average_rate for m in stream.metadata] == [None, None]
        assert stream.timeline.offsets == (0, Fraction(1, 5), Fraction(2, 5))
        assert stream.diagnostics.last_child_status == "eof"


def test_handwritten_partition_oracle_for_all_fractional_seam_combinations(decoder):
    clips = (
        NativeConcatClip(decoder.paths[0], start=-2, end=1),
        NativeConcatClip(decoder.paths[1], start=10, end=12),
        NativeConcatClip(decoder.paths[0], start=Fraction(1, 7), end=Fraction(8, 7)),
    )
    timeline = NativeConcatTimeline(clips)
    for left in (Fraction(0), Fraction(1, 7), Fraction(3), Fraction(7, 2), Fraction(5), Fraction(6)):
        for right in (Fraction(0), Fraction(3), Fraction(33, 7), Fraction(5), Fraction(6)):
            if left > right:
                continue
            spans = timeline.map_span(left, right)
            expected = []
            for index, (offset, width, origin) in enumerate(((0, 3, -2), (3, 2, 10), (5, 1, Fraction(1, 7)))):
                lo, hi = max(left, offset), min(right, offset + width)
                if lo < hi:
                    expected.append((index, lo, hi, origin + lo - offset, origin + hi - offset))
            assert [
                (s.clip_index, s.global_start, s.global_end, s.local_start, s.local_end) for s in spans
            ] == expected
            assert sum((s.global_end - s.global_start for s in spans), Fraction()) == right - left


@pytest.mark.parametrize("fault", ["missing", "corrupt", "no-video"])
@pytest.mark.parametrize("index", [0, 1])
def test_first_and_late_source_setup_failure_never_starts_playback(decoder, fault, index):
    if fault == "missing":
        decoder.paths[index].unlink()
    elif fault == "corrupt":
        decoder.paths[index].write_bytes(b"unsupported signature" * 2)
    else:

        def missing(container, source):
            if source == index:
                container.streams.video = []

        decoder.hook = missing
    stream = NativeConcatStream(decoder.clips)
    with pytest.raises(ScanError), stream:
        pass
    assert stream.diagnostics.closed and stream.diagnostics.status == "error"
    assert all(c.closed and c.decoded == 0 for c in decoder.containers)
    assert all(r.handle.closed for r in decoder.readers)


@pytest.mark.parametrize(
    "field,value",
    [
        ("st_ino", 0),
        ("st_ino", -1),
        ("st_ino", True),
        ("st_ino", 1 << 128),
        ("st_dev", -1),
        ("st_dev", None),
        ("st_dev", 1 << 128),
        ("st_size", -1),
        ("st_size", 1 << 41),
        ("st_mtime_ns", 1 << 63),
        ("st_mode", 0),
        ("st_file_attributes", 0x400),
    ],
)
def test_stable_identity_is_mandatory_before_acquiring_native_handles(decoder, monkeypatch, field, value):
    stream = NativeConcatStream(decoder.clips)
    actual = decoder.paths[0].lstat()
    fields = {
        name: getattr(actual, name, 0)
        for name in ("st_ino", "st_dev", "st_size", "st_mtime_ns", "st_mode", "st_file_attributes")
    }
    fields[field] = value
    monkeypatch.setattr(Path, "lstat", lambda *_: SimpleNamespace(**fields))
    with pytest.raises((ScanError, ConfigurationError)), stream:
        pass
    assert not decoder.containers and stream.diagnostics.closed


@pytest.mark.parametrize("boundary", ["captured", "postopen"])
def test_identity_reconciliation_checks_actual_open_handle_and_postopen_path(decoder, monkeypatch, boundary):
    if boundary == "captured":
        monkeypatch.setattr(
            native.NativeVideoStream,
            "_captured_identity",
            lambda _: (0, 1, 16, 0),
        )
    else:
        original = concat._identity
        calls = []

        def changed(path):
            calls.append(path)
            observed = original(path)
            return (*observed[:-1], observed[-1] + 1) if len(calls) == 4 else observed

        monkeypatch.setattr(concat, "_identity", changed)
    stream = NativeConcatStream(decoder.clips)
    with pytest.raises(ScanError, match="identity changed while opening"), stream:
        pass
    assert stream.diagnostics.closed and stream.diagnostics.activations == 1
    assert all(c.closed for c in decoder.containers)


def test_seek_changed_source_fails_after_closing_old_child_without_new_acquisition(decoder):
    with NativeConcatStream(decoder.clips) as stream:
        next(stream)
        decoder.paths[1].write_bytes(b"\x1aE\xdf\xa3" + b"changed" * 10)
        with pytest.raises(ScanError, match="changed between"):
            stream.seek(Fraction(1, 5))
        assert stream.diagnostics.closed and stream.diagnostics.status == "error"
        assert stream.diagnostics.activations == 3 and stream.diagnostics.generation == 1


def test_fractional_native_tick_preflight_and_overflow_do_not_replace_active_child(decoder):
    # A huge second declared interval is representable; a seek near its far end is not in native ticks.
    clips = (decoder.clips[0], NativeConcatClip(decoder.paths[1], start=0, end=(1 << 63) - 1))
    with NativeConcatStream(clips, NativeConcatConfig(end=Fraction(1, 5))) as stream:
        next(stream)
        before = stream.diagnostics
        with pytest.raises(ConfigurationError, match="native seek offset"):
            stream.seek(Fraction(1 << 62) + Fraction(1, 5))
        assert stream.diagnostics == before
        stream.seek(Fraction(81, 2000), end=Fraction(1, 5))
        assert decoder.containers[-1].seek_calls[0][0] == 5040
        assert [f.native.pts for f in stream] == [5100]


@pytest.mark.parametrize(
    "field,cap,status,scope",
    [
        ("max_decoded_frames", 4, "decode_limit", "total"),
        ("max_source_decoded_frames", 4, "decode_limit", "source"),
        ("max_total_pixels", 4, "pixel_limit", "total"),
        ("max_source_total_pixels", 4, "pixel_limit", "source"),
    ],
)
def test_zero_remaining_work_is_rejected_before_new_source_or_seek(decoder, field, cap, status, scope):
    with NativeConcatStream(
        decoder.clips, NativeConcatConfig(limits=NativeConcatLimits(**{field: cap}))
    ) as s:
        if "pixels" in field:
            next(s)
            before = s.diagnostics
            with pytest.raises(ScanError, match="budget"):
                s.seek(0)
            assert s.diagnostics == before
        elif "source" in field:
            list(s)
            # Both source EOF boundary frames are charged to their occurrences; reset cannot replenish either.
            before = s.diagnostics
            with pytest.raises(ScanError, match="budget"):
                s.reset()
            assert s.diagnostics == before
        else:
            assert len(list(s)) == 3
            assert s.diagnostics.status == status and s.diagnostics.limit_scope == scope
            assert s.diagnostics.source_activations == (2, 1)


def test_stride_discarded_limit_frame_is_charged_without_return_or_source_advance(decoder):
    config = NativeConcatConfig(frame_step=3, limits=NativeConcatLimits(max_rgb_frames=2))
    with NativeConcatStream(decoder.clips, config) as stream:
        assert len(list(stream)) == 1
        assert stream.diagnostics.owned_rgb_frames == 2 and stream.diagnostics.status == "rgb_limit"
        assert stream.diagnostics.source_activations == (2, 1)


@pytest.mark.parametrize("failure_kind", [MemoryError, KeyboardInterrupt, SystemExit])
def test_output_allocation_failure_charges_attempt_and_all_performed_native_work(
    decoder, monkeypatch, failure_kind
):
    failure = failure_kind("allocation failed")

    def allocate(*_):
        raise failure

    monkeypatch.setattr(concat, "NativeConcatFrame", allocate)
    stream = NativeConcatStream(decoder.clips)
    expected = ScanError if isinstance(failure, Exception) else failure_kind
    with pytest.raises(expected) as raised, stream:
        next(stream)
    assert raised.value.__cause__ is failure if isinstance(failure, Exception) else raised.value is failure
    assert stream.diagnostics.closed and stream.diagnostics.decoded_frames == 1
    assert stream.diagnostics.owned_rgb_frames == stream.diagnostics.returned_frames == 1
    assert stream.diagnostics.decoded_pixels_observed == 4
    assert stream.diagnostics.status == ("error" if isinstance(failure, Exception) else "interrupted")


@pytest.mark.parametrize("failure_kind", [MemoryError, KeyboardInterrupt, SystemExit])
def test_child_rgb_conversion_failure_counts_decode_and_closes(decoder, failure_kind):
    failure = failure_kind("RGB conversion failed")

    def convert():
        raise failure

    decoder.frames[0][0].to_image = convert
    stream = NativeConcatStream(decoder.clips)
    with pytest.raises(ScanError if isinstance(failure, Exception) else failure_kind), stream:
        next(stream)
    assert stream.diagnostics.closed and stream.diagnostics.decoded_frames == 1
    assert stream.diagnostics.owned_rgb_frames == stream.diagnostics.returned_frames == 0
    assert stream.diagnostics.decoded_pixels_observed == 4


@pytest.mark.parametrize("primary_kind", [KeyboardInterrupt, SystemExit, RuntimeError])
@pytest.mark.parametrize("secondary_kind", [KeyboardInterrupt, SystemExit, MemoryError])
def test_finally_work_observation_preserves_control_identity(
    decoder, monkeypatch, primary_kind, secondary_kind
):
    primary, secondary = primary_kind("primary"), secondary_kind("secondary")
    decoder.frames[0][:] = [primary]
    stream = NativeConcatStream(decoder.clips)
    original = stream._account
    failed = []

    def observe():
        original()
        if (
            stream._last_child_status
            in (native.NativeVideoStatus.ERROR, native.NativeVideoStatus.INTERRUPTED)
            and not failed
        ):
            failed.append(True)
            raise secondary

    monkeypatch.setattr(stream, "_account", observe)
    expected = (
        primary
        if not isinstance(primary, Exception)
        else secondary
        if not isinstance(secondary, Exception)
        else ScanError
    )
    with pytest.raises(expected if isinstance(expected, type) else type(expected)) as raised, stream:
        next(stream)
    if not isinstance(expected, type):
        assert raised.value is expected
    assert failed and stream.diagnostics.closed


def test_child_identity_accessor_requires_actual_captured_open(decoder):
    child = native.NativeVideoStream(decoder.paths[0])
    with pytest.raises(ConfigurationError, match="identity"):
        child._captured_identity()
    with child:
        assert child._captured_identity() == concat._identity(decoder.paths[0])


def test_native_secondary_io_open_is_still_denied_in_concat_probe(decoder, monkeypatch):
    def reject(reader, **kwargs):
        decoder.readers.append(reader)
        kwargs["io_open"]("https://example.org/segment.ts", 0, {})

    monkeypatch.setattr(native, "_load_av", lambda: SimpleNamespace(open=reject))
    with pytest.raises(ScanError), NativeConcatStream(decoder.clips):
        pass
    assert decoder.readers[0].handle.closed


@pytest.mark.parametrize("value", ["invented", [], 1])
def test_last_child_diagnostics_are_explicit_and_validated(decoder, value):
    with pytest.raises(ConfigurationError):
        replace(NativeConcatStream(decoder.clips).diagnostics, last_child_status=value)


def test_end_of_source_is_not_an_exception_that_can_hide_failed_work_observation(decoder, monkeypatch):
    stream = NativeConcatStream(decoder.clips)
    original = stream._account
    failed = []

    def observe():
        original()
        if stream._last_child_status is native.NativeVideoStatus.RANGE_END and not failed:
            failed.append(True)
            raise MemoryError("observation allocation failed")

    monkeypatch.setattr(stream, "_account", observe)
    with pytest.raises(ScanError, match="allocation"), stream:
        list(stream)
    assert failed and stream.diagnostics.closed and stream.diagnostics.status == "error"
    assert stream.diagnostics.source_activations == (2, 1)


def test_failed_close_observation_still_checks_actual_closure(decoder, monkeypatch):
    stream = NativeConcatStream(decoder.clips)
    with stream:
        next(stream)
        original = stream._account
        failed = []

        def observe():
            original()
            if stream._last_child_status is native.NativeVideoStatus.CLOSED and not failed:
                failed.append(True)
                raise MemoryError("close observation failed")

        monkeypatch.setattr(stream, "_account", observe)
        with pytest.raises(MemoryError, match="close observation"):
            stream.close()
        assert stream.diagnostics.closed and stream.diagnostics.status == "error"
        assert stream._child is None


@pytest.mark.parametrize("failure_kind", [MemoryError, KeyboardInterrupt, SystemExit])
def test_close_acknowledgement_failure_retains_owned_child_for_retry(decoder, monkeypatch, failure_kind):
    stream = NativeConcatStream(decoder.clips)
    with stream:
        next(stream)
        original = native.NativeVideoStream.diagnostics.fget
        child = stream._child
        calls = []
        failure = failure_kind("acknowledgement unavailable")

        def diagnostic(instance):
            observed = original(instance)
            if instance is child and observed.closed:
                calls.append(True)
                if len(calls) == 2:
                    raise failure
            return observed

        monkeypatch.setattr(native.NativeVideoStream, "diagnostics", property(diagnostic))
        with pytest.raises(failure_kind) as raised:
            stream.close()
        assert raised.value is failure and stream._child is child
        assert all(c.closed for c in decoder.containers)
        stream.close()
        assert stream._child is None and stream.diagnostics.closed


def test_composite_rejects_unexpected_child_terminal_status_instead_of_advancing(decoder):
    with NativeConcatStream(decoder.clips) as stream:
        next(stream)
        stream._child.close()  # Fault injection: a backend stopped without EOF or RANGE_END.
        with pytest.raises(ScanError, match="without logical exhaustion"):
            next(stream)
        assert stream.diagnostics.closed and stream.diagnostics.source_activations == (2, 1)


def test_low_level_work_and_ownership_guards_defend_against_child_contract_violation(decoder):
    stream = NativeConcatStream(decoder.clips)
    stream._account()
    with stream:
        next(stream)
        with pytest.raises(ScanError, match="closure is unknown"):
            stream._activate(1, (Fraction(5), Fraction(26, 5)))
        child = stream._child
        stream._accounted = (child.diagnostics.decoded_frames + 1, 0, 0)
        with pytest.raises(ScanError, match="counters decreased"):
            stream._account()
        stream._accounted = (
            child.diagnostics.decoded_frames,
            child.diagnostics.returned_frames,
            child.diagnostics.decoded_pixels_observed,
        )


def test_paths_have_utf8_and_resolved_length_admission_before_native_work(decoder, monkeypatch):
    with pytest.raises(ConfigurationError, match="4096"):
        NativeConcatClip("\u4e2d" * 2000, start=0, end=1)
    monkeypatch.setattr(concat, "_source_path", lambda _: Path("a" * 4097))
    with pytest.raises(ConfigurationError, match="resolved"):
        NativeConcatClip("short", start=0, end=1)
    assert not decoder.containers
