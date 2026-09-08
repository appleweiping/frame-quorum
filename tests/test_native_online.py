"""Independent online decision, ownership and bounded lookahead oracles."""

from fractions import Fraction
from pathlib import Path
from typing import ClassVar

import pytest

import frame_quorum.native_online as module
from frame_quorum.errors import ConfigurationError
from frame_quorum.native_online import (
    NativeOnlineLimits,
    NativePixelChangeEnd,
    NativePixelChangeStream,
    NativePixelChangeUpdate,
)
from frame_quorum.native_pixel_changes import PixelChangeDetectionConfig
from frame_quorum.native_video import (
    NativeVideoConfig,
    NativeVideoDiagnostics,
    NativeVideoFrame,
    NativeVideoMetadata,
    NativeVideoStatus,
)


@pytest.fixture
def decoder(monkeypatch):
    class Decoder:
        values: ClassVar = [0, 0, 255, 255, 0, 255]
        instances: ClassVar = []
        failure = None

        def __init__(self, path, config):
            self.config = config
            self.count = self.calls = 0
            self.closed = False
            self.status = NativeVideoStatus.NEW
            self.instances.append(self)
            self.metadata = NativeVideoMetadata(
                Path(path),
                32,
                "matroska",
                "ffv1",
                0,
                1,
                1,
                Fraction(1, 1000),
                5000,
                None,
                None,
                None,
            )

        @property
        def diagnostics(self):
            return NativeVideoDiagnostics(self.status, self.count, self.count, self.count, 0, self.closed)

        def __enter__(self):
            self.status = NativeVideoStatus.OPEN
            return self

        def __next__(self):
            if self.closed:
                raise StopIteration
            self.calls += 1
            if self.failure is not None:
                raise self.failure
            if self.count == len(self.values):
                self.status = NativeVideoStatus.EOF
                self.closed = True
                raise StopIteration
            if self.count == self.config.max_frames:
                self.status = NativeVideoStatus.FRAME_LIMIT
                self.closed = True
                raise StopIteration
            index = self.count
            self.count += 1
            if self.count == self.config.max_frames:
                self.status = NativeVideoStatus.FRAME_LIMIT
                self.closed = True
            return NativeVideoFrame(
                5000 + index * index,
                Fraction(1, 1000),
                index * self.config.frame_step,
                index,
                0,
                1,
                1,
                bytes([self.values[index]]) * 3,
            )

        def close(self):
            self.closed = True
            if self.status in (NativeVideoStatus.NEW, NativeVideoStatus.OPEN):
                self.status = NativeVideoStatus.CLOSED

    monkeypatch.setattr(module, "NativeVideoStream", Decoder)
    return Decoder


def test_content_delivery_waits_for_real_tail_and_never_retracts(decoder):
    config = PixelChangeDetectionConfig(value_only=True, min_scene_samples=2)
    with NativePixelChangeStream("unused.mkv", detection=config) as stream:
        first = next(stream)
        assert type(first) is NativePixelChangeUpdate
        assert first.sample.sample.sample_index == 0
        assert first.observed_through_sample_index == 1
        assert decoder.instances[-1].count == 2
        events = [first, *stream]
        rows = [row for row in events if type(row) is NativePixelChangeUpdate]
        assert [row.statistic.weighted_score for row in rows] == [0, 0, 1, 0, 1, 1]
        assert [row.sample.sample.sample_index for row in rows if row.statistic.accepted] == [2, 4]
        assert rows[-1].statistic.reason == "short_previous_scene"
        assert rows[2].closed_scene.end_time == Fraction(5004, 1000)
        assert rows[4].closed_scene.start_position == 2
        end = events[-1]
        assert type(end) is NativePixelChangeEnd
        assert end.final_scene.start_position == 4 and end.final_scene.end_position == 6
        assert end.final_scene.end_time is None
        assert end.diagnostics.status == "finished" and end.diagnostics.video.closed
        assert end.source_verified is False


def test_eof_rejects_candidate_whose_tail_never_matures(decoder):
    decoder.values = [0, 0, 0, 255]
    with NativePixelChangeStream(
        "unused.mkv", detection=PixelChangeDetectionConfig(value_only=True, min_scene_samples=2)
    ) as stream:
        events = list(stream)
    rows = events[:-1]
    assert rows[3].statistic.candidate and not rows[3].statistic.accepted
    assert rows[3].statistic.reason == "short_final_scene"
    assert events[-1].final_scene.sample_count == 4


def test_adaptive_complete_window_and_confirmation_coordinates(decoder):
    decoder.values = [0, 0, 0, 255, 255, 255, 255]
    with NativePixelChangeStream(
        "unused.mkv",
        detection=PixelChangeDetectionConfig(
            detector="adaptive",
            value_only=True,
            window_radius=1,
            min_content=0.5,
        ),
    ) as stream:
        events = list(stream)
    rows = events[:-1]
    assert [row.statistic.score for row in rows] == [None, None, 0, 1_000_000, 0, 0, None]
    assert [row.sample.sample.sample_index for row in rows if row.statistic.accepted] == [3]
    assert rows[3].observed_through_sample_index == 4
    assert rows[3].observed_through_time == Fraction(5016, 1000)


def test_cancel_discards_pending_without_success_end_or_extra_decode(decoder):
    with NativePixelChangeStream(
        "unused.mkv", detection=PixelChangeDetectionConfig(min_scene_samples=3)
    ) as stream:
        next(stream)
        before = decoder.instances[-1].calls
        stream.cancel()
        assert list(stream) == []
        assert stream.diagnostics.status == "cancelled"
        assert stream.diagnostics.video.closed
        assert decoder.instances[-1].calls == before


def test_invalid_buffer_capacity_precedes_decoder_construction(decoder):
    with pytest.raises(ConfigurationError, match="buffer"):
        NativePixelChangeStream(
            "unused.mkv",
            detection=PixelChangeDetectionConfig(window_radius=10, detector="adaptive"),
            limits=NativeOnlineLimits(max_buffered_samples=10),
        )
    assert decoder.instances == []


def test_failure_closes_owned_decoder_and_no_success_end(decoder):
    decoder.failure = KeyboardInterrupt("control")
    with pytest.raises(KeyboardInterrupt, match="control"), NativePixelChangeStream("unused.mkv") as stream:
        next(stream)
    assert decoder.instances[-1].closed
    assert stream.diagnostics.status == "interrupted"


def test_empty_and_budget_termination_are_distinct(decoder):
    decoder.values = []
    with NativePixelChangeStream("unused.mkv") as stream:
        events = list(stream)
    assert len(events) == 1 and events[0].final_scene is None
    assert events[0].diagnostics.video.status is NativeVideoStatus.EOF
    decoder.values = [0, 255, 0]
    with NativePixelChangeStream("unused.mkv", video=NativeVideoConfig(max_frames=2)) as stream:
        events = list(stream)
    assert events[-1].diagnostics.video.status is NativeVideoStatus.FRAME_LIMIT
    assert events[-1].final_scene.end_time is None
    assert decoder.instances[-1].calls == 2


def test_frozen_fsum_operation_order_matches_long_seeded_oracle():
    import math
    import random

    from frame_quorum.scene_detection import _adaptive_scores

    # Independently retained pre-extraction numerical specification; do not
    # replace this with a call to the new rolling state implementation.
    randomizer = random.Random(4301)
    content = [0.0, *(randomizer.random() ** 7 for _ in range(3000))]
    for radius in (1, 2, 17, 256):
        expected = [None] * len(content)
        total = math.fsum(content[1 : 2 * radius + 2])
        for position in range(radius + 1, len(content) - radius):
            average = max(0.0, (total - content[position]) / (2 * radius))
            expected[position] = min(1_000_000.0, content[position] / max(average, 1e-12))
            if position + 1 < len(content) - radius:
                total = math.fsum((total, -content[position - radius], content[position + radius + 1]))
        assert _adaptive_scores(content, radius) == expected


@pytest.mark.parametrize("mode", ["content", "adaptive"])
@pytest.mark.parametrize("minimum", [1, 2, 4])
def test_complete_decisions_and_buffer_peak_match_independent_short_prefixes(decoder, mode, minimum):
    import itertools
    import math

    for values in itertools.product((0, 255), repeat=5):
        decoder.values = list(values)
        config = PixelChangeDetectionConfig(
            detector=mode,
            value_only=True,
            min_scene_samples=minimum,
            window_radius=1,
            adaptive_ratio=2,
            min_content=0.5,
        )
        with NativePixelChangeStream("unused.mkv", detection=config) as stream:
            events = list(stream)
        distances = [0, *[abs(a - b) / 255 for a, b in itertools.pairwise(values)]]
        scores = (
            distances
            if mode == "content"
            else [
                None,
                None,
                *[
                    min(
                        1_000_000,
                        distances[i] / max(math.fsum((distances[i - 1], distances[i + 1])) / 2, 1e-12),
                    )
                    for i in (2, 3)
                ],
                None,
            ]
        )
        prior = 0
        for index, (row, score) in enumerate(zip(events[:-1], scores, strict=True)):
            candidate = (
                index > 0
                and score is not None
                and (score >= 0.3 if mode == "content" else score >= 2 and distances[index] >= 0.5)
            )
            accepted = candidate and index - prior >= minimum and len(values) - index >= minimum
            assert row.statistic.score == score
            assert row.statistic.candidate == candidate
            assert row.statistic.accepted == accepted
            if accepted:
                prior = index
        assert events[-1].diagnostics.peak_buffer_slots <= stream.required_buffer_slots
        assert stream._pending == {} and stream._measure.previous is None


def test_long_consumption_has_fixed_pending_storage_and_no_rgb_history(decoder):
    decoder.values = [0, 255] * 250
    with NativePixelChangeStream(
        "unused.mkv",
        detection=PixelChangeDetectionConfig(detector="adaptive", window_radius=3, min_scene_samples=8),
    ) as stream:
        first = next(stream)
        before = decoder.instances[-1].calls
        assert first.observed_through_sample_index == 7
        assert stream.diagnostics.observed_samples == 8
        assert decoder.instances[-1].calls == before  # No producer exists while consumer is idle.
        count = 1
        for event in stream:
            if type(event) is NativePixelChangeUpdate:
                count += 1
                assert len(stream._pending) <= stream.confirmation_lag
        assert count == 500
        assert stream.diagnostics.peak_buffer_slots == stream.required_buffer_slots


@pytest.mark.parametrize("name", ["max_buffered_samples", "max_line_bytes", "max_output_bytes"])
@pytest.mark.parametrize("value", [True, 0, -1, 1.5, float("nan"), float("inf"), 10**1000])
def test_limits_reject_before_io(name, value, decoder):
    with pytest.raises(ConfigurationError):
        NativeOnlineLimits(**{name: value})
    assert decoder.instances == []


def test_readonly_configuration_and_owner_lifecycle(decoder):
    stream = NativePixelChangeStream("unused.mkv")
    with pytest.raises(ConfigurationError, match="active"):
        next(stream)
    with pytest.raises(ConfigurationError, match="active"):
        iter(stream)
    with pytest.raises(AttributeError):
        stream.detection = PixelChangeDetectionConfig()
    with stream:
        with pytest.raises(ConfigurationError, match="twice"):
            stream.__enter__()
        next(stream)
        stream.close()
        assert list(stream) == []
        assert stream.diagnostics.status == "closed"
    with pytest.raises(ConfigurationError, match="active"):
        next(stream)


def test_cancel_during_measurement_and_reentrant_close_are_safe(decoder, monkeypatch):
    stream = NativePixelChangeStream("unused.mkv")
    original = stream._measure.measure

    def measure(frame):
        stream.cancel()
        return original(frame)

    monkeypatch.setattr(stream._measure, "measure", measure)
    with stream:
        assert list(stream) == []
    assert stream.diagnostics.status == "cancelled" and stream.diagnostics.confirmed_samples == 0
    stream = NativePixelChangeStream("unused.mkv")
    monkeypatch.setattr(stream._measure, "measure", lambda frame: stream.close())
    with pytest.raises(ConfigurationError, match="reentrant"), stream:
        next(stream)
    assert decoder.instances[-1].closed


def test_jsonl_footer_hash_and_exact_newline_inclusive_budget(decoder):
    import hashlib
    import json
    from dataclasses import replace

    from frame_quorum.errors import OutputError

    decoder.values = []
    limits = NativeOnlineLimits(max_output_bytes=9999)

    def encode(bounds):
        with NativePixelChangeStream("unused.mkv", limits=bounds) as stream:
            return list(module.iter_native_pixel_change_jsonl(stream))

    lines = encode(limits)
    size = sum(map(len, lines))
    assert 1000 <= size < 9999  # Same-width decimal budget in the canonical header.
    exact = encode(replace(limits, max_output_bytes=size))
    assert sum(map(len, exact)) == size
    assert json.loads(exact[-1])["sha256"] == hashlib.sha256(exact[0]).hexdigest()
    assert all(line.endswith(b"\n") for line in exact)
    with NativePixelChangeStream("unused.mkv", limits=replace(limits, max_output_bytes=size - 1)) as stream:
        chunks = module.iter_native_pixel_change_jsonl(stream)
        assert json.loads(next(chunks))["schema_version"] == 1
        with pytest.raises(OutputError, match="budget"):
            next(chunks)


def test_sink_failure_closes_native_without_generator_garbage_collection(decoder, monkeypatch, tmp_path):
    def failed_bundle(target, files, maximum):
        lines = files[0][1]
        next(lines)
        next(lines)
        assert not decoder.instances[-1].closed
        raise OSError("short sink")

    monkeypatch.setattr(module, "_bundle", failed_bundle)
    stream = NativePixelChangeStream("unused.mkv")
    with pytest.raises(OSError, match="short sink"):
        module.write_native_pixel_change_stream(stream, tmp_path / "output")
    assert decoder.instances[-1].closed and stream._measure.previous is None


def test_cancelled_jsonl_has_no_success_footer(decoder):
    from frame_quorum.errors import ScanError

    with NativePixelChangeStream("unused.mkv") as stream:
        lines = module.iter_native_pixel_change_jsonl(stream)
        next(lines)
        stream.cancel()
        with pytest.raises(ScanError, match="without successful"):
            next(lines)


@pytest.mark.parametrize("primary", [OSError("primary"), KeyboardInterrupt("primary"), SystemExit("primary")])
@pytest.mark.parametrize("cleanup", [OSError("cleanup"), KeyboardInterrupt("cleanup"), SystemExit("cleanup")])
def test_primary_and_control_cleanup_precedence(decoder, monkeypatch, primary, cleanup):
    stream = NativePixelChangeStream("unused.mkv")
    original = stream._release

    def release():
        original()
        raise cleanup

    monkeypatch.setattr(stream, "_release", release)
    decoder.failure = primary
    expected = cleanup if isinstance(primary, Exception) and not isinstance(cleanup, Exception) else primary
    with pytest.raises(type(expected)) as raised, stream:
        next(stream)
    assert raised.value is expected
    assert decoder.instances[-1].closed and stream._pending == {}


def test_cleanup_failure_with_no_active_error_is_not_successful_end(decoder, monkeypatch):
    stream = NativePixelChangeStream("unused.mkv")
    original = stream._release

    def release():
        original()
        raise OSError("release failed")

    monkeypatch.setattr(stream, "_release", release)
    observed = []
    with pytest.raises(OSError, match="release failed"), stream:
        observed.extend(stream)
    assert not any(type(event) is NativePixelChangeEnd for event in observed)
    assert stream.diagnostics.status == "error" and decoder.instances[-1].closed


def test_mixed_consumption_cannot_emit_a_valid_footer(decoder):
    with NativePixelChangeStream("unused.mkv") as stream:
        lines = module.iter_native_pixel_change_jsonl(stream)
        next(lines)
        next(stream)
        with pytest.raises(ConfigurationError, match="outside"):
            next(lines)
    decoder.values = [0]
    with NativePixelChangeStream("unused.mkv") as stream:
        lines = module.iter_native_pixel_change_jsonl(stream)
        next(lines)
        next(stream)
        with pytest.raises(ConfigurationError, match="outside"):
            next(lines)


def test_owner_thread_check_has_no_decode_side_effect(decoder):
    from concurrent.futures import ThreadPoolExecutor

    with NativePixelChangeStream("unused.mkv") as stream, ThreadPoolExecutor(max_workers=1) as executor:
        with pytest.raises(ConfigurationError, match="single-owner"):
            executor.submit(next, stream).result()
        assert decoder.instances[-1].calls == 0
        executor.submit(stream.cancel).result()
        assert list(stream) == []


@pytest.mark.parametrize(
    "name,value",
    [
        ("video", {}),
        ("measurement", {}),
        ("detection", {}),
        ("pixel_limits", {}),
        ("limits", {}),
    ],
)
def test_public_wrong_options_rejected_without_open(decoder, name, value):
    with pytest.raises(ConfigurationError):
        NativePixelChangeStream("unused.mkv", **{name: value})
    assert decoder.instances == []


def test_pair_byte_admission_and_cancel_before_enter(decoder):
    from frame_quorum import PixelChangeLimits

    with pytest.raises(ConfigurationError, match="pair"):
        NativePixelChangeStream("unused.mkv", pixel_limits=PixelChangeLimits(max_pair_rgb_bytes=6))
    stream = NativePixelChangeStream("unused.mkv")
    stream.cancel()
    with pytest.raises(ConfigurationError, match="cancellation"):
        stream.__enter__()
    stream.close()
    assert decoder.instances[-1].closed


def test_wrong_stream_and_consumed_jsonl_rejected(decoder, tmp_path):
    with pytest.raises(ConfigurationError):
        list(module.iter_native_pixel_change_jsonl(object()))
    with pytest.raises(ConfigurationError):
        module.write_native_pixel_change_stream(object(), tmp_path / "output")
    stream = NativePixelChangeStream("unused.mkv")
    with pytest.raises(ConfigurationError, match="fresh"):
        list(module.iter_native_pixel_change_jsonl(stream))
    with stream:
        next(stream)
        with pytest.raises(ConfigurationError, match="fresh"):
            list(module.iter_native_pixel_change_jsonl(stream))


@pytest.mark.parametrize(
    "field,value",
    [
        ("status", "unknown"),
        ("status", None),
        ("observed_samples", True),
        ("confirmed_samples", 99),
        ("measurement_pixels", -1),
        ("peak_buffer_slots", 65_537),
        ("video", object()),
    ],
)
def test_diagnostics_reject_invalid_public_records(decoder, field, value):
    from dataclasses import replace

    with NativePixelChangeStream("unused.mkv") as stream:
        end = list(stream)[-1]
    with pytest.raises(ConfigurationError):
        replace(end.diagnostics, **{field: value})


@pytest.mark.parametrize(
    "field,value",
    [
        ("sample", object()),
        ("statistic", object()),
        ("observed_through_sample_index", -1),
        ("observed_through_sample_index", True),
        ("observed_through_time", 0.5),
        ("observed_through_time", Fraction(0)),
        ("observed_through_time", Fraction(10**100)),
        ("closed_scene", None),
    ],
)
def test_update_rejects_invalid_public_records(decoder, field, value):
    from dataclasses import replace

    with NativePixelChangeStream(
        "unused.mkv", detection=PixelChangeDetectionConfig(value_only=True)
    ) as stream:
        cut = list(stream)[2]
    assert cut.statistic.accepted
    with pytest.raises(ConfigurationError):
        replace(cut, **{field: value})


def test_public_scene_and_end_cross_fields_reject_contradictions(decoder):
    from dataclasses import replace

    with NativePixelChangeStream(
        "unused.mkv", detection=PixelChangeDetectionConfig(value_only=True)
    ) as stream:
        events = list(stream)
    cut, end = events[2], events[-1]
    with pytest.raises(ConfigurationError, match="closed scene"):
        replace(cut, closed_scene=replace(cut.closed_scene, end_position=3))
    for diagnostics, scene in (
        (object(), end.final_scene),
        (replace(end.diagnostics, status="error"), end.final_scene),
        (replace(end.diagnostics, confirmed_samples=0), end.final_scene),
        (replace(end.diagnostics, video=replace(end.diagnostics.video, closed=False)), end.final_scene),
        (end.diagnostics, None),
        (end.diagnostics, cut.closed_scene),
    ):
        with pytest.raises(ConfigurationError):
            module.NativePixelChangeEnd(diagnostics, scene)


def test_unsuccessful_source_stop_cannot_finalize_pending_tail(decoder):
    from frame_quorum.errors import ScanError

    decoder.failure = StopIteration()
    with NativePixelChangeStream("unused.mkv") as stream, pytest.raises(ScanError, match="terminate"):
        next(stream)
    assert stream.diagnostics.status == "error"


def test_setup_failure_attempts_cleanup(decoder, monkeypatch):
    def fail(self):
        raise OSError("setup failure")

    monkeypatch.setattr(decoder, "__enter__", fail)
    stream = NativePixelChangeStream("unused.mkv")
    with pytest.raises(OSError, match="setup"):
        stream.__enter__()
    assert decoder.instances[-1].closed


def test_cancel_checked_after_native_return_before_measurement(decoder, monkeypatch):
    original = decoder.__next__
    stream = NativePixelChangeStream("unused.mkv")

    def returned(self):
        frame = original(self)
        stream.cancel()
        return frame

    monkeypatch.setattr(decoder, "__next__", returned)
    with stream:
        assert list(stream) == []
    assert stream.diagnostics.measurement_pixels == 0
    assert stream.diagnostics.video.returned_frames == 1


@pytest.mark.parametrize("event_type", ["NativePixelChangeUpdate", "NativePixelChangeEnd"])
def test_cancel_during_event_construction_does_not_publish_event(decoder, monkeypatch, event_type):
    stream = NativePixelChangeStream("unused.mkv")
    original = getattr(module, event_type)

    def construct(*args):
        result = original(*args)
        stream.cancel()
        return result

    monkeypatch.setattr(module, event_type, construct)
    with stream:
        events = list(stream)
    assert len(events) == (0 if event_type == "NativePixelChangeUpdate" else 6)
    assert stream.diagnostics.status == "cancelled"


def test_close_failure_is_visible_and_explicit_close_can_retry(decoder, monkeypatch):
    stream = NativePixelChangeStream("unused.mkv")
    with stream:
        source = decoder.instances[-1]
        original = source.close

        def fail():
            raise OSError("close failure")

        monkeypatch.setattr(source, "close", fail)
        with pytest.raises(OSError, match="close failure"):
            stream.close()
        assert stream.diagnostics.status == "error" and not stream.diagnostics.video.closed
        monkeypatch.setattr(source, "close", original)
        stream.close()
        assert stream.diagnostics.video.closed


def test_signed_baseline_pixel_cache_golden_bytes_unchanged():
    import hashlib
    import runpy

    from frame_quorum.native_measurements import NativeMeasurementLimits, _bounded_cache_lines
    from frame_quorum.native_pixel_changes import _lines

    fixture = runpy.run_path(str(Path(__file__).with_name("test_native_pixel_changes.py")))["data"]()
    raw = b"".join(_bounded_cache_lines(_lines(fixture), len(fixture.changes), NativeMeasurementLimits()))
    # Independently generated from the signed 2538a06 wheel (SHA256 1c4d954a...)
    # under -I -S, with no source package on its import path, before this assertion.
    assert len(raw) == 3617
    assert (
        hashlib.sha256(raw).hexdigest() == "b863f789598b2517db4eb0bd63896e860f7d63c72990c2cd1e76050b7eba62e8"
    )


def test_negative_and_equal_exact_pts_are_not_interpolated(decoder, monkeypatch):
    from dataclasses import replace

    original = decoder.__next__
    pts = (-10, -10, -1, 0, 0, 4)

    def coordinate(self):
        frame = original(self)
        return replace(frame, pts=pts[frame.sample_index])

    monkeypatch.setattr(decoder, "__next__", coordinate)
    with NativePixelChangeStream(
        "unused.mkv", detection=PixelChangeDetectionConfig(value_only=True)
    ) as stream:
        events = list(stream)
    assert [row.sample.sample.presentation_time for row in events[:-1]] == [
        Fraction(pts, 1000) for pts in pts
    ]
    assert events[2].closed_scene.end_time == Fraction(-1, 1000)
    assert events[4].closed_scene.end_time == 0


@pytest.mark.parametrize("field,value", [("sample_index", 2), ("generation", 1)])
def test_source_identity_mismatch_is_failure_not_a_completed_prefix(decoder, monkeypatch, field, value):
    from dataclasses import replace

    from frame_quorum.errors import ScanError

    original = decoder.__next__

    def coordinate(self):
        frame = original(self)
        return replace(frame, **{field: value})

    monkeypatch.setattr(decoder, "__next__", coordinate)
    with NativePixelChangeStream("unused.mkv") as stream, pytest.raises((ConfigurationError, ScanError)):
        next(stream)
    assert stream.diagnostics.status == "error" and stream.diagnostics.confirmed_samples == 0


def test_cli_stdout_short_write_fails_and_closes_owned_source(decoder, monkeypatch):
    import argparse
    import io

    from frame_quorum import cli
    from frame_quorum.errors import OutputError

    class ShortSink(io.StringIO):
        def write(self, text):
            return max(0, len(text) - 1)

    args = cli.build_parser().parse_args(["native-change-stream", "unused.mkv"])
    assert isinstance(args, argparse.Namespace)
    monkeypatch.setattr(cli.sys, "stdout", ShortSink())
    with pytest.raises(OutputError, match="short write"):
        cli._handle_change_stream(args)
    assert decoder.instances[-1].closed


def test_cli_stdout_oserror_preserves_failure_and_decoder_cleanup(decoder, monkeypatch):
    import io

    from frame_quorum import cli
    from frame_quorum.errors import OutputError

    class FailedSink(io.StringIO):
        def write(self, text):
            raise OSError("broken sink")

    args = cli.build_parser().parse_args(["native-change-stream", "unused.mkv"])
    monkeypatch.setattr(cli.sys, "stdout", FailedSink())
    with pytest.raises(OutputError, match="write failed"):
        cli._handle_change_stream(args)
    assert decoder.instances[-1].closed


def test_cli_binary_stdout_preserves_canonical_lf_and_flushes_each_line(decoder, monkeypatch):
    import hashlib
    import io
    import json
    from types import SimpleNamespace

    from frame_quorum import cli

    class Sink(io.BytesIO):
        def __init__(self):
            super().__init__()
            self.flushes = 0

        def flush(self):
            self.flushes += 1
            super().flush()

    sink = Sink()
    args = cli.build_parser().parse_args(["native-change-stream", "unused.mkv"])
    monkeypatch.setattr(cli.sys, "stdout", SimpleNamespace(buffer=sink))
    assert cli._handle_change_stream(args) == 0
    raw = sink.getvalue()
    lines = raw.splitlines(keepends=True)
    assert b"\r\n" not in raw and sink.flushes == len(lines) == 8
    assert json.loads(lines[-1])["sha256"] == hashlib.sha256(b"".join(lines[:-1])).hexdigest()
