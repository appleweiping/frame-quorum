from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, replace
from fractions import Fraction
from pathlib import Path

import pytest

import frame_quorum.native_scenes as module
from frame_quorum import (
    DetectionConfig,
    Frame,
    FrameMetrics,
    NativeDetectorStatistic,
    NativeScene,
    NativeSceneConfig,
    NativeSceneResult,
    NativeSceneSample,
    NativeSceneStatistic,
    NativeVideoConfig,
    NativeVideoDiagnostics,
    NativeVideoFrame,
    NativeVideoMetadata,
    NativeVideoStatus,
    detect_native_scenes,
    detect_scenes,
)
from frame_quorum.errors import ConfigurationError


def metric(value):
    return FrameMetrics(0, value, 0, 0, 0, value, value, value)


@pytest.fixture
def source(monkeypatch):
    """A deterministic owned source, independent of codecs and image paths."""
    streams = []

    def install(values, *, pts=None, status=NativeVideoStatus.EOF, decode_indices=None):
        times = list(range(len(values))) if pts is None else pts
        indices = list(range(len(values))) if decode_indices is None else decode_indices

        class Stream:
            def __init__(self, path, config):
                self.closed = False
                self.config = config
                self.count = 0
                self.metadata = NativeVideoMetadata(
                    Path(path),
                    20,
                    "matroska",
                    "controlled",
                    0,
                    1,
                    1,
                    Fraction(1, 1000),
                    times[0] if times else None,
                    None,
                    None,
                    None,
                )
                streams.append(self)

            def __enter__(self):
                return self

            def __exit__(self, *args):
                self.closed = True

            def __iter__(self):
                for position, (pts, decode_index, _value) in enumerate(
                    zip(times, indices, values, strict=True)
                ):
                    self.count += 1
                    yield NativeVideoFrame(pts, Fraction(1, 1000), decode_index, position, 0, 1, 1, bytes(3))

            @property
            def diagnostics(self):
                return NativeVideoDiagnostics(
                    status, indices[-1] + 1 if indices else 0, self.count, self.count, 0, self.closed
                )

        monkeypatch.setattr(module, "NativeVideoStream", Stream)
        monkeypatch.setattr(NativeVideoFrame, "measure", lambda frame: metric(values[frame.sample_index]))
        return streams

    return install


@pytest.mark.parametrize("name", ["content", "color", "luminance", "adaptive", "threshold"])
@pytest.mark.parametrize("minimum", [1, 2, 4])
def test_all_five_detectors_preserve_offline_scores_candidates_reasons_and_partition(source, name, minimum):
    values = [1, 1, 0, 0, 0.04, 0.06, 1, 1, 0, 0, 0, 0.8, 0.8]
    pts = [5000 + index * index for index in range(len(values))]
    streams = source(values, pts=pts)
    detector = DetectionConfig(
        detector=name, min_scene_frames=minimum, window_radius=1, include_final_fade=True
    )
    native = detect_native_scenes("input.mkv", NativeSceneConfig(detectors=(detector,)))
    # The independent legacy oracle has actual image-domain records but receives
    # the same hand-authored measurements. Production never fabricates these.
    offline = detect_scenes(
        tuple(
            Frame(i, Path(f"{i}.png"), f"{i}.png", None, 1, 1, 1, metric(value))
            for i, value in enumerate(values)
        ),
        detector,
    )
    assert streams[0].closed
    assert native.cut_positions == offline.cut_indices
    for native_row, original in zip(native.statistics, offline.statistics, strict=True):
        (evidence,) = native_row.detectors
        assert native_row.content_score == original.content_score
        assert evidence.score == original.detector_score
        assert evidence.candidate == original.candidate
        assert evidence.qualified == original.accepted
        assert evidence.reason == original.reason
    assert [(scene.start_position, scene.end_position) for scene in native.scenes] == [
        (scene.start_index, scene.end_index) for scene in offline.scenes
    ]
    assert native.cut_times == tuple(Fraction(pts[i], 1000) for i in offline.cut_indices)
    assert native.scenes[-1].end_time is None
    assert native.scenes[-1].last_sample_time == Fraction(pts[-1], 1000)


def test_ensemble_votes_qualified_candidates_then_applies_own_minimum(source):
    source([0, 1, 1, 0, 0, 1, 0])
    configs = (
        DetectionConfig(detector="luminance", threshold=0.5),
        DetectionConfig(detector="color", threshold=0.5, min_scene_frames=2),
    )
    result = detect_native_scenes("input.mkv", NativeSceneConfig(detectors=configs, minimum_votes=2))
    assert result.cut_positions == (3, 5)
    first = result.statistics[1]
    assert [item.candidate for item in first.detectors] == [True, True]
    assert [item.qualified for item in first.detectors] == [True, False]
    assert first.votes == 1 and first.reason == "insufficient_votes"
    # Union is exact-position voting, not temporal clustering. New union cuts
    # may require additional suppression to keep the final sample-count minimum.
    result = detect_native_scenes("input.mkv", NativeSceneConfig(detectors=configs, min_scene_samples=3))
    assert result.cut_positions == (3,)
    assert result.statistics[1].reason == "short_previous_scene"
    assert result.statistics[6].reason == "short_final_scene"
    assert result.statistics[5].reason == "short_previous_scene"


def test_vfr_stride_generation_and_equal_time_mapping_are_exact(source):
    pts = [5000, 5000, 5251, 5600]
    source([0, 1, 0, 1], pts=pts, decode_indices=[4, 7, 10, 13])
    result = detect_native_scenes(
        "input.mkv",
        NativeSceneConfig(
            video=NativeVideoConfig(start=Fraction(5), frame_step=3),
            detectors=(DetectionConfig(detector="luminance", threshold=0.5),),
        ),
    )
    assert result.cut_positions == (1, 2, 3)
    assert result.cut_times == (Fraction(5), Fraction(5251, 1000), Fraction(28, 5))
    assert result.scenes[0].start_time == result.scenes[0].end_time == 5
    assert [row.sample.decode_index for row in result.statistics] == [4, 7, 10, 13]
    assert [row.sample.sample_index for row in result.statistics] == [0, 1, 2, 3]
    assert {row.sample.generation for row in result.statistics} == {0}
    output = json.loads(json.dumps(result.to_dict(), allow_nan=False))
    assert output["scenes"][1]["end_time"] == {"numerator": 5251, "denominator": 1000}
    assert "rgb" not in output["statistics"][0]["sample"]
    assert "path" not in output["statistics"][0]["sample"]


@pytest.mark.parametrize(
    "status", [NativeVideoStatus.EOF, NativeVideoStatus.FRAME_LIMIT, NativeVideoStatus.DECODE_LIMIT]
)
def test_terminal_limits_and_eof_never_fabricate_endpoint_even_when_end_requested(source, status):
    source([0, 1], pts=[5000, 5040], status=status)
    video = NativeVideoConfig(
        end=Fraction(10),
        max_frames=2 if status is NativeVideoStatus.FRAME_LIMIT else 10,
        max_decoded_frames=2 if status is NativeVideoStatus.DECODE_LIMIT else 20,
    )
    result = detect_native_scenes("input.mkv", NativeSceneConfig(video=video))
    assert result.diagnostics.status is status
    assert result.scenes[-1].end_time is None
    assert result.scenes[-1].end_reason == "unknown"


def test_only_observed_range_end_can_supply_requested_tail_endpoint(source):
    source([0, 1], pts=[5000, 5040], status=NativeVideoStatus.RANGE_END)
    result = detect_native_scenes(
        "input.mkv", NativeSceneConfig(video=NativeVideoConfig(end=Fraction(101, 20)))
    )
    assert result.scenes[-1].end_time == Fraction(101, 20)
    assert result.scenes[-1].end_reason == "requested_end"


@pytest.mark.parametrize("name", ["content", "color", "luminance", "adaptive", "threshold"])
def test_empty_selection_is_valid_without_scenes_or_endpoints(source, name):
    source([])
    result = detect_native_scenes("input.mkv", NativeSceneConfig(detectors=(DetectionConfig(detector=name),)))
    assert result.statistics == result.scenes == result.cut_positions == result.cut_times == ()
    assert result.to_dict()["sample_count"] == 0


def test_measured_callback_is_immutable_rgb_free_and_decoder_closes_on_success(source):
    streams = source([0, 1])
    observed = []

    def callback(sample):
        assert not streams[0].closed
        assert not hasattr(sample, "rgb")
        with pytest.raises(FrozenInstanceError):
            sample.pts = 99
        observed.append(sample)

    result = detect_native_scenes("input.mkv", on_sample=callback)
    assert observed == [row.sample for row in result.statistics]
    assert streams[0].closed


@pytest.mark.parametrize("error", [RuntimeError("observer failed"), KeyboardInterrupt(), SystemExit(4)])
def test_callback_primary_errors_propagate_after_cleanup(source, error):
    streams = source([0, 1])

    def fail(sample):
        raise error

    with pytest.raises(type(error)) as caught:
        detect_native_scenes("input.mkv", on_sample=fail)
    assert caught.value is error
    assert streams[0].closed and streams[0].count == 1


@pytest.mark.parametrize("bad", [False, 1, object()])
def test_bad_callback_value_rejects_and_closes(source, bad):
    streams = source([0, 1])
    with pytest.raises(ConfigurationError, match="return None"):
        detect_native_scenes("input.mkv", on_sample=lambda sample: bad)
    assert streams[0].closed


@pytest.mark.parametrize("cleanup_failure", [False, True])
def test_returned_coroutine_is_closed_before_callback_contract_error(source, cleanup_failure):
    streams = source([0, 1])
    closed = []

    class Suspend:
        def __await__(self):
            yield

    async def invalid():
        try:
            await Suspend()
        finally:
            closed.append(True)
            if cleanup_failure:
                raise RuntimeError("private cleanup detail")

    def callback(sample):
        coroutine = invalid()
        coroutine.send(None)
        return coroutine

    with pytest.raises(ConfigurationError, match="return None"):
        detect_native_scenes("input.mkv", on_sample=callback)
    assert closed == [True] and streams[0].closed


def test_async_or_noncallable_callback_rejected_before_decode(source):
    streams = source([])

    async def callback(sample):
        pass

    class AsyncCallback:
        async def __call__(self, sample):
            pass

    for bad in (callback, AsyncCallback(), 1, False):
        with pytest.raises(ConfigurationError, match="synchronous"):
            detect_native_scenes("input.mkv", on_sample=bad)
    assert not streams


def test_measurement_failure_closes_stream_without_observer_or_partial_result(source, monkeypatch):
    streams = source([0, 1])
    observed = []
    error = RuntimeError("measurement failed")

    def fail(frame):
        raise error

    monkeypatch.setattr(NativeVideoFrame, "measure", fail)
    with pytest.raises(RuntimeError) as caught:
        detect_native_scenes("input.mkv", on_sample=observed.append)
    assert caught.value is error and not observed
    assert streams[0].closed and streams[0].count == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"video": None},
        {"detectors": []},
        {"detectors": ()},
        {"detectors": (None,)},
        {"detectors": (DetectionConfig(),) * 2},
        {"detectors": (DetectionConfig(detector="wrong"),)},
        {"minimum_votes": True},
        {"minimum_votes": 2},
        {"min_scene_samples": False},
        {"min_scene_samples": 10**1000},
        {"detectors": (DetectionConfig(max_frames=1),)},
        {
            "video": NativeVideoConfig(max_frames=200_001),
            "detectors": tuple(
                DetectionConfig(detector=name)
                for name in ["content", "color", "luminance", "adaptive", "threshold"]
            ),
        },
    ],
)
def test_config_rejects_invalid_types_and_predecode_work_amplification(kwargs):
    with pytest.raises(ConfigurationError):
        NativeSceneConfig(**kwargs)


def test_config_ceiling_is_inclusive_and_serialization_detached():
    config = NativeSceneConfig(video=NativeVideoConfig(max_frames=1_000_000))
    assert config.video.max_frames == 1_000_000
    value = config.to_dict()
    value["video"]["max_frames"] = 0
    value["detectors"][0]["window_radius"] = 0
    assert config.video.max_frames == 1_000_000 and config.detectors[0].window_radius == 2


def test_invalid_top_level_configuration_rejected_without_decode(source):
    streams = source([])
    with pytest.raises(ConfigurationError, match="config"):
        detect_native_scenes("input.mkv", {})
    assert not streams


@pytest.fixture
def result(source):
    source([0, 0, 1, 1])
    return detect_native_scenes(
        "input.mkv", NativeSceneConfig(detectors=(DetectionConfig(detector="luminance"),))
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"pts": True},
        {"pts": 2**63},
        {"time_base": 0.1},
        {"time_base": Fraction(0)},
        {"decode_index": -1},
        {"sample_index": True},
        {"generation": -1},
        {"metrics": {}},
        {"metrics": metric(float("nan"))},
    ],
)
def test_sample_rejects_malformed_fields(result, changes):
    with pytest.raises(ConfigurationError):
        replace(result.statistics[0].sample, **changes)


def test_native_products_beyond_int64_are_preserved_without_float():
    sample = NativeSceneSample(2**63 - 1, Fraction(2**63 - 1), 0, 0, 0, metric(0))
    instant = (2**63 - 1) ** 2
    assert sample.presentation_time == instant
    scene = NativeScene(0, 0, 1, sample.presentation_time, sample.presentation_time, None, "unknown")
    assert scene.to_dict()["start_time"]["numerator"] == instant


@pytest.mark.parametrize(
    "changes",
    [
        {"detector": []},
        {"detector": "other"},
        {"score": None},
        {"score": 2},
        {"score": True},
        {"score": 10**1000},
        {"candidate": 1},
        {"qualified": 1},
        {"reason": []},
        {"reason": "unknown"},
        {"qualified": True},
        {"candidate": True},
    ],
)
def test_detector_statistic_rejects_inconsistent_fields(result, changes):
    with pytest.raises(ConfigurationError):
        replace(result.statistics[0].detectors[0], **changes)


def test_adaptive_candidate_cannot_have_absent_score():
    with pytest.raises(ConfigurationError):
        NativeDetectorStatistic("adaptive", None, True, False, "short_previous_scene")
    with pytest.raises(ConfigurationError):
        NativeDetectorStatistic("adaptive", None, True, True, "adaptive_peak")


@pytest.mark.parametrize("qualified, reason", [(True, "below_threshold"), (False, "distance_threshold")])
def test_qualified_flag_must_agree_with_detector_reason(qualified, reason):
    with pytest.raises(ConfigurationError, match="qualified flag"):
        NativeDetectorStatistic("luminance", 1.0, True, qualified, reason)


@pytest.mark.parametrize(
    "changes",
    [
        {"sample": None},
        {"content_score": float("inf")},
        {"detectors": []},
        {"detectors": ()},
        {"detectors": (None,)},
        {"accepted": 1},
        {"reason": []},
        {"reason": "fake"},
        {"reason": "quorum", "accepted": True},
    ],
)
def test_aggregate_statistic_rejects_inconsistent_fields(result, changes):
    with pytest.raises(ConfigurationError):
        replace(result.statistics[0], **changes)


def test_aggregate_names_unique(result):
    evidence = result.statistics[0].detectors[0]
    with pytest.raises(ConfigurationError):
        replace(result.statistics[0], detectors=(evidence, evidence))


@pytest.mark.parametrize(
    "changes",
    [
        {"ordinal": True},
        {"start_position": -1},
        {"end_position": 0},
        {"start_time": 0.0},
        {"start_time": Fraction(2**128)},
        {"start_time": Fraction(1, 2**64)},
        {"last_sample_time": Fraction(-1)},
        {"end_time": Fraction(-1)},
        {"end_time": None},
        {"end_reason": []},
        {"end_reason": "unknown"},
    ],
)
def test_scene_requires_exact_coherent_endpoints(result, changes):
    with pytest.raises(ConfigurationError):
        replace(result.scenes[0], **changes)


@pytest.mark.parametrize(
    "change",
    [
        "config",
        "metadata",
        "diagnostics",
        "live",
        "error",
        "generation",
        "statistics",
        "excess_statistics",
        "scenes",
        "excess_scenes",
        "returned",
        "decode_budget",
        "range_without_end",
        "frame_limit",
        "decode_limit",
        "row_type",
        "sample_index",
        "sample_generation",
        "decode_index",
        "stride",
        "time_order",
        "interval",
        "detector_order",
        "decision",
        "first_content",
        "scene_type",
        "partition",
    ],
)
def test_result_rejects_structurally_contradictory_records(result, change):
    kwargs = {}
    rows = list(result.statistics)
    if change in {"config", "metadata", "diagnostics"}:
        kwargs[change] = None
    elif change in {
        "live",
        "error",
        "generation",
        "returned",
        "decode_budget",
        "range_without_end",
        "frame_limit",
        "decode_limit",
    }:
        changes = {
            "live": {"closed": False},
            "error": {"status": NativeVideoStatus.ERROR},
            "generation": {"generation": 1},
            "returned": {"returned_frames": 0},
            "decode_budget": {"decoded_frames": 100_001},
            "range_without_end": {"status": NativeVideoStatus.RANGE_END},
            "frame_limit": {"status": NativeVideoStatus.FRAME_LIMIT},
            "decode_limit": {"status": NativeVideoStatus.DECODE_LIMIT},
        }
        kwargs["diagnostics"] = replace(result.diagnostics, **changes[change])
    elif change in {"statistics", "scenes"}:
        kwargs[change] = []
    elif change == "excess_statistics":
        kwargs["statistics"] = result.statistics * 2501
    elif change == "excess_scenes":
        kwargs["scenes"] = result.scenes * 3
    elif change == "row_type":
        rows[1] = None
    elif change in {"sample_index", "sample_generation", "decode_index", "stride", "time_order"}:
        changes = {
            "sample_index": {"sample_index": 0},
            "sample_generation": {"generation": 1},
            "decode_index": {"decode_index": 100},
            "stride": {"decode_index": 0},
            "time_order": {"pts": -1},
        }
        rows[1] = replace(rows[1], sample=replace(rows[1].sample, **changes[change]))
    elif change == "interval":
        kwargs["config"] = replace(result.config, video=replace(result.config.video, start=Fraction(1)))
    elif change == "detector_order":
        kwargs["config"] = replace(result.config, detectors=(DetectionConfig(detector="color"),))
    elif change == "decision":
        rows[2] = replace(rows[2], accepted=False, reason="insufficient_votes")
    elif change == "first_content":
        rows[0] = replace(rows[0], content_score=0.1)
    elif change == "scene_type":
        kwargs["scenes"] = (None,)
    elif change == "partition":
        kwargs["scenes"] = result.scenes[:1]
    kwargs.setdefault("statistics", tuple(rows))
    with pytest.raises(ConfigurationError):
        replace(result, **kwargs)


def test_result_public_constructors_and_roundtrip_detached(result):
    assert type(result) is NativeSceneResult
    assert type(result.statistics[0]) is NativeSceneStatistic
    assert type(result.scenes[0]) is NativeScene
    serialized = result.to_dict()
    serialized["statistics"][0]["sample"]["pts"] = 99
    assert result.statistics[0].sample.pts == 0
    assert sum(scene.sample_count for scene in result.scenes) == 4
