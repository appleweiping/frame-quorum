from __future__ import annotations

import copy
from dataclasses import replace
from fractions import Fraction

import pytest

from frame_quorum import (
    DetectionConfig,
    NativeConcatConfig,
    NativeConcatLimits,
    NativeConcatSceneConfig,
    detect_native_concat_scenes,
)
from frame_quorum.errors import ConfigurationError
from tests.test_native_concat import decoder as _decoder_fixture

decoder = _decoder_fixture


def forge(value, **changes):
    result = copy.deepcopy(value)
    for key, changed in changes.items():
        object.__setattr__(result, key, changed)
    return result


@pytest.mark.parametrize(
    "changes",
    [
        {"detectors": ()},
        {"detectors": (DetectionConfig(), DetectionConfig())},
        {"detectors": (DetectionConfig(max_frames=10),)},
        {
            "video": NativeConcatConfig(limits=NativeConcatLimits(max_frames=500001)),
            "detectors": (DetectionConfig(detector="color"), DetectionConfig(detector="luminance")),
        },
    ],
)
def test_independent_detector_count_and_work_capacity_admission(changes):
    with pytest.raises(ConfigurationError):
        NativeConcatSceneConfig(**changes)


@pytest.mark.parametrize("kind", ["value", "async-function", "async-instance"])
def test_invalid_callback_is_rejected_before_any_source_open(decoder, kind):
    async def callback(sample):
        return None

    class Observer:
        async def __call__(self, sample):
            return None

    value = {"value": 3, "async-function": callback, "async-instance": Observer()}[kind]
    with pytest.raises(ConfigurationError, match="synchronous"):
        detect_native_concat_scenes(decoder.clips, on_sample=value)
    assert not decoder.containers


def test_direct_wrong_config_and_callback_value_preserve_contract(decoder):
    with pytest.raises(ConfigurationError, match="config"):
        detect_native_concat_scenes(decoder.clips, config=NativeConcatConfig())
    assert not decoder.containers
    with pytest.raises(ConfigurationError, match="return None"):
        detect_native_concat_scenes(decoder.clips, on_sample=lambda sample: 1)
    assert all(item.closed for item in decoder.containers)


@pytest.mark.parametrize(
    "kind",
    [
        "config",
        "timeline",
        "metadata-count",
        "metadata-path",
        "statistics",
        "scenes",
        "sample",
        "native",
        "metrics",
        "evidence",
        "source",
        "index",
        "order",
        "timeline-offset-type",
        "timeline-offset-value",
        "timeline-digest-value",
    ],
)
def test_result_geometry_type_and_observation_identity_boundaries(decoder, kind):
    result = detect_native_concat_scenes(decoder.clips)
    row = result.statistics[0]
    if kind == "config":
        bad = forge(result, config=None)
    elif kind == "timeline":
        bad = forge(result, timeline=None)
    elif kind == "metadata-count":
        bad = forge(result, metadata=result.metadata[:1])
    elif kind == "metadata-path":
        bad = forge(
            result, metadata=(replace(result.metadata[0], path=result.metadata[1].path), result.metadata[1])
        )
    elif kind == "statistics":
        bad = forge(result, statistics=(None, *result.statistics[1:]))
    elif kind == "scenes":
        bad = forge(result, scenes=(None,))
    elif kind.startswith("timeline-"):
        change = (
            {"offsets": (0, Fraction(1, 5), Fraction(2, 5))}
            if kind == "timeline-offset-type"
            else {"offsets": (Fraction(0), Fraction(1, 6), Fraction(2, 5))}
            if kind == "timeline-offset-value"
            else {"digest": "0" * 64}
        )
        bad = forge(result, timeline=forge(result.timeline, **change))
    else:
        if kind == "sample":
            row = forge(row, sample=None)
        elif kind == "native":
            row = forge(row, sample=forge(row.sample, native=None))
        elif kind == "metrics":
            row = forge(row, sample=forge(row.sample, native=forge(row.sample.native, metrics=None)))
        elif kind == "evidence":
            row = forge(row, detectors=(None,))
        elif kind == "source":
            row = replace(row, sample=replace(row.sample, clip_index=2))
        elif kind == "index":
            row = replace(row, sample=replace(row.sample, native=replace(row.sample.native, sample_index=3)))
        elif kind == "order":
            second = result.statistics[1]
            second = replace(
                second, sample=replace(second.sample, native=replace(second.sample.native, decode_index=0))
            )
            row = result.statistics[0]
            result = forge(result, statistics=(row, second, *result.statistics[2:]))
        bad = forge(result, statistics=(row, *result.statistics[1:]))
    with pytest.raises(ConfigurationError):
        bad.to_dict()


@pytest.mark.parametrize(
    "change",
    [
        {"status": "frame_limit"},
        {"closed": False},
        {"generation": 1, "seeks": 1},
        {"cleanup_errors": ("file.close",)},
        {"limit_scope": "total"},
        {"returned_frames": 5},
        {"source_activations": (1, 2), "activations": 3},
        {"opened_source_bytes": 65},
    ],
)
def test_result_requires_positive_complete_counter_and_cleanup_acknowledgement(decoder, change):
    result = detect_native_concat_scenes(decoder.clips)
    with pytest.raises(ConfigurationError):
        replace(result, diagnostics=replace(result.diagnostics, **change))


@pytest.mark.parametrize(
    "name,maximum",
    [
        ("max_manifest_source_bytes", 31),
        ("max_opened_source_bytes", 63),
        ("max_decoded_frames", 7),
        ("max_rgb_frames", 5),
        ("max_total_pixels", 31),
        ("max_source_bytes", 15),
        ("max_frame_pixels", 3),
        ("max_source_decoded_frames", 3),
        ("max_source_rgb_frames", 2),
        ("max_source_total_pixels", 15),
        ("max_sources", 1),
        ("max_span_parts", 1),
    ],
)
def test_result_cannot_be_relabelled_with_budgets_below_observed_work(decoder, name, maximum):
    result = detect_native_concat_scenes(decoder.clips)
    limits = replace(result.config.video.limits, **{name: maximum})
    options = replace(result.config, video=replace(result.config.video, limits=limits))
    with pytest.raises(ConfigurationError):
        replace(result, config=options)


@pytest.mark.parametrize(
    "change",
    [
        {"start_time": 0},
        {"start_time": Fraction(-1)},
        {"last_sample_time": Fraction(1)},
        {"end_reason": "unknown"},
        {"spans": ()},
    ],
)
def test_standalone_scene_exact_time_endpoint_and_span_contract(decoder, change):
    result = detect_native_concat_scenes(decoder.clips)
    with pytest.raises(ConfigurationError):
        replace(result.scenes[0], **change)


@pytest.mark.parametrize(
    "kind", ["missing", "geometry", "outside", "other-timeline", "overlapping-partition"]
)
def test_scene_span_geometry_is_exact_not_only_well_typed(decoder, kind):
    result = detect_native_concat_scenes(decoder.clips)
    scene = result.scenes[0]
    if kind == "missing":
        scene = replace(scene, spans=scene.spans[:1])
    elif kind == "geometry":
        scene = replace(scene, spans=(replace(scene.spans[0], global_start=Fraction(1, 100)), scene.spans[1]))
    elif kind == "outside":
        scene = forge(scene, spans=(forge(scene.spans[0], global_start=Fraction(-1)), scene.spans[1]))
    elif kind == "other-timeline":
        timeline = replace(
            result.timeline, clips=(replace(result.timeline.clips[0], start=4), result.timeline.clips[1])
        )
        scene = forge(scene, spans=tuple(forge(s, timeline=timeline) for s in scene.spans))
    rows = (scene, scene, scene) if kind == "overlapping-partition" else (scene,)
    with pytest.raises(ConfigurationError):
        replace(result, scenes=rows)


def test_successful_standalone_scene_serialization_and_alternate_shared_manifest(decoder):
    result = detect_native_concat_scenes(decoder.clips)
    assert result.scenes[0].to_dict() == result.to_dict()["scenes"][0]
    timeline = replace(result.timeline)
    scene = replace(
        result.scenes[0], spans=tuple(replace(s, timeline=timeline) for s in result.scenes[0].spans)
    )
    assert replace(result, scenes=(scene,)).to_dict() == result.to_dict()


def test_relabelled_window_must_fit_declared_duration(decoder):
    result = detect_native_concat_scenes(decoder.clips)
    with pytest.raises(ConfigurationError, match="outside"):
        replace(result, config=replace(result.config, video=replace(result.config.video, end=Fraction(1))))
