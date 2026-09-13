from __future__ import annotations

import json
from dataclasses import replace
from fractions import Fraction

import pytest

import frame_quorum as fq
from frame_quorum import DetectionConfig, NativeConcatConfig, NativeConcatLimits
from frame_quorum.errors import ConfigurationError, ScanError
from tests.test_native_concat import decoder as _decoder_fixture

decoder = _decoder_fixture


def api(name):
    assert hasattr(fq, name), f"missing public composite scene API: {name}"
    assert name in fq.__all__
    return getattr(fq, name)


@pytest.mark.parametrize("step", [1, 2, 3, 7])
def test_complete_pipeline_preserves_native_identity_and_does_not_force_seam_cut(decoder, step):
    config = api("NativeConcatSceneConfig")(
        video=NativeConcatConfig(frame_step=step),
        detectors=(DetectionConfig(detector="content"),),
    )
    observed = []
    result = api("detect_native_concat_scenes")(decoder.clips, config, on_sample=observed.append)
    expected = [(c, p, i) for c in range(2) for i, p in enumerate([5000, 5040, 5100])][::step]
    assert [
        (r.sample.clip_index, r.sample.native.pts, r.sample.native.decode_index) for r in result.statistics
    ] == expected
    assert [r.sample for r in result.statistics] == observed
    assert [r.sample.sample_index for r in result.statistics] == list(range(len(expected)))
    assert all(
        r.sample.activation == 1 and r.sample.generation == r.sample.native.generation == 0
        for r in result.statistics
    )
    assert all(r.sample.timeline_digest == result.timeline.digest for r in result.statistics)
    assert all(
        m.stream_index == 2 for m in result.metadata
    )  # Video selector is zero, container index is two.
    assert result.cut_positions == result.cut_times == ()
    assert len(result.scenes) == 1 and len(result.scenes[0].spans) == 2
    assert result.scenes[0].end_time == Fraction(2, 5)
    assert result.scenes[0].end_reason == "declared_end"
    assert result.diagnostics.closed and result.diagnostics.activations == 4
    assert result.diagnostics.status == "intervals_exhausted"
    document = json.loads(json.dumps(result.to_dict()))
    assert document["execution"] == "offline_declared_concat"
    assert document["source_verified"] is False and document["coverage"] == "declared_only"
    assert all(c.closed for c in decoder.containers)


@pytest.mark.parametrize("maximum", [1, 3, 6])
def test_budget_prefix_is_never_returned_as_complete_scene_analysis(decoder, maximum):
    config = api("NativeConcatSceneConfig")(
        video=NativeConcatConfig(limits=NativeConcatLimits(max_frames=maximum))
    )
    with pytest.raises(ScanError, match="incomplete") as caught:
        api("detect_native_concat_scenes")(decoder.clips, config)
    assert caught.value.native_concat_cleanup.diagnostics.closed
    assert caught.value.native_concat_cleanup.diagnostics.status == "frame_limit"


@pytest.mark.parametrize(
    "limits", [NativeConcatLimits(max_activations=3), NativeConcatLimits(max_frames=500001)]
)
def test_known_impossible_activation_or_mapping_work_rejects_before_open(decoder, limits):
    config = api("NativeConcatSceneConfig")(video=NativeConcatConfig(limits=limits))
    with pytest.raises(ConfigurationError):
        api("detect_native_concat_scenes")(decoder.clips, config)
    assert not decoder.containers


def test_terminal_empty_range_still_probes_but_does_not_decode_or_make_scenes(decoder):
    config = api("NativeConcatSceneConfig")(video=NativeConcatConfig(start=Fraction(2, 5)))
    result = api("detect_native_concat_scenes")(decoder.clips, config)
    assert result.statistics == result.scenes == ()
    assert result.diagnostics.activations == 2 and result.diagnostics.decoded_frames == 0
    assert all(c.closed for c in decoder.containers)


@pytest.mark.parametrize(
    "error", [ValueError("callback"), KeyboardInterrupt("callback"), SystemExit("callback")]
)
def test_callback_failure_preserves_identity_and_explicit_cleanup_owner(decoder, error):
    def fail(sample):
        raise error

    with pytest.raises(type(error)) as caught:
        api("detect_native_concat_scenes")(decoder.clips, on_sample=fail)
    assert caught.value is error
    owner = error.native_concat_cleanup
    error.__traceback__ = None
    assert owner.diagnostics.closed and owner.diagnostics.returned_frames == 1
    count = len(decoder.containers)
    owner.close()
    assert len(decoder.containers) == count


@pytest.mark.parametrize(
    "changed",
    [
        {"minimum_votes": True},
        {"minimum_votes": 2},
        {"min_scene_samples": 0},
        {"detectors": []},
        {"video": None},
    ],
)
def test_config_exact_admission(changed):
    with pytest.raises(ConfigurationError):
        api("NativeConcatSceneConfig")(**changed)


def test_callback_value_and_native_coroutine_return_are_rejected_and_closed(decoder):
    pending = []

    async def invalid():
        return None

    def callback(sample):
        value = invalid()
        pending.append(value)
        return value

    with pytest.raises(ConfigurationError, match="return None"):
        api("detect_native_concat_scenes")(decoder.clips, on_sample=callback)
    assert pending[0].cr_frame is None
    assert all(c.closed for c in decoder.containers)


def test_returned_models_reject_forged_native_identity_and_decision_replay(decoder):
    result = api("detect_native_concat_scenes")(decoder.clips)
    first = result.statistics[0]
    with pytest.raises(ConfigurationError):
        replace(
            result,
            statistics=(replace(first, sample=replace(first.sample, activation=2)), *result.statistics[1:]),
        )
    with pytest.raises(ConfigurationError):
        replace(result, statistics=(replace(first, content_score=0.25), *result.statistics[1:]))
    with pytest.raises(ConfigurationError):
        replace(result, scenes=())
