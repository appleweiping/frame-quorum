from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from fractions import Fraction

import pytest

import frame_quorum.native_concat_scenes as scenes
from frame_quorum import NativeConcatSceneConfig, NativeVideoFrame, detect_native_concat_scenes
from frame_quorum.errors import ConfigurationError, ScanError
from tests.test_native_concat import decoder as _decoder_fixture

decoder = _decoder_fixture


@pytest.mark.parametrize("stage", ["enter", "callback"])
@pytest.mark.parametrize("primary_type", [ValueError, KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("cleanup_type", [RuntimeError, KeyboardInterrupt, SystemExit])
def test_failed_cleanup_owner_is_explicit_retryable_and_original_control_priority_is_preserved(
    decoder, stage, primary_type, cleanup_type
):
    primary, secondary = primary_type("primary"), cleanup_type("cleanup")
    allow_close = False
    seen = []

    def hook(container, index):
        if len(decoder.containers) != (1 if stage == "enter" else 3):
            return
        close = container.close

        def fail_close():
            if not allow_close:
                raise secondary
            close()

        container.close = fail_close

    def callback(sample):
        seen.append(sample)
        raise primary

    decoder.hook = hook
    with pytest.raises(BaseException) as caught:
        detect_native_concat_scenes(decoder.clips, on_sample=callback)
    selected = caught.value
    if stage == "callback":
        assert selected is (
            primary if not isinstance(primary, Exception) or isinstance(secondary, Exception) else secondary
        )
    elif not isinstance(secondary, Exception):
        assert selected is secondary
    else:
        assert isinstance(selected, ScanError)
    owner = selected.native_concat_cleanup
    selected.__traceback__ = None
    primary.__traceback__ = secondary.__traceback__ = None
    assert not owner.diagnostics.closed
    assert "container.close" in owner.diagnostics.cleanup_errors
    before = len(decoder.containers), len(seen), owner.diagnostics.activations
    with ThreadPoolExecutor(max_workers=1) as pool, pytest.raises(ConfigurationError, match="single-owner"):
        pool.submit(owner.close).result()
    allow_close = True
    owner.close()
    assert owner.diagnostics.closed
    assert "container.close" in owner.diagnostics.cleanup_errors  # Historical failure is not erased.
    assert (len(decoder.containers), len(seen), owner.diagnostics.activations) == before


@pytest.mark.parametrize("seam", ["measure", "statistics", "scenes", "result"])
@pytest.mark.parametrize("error_type", [MemoryError, KeyboardInterrupt])
def test_measurement_and_postdecode_allocation_failures_keep_same_closed_owner(
    decoder, monkeypatch, seam, error_type
):
    error = error_type("injected")

    def fail(*args, **kwargs):
        raise error

    if seam == "measure":
        monkeypatch.setattr(NativeVideoFrame, "measure", fail)
    else:
        monkeypatch.setattr(
            scenes,
            {"statistics": "_statistics", "scenes": "_scenes", "result": "NativeConcatSceneResult"}[seam],
            fail,
        )
    with pytest.raises(error_type) as caught:
        detect_native_concat_scenes(decoder.clips)
    assert caught.value is error and error.native_concat_cleanup.diagnostics.closed


class Hostile:
    def __getattribute__(self, name):
        raise AssertionError("unexpected attribute hook")

    def __len__(self):
        raise AssertionError("unexpected length hook")

    def __eq__(self, other):
        raise AssertionError("unexpected equality hook")

    def __hash__(self):
        raise AssertionError("unexpected hash hook")


@pytest.mark.parametrize(
    "path",
    [
        "config.video",
        "config.detectors",
        "config.minimum_votes",
        "config.min_scene_samples",
        "config.detectors.0.detector",
        "config.detectors.0.threshold",
        "timeline.clips",
        "timeline.offsets",
        "timeline.digest",
        "timeline.clips.0.path",
        "timeline.clips.0.start",
        "metadata",
        "metadata.0.path",
        "metadata.0.width",
        "metadata.0.time_base",
        "diagnostics.status",
        "diagnostics.limit_scope",
        "diagnostics.source_decoded_frames",
        "diagnostics.cleanup_errors",
        "statistics",
        "scenes",
        "statistics.0.sample.timeline_digest",
        "statistics.0.sample.native.metrics.perceptual_hash",
        "statistics.0.sample.native.metrics.mean_red",
        "statistics.0.sample.presentation_time",
        "statistics.0.sample.native.pts",
        "statistics.0.detectors",
        "statistics.0.detectors.0.detector",
        "statistics.0.detectors.0.score",
        "statistics.0.content_score",
        "scenes.0.spans",
        "scenes.0.spans.0.clip_index",
        "scenes.0.spans.0.global_end",
        "scenes.0.spans.0.timeline",
    ],
)
def test_forged_nested_values_reject_before_user_length_equality_or_attribute_hooks(decoder, path):
    result = copy.deepcopy(detect_native_concat_scenes(decoder.clips))
    target = result
    parts = path.split(".")
    for part in parts[:-1]:
        target = target[int(part)] if part.isdigit() else getattr(target, part)
    object.__setattr__(target, parts[-1], Hostile())
    with pytest.raises(ConfigurationError):
        result.to_dict()


def test_result_rejects_equal_but_independently_owned_per_span_timelines(decoder):
    result = detect_native_concat_scenes(decoder.clips)
    scene = result.scenes[0]
    second = replace(scene.spans[1], timeline=replace(result.timeline))
    bad = replace(scene, spans=(scene.spans[0], second))
    with pytest.raises(ConfigurationError, match="share one"):
        replace(result, scenes=(bad,))


def test_declared_end_beyond_timeline_is_not_silently_clipped(decoder):
    from frame_quorum import NativeConcatConfig

    with pytest.raises(ConfigurationError, match="outside"):
        detect_native_concat_scenes(
            decoder.clips, NativeConcatSceneConfig(video=NativeConcatConfig(end=Fraction(1)))
        )
    assert not decoder.containers
