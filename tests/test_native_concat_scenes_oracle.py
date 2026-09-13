"""Independent detector arithmetic and interval geometry, through the real owner.

Fake codec frames supply a controlled measurement corpus. This is not a codec or
media accuracy test; the separate integration suite decodes real lossless video.
"""

from __future__ import annotations

from fractions import Fraction

import pytest

from frame_quorum import (
    DetectionConfig,
    NativeConcatClip,
    NativeConcatConfig,
    NativeConcatSceneConfig,
    NativeVideoFrame,
    detect_native_concat_scenes,
)
from tests._concat_scene_oracle import hand_detector, records
from tests.test_native_concat import decoder as _decoder_fixture
from tests.test_native_video import FakeFrame

decoder = _decoder_fixture


@pytest.mark.parametrize("pattern", range(20))
@pytest.mark.parametrize("profile", ["content", "color", "luminance", "adaptive", "threshold", "quorum"])
@pytest.mark.parametrize("step,minimum", [(1, 1), (1, 3), (3, 1), (3, 3)])
def test_complete_measurement_sequence_matches_handwritten_evidence_and_partition(
    decoder, monkeypatch, pattern, profile, step, minimum
):
    corpus = records(pattern)
    divide = len(corpus) // 2
    clips = (
        NativeConcatClip(decoder.paths[0], start=5, end=10),
        NativeConcatClip(decoder.paths[1], start=20, end=25),
    )
    native_stamps = [5000 + i * 100 for i in range(divide)] + [
        20000 + i * 100 for i in range(len(corpus) - divide)
    ]
    decoder.frames[0][:] = [FakeFrame(p) for p in native_stamps[:divide]]
    decoder.frames[1][:] = [FakeFrame(p) for p in native_stamps[divide:]]
    measured = dict(zip(native_stamps, corpus, strict=True))
    monkeypatch.setattr(NativeVideoFrame, "measure", lambda frame: measured[frame.pts])
    names = ["content", "color", "luminance", "adaptive", "threshold"] if profile == "quorum" else [profile]
    configs = tuple(
        DetectionConfig(
            detector=name,
            threshold=0.3,
            min_scene_frames=minimum,
            window_radius=1,
            adaptive_ratio=2.3,
            include_final_fade=True,
        )
        for name in names
    )
    quorum = 3 if profile == "quorum" else 1
    result = detect_native_concat_scenes(
        clips,
        NativeConcatSceneConfig(
            video=NativeConcatConfig(frame_step=step),
            detectors=configs,
            minimum_votes=quorum,
            min_scene_samples=minimum,
        ),
    )
    selected = corpus[::step]
    expected = [hand_detector(selected, config) for config in configs]
    positions = list(range(len(corpus)))[::step]
    times = [Fraction(i, 10) if i < divide else 5 + Fraction(i - divide, 10) for i in positions]
    assert [r.sample.native.pts for r in result.statistics] == native_stamps[::step]
    assert [r.sample.presentation_time for r in result.statistics] == times
    cuts = []
    previous = 0
    for i, actual in enumerate(result.statistics):
        assert actual.content_score == pytest.approx(expected[0][0][i], rel=1e-11, abs=1e-12)
        votes = 0
        for evidence, (_content, scores, candidates, decisions) in zip(
            actual.detectors, expected, strict=True
        ):
            assert (
                evidence.score is None
                if scores[i] is None
                else evidence.score == pytest.approx(scores[i], rel=1e-11, abs=1e-12)
            )
            assert evidence.candidate == (i in candidates)
            assert (evidence.qualified, evidence.reason) == decisions[i]
            votes += decisions[i][0]
        if i == 0:
            decision = False, "sequence_start"
        elif votes < quorum:
            decision = False, "insufficient_votes"
        elif i - previous < minimum:
            decision = False, "short_previous_scene"
        elif len(selected) - i < minimum:
            decision = False, "short_final_scene"
        else:
            decision = True, "quorum"
            cuts.append(i)
            previous = i
        assert actual.votes == votes and (actual.accepted, actual.reason) == decision
    assert result.cut_positions == tuple(cuts)
    assert result.cut_times == tuple(times[i] for i in cuts)
    if not selected:
        assert result.scenes == ()
        return
    for ordinal, (scene, left, right) in enumerate(
        zip(result.scenes, [0, *cuts], [*cuts, len(selected)], strict=True)
    ):
        endpoint = times[right] if right < len(selected) else Fraction(10)
        assert (
            scene.ordinal,
            scene.start_position,
            scene.end_position,
            scene.start_time,
            scene.last_sample_time,
            scene.end_time,
            scene.end_reason,
        ) == (
            ordinal,
            left,
            right,
            times[left],
            times[right - 1],
            endpoint,
            "cut" if right < len(selected) else "declared_end",
        )
        spans = []
        for source, offset, origin in ((0, Fraction(0), Fraction(5)), (1, Fraction(5), Fraction(20))):
            a, b = max(times[left], offset), min(endpoint, offset + 5)
            if a < b:
                spans.append((source, a, b, origin + a - offset, origin + b - offset))
        assert [
            (s.clip_index, s.global_start, s.global_end, s.local_start, s.local_end) for s in scene.spans
        ] == spans


def test_equal_native_times_can_have_nonempty_sample_scene_without_positive_time_span(decoder, monkeypatch):
    from frame_quorum import FrameMetrics

    decoder.frames[0][:] = [FakeFrame(5000), FakeFrame(5100), FakeFrame(5100), FakeFrame(5190)]
    decoder.frames[1][:] = []
    gray = iter([0.0, 1.0, 0.0, 0.0])

    def measure(frame):
        value = next(gray)
        return FrameMetrics(0, value, 0.0, 0.0, 0.0, value, value, value)

    monkeypatch.setattr(NativeVideoFrame, "measure", measure)
    result = detect_native_concat_scenes(
        decoder.clips, NativeConcatSceneConfig(detectors=(DetectionConfig(detector="luminance"),))
    )
    assert result.cut_positions == (1, 2)
    assert result.scenes[1].sample_count == 1
    assert result.scenes[1].start_time == result.scenes[1].end_time == Fraction(1, 10)
    assert result.scenes[1].spans == ()
    assert result.scenes[1].to_dict()["spans"] == []
