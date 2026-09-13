"""Independent consistency checks against observed native concat execution."""

from dataclasses import replace
from fractions import Fraction

import pytest

from frame_quorum import NativeConcatConfig, NativeConcatSceneConfig, detect_native_concat_scenes
from frame_quorum.errors import ConfigurationError, ScanError
from tests.test_native_concat import decoder as _decoder_fixture
from tests.test_native_video import FakeFrame

decoder = _decoder_fixture


@pytest.mark.parametrize("subrange", [False, True])
def test_result_terminal_status_matches_exact_declared_endpoint(decoder, subrange):
    video = NativeConcatConfig(end=Fraction(1, 10)) if subrange else NativeConcatConfig()
    result = detect_native_concat_scenes(decoder.clips, NativeConcatSceneConfig(video=video))
    expected = "range_end" if subrange else "intervals_exhausted"
    assert result.diagnostics.status == expected
    false_status = "intervals_exhausted" if subrange else "range_end"
    with pytest.raises(ConfigurationError):
        replace(result, diagnostics=replace(result.diagnostics, status=false_status)).to_dict()


@pytest.mark.parametrize("dimension", ["width", "height"])
def test_result_metadata_preserves_required_equal_source_dimensions(decoder, dimension):
    result = detect_native_concat_scenes(decoder.clips)
    first, second = result.metadata
    assert (first.width, first.height) == (second.width, second.height) == (2, 2)
    different = replace(second, **{dimension: 3})
    with pytest.raises(ConfigurationError):
        replace(result, metadata=(first, different)).to_dict()


@pytest.mark.parametrize("change", ["global-step", "returned-ceiling", "rgb-ceiling", "source-rgb-ceiling"])
def test_result_cannot_relabel_global_stride_or_no_peek_completeness(decoder, change):
    result = detect_native_concat_scenes(decoder.clips)
    assert result.diagnostics.returned_frames == result.diagnostics.owned_rgb_frames == 6
    if change == "global-step":
        video = replace(result.config.video, frame_step=2)
    else:
        name, maximum = {
            "returned-ceiling": ("max_frames", 6),
            "rgb-ceiling": ("max_rgb_frames", 6),
            "source-rgb-ceiling": ("max_source_rgb_frames", 3),
        }[change]
        video = replace(result.config.video, limits=replace(result.config.video.limits, **{name: maximum}))
    config = replace(result.config, video=video)
    if change == "global-step":
        actual = detect_native_concat_scenes(decoder.clips, config)
        assert actual.diagnostics.owned_rgb_frames == 6
        assert actual.diagnostics.returned_frames == 3
    else:
        with pytest.raises(ScanError) as failure:
            detect_native_concat_scenes(decoder.clips, config)
        assert failure.value.native_concat_cleanup.diagnostics.status in {"frame_limit", "rgb_limit"}
    with pytest.raises(ConfigurationError):
        replace(result, config=config).to_dict()


def test_selected_native_rgb_index_obeys_known_global_stride_counter(decoder):
    config = NativeConcatSceneConfig(video=NativeConcatConfig(frame_step=2))
    result = detect_native_concat_scenes(decoder.clips, config)
    assert result.diagnostics.source_rgb_frames == (3, 3)
    assert [(r.sample.clip_index, r.sample.native.sample_index) for r in result.statistics] == [
        (0, 0),
        (0, 2),
        (1, 1),
    ]
    last = result.statistics[-1]
    # RGB positions are 0, 2, 4. Relabeling the last child index to 0 means
    # global RGB position 3, which stride 2 does not select.
    native = replace(last.sample.native, sample_index=0)
    row = replace(last, sample=replace(last.sample, native=native))
    with pytest.raises(ConfigurationError):
        replace(result, statistics=(*result.statistics[:-1], row)).to_dict()


@pytest.mark.parametrize("last_status", ["frame_limit", "decode_limit", "error", "interrupted", None])
def test_complete_result_rejects_unfinished_or_unknown_last_native_child(decoder, last_status):
    result = detect_native_concat_scenes(decoder.clips)
    assert result.diagnostics.activations == 4
    assert result.diagnostics.last_child_status == "range_end"
    with pytest.raises(ConfigurationError):
        replace(result, diagnostics=replace(result.diagnostics, last_child_status=last_status)).to_dict()


@pytest.mark.parametrize("step", [1, 2, 3, 4, 7])
def test_complete_range_boundary_can_exactly_consume_decode_and_pixel_budgets(decoder, step):
    config = NativeConcatSceneConfig()
    limits = replace(
        config.video.limits,
        max_decoded_frames=8,
        max_source_decoded_frames=4,
        max_total_pixels=32,
        max_source_total_pixels=16,
    )
    config = replace(config, video=replace(config.video, limits=limits, frame_step=step))
    result = detect_native_concat_scenes(decoder.clips, config)
    assert result.diagnostics.decoded_frames == 8
    assert result.diagnostics.source_decoded_frames == (4, 4)
    assert result.diagnostics.decoded_pixels_observed == 32
    assert result.diagnostics.source_pixels == (16, 16)
    assert result.diagnostics.last_child_status == "range_end"
    assert result.diagnostics.returned_frames == (6 + step - 1) // step
    result.to_dict()


@pytest.mark.parametrize("outside_dimensions", [(1, 1), (3, 2)])
@pytest.mark.parametrize("preroll", [False, True])
def test_non_rgb_boundary_and_preroll_pixels_need_not_match_metadata(decoder, outside_dimensions, preroll):
    width, height = outside_dimensions
    for frames in decoder.frames:
        frames[-1] = FakeFrame(5200, width=width, height=height)
        if preroll:
            frames.insert(0, FakeFrame(4900, width=width, height=height))
    result = detect_native_concat_scenes(decoder.clips)
    decoded = 4 + int(preroll)
    pixels = 12 + (1 + int(preroll)) * width * height
    assert result.diagnostics.source_decoded_frames == (decoded, decoded)
    assert result.diagnostics.source_rgb_frames == (3, 3)
    assert result.diagnostics.source_pixels == (pixels, pixels)
    assert result.diagnostics.returned_frames == 6
    result.to_dict()
