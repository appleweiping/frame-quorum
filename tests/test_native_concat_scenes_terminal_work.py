from dataclasses import replace
from fractions import Fraction

import pytest

from frame_quorum import NativeConcatConfig, NativeConcatSceneConfig, detect_native_concat_scenes
from frame_quorum.errors import ConfigurationError
from tests.test_native_concat import decoder as _decoder_fixture

decoder = _decoder_fixture


@pytest.mark.parametrize("probe_only", [False, True])
@pytest.mark.parametrize("with_rgb", [False, True])
def test_probe_only_source_cannot_claim_decoded_or_rgb_work(decoder, probe_only, with_rgb):
    video = (
        NativeConcatConfig(start=Fraction(2, 5)) if probe_only else NativeConcatConfig(end=Fraction(1, 10))
    )
    result = detect_native_concat_scenes(decoder.clips, NativeConcatSceneConfig(video=video))
    d = result.diagnostics
    assert d.source_activations[1] == 1
    assert (d.source_decoded_frames[1], d.source_rgb_frames[1], d.source_pixels[1]) == (0, 0, 0)
    rgb, pixels = int(with_rgb), 4 if with_rgb else 1
    forged = replace(
        d,
        decoded_frames=d.decoded_frames + 1,
        owned_rgb_frames=d.owned_rgb_frames + rgb,
        decoded_pixels_observed=d.decoded_pixels_observed + pixels,
        source_decoded_frames=(d.source_decoded_frames[0], 1),
        source_rgb_frames=(d.source_rgb_frames[0], rgb),
        source_pixels=(d.source_pixels[0], pixels),
    )
    with pytest.raises(ConfigurationError):
        replace(result, diagnostics=forged).to_dict()


@pytest.mark.parametrize("mode", ["full", "first-range", "at-seam", "second-range", "probe-only"])
def test_current_clip_matches_actual_complete_owner_position(decoder, mode):
    video, expected = {
        "full": (NativeConcatConfig(), None),
        "first-range": (NativeConcatConfig(end=Fraction(1, 10)), 1),
        "at-seam": (NativeConcatConfig(end=Fraction(1, 5)), 1),
        "second-range": (NativeConcatConfig(end=Fraction(3, 10)), None),
        "probe-only": (NativeConcatConfig(start=Fraction(2, 5)), None),
    }[mode]
    result = detect_native_concat_scenes(decoder.clips, NativeConcatSceneConfig(video=video))
    assert result.diagnostics.current_clip is expected
    bad = None if expected == 1 else 0
    with pytest.raises(ConfigurationError):
        replace(result, diagnostics=replace(result.diagnostics, current_clip=bad)).to_dict()


@pytest.mark.parametrize("last_status", ["eof", "range_end", None])
def test_probe_only_terminal_child_did_not_play_or_exhaust(decoder, last_status):
    video = NativeConcatConfig(start=Fraction(2, 5))
    result = detect_native_concat_scenes(decoder.clips, NativeConcatSceneConfig(video=video))
    assert result.diagnostics.last_child_status == "closed"
    assert result.diagnostics.source_activations == (1, 1)
    with pytest.raises(ConfigurationError):
        replace(result, diagnostics=replace(result.diagnostics, last_child_status=last_status)).to_dict()


def test_observed_pixels_cover_known_owned_rgb_dimensions(decoder):
    result = detect_native_concat_scenes(decoder.clips)
    d = result.diagnostics
    assert (d.source_decoded_frames[0], d.source_rgb_frames[0], d.source_pixels[0]) == (4, 3, 16)
    assert result.metadata[0].width * result.metadata[0].height == 4
    # Owned RGB dimensions are verified; non-RGB preroll/sentinel frames may
    # have other dimensions but necessarily contribute at least one pixel.
    minimal_known = 3 * 4 + (4 - 3)
    assert minimal_known == 13
    forged = replace(
        d,
        source_pixels=(4, d.source_pixels[1]),
        decoded_pixels_observed=4 + d.source_pixels[1],
    )
    with pytest.raises(ConfigurationError):
        replace(result, diagnostics=forged).to_dict()
