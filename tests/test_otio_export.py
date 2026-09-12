from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest

from frame_quorum.errors import ConfigurationError, OutputError
from frame_quorum.otio_export import (
    OTIOCut,
    OTIOExportConfig,
    OTIOMedia,
    render_otio,
    write_otio_bundle,
)


def tracks(document):
    return document["tracks"]["children"]


def seconds(time):
    assert type(time["value"]) is int and type(time["rate"]) is int
    return Fraction(time["value"], time["rate"])


def test_disjoint_cuts_have_exact_source_ranges_and_contiguous_record_duration(tmp_path):
    media = OTIOMedia(tmp_path / 'missing #"é.nut', Fraction(0))
    document = json.loads(
        render_otio((OTIOCut(Fraction(4), Fraction(6)), OTIOCut(Fraction(10), Fraction(11))), media)
    )
    video = tracks(document)[0]
    assert video["kind"] == "Video" and len(tracks(document)) == 1
    ranges = [clip["source_range"] for clip in video["children"]]
    assert [seconds(value["start_time"]) for value in ranges] == [4, 10]
    assert [seconds(value["duration"]) for value in ranges] == [2, 1]
    assert sum(seconds(value["duration"]) for value in ranges) == 3
    reference = video["children"][0]["media_references"]["DEFAULT_MEDIA"]
    assert reference["target_url"] == media.path.as_uri()
    assert reference["available_range"] is None
    assert document["metadata"]["frame_quorum"]["source_verified"] is False
    assert not media.path.exists()


def test_ntsc_and_audio_ticks_use_exact_shared_integer_rate_and_explicit_origin(tmp_path):
    origin = Fraction(-5)
    cuts = (
        OTIOCut(origin, origin + Fraction(1001, 30000), "NTSC"),
        OTIOCut(origin + Fraction(1, 10), origin + Fraction(1, 10) + Fraction(1, 48000), "audio"),
    )
    document = json.loads(
        render_otio(cuts, OTIOMedia(tmp_path / "a.nut", origin), OTIOExportConfig(include_audio=True))
    )
    video, audio = tracks(document)
    assert (video["kind"], audio["kind"]) == ("Video", "Audio")
    assert video["children"] == audio["children"]
    assert document["global_start_time"] == {"OTIO_SCHEMA": "RationalTime.1", "rate": 240000, "value": 0}
    assert video["children"][0]["source_range"]["duration"]["value"] == 8008
    assert video["children"][1]["source_range"]["duration"]["value"] == 5
    assert document["metadata"]["frame_quorum"]["audio"] == "caller_declared_unverified"


def test_available_range_is_explicit_not_inferred_from_last_cut(tmp_path):
    media = OTIOMedia(tmp_path / "a.nut", Fraction(5), Fraction(3), Fraction(15))
    document = json.loads(render_otio((OTIOCut(Fraction(6), Fraction(8)),), media))
    reference = tracks(document)[0]["children"][0]["media_references"]["DEFAULT_MEDIA"]
    assert seconds(reference["available_range"]["start_time"]) == -2
    assert seconds(reference["available_range"]["duration"]) == 12
    with pytest.raises(ConfigurationError, match="available"):
        render_otio((OTIOCut(Fraction(2), Fraction(4)),), media)


def test_binary64_exact_integer_edge_and_accumulated_record_bounds(tmp_path):
    media = OTIOMedia(tmp_path / "a.nut", Fraction(0))
    limit = 2**53 - 1
    document = json.loads(render_otio((OTIOCut(Fraction(limit - 1), Fraction(limit)),), media))
    assert tracks(document)[0]["children"][0]["source_range"]["start_time"]["value"] == limit - 1
    with pytest.raises(ConfigurationError, match="tick"):
        render_otio((OTIOCut(Fraction(limit), Fraction(limit + 1)),), media)
    # Both source coordinates fit, but their negative-to-positive duration does not.
    with pytest.raises(ConfigurationError, match="tick"):
        render_otio((OTIOCut(Fraction(-limit), Fraction(limit)),), media)
    with pytest.raises(ConfigurationError, match="accumulated"):
        render_otio((OTIOCut(Fraction(-limit), Fraction(0)), OTIOCut(Fraction(0), Fraction(limit))), media)
    negative = json.loads(render_otio((OTIOCut(Fraction(-limit), Fraction(-limit + 1)),), media))
    value = tracks(negative)[0]["children"][0]["source_range"]["start_time"]["value"]
    assert type(value) is int and value == -limit and int(float(value)) == value
    with pytest.raises(ConfigurationError, match="tick"):
        render_otio(
            (OTIOCut(Fraction(0), Fraction(1)),),
            replace(media, available_start=Fraction(-limit), available_end=Fraction(limit)),
        )


def test_lcm_ceiling_is_checked_without_quantizing(tmp_path):
    media = OTIOMedia(tmp_path / "a.nut", Fraction(0))
    with pytest.raises(ConfigurationError, match="tick rate"):
        render_otio((OTIOCut(Fraction(1, 32749), Fraction(1, 32719)),), media)
    with pytest.raises(ConfigurationError, match="tick rate"):
        render_otio((OTIOCut(Fraction(0), Fraction(1, 1001)),), media, OTIOExportConfig(max_tick_rate=1000))


@pytest.mark.parametrize(
    "start,end", [(0.0, 1), (True, 1), (0, float("inf")), (1, 1), (2, 1), (10**1000, 10**1000 + 1)]
)
def test_bad_cut_coordinates_are_rejected(start, end):
    with pytest.raises(ConfigurationError):
        OTIOCut(start, end)


@pytest.mark.parametrize(
    "option",
    [
        {"include_audio": 1},
        {"title": "x\n"},
        {"title": ""},
        {"max_clips": True},
        {"max_clips": 10001},
        {"max_tick_rate": 10**1000},
        {"max_output_bytes": 64 * 1024 * 1024 + 1},
    ],
)
def test_bad_config(option):
    with pytest.raises(ConfigurationError):
        OTIOExportConfig(**option)


def test_all_input_admission_precedes_output_and_preserves_tuple(tmp_path):
    media = OTIOMedia(tmp_path / "a.nut", Fraction(0))
    first = OTIOCut(Fraction(0), Fraction(1))
    cuts = (first, OTIOCut(Fraction(1), Fraction(2)))
    before = repr(cuts)
    with pytest.raises(ConfigurationError, match="clips"):
        render_otio(cuts, media, OTIOExportConfig(max_clips=1))
    for bad in ((), [first], (first, "bad"), (first, OTIOCut(Fraction(0), Fraction(1)))):
        with pytest.raises(ConfigurationError):
            write_otio_bundle(bad, media, tmp_path / "report")
    assert repr(cuts) == before
    assert list(tmp_path.iterdir()) == []


def test_output_limit_includes_newline_and_bundle_audit_before_any_staging(tmp_path):
    media = OTIOMedia(tmp_path / "a.nut", Fraction(0))
    cuts = (OTIOCut(Fraction(0), Fraction(1)),)
    raw = render_otio(cuts, media).encode("ascii")
    assert raw.endswith(b"\n")
    assert render_otio(cuts, media, OTIOExportConfig(max_output_bytes=len(raw))).encode("ascii") == raw
    with pytest.raises(OutputError, match="byte"):
        render_otio(cuts, media, OTIOExportConfig(max_output_bytes=len(raw) - 1))
    with pytest.raises(OutputError, match="byte"):
        write_otio_bundle(cuts, media, tmp_path / "report", OTIOExportConfig(max_output_bytes=len(raw)))
    assert list(tmp_path.iterdir()) == []


def test_bundle_matches_rendered_bytes_digest_and_refuses_existing_directory(tmp_path):
    cuts = (OTIOCut(Fraction(0), Fraction(1)),)
    media = OTIOMedia(tmp_path / "a.nut", Fraction(0))
    target = tmp_path / "report"
    result = write_otio_bundle(cuts, media, target)
    raw = (target / "scenes.otio").read_bytes()
    assert raw == render_otio(cuts, media).encode("ascii")
    assert result.otio_sha256 == hashlib.sha256(raw).hexdigest()
    assert result.total_output_bytes == sum(path.stat().st_size for path in target.iterdir())
    audit = json.loads((target / "audit.json").read_bytes())
    assert audit["otio_sha256"] == result.otio_sha256
    assert audit["source_verified"] is False
    with pytest.raises(OutputError):
        write_otio_bundle(cuts, media, target)
    assert (target / "scenes.otio").read_bytes() == raw


def test_unknown_available_requires_paired_bounds_and_absolute_media_path(tmp_path):
    for values in ((Fraction(0), None), (None, Fraction(1)), (Fraction(2), Fraction(1))):
        with pytest.raises(ConfigurationError):
            OTIOMedia(tmp_path / "a.nut", Fraction(0), *values)
    with pytest.raises(ConfigurationError):
        OTIOMedia(Path("relative.nut"), Fraction(0))
    with pytest.raises(ConfigurationError):
        OTIOMedia(tmp_path / "a.nut", 0.0)
    media = OTIOMedia(tmp_path / "a.nut", Fraction(0))
    assert replace(media, origin=Fraction(1)).origin == 1


@pytest.mark.parametrize("options", [{"config": {}}, {"media": "a"}])
def test_wrong_public_configuration_types(tmp_path, options):
    arguments = {
        "cuts": (OTIOCut(Fraction(0), Fraction(1)),),
        "media": OTIOMedia(tmp_path / "a", Fraction(0)),
    }
    arguments.update(options)
    with pytest.raises(ConfigurationError):
        render_otio(**arguments)


@pytest.mark.parametrize("name", ["x\x00", "x\x7f", "\ud800", "x" * 129, 3])
def test_names_reject_invalid_unicode_controls_and_lengths(name):
    with pytest.raises(ConfigurationError):
        OTIOCut(Fraction(0), Fraction(1), name)


@pytest.mark.parametrize("phase", ["preflight", "write"])
@pytest.mark.parametrize("failure", [OSError, KeyboardInterrupt, SystemExit])
def test_encoder_failure_or_control_never_publishes_partial_bundle(tmp_path, monkeypatch, phase, failure):
    import frame_quorum.otio_export as module

    original = module._pieces
    calls = 0

    def broken(prepared):
        nonlocal calls
        calls += 1
        for index, piece in enumerate(original(prepared)):
            if index == 1 and calls == (1 if phase == "preflight" else 2):
                raise failure("controlled encoder failure")
            yield piece

    monkeypatch.setattr(module, "_pieces", broken)
    with pytest.raises((failure, OutputError)):
        module.write_otio_bundle(
            (OTIOCut(Fraction(0), Fraction(1)),), OTIOMedia(tmp_path / "a", Fraction(0)), tmp_path / "report"
        )
    assert list(tmp_path.iterdir()) == []


def test_lost_publication_acknowledgment_keeps_complete_published_directory(tmp_path, monkeypatch):
    import frame_quorum.native_measurements as publication

    original = publication._publish

    def lost(stage, target):
        original(stage, target)
        raise OSError("lost acknowledgment")

    monkeypatch.setattr(publication, "_publish", lost)
    target = tmp_path / "report"
    with pytest.raises(OutputError) as caught:
        write_otio_bundle(
            (OTIOCut(Fraction(0), Fraction(1)),), OTIOMedia(tmp_path / "a", Fraction(0)), target
        )
    assert json.loads((target / "scenes.otio").read_bytes())["OTIO_SCHEMA"] == "Timeline.1"
    assert json.loads((target / "audit.json").read_bytes())["source_verified"] is False
    assert caught.value.__cause__.__notes__
    assert list(tmp_path.iterdir()) == [target]


def test_pure_render_in_fresh_process_never_imports_optional_decoders_or_otio(tmp_path):
    script = """
import importlib.abc, json, sys
from fractions import Fraction
from pathlib import Path
class Deny(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('av', 'opentimelineio'):
            raise AssertionError('pure export imported optional dependency')
sys.meta_path.insert(0, Deny())
from frame_quorum import OTIOCut, OTIOMedia, render_otio
media = OTIOMedia(Path(sys.argv[1]), Fraction(0))
document = json.loads(render_otio((OTIOCut(Fraction(0), Fraction(1)),), media))
assert document['tracks']['children'][0]['kind'] == 'Video'
assert 'av' not in sys.modules and 'opentimelineio' not in sys.modules
print('pure export with missing source and denied optional imports passed')
"""
    process = subprocess.run(
        [sys.executable, "-B", "-I", "-c", script, str(tmp_path / "missing.nut")],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    assert process.stdout.strip() == "pure export with missing source and denied optional imports passed"
    assert list(tmp_path.iterdir()) == []
