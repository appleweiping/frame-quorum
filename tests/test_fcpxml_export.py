"""Independent rational-time and publication contracts for cuts-only FCPXML."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from contextlib import contextmanager
from fractions import Fraction
from xml.etree import ElementTree

import pytest

from frame_quorum import OTIOCut, OTIOMedia
from frame_quorum.errors import ConfigurationError, OutputError
from frame_quorum.fcpxml_export import FCPXMLExportConfig, render_fcpxml, write_fcpxml_bundle


def _fixture(tmp_path):
    frame = Fraction(1001, 24000)
    origin = Fraction(5)
    media = OTIOMedia(
        tmp_path / 'missing & "é.nut',
        origin,
        origin,
        origin + 20 * frame,
    )
    cuts = (
        OTIOCut(origin + 2 * frame, origin + 6 * frame, 'One & "<'),
        OTIOCut(origin + 10 * frame, origin + 13 * frame, "Deux é"),
    )
    config = FCPXMLExportConfig(frame_rate=Fraction(24000, 1001), width=1920, height=1080)
    return frame, media, cuts, config


def _seconds(value):
    assert value.endswith("s")
    return Fraction(value[:-1])


def test_missing_source_exact_rational_project_xml_has_contiguous_record_and_source_gap(tmp_path):
    frame, media, cuts, config = _fixture(tmp_path)
    xml = render_fcpxml(cuts, media, config)
    assert "One &amp; &quot;&lt;" in xml
    assert "missing%20%26%20%22%C3%A9.nut" in xml
    root = ElementTree.fromstring(xml)
    assert root.tag == "fcpxml" and root.attrib == {"version": "1.9"}
    fmt, asset = root.findall("./resources/*")
    assert fmt.tag == "format" and fmt.attrib["frameDuration"] == "1001/24000s"
    assert (fmt.attrib["width"], fmt.attrib["height"]) == ("1920", "1080")
    assert asset.tag == "asset" and asset.attrib["hasVideo"] == "1"
    assert "hasAudio" not in asset.attrib
    assert _seconds(asset.attrib["start"]) == 0
    assert _seconds(asset.attrib["duration"]) == 20 * frame
    assert asset.find("media-rep").attrib["src"] == media.path.as_uri()
    sequence = root.find("./library/event/project/sequence")
    assert sequence is not None and _seconds(sequence.attrib["duration"]) == 7 * frame
    clips = sequence.findall("./spine/asset-clip")
    assert len(clips) == 2 and [clip.attrib["name"] for clip in clips] == [cut.name for cut in cuts]
    assert [_seconds(clip.attrib["offset"]) for clip in clips] == [0, 4 * frame]
    assert [_seconds(clip.attrib["start"]) for clip in clips] == [2 * frame, 10 * frame]
    assert [_seconds(clip.attrib["duration"]) for clip in clips] == [4 * frame, 3 * frame]
    assert [clip.attrib["srcEnable"] for clip in clips] == ["video", "video"]
    assert not media.path.exists()


def test_off_lattice_or_unknown_availability_rejected_without_output(tmp_path):
    frame, media, cuts, config = _fixture(tmp_path)
    with pytest.raises(ConfigurationError, match="frame lattice"):
        write_fcpxml_bundle(
            (OTIOCut(media.origin + frame / 2, media.origin + frame),),
            media,
            tmp_path / "off-lattice",
            config,
        )
    with pytest.raises(ConfigurationError, match="available"):
        write_fcpxml_bundle(cuts, OTIOMedia(media.path, media.origin), tmp_path / "unknown", config)
    with pytest.raises(ConfigurationError, match="precede media origin"):
        write_fcpxml_bundle(
            (OTIOCut(Fraction(0), Fraction(1, 25)),),
            OTIOMedia(media.path, Fraction(1), Fraction(0), Fraction(1)),
            tmp_path / "negative-source",
            FCPXMLExportConfig(Fraction(25), 8, 6),
        )
    assert sorted(tmp_path.iterdir()) == []


def test_wrong_config_and_oversized_xml_time_are_rejected(tmp_path):
    import frame_quorum.fcpxml_export as module

    _, media, cuts, _ = _fixture(tmp_path)
    with pytest.raises(ConfigurationError, match="FCPXMLExportConfig"):
        render_fcpxml(cuts, media, {})
    with pytest.raises(ConfigurationError, match="64-bit numerator"):
        module._time(Fraction(1, 2**31))


def test_bundle_digest_budget_and_no_replace(tmp_path):
    _, media, cuts, config = _fixture(tmp_path)
    target = tmp_path / "editor"
    result = write_fcpxml_bundle(cuts, media, target, config)
    raw = (target / "scenes.fcpxml").read_bytes()
    assert raw == render_fcpxml(cuts, media, config).encode("utf-8")
    assert (
        render_fcpxml(
            cuts,
            media,
            FCPXMLExportConfig(config.frame_rate, config.width, config.height, max_output_bytes=len(raw)),
        ).encode("utf-8")
        == raw
    )
    with pytest.raises(OutputError, match="byte"):
        render_fcpxml(
            cuts,
            media,
            FCPXMLExportConfig(config.frame_rate, config.width, config.height, max_output_bytes=len(raw) - 1),
        )
    assert result.fcpxml_sha256 == hashlib.sha256(raw).hexdigest()
    audit = json.loads((target / "audit.json").read_bytes())
    assert audit["fcpxml_sha256"] == result.fcpxml_sha256
    assert audit["source_verified"] is False
    assert audit["editor_import_verified"] is False
    assert result.total_output_bytes == sum(path.stat().st_size for path in target.iterdir())
    with pytest.raises(OutputError):
        write_fcpxml_bundle(cuts, media, target, config)
    assert (target / "scenes.fcpxml").read_bytes() == raw
    with pytest.raises(OutputError, match="byte"):
        write_fcpxml_bundle(
            cuts,
            media,
            tmp_path / "too-small",
            FCPXMLExportConfig(
                frame_rate=config.frame_rate,
                width=config.width,
                height=config.height,
                max_output_bytes=result.total_output_bytes - 1,
            ),
        )
    assert not (tmp_path / "too-small").exists()


@pytest.mark.parametrize(
    "changes",
    [
        {"frame_rate": 0},
        {"frame_rate": 25.0},
        {"frame_rate": Fraction(2**31)},
        {"width": True},
        {"height": 0},
        {"title": "bad\nname"},
        {"title": "bad\ufffe"},
        {"max_clips": 10_001},
        {"max_output_bytes": 64 * 1024 * 1024 + 1},
    ],
)
def test_invalid_config_rejected(changes):
    values = {"frame_rate": Fraction(25), "width": 8, "height": 6}
    values.update(changes)
    with pytest.raises(ConfigurationError):
        FCPXMLExportConfig(**values)


def test_unknown_outside_overlap_and_xml_name_rejected_before_stage(tmp_path):
    frame, media, cuts, config = _fixture(tmp_path)
    invalid = (
        (cuts[1], cuts[0]),
        (OTIOCut(media.origin + 19 * frame, media.origin + 21 * frame),),
        (OTIOCut(media.origin, media.origin + frame, "bad\ufffe"),),
    )
    for index, bad in enumerate(invalid):
        with pytest.raises(ConfigurationError):
            write_fcpxml_bundle(bad, media, tmp_path / f"invalid-{index}", config)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("phase", ["preflight", "write"])
@pytest.mark.parametrize("failure", [OSError, KeyboardInterrupt, SystemExit])
def test_encoder_failure_never_publishes_partial_bundle(tmp_path, monkeypatch, phase, failure):
    import frame_quorum.fcpxml_export as module

    _, media, cuts, config = _fixture(tmp_path)
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
        module.write_fcpxml_bundle(cuts, media, tmp_path / "report", config)
    assert list(tmp_path.iterdir()) == []


def test_short_write_never_publishes_partial_bundle(tmp_path, monkeypatch):
    import frame_quorum.native_measurements as publication

    _, media, cuts, config = _fixture(tmp_path)
    original = publication._managed_file

    @contextmanager
    def short(path, owned):
        with original(path, owned) as handle:

            class Partial:
                def write(self, data):
                    return handle.write(data[:-1])

            yield Partial()

    monkeypatch.setattr(publication, "_managed_file", short)
    with pytest.raises(OutputError, match="short write"):
        write_fcpxml_bundle(cuts, media, tmp_path / "report", config)
    assert list(tmp_path.iterdir()) == []


def test_cli_requires_explicit_rate_size_origin_and_availability():
    from frame_quorum.cli import build_parser

    parser = build_parser()
    with pytest.raises(SystemExit) as caught:
        parser.parse_args(["native-fcpxml", "source.nut", "--output-dir", "new"])
    assert caught.value.code == 2
    args = parser.parse_args(
        [
            "native-fcpxml",
            "source.nut",
            "--output-dir",
            "new",
            "--media-origin",
            "0",
            "--available-start",
            "0",
            "--available-end",
            "6/25",
            "--frame-rate",
            "25",
            "--width",
            "8",
            "--height",
            "6",
        ]
    )
    assert args.frame_rate == 25 and args.width == 8 and args.height == 6


def test_optimized_pure_api_keeps_validation(tmp_path):
    script = """
from fractions import Fraction
from pathlib import Path
from frame_quorum import FCPXMLExportConfig, OTIOCut, OTIOMedia, render_fcpxml
from frame_quorum.errors import ConfigurationError
media = OTIOMedia(Path(__import__('sys').argv[1]), Fraction(0), Fraction(0), Fraction(1))
config = FCPXMLExportConfig(Fraction(25), 8, 6)
try:
    render_fcpxml((OTIOCut(Fraction(0), Fraction(1, 50)),), media, config)
except ConfigurationError as error:
    if 'frame lattice' not in str(error):
        raise AssertionError('unexpected validation error')
else:
    raise AssertionError('off-lattice cut accepted')
print('optimized validation passed')
"""
    process = subprocess.run(
        [sys.executable, "-B", "-O", "-c", script, str(tmp_path / "missing.nut")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode == 0, process.stderr
    assert process.stdout.strip() == "optimized validation passed"


def test_real_native_cli_exports_two_frame_aligned_cfr_scenes(tmp_path, capsys):
    av = pytest.importorskip("av")
    from PIL import Image

    from frame_quorum.cli import main

    source = tmp_path / "neutral.nut"
    with av.open(str(source), "w", format="nut") as container:
        stream = container.add_stream("ffv1", rate=25)
        stream.width, stream.height, stream.pix_fmt = 8, 6, "bgr0"
        for index in range(6):
            with Image.new("RGB", (8, 6), (0, 0, 0) if index < 3 else (255, 255, 255)) as image:
                frame = av.VideoFrame.from_image(image)
            frame.pts, frame.time_base = index, Fraction(1, 25)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    with av.open(str(source)) as container:
        assert [frame.pts * frame.time_base for frame in container.decode(video=0)] == [
            Fraction(index, 25) for index in range(6)
        ]
    output = tmp_path / "editor"
    arguments = [
        "native-fcpxml",
        str(source),
        "--detectors",
        "luminance",
        "--media-origin",
        "0",
        "--available-start",
        "0",
        "--available-end",
        "6/25",
        "--frame-rate",
        "25",
        "--width",
        "8",
        "--height",
        "6",
        "--final-end",
        "6/25",
        "--output-dir",
        str(output),
    ]
    assert main(arguments) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["clip_count"] == 2 and result["editor_import_verified"] is False
    root = ElementTree.parse(output / "scenes.fcpxml").getroot()
    clips = root.findall("./library/event/project/sequence/spine/asset-clip")
    assert [clip.attrib["start"] for clip in clips] == ["0s", "3/25s"]
    assert [clip.attrib["offset"] for clip in clips] == ["0s", "3/25s"]
    assert [clip.attrib["duration"] for clip in clips] == ["3/25s", "3/25s"]
    assert main([*arguments, "--frame-step", "2", "--output-dir", str(tmp_path / "bad")]) == 2
    assert not (tmp_path / "bad").exists()
