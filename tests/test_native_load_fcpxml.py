"""Independent CSV scene-start import and native FCPXML composition contracts."""

from __future__ import annotations

import hashlib
import json
import os
from contextlib import contextmanager
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree

import pytest

import frame_quorum
from frame_quorum.cli import build_parser
from frame_quorum.errors import ConfigurationError, OutputError, ScanError
from frame_quorum.fcpxml_export import FCPXMLExportConfig, render_fcpxml
from frame_quorum.native_video import NativeVideoConfig, NativeVideoFrame, NativeVideoStatus
from frame_quorum.otio_export import OTIOCut, OTIOMedia

# Column order and names match the frozen PySceneDetect write_scene_list at
# 24953b0bf76af17c450bc143d330eea48fc5e276. The data below is fixed
# independently of Frame's parser and intentionally contains stale end data.
_HEADER = (
    b"Scene Number,Start Frame,Start Timecode,Start Time (seconds),End Frame,"
    b"End Timecode,End Time (seconds),Length (frames),Length (timecode),Length (seconds)\n"
)
_ROWS = (
    b"1,1,00:00:00.000,0.000,2,00:00:00.080,0.080,2,00:00:00.080,0.080\n",
    b"2,3,00:00:00.080,0.080,5,00:00:00.200,0.200,3,00:00:00.120,0.120\n",
    b"3,6,00:00:00.200,0.200,7,00:00:00.280,0.280,2,00:00:00.080,0.080\n",
)
_FROZEN_CSV = _HEADER + b"".join(_ROWS)


def _csv(tmp_path: Path, data: bytes = _FROZEN_CSV, name: str = "scenes.csv") -> Path:
    path = tmp_path / name
    path.write_bytes(data)
    return path


def _seconds(value: str) -> Fraction:
    assert value.endswith("s")
    return Fraction(value[:-1])


def _cfr_video(
    tmp_path: Path, *, pts: tuple[int, ...] = tuple(range(7))
) -> tuple[Path, tuple[Fraction, ...]]:
    av = pytest.importorskip("av")
    from PIL import Image

    source = tmp_path / "colors.nut"
    colors = tuple((index * 23, index * 17, index * 11) for index in range(len(pts)))
    with av.open(str(source), "w", format="nut") as container:
        stream = container.add_stream("ffv1", rate=25)
        stream.width, stream.height, stream.pix_fmt = 8, 6, "bgr0"
        for value, color in zip(pts, colors, strict=True):
            with Image.new("RGB", (8, 6), color) as image:
                frame = av.VideoFrame.from_image(image)
            frame.pts, frame.time_base = value, Fraction(1, 25)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    with av.open(str(source)) as container:
        decoded = tuple(container.decode(video=0))
        oracle_pts = tuple(Fraction(frame.pts) * frame.time_base for frame in decoded)
        assert len(decoded) == len(pts)
        assert [(frame.width, frame.height) for frame in decoded] == [(8, 6)] * len(pts)
        for frame, color in zip(decoded, colors, strict=True):
            with frame.to_image() as image:
                assert image.convert("RGB").getpixel((0, 0)) == color
    return source, oracle_pts


def test_public_csv_loader_is_available(tmp_path):
    csv_path = tmp_path / "scenes.csv"
    csv_path.write_bytes(b"Scene Number,Start Frame\n1,1\n")
    loaded = frame_quorum.load_native_scene_csv(csv_path)
    assert loaded.start_ordinals == (0,)


def test_cli_registers_native_load_fcpxml():
    parsed = build_parser().parse_args(
        [
            "native-load-fcpxml",
            "source.nut",
            "--scene-csv",
            "scenes.csv",
            "--frame-rate",
            "25",
            "--final-end",
            "7/25",
            "--output-dir",
            "editor",
        ]
    )
    assert parsed.command == "native-load-fcpxml"


@pytest.mark.parametrize(
    "data",
    [
        _FROZEN_CSV,
        b"Timecode List:,00:00:00.080,00:00:00.200\n" + _FROZEN_CSV,
        b"Timecode List:\n" + _FROZEN_CSV,
        b"\xef\xbb\xbf" + _FROZEN_CSV.replace(b"\n", b"\r\n"),
        b"Note,Start Frame,Scene Number\n"
        b'"ignored, even when quoted",1,1\n'
        b'"stale, ignored",3,2\n'
        b'"also stale",6,3\n',
        b"Note,Start Frame,Scene Number\r\n"
        b'"say ""hi"", now",1,1\r\n'
        b'"line one\r\nline two",3,2\r\n'
        b'"quoted, ignored",6,3\r\n',
    ],
)
def test_frozen_writer_profile_and_safe_extra_columns(data: bytes, tmp_path: Path) -> None:
    loaded = frame_quorum.load_native_scene_csv(_csv(tmp_path, data))
    assert loaded.start_ordinals == (0, 2, 5)
    assert loaded.csv_sha256 == hashlib.sha256(data).hexdigest()
    assert loaded.row_count == 3


def test_one_scene_has_no_cuts(tmp_path: Path) -> None:
    data = _HEADER + _ROWS[0]
    loaded = frame_quorum.load_native_scene_csv(_csv(tmp_path, data))
    assert loaded.start_ordinals == (0,)
    assert loaded.row_count == 1


def test_one_scene_default_writer_blank_prelude(tmp_path: Path) -> None:
    loaded = frame_quorum.load_native_scene_csv(_csv(tmp_path, b"\n" + _HEADER + _ROWS[0]))
    assert loaded.start_ordinals == (0,)


@pytest.mark.parametrize(
    "data",
    [
        b"Scene Number,Scene Number,Start Frame\n1,1,1\n",
        b"Scene Number,Other\n1,1\n",
        b"Scene Number,Start Frame\n",
        b"Scene Number,Start Frame\n1,1\n\n2,3\n",
        b"Scene Number,Start Frame\n1,1,extra\n",
        b"Scene Number,Start Frame\n1,1\n2\n",
        b'Scene Number,Start Frame\n1,"unterminated\n',
        b"Scene Number,Start Frame\n1,1\n3,3\n",
        b"Scene Number,Start Frame\n1,0\n",
        b"Scene Number,Start Frame\n1,+1\n",
        b"Scene Number,Start Frame\n1, 1\n",
        b"Scene Number,Start Frame\n1,1\n2,1\n",
        b"Scene Number,Start Frame\n1,1\n2,6\n3,3\n",
        b"Scene Number,Start Frame\n1,1.0\n",
        b"Scene Number,Start Frame\n1,\xef\n",
        b"Timecode List:\nTimecode List:\nScene Number,Start Frame\n1,1\n",
        b"Scene Number,Start Frame\n1,1\n2," + b"9" * 1000 + b"\n",
    ],
)
def test_bad_csv_is_rejected_without_cell_echo(data: bytes, tmp_path: Path) -> None:
    path = _csv(tmp_path, data)
    with pytest.raises((ConfigurationError, ScanError)) as caught:
        frame_quorum.load_native_scene_csv(path)
    assert "unterminated" not in str(caught.value)


@pytest.mark.parametrize(
    "data",
    [
        b'Scene Number,Start Frame,Note\n1,1,unquoted"bad\n',
        b"Scene Number,Start Frame\r1,1\r",
        b'Scene Number,Start Frame,Note\n1,1,"bad\rnote"\n',
    ],
    ids=("bare-quote-in-unquoted-field", "cr-only-record-separators", "bare-cr-in-quoted-field"),
)
def test_csv_lexical_profile_rejects_non_rfc_records(data: bytes, tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError):
        frame_quorum.load_native_scene_csv(_csv(tmp_path, data))


def test_csv_byte_and_scene_limits_and_symlink(tmp_path: Path) -> None:
    path = _csv(tmp_path)
    with pytest.raises((ConfigurationError, ScanError)):
        frame_quorum.load_native_scene_csv(path, max_input_bytes=len(_FROZEN_CSV) - 1)
    with pytest.raises(ConfigurationError):
        frame_quorum.load_native_scene_csv(path, max_scenes=2)
    link = tmp_path / "linked.csv"
    try:
        link.symlink_to(path)
    except (NotImplementedError, OSError):
        pytest.skip("local Windows configuration does not allow symlinks")
    with pytest.raises((ConfigurationError, ScanError)):
        frame_quorum.load_native_scene_csv(link)


def test_real_cfr_csv_partition_matches_direct_pyav_and_existing_serializer(
    tmp_path: Path, monkeypatch
) -> None:
    import frame_quorum.cli as cli
    import frame_quorum.native_scenes as detector

    source, pts = _cfr_video(tmp_path)
    assert pts == tuple(Fraction(index, 25) for index in range(7))
    csv_path = _csv(tmp_path)

    def forbidden_detector(*_args, **_kwargs):
        raise AssertionError("imported cuts must not run a detector")

    monkeypatch.setattr(cli, "detect_native_scenes", forbidden_detector)
    monkeypatch.setattr(detector, "detect_native_scenes", forbidden_detector)
    target = tmp_path / "editor"
    result = frame_quorum.write_loaded_fcpxml_bundle(
        source, csv_path, target, frame_rate=Fraction(25), final_end=Fraction(7, 25)
    )
    assert target.is_dir()
    assert {path.name for path in target.iterdir()} == {"scenes.fcpxml", "audit.json"}
    xml_bytes = (target / "scenes.fcpxml").read_bytes()
    clips = ElementTree.fromstring(xml_bytes).findall("./library/event/project/sequence/spine/asset-clip")
    assert len(clips) == 3
    assert [_seconds(clip.attrib["start"]) for clip in clips] == [
        Fraction(0),
        Fraction(2, 25),
        Fraction(5, 25),
    ]
    assert [_seconds(clip.attrib["offset"]) for clip in clips] == [
        Fraction(0),
        Fraction(2, 25),
        Fraction(5, 25),
    ]
    assert [_seconds(clip.attrib["duration"]) for clip in clips] == [
        Fraction(2, 25),
        Fraction(3, 25),
        Fraction(2, 25),
    ]
    media = OTIOMedia(source.absolute(), Fraction(0), Fraction(0), Fraction(7, 25))
    cuts = (
        OTIOCut(Fraction(0), Fraction(2, 25)),
        OTIOCut(Fraction(2, 25), Fraction(5, 25)),
        OTIOCut(Fraction(5, 25), Fraction(7, 25)),
    )
    assert xml_bytes == render_fcpxml(cuts, media, FCPXMLExportConfig(Fraction(25), 8, 6)).encode()
    audit = json.loads((target / "audit.json").read_bytes())
    assert audit["csv_sha256"] == hashlib.sha256(_FROZEN_CSV).hexdigest()
    assert audit["scene_count"] == 3 and audit["frame_count"] == 7
    assert audit["fcpxml_bytes"] == len(xml_bytes)
    assert audit["fcpxml_sha256"] == hashlib.sha256(xml_bytes).hexdigest()
    for key in ("source_verified", "source_content_authenticated", "editor_import_verified"):
        assert audit[key] is False
    assert result.fcpxml_sha256 == audit["fcpxml_sha256"]
    assert result.total_output_bytes == sum(path.stat().st_size for path in target.iterdir())


def test_editing_start_frames_changes_cuts_without_detector(tmp_path: Path, monkeypatch) -> None:
    import frame_quorum.native_scenes as detector

    source, pts = _cfr_video(tmp_path)
    assert len(pts) == 7
    edited = _HEADER + _ROWS[0] + _ROWS[1].replace(b"2,3,", b"2,4,", 1) + _ROWS[2]
    csv_path = _csv(tmp_path, edited)
    monkeypatch.setattr(
        detector, "detect_native_scenes", lambda *_args, **_kwargs: pytest.fail("detector called")
    )
    target = tmp_path / "edited"
    frame_quorum.write_loaded_fcpxml_bundle(
        source, csv_path, target, frame_rate=Fraction(25), final_end=Fraction(7, 25)
    )
    clips = ElementTree.parse(target / "scenes.fcpxml").findall(
        "./library/event/project/sequence/spine/asset-clip"
    )
    assert [_seconds(clip.attrib["start"]) for clip in clips] == [
        Fraction(0),
        Fraction(3, 25),
        Fraction(5, 25),
    ]
    assert [_seconds(clip.attrib["duration"]) for clip in clips] == [
        Fraction(3, 25),
        Fraction(2, 25),
        Fraction(2, 25),
    ]


@pytest.mark.parametrize(
    "starts, final_end, pts",
    [
        ((1, 8), Fraction(7, 25), tuple(range(7))),  # cut at EOF
        ((1, 3), Fraction(8, 25), tuple(range(7))),  # wrong exclusive endpoint
        ((1, 3), Fraction(7, 25), (0, 1, 2, 4, 5, 6, 7)),  # VFR cadence
    ],
)
def test_invalid_media_partition_never_stages(
    starts: tuple[int, ...], final_end: Fraction, pts: tuple[int, ...], tmp_path: Path
) -> None:
    source, observed = _cfr_video(tmp_path, pts=pts)
    assert len(observed) == len(pts)
    rows = b"".join(f"{index},{start}\n".encode() for index, start in enumerate(starts, 1))
    csv_path = _csv(tmp_path, b"Scene Number,Start Frame\n" + rows)
    target = tmp_path / "invalid"
    with pytest.raises((ConfigurationError, ScanError)):
        frame_quorum.write_loaded_fcpxml_bundle(
            source, csv_path, target, frame_rate=Fraction(25), final_end=final_end
        )
    assert not target.exists()


def test_invalid_csv_is_rejected_before_decoder_open(tmp_path: Path, monkeypatch) -> None:
    import av

    csv_path = _csv(tmp_path, b"Scene Number,Start Frame\n1,0\n")
    monkeypatch.setattr(av, "open", lambda *_args, **_kwargs: pytest.fail("decoder opened"))
    target = tmp_path / "no-output"
    with pytest.raises((ConfigurationError, ScanError)):
        frame_quorum.write_loaded_fcpxml_bundle(
            tmp_path / "missing.nut", csv_path, target, frame_rate=Fraction(25), final_end=Fraction(7, 25)
        )
    assert not target.exists()


def test_csv_change_during_read_is_rejected(tmp_path: Path, monkeypatch) -> None:
    csv_path = _csv(tmp_path)
    original = Path.open

    class MutatingRead:
        def __init__(self, handle):
            self.handle = handle
            self.done = False

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return self.handle.__exit__(*args)

        def __getattr__(self, name):
            return getattr(self.handle, name)

        def read(self, *args):
            data = self.handle.read(*args)
            if not self.done:
                self.done = True
                info = csv_path.stat()
                os.utime(
                    csv_path,
                    ns=(info.st_atime_ns, info.st_mtime_ns + 1_000_000_000),
                )
            return data

    def mutating_open(path, *args, **kwargs):
        handle = original(path, *args, **kwargs)
        return MutatingRead(handle) if path == csv_path and args and args[0] == "rb" else handle

    monkeypatch.setattr(Path, "open", mutating_open)
    with pytest.raises(ScanError, match="changed"):
        frame_quorum.load_native_scene_csv(csv_path)


def test_equal_pts_fails_cfr_even_with_contiguous_decode_ordinals(tmp_path: Path, monkeypatch) -> None:
    import frame_quorum.native_load_fcpxml as module

    csv_path = _csv(tmp_path, b"Scene Number,Start Frame\n1,1\n2,3\n")
    frames = tuple(
        NativeVideoFrame(
            pts,
            Fraction(1, 25),
            index,
            index,
            0,
            8,
            6,
            bytes(8 * 6 * 3),
        )
        for index, pts in enumerate((0, 1, 1))
    )

    class StubStream:
        def __init__(self, path, _config):
            self.metadata = SimpleNamespace(path=Path(path), width=8, height=6)
            self.diagnostics = SimpleNamespace(
                status=NativeVideoStatus.EOF,
                closed=True,
                cleanup_errors=(),
                generation=0,
                decoded_frames=len(frames),
                returned_frames=len(frames),
            )

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def __iter__(self):
            return iter(frames)

    monkeypatch.setattr(module, "NativeVideoStream", StubStream)
    target = tmp_path / "equal-pts"
    with pytest.raises(ConfigurationError, match="CFR"):
        frame_quorum.write_loaded_fcpxml_bundle(
            tmp_path / "stub.nut",
            csv_path,
            target,
            frame_rate=Fraction(25),
            final_end=Fraction(3, 25),
        )
    assert not target.exists()


@pytest.mark.parametrize(
    "limits",
    [
        NativeVideoConfig(frame_step=2),
        NativeVideoConfig(video_stream=1),
        NativeVideoConfig(start=Fraction(0)),
        NativeVideoConfig(end=Fraction(1)),
        NativeVideoConfig(max_frames=3),
        NativeVideoConfig(max_decoded_frames=3),
        NativeVideoConfig(max_total_pixels=8 * 6 * 3),
    ],
)
def test_invalid_or_insufficient_video_limits_do_not_publish(
    limits: NativeVideoConfig, tmp_path: Path
) -> None:
    source, _ = _cfr_video(tmp_path)
    target = tmp_path / "limited"
    with pytest.raises((ConfigurationError, ScanError)):
        frame_quorum.write_loaded_fcpxml_bundle(
            source,
            _csv(tmp_path),
            target,
            frame_rate=Fraction(25),
            final_end=Fraction(7, 25),
            video_limits=limits,
        )
    assert not target.exists()


def test_exact_frame_budget_still_establishes_eof(tmp_path: Path) -> None:
    source, _ = _cfr_video(tmp_path)
    target = tmp_path / "exact-limit"
    result = frame_quorum.write_loaded_fcpxml_bundle(
        source,
        _csv(tmp_path),
        target,
        frame_rate=Fraction(25),
        final_end=Fraction(7, 25),
        video_limits=NativeVideoConfig(max_frames=7, max_decoded_frames=8),
    )
    assert result.frame_count == 7 and result.scene_count == 3


def test_existing_target_preserved(tmp_path: Path) -> None:
    source, _ = _cfr_video(tmp_path)
    target = tmp_path / "editor"
    target.mkdir()
    sentinel = target / "foreign.txt"
    sentinel.write_bytes(b"user-owned")
    with pytest.raises(OutputError):
        frame_quorum.write_loaded_fcpxml_bundle(
            source, _csv(tmp_path), target, frame_rate=Fraction(25), final_end=Fraction(7, 25)
        )
    assert sentinel.read_bytes() == b"user-owned"


@pytest.mark.parametrize("case", ["existing", "missing-parent"])
def test_invalid_output_target_rejects_before_decoder(tmp_path: Path, monkeypatch, case: str) -> None:
    import frame_quorum.native_load_fcpxml as module

    csv_path = _csv(tmp_path)
    if case == "existing":
        target = tmp_path / "existing"
        target.mkdir()
        (target / "foreign.txt").write_bytes(b"preserve")
    else:
        target = tmp_path / "missing-parent" / "output"

    def forbidden_decoder(*_args, **_kwargs):
        pytest.fail("decoder opened before output target preflight")

    monkeypatch.setattr(module, "NativeVideoStream", forbidden_decoder)
    with pytest.raises(OutputError):
        frame_quorum.write_loaded_fcpxml_bundle(
            tmp_path / "source.nut",
            csv_path,
            target,
            frame_rate=Fraction(25),
            final_end=Fraction(7, 25),
        )
    if case == "existing":
        assert (target / "foreign.txt").read_bytes() == b"preserve"
    else:
        assert not target.parent.exists()


def test_cli_end_to_end_and_rejects_window_before_decode(tmp_path: Path, monkeypatch, capsys) -> None:
    import av

    from frame_quorum.cli import main

    source, _ = _cfr_video(tmp_path)
    csv_path = _csv(tmp_path)
    target = tmp_path / "editor"
    arguments = [
        "native-load-fcpxml",
        str(source),
        "--scene-csv",
        str(csv_path),
        "--frame-rate",
        "25",
        "--final-end",
        "7/25",
        "--output-dir",
        str(target),
    ]
    assert main(arguments) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["scene_count"] == 3 and payload["frame_count"] == 7
    assert payload["source_verified"] is False
    original = av.open
    monkeypatch.setattr(av, "open", lambda *_args, **_kwargs: pytest.fail("decoder opened"))
    denied = tmp_path / "denied"
    invalid = [*arguments[:-1], str(denied), "--frame-step", "2"]
    assert main(invalid) == 2
    assert not denied.exists()
    monkeypatch.setattr(av, "open", original)


def test_cli_stdout_short_write_keeps_published_bundle(tmp_path: Path, monkeypatch) -> None:
    import frame_quorum.cli as cli

    source, _ = _cfr_video(tmp_path)
    target = tmp_path / "published"
    arguments = [
        "native-load-fcpxml",
        str(source),
        "--scene-csv",
        str(_csv(tmp_path)),
        "--frame-rate",
        "25",
        "--final-end",
        "7/25",
        "--output-dir",
        str(target),
    ]

    class ShortStdout:
        def write(self, message):
            return len(message) - 1

        def flush(self):
            return None

    with monkeypatch.context() as patch:
        patch.setattr(cli.sys, "stdout", ShortStdout())
        assert cli.main(arguments) == 2
    assert {path.name for path in target.iterdir()} == {"scenes.fcpxml", "audit.json"}


def test_racing_foreign_target_is_preserved(tmp_path: Path, monkeypatch) -> None:
    import frame_quorum.native_measurements as publication

    source, _ = _cfr_video(tmp_path)
    target = tmp_path / "race"
    original = publication._publish

    def race(stage, destination):
        destination.mkdir()
        (destination / "foreign.txt").write_bytes(b"foreign")
        original(stage, destination)

    monkeypatch.setattr(publication, "_publish", race)
    with pytest.raises(OutputError):
        frame_quorum.write_loaded_fcpxml_bundle(
            source, _csv(tmp_path), target, frame_rate=Fraction(25), final_end=Fraction(7, 25)
        )
    assert (target / "foreign.txt").read_bytes() == b"foreign"


def test_stage_close_failure_does_not_publish(tmp_path: Path, monkeypatch) -> None:
    import frame_quorum.native_measurements as publication

    source, _ = _cfr_video(tmp_path)
    target = tmp_path / "close-failure"
    original = publication._managed_file

    @contextmanager
    def bad_close(path, owned):
        with original(path, owned) as handle:
            yield handle
        raise OSError("injected close failure")

    monkeypatch.setattr(publication, "_managed_file", bad_close)
    with pytest.raises(OutputError):
        frame_quorum.write_loaded_fcpxml_bundle(
            source, _csv(tmp_path), target, frame_rate=Fraction(25), final_end=Fraction(7, 25)
        )
    assert not target.exists()


@pytest.mark.parametrize("name", ["scenes.fcpxml", "audit.json"])
@pytest.mark.parametrize("mode", ["corrupt", "silent-short"])
def test_staged_byte_reconciliation_prevents_publish(
    tmp_path: Path, monkeypatch, mode: str, name: str
) -> None:
    import frame_quorum.native_measurements as publication

    source, _ = _cfr_video(tmp_path)
    csv_path = _csv(tmp_path)
    target = tmp_path / "editor"
    original = publication._managed_file

    @contextmanager
    def altered(path, owned):
        with original(path, owned) as handle:

            class Writer:
                def write(self, data):
                    if path.name == name:
                        changed = (b"X" + data[1:]) if mode == "corrupt" else data[:-1]
                        handle.write(changed)
                        return len(data)
                    return handle.write(data)

            yield Writer()

    monkeypatch.setattr(publication, "_managed_file", altered)
    with pytest.raises(OutputError):
        frame_quorum.write_loaded_fcpxml_bundle(
            source, csv_path, target, frame_rate=Fraction(25), final_end=Fraction(7, 25)
        )
    assert not target.exists()
    assert not list(tmp_path.glob(".frame-quorum-measurements-*"))


def test_lost_publication_ack_keeps_complete_bundle(tmp_path: Path, monkeypatch) -> None:
    import frame_quorum.native_measurements as publication

    source, _ = _cfr_video(tmp_path)
    target = tmp_path / "editor"
    original = publication._publish

    def lost_return(stage, destination):
        original(stage, destination)
        raise OSError("lost return after successful rename")

    monkeypatch.setattr(publication, "_publish", lost_return)
    with pytest.raises(OutputError, match="inspect"):
        frame_quorum.write_loaded_fcpxml_bundle(
            source, _csv(tmp_path), target, frame_rate=Fraction(25), final_end=Fraction(7, 25)
        )
    assert {path.name for path in target.iterdir()} == {"scenes.fcpxml", "audit.json"}
    audit = json.loads((target / "audit.json").read_bytes())
    assert audit["fcpxml_sha256"] == hashlib.sha256((target / "scenes.fcpxml").read_bytes()).hexdigest()


def test_cli_lost_publication_ack_reports_inspect_target(tmp_path: Path, monkeypatch, capsys) -> None:
    import frame_quorum.native_measurements as publication
    from frame_quorum.cli import main

    source, _ = _cfr_video(tmp_path)
    target = tmp_path / "published"
    original = publication._publish

    def lost_return(stage, destination):
        original(stage, destination)
        raise OSError("lost return after successful rename")

    monkeypatch.setattr(publication, "_publish", lost_return)
    assert (
        main(
            [
                "native-load-fcpxml",
                str(source),
                "--scene-csv",
                str(_csv(tmp_path)),
                "--frame-rate",
                "25",
                "--final-end",
                "7/25",
                "--output-dir",
                str(target),
            ]
        )
        == 2
    )
    stderr = capsys.readouterr().err
    assert "inspect" in stderr and str(target) in stderr
    assert {path.name for path in target.iterdir()} == {"scenes.fcpxml", "audit.json"}
