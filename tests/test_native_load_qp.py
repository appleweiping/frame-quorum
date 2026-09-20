"""Imported scene starts to encoder QP instructions; RED first."""

from __future__ import annotations

import hashlib
import json
import os
import runpy
from contextlib import contextmanager
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

import frame_quorum
from frame_quorum.cli import build_parser, main
from frame_quorum.errors import ConfigurationError, OutputError, ScanError
from frame_quorum.native_video import NativeVideoConfig


def test_loaded_qp_public_api_exists() -> None:
    assert callable(frame_quorum.write_loaded_qp_bundle)


def test_loaded_qp_cli_exists() -> None:
    args = build_parser().parse_args(
        ["native-load-qp", "source.nut", "--scene-csv", "scenes.csv", "--output-dir", "new-qp"]
    )
    assert args.command == "native-load-qp"


_CSV = b"Scene Number,Start Frame,End Frame\n1,1,999\n2,3,999\n3,6,999\n"
_QP = b"0 I -1\n2 I -1\n5 I -1\n"


def _csv(tmp_path: Path, content: bytes = _CSV) -> Path:
    path = tmp_path / "scenes.csv"
    path.write_bytes(content)
    return path


def _video(path: Path, pts: tuple[int, ...]) -> tuple[tuple[int, Fraction, bytes], ...]:
    av = pytest.importorskip("av")
    values = tuple(index * 29 for index in range(len(pts)))
    with av.open(str(path), "w", format="matroska") as container:
        stream = container.add_stream("ffv1", rate=25)
        stream.width, stream.height, stream.pix_fmt = 8, 6, "bgr0"
        stream.time_base = stream.codec_context.time_base = Fraction(1, 1000)
        for value, timestamp in zip(values, pts, strict=True):
            with Image.new("RGB", (8, 6), (value, value, value)) as image:
                frame = av.VideoFrame.from_image(image)
            frame.pts, frame.time_base = timestamp, Fraction(1, 1000)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    with av.open(str(path)) as container:
        rows = []
        for frame in container.decode(video=0):
            with frame.to_rgb().to_image() as image:
                rows.append((frame.pts, Fraction(frame.time_base), image.tobytes()))
    assert len(rows) == len(pts)
    assert [row[:2] for row in rows] == [(value, Fraction(1, 1000)) for value in pts]
    assert [row[2] for row in rows] == [bytes((value,)) * (8 * 6 * 3) for value in values]
    return tuple(rows)


@pytest.mark.parametrize("pts", [(0, 40, 80, 120, 160, 200, 240), (5000, 5040, 5110, 5180, 5270, 5310, 5470)])
def test_imported_qp_direct_decoder_oracle_and_distinct_audit(tmp_path: Path, pts) -> None:
    source = tmp_path / "source.mkv"
    assert len(_video(source, pts)) == 7
    csv_path = _csv(tmp_path, b"Timecode List:,ignored\n" + _CSV)
    result = frame_quorum.write_loaded_qp_bundle(source, csv_path, tmp_path / "loaded")
    assert result.frame_count == 7 and result.cut_count == 2
    assert result.csv_sha256 == hashlib.sha256(csv_path.read_bytes()).hexdigest()
    assert result.output_path.read_bytes() == _QP
    audit_bytes = (tmp_path / "loaded" / "audit.json").read_bytes()
    audit = json.loads(audit_bytes)
    assert audit == {
        "kind": "frame-quorum-loaded-qp",
        "schema_version": 1,
        "qp_format": "x264-x265-qpfile-v1",
        "video_stream": 0,
        "frame_count": 7,
        "cut_count": 2,
        "csv_bytes": len(csv_path.read_bytes()),
        "csv_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        "qp_bytes": len(_QP),
        "qp_sha256": hashlib.sha256(_QP).hexdigest(),
        "detector_performed": False,
        "source_content_authenticated": False,
        "encoder_input_verified": False,
    }
    assert audit_bytes == (
        json.dumps(audit, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("ascii")
    assert result.total_output_bytes == len(_QP) + len(audit_bytes)
    with pytest.raises(OutputError):
        frame_quorum.write_loaded_qp_bundle(source, csv_path, tmp_path / "loaded")


def test_edited_csv_changes_only_requested_cut_and_no_detector(tmp_path: Path, monkeypatch, capsys) -> None:
    import frame_quorum.cli as cli

    source = tmp_path / "source.mkv"
    _video(source, (0, 40, 80, 120, 160, 200, 240))
    csv_path = _csv(tmp_path, _CSV.replace(b"2,3,999", b"2,4,999"))

    def forbidden(*args, **kwargs):
        raise AssertionError("detector must not be called")

    monkeypatch.setattr(cli, "detect_native_scenes", forbidden)
    target = tmp_path / "loaded"
    assert main(["native-load-qp", str(source), "--scene-csv", str(csv_path), "-o", str(target)]) == 0
    assert (target / "scenes.qp").read_bytes() == b"0 I -1\n3 I -1\n5 I -1\n"
    assert json.loads(capsys.readouterr().out)["detector_performed"] is False


def test_one_frame_no_cut_and_limit_proves_eof(tmp_path: Path) -> None:
    source = tmp_path / "source.mkv"
    _video(source, (300,))
    csv_path = _csv(tmp_path, b"Scene Number,Start Frame\n1,1\n")
    result = frame_quorum.write_loaded_qp_bundle(
        source,
        csv_path,
        tmp_path / "single",
        video_limits=NativeVideoConfig(max_frames=1, max_decoded_frames=2),
    )
    assert result.output_path.read_bytes() == b"0 I -1\n"
    with pytest.raises(ConfigurationError):
        frame_quorum.write_loaded_qp_bundle(
            source,
            csv_path,
            tmp_path / "no-eof-allowance",
            video_limits=NativeVideoConfig(max_frames=1, max_decoded_frames=1),
        )


@pytest.mark.parametrize(
    "csv_bytes",
    [
        b"Scene Number,Start Frame\n1,0\n",
        b"Scene Number,Start Frame\n1,1\n2,1\n",
        b"Scene Number,Start Frame\n1,1\n3,3\n",
        b"Scene Number,Start Frame\n1,1\n2, 3\n",
        b"Scene Number,Start Frame,Start Frame\n1,1,2\n",
        b'Scene Number,Start Frame\n1,1\n2,"3\n',
        b"\xff\xfe",
    ],
)
def test_invalid_csv_fails_before_video_open(tmp_path: Path, monkeypatch, csv_bytes: bytes) -> None:
    import frame_quorum.native_load_qp as loaded_qp

    def forbidden(*args, **kwargs):
        raise AssertionError("decoder must not open")

    monkeypatch.setattr(loaded_qp, "_complete_video", forbidden)
    with pytest.raises((ConfigurationError, ScanError)):
        frame_quorum.write_loaded_qp_bundle(tmp_path / "missing", _csv(tmp_path, csv_bytes), tmp_path / "out")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    "config",
    [
        NativeVideoConfig(start=Fraction(1, 25)),
        NativeVideoConfig(end=Fraction(1, 5)),
        NativeVideoConfig(frame_step=2),
        NativeVideoConfig(video_stream=1),
        NativeVideoConfig(max_frames=100_001),
        NativeVideoConfig(max_frames=3, max_decoded_frames=4),
    ],
)
def test_invalid_video_profile_or_limit_never_publishes(tmp_path: Path, config: NativeVideoConfig) -> None:
    source = tmp_path / "source.mkv"
    _video(source, (0, 40, 80, 120, 160, 200, 240))
    with pytest.raises((ConfigurationError, ScanError)):
        frame_quorum.write_loaded_qp_bundle(source, _csv(tmp_path), tmp_path / "out", video_limits=config)
    assert not (tmp_path / "out").exists()


def test_cut_at_or_after_eof_and_budget_boundaries(tmp_path: Path) -> None:
    source = tmp_path / "source.mkv"
    _video(source, (0, 40, 80, 120, 160, 200, 240))
    with pytest.raises(ConfigurationError):
        frame_quorum.write_loaded_qp_bundle(
            source,
            _csv(tmp_path, b"Scene Number,Start Frame\n1,1\n2,8\n"),
            tmp_path / "past-end",
        )
    csv_path = _csv(tmp_path)
    ordinary = frame_quorum.write_loaded_qp_bundle(source, csv_path, tmp_path / "ordinary")
    frame_quorum.write_loaded_qp_bundle(
        source, csv_path, tmp_path / "exact", max_output_bytes=ordinary.total_output_bytes
    )
    with pytest.raises(OutputError):
        frame_quorum.write_loaded_qp_bundle(
            source, csv_path, tmp_path / "one-under", max_output_bytes=ordinary.total_output_bytes - 1
        )
    with pytest.raises(OutputError, match="QP output"):
        frame_quorum.write_loaded_qp_bundle(source, csv_path, tmp_path / "qp-too-small", max_output_bytes=1)
    assert not (tmp_path / "past-end").exists() and not (tmp_path / "one-under").exists()


def test_wrong_video_config_is_rejected_before_decode(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="video_limits"):
        frame_quorum.write_loaded_qp_bundle(
            tmp_path / "missing", tmp_path / "missing.csv", tmp_path / "out", video_limits=object()
        )
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("mode", ["empty", "ordinal", "dimensions"])
def test_decoder_contract_rejects_forged_frame_or_empty_eof(tmp_path: Path, monkeypatch, mode) -> None:
    import frame_quorum.native_load_qp as loaded_qp

    class FakeStream:
        metadata = SimpleNamespace(width=8, height=6)

        def __init__(self, path, config):
            self.diagnostics = SimpleNamespace(
                status=loaded_qp.NativeVideoStatus.EOF,
                closed=False,
                cleanup_errors=(),
                generation=0,
                decoded_frames=0 if mode == "empty" else 1,
                returned_frames=0 if mode == "empty" else 1,
            )

        def __enter__(self):
            return self

        def _captured_identity(self):
            return (0, 0, 0, 0)

        def __exit__(self, kind, error, traceback):
            self.diagnostics.closed = True

        def __iter__(self):
            if mode != "empty":
                yield SimpleNamespace(
                    decode_index=1 if mode == "ordinal" else 0,
                    sample_index=0,
                    generation=0,
                    width=9 if mode == "dimensions" else 8,
                    height=6,
                )

    monkeypatch.setattr(loaded_qp, "NativeVideoStream", FakeStream)
    with pytest.raises(ScanError):
        frame_quorum.write_loaded_qp_bundle(
            tmp_path / "synthetic", _csv(tmp_path, b"Scene Number,Start Frame\n1,1\n"), tmp_path / "out"
        )
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("mutation", ["mtime", "rename"])
def test_source_mutation_after_buffered_eof_is_rejected(tmp_path: Path, monkeypatch, mutation) -> None:
    import frame_quorum.native_load_qp as loaded_qp

    source = tmp_path / "source.mkv"
    _video(source, (0, 40, 80, 120, 160, 200, 240))
    original = loaded_qp.NativeVideoStream

    class MutatingStream:
        def __init__(self, path, config):
            self.inner = original(path, config)
            self.path = self.inner.path

        def __enter__(self):
            self.inner.__enter__()
            return self

        def __exit__(self, kind, error, traceback):
            result = self.inner.__exit__(kind, error, traceback)
            if mutation == "rename":
                self.path.rename(tmp_path / "moved.mkv")
            return result

        def __iter__(self):
            yield from self.inner
            if mutation == "mtime":
                info = self.path.stat()
                os.utime(self.path, ns=(info.st_atime_ns, info.st_mtime_ns + 1_000_000_000))

        def _captured_identity(self):
            return self.inner._captured_identity()

        @property
        def metadata(self):
            return self.inner.metadata

        @property
        def diagnostics(self):
            return self.inner.diagnostics

    monkeypatch.setattr(loaded_qp, "NativeVideoStream", MutatingStream)
    with pytest.raises(ScanError, match="changed"):
        frame_quorum.write_loaded_qp_bundle(source, _csv(tmp_path), tmp_path / "out")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("filename", ["scenes.qp", "audit.json"])
@pytest.mark.parametrize("mode", ["corrupt", "short"])
def test_staged_bytes_are_independently_reconciled(tmp_path: Path, monkeypatch, filename, mode) -> None:
    import frame_quorum.native_measurements as publication

    source = tmp_path / "source.mkv"
    _video(source, (0, 40, 80, 120, 160, 200, 240))
    original = publication._managed_file

    @contextmanager
    def altered(path, owned):
        with original(path, owned) as handle:

            class Writer:
                def write(self, data):
                    if path.name == filename:
                        changed = (b"X" + data[1:]) if mode == "corrupt" else data[:-1]
                        handle.write(changed)
                        return len(data)
                    return handle.write(data)

            yield Writer()

    monkeypatch.setattr(publication, "_managed_file", altered)
    with pytest.raises(OutputError):
        frame_quorum.write_loaded_qp_bundle(source, _csv(tmp_path), tmp_path / "out")
    assert not (tmp_path / "out").exists()
    assert not list(tmp_path.glob(".frame-quorum-measurements-*"))


def test_lost_publication_ack_retains_complete_bundle(tmp_path: Path, monkeypatch) -> None:
    import frame_quorum.native_measurements as publication

    source = tmp_path / "source.mkv"
    _video(source, (0, 40, 80, 120, 160, 200, 240))
    original = publication._publish

    def lost(stage, target):
        original(stage, target)
        raise OSError("lost acknowledgment")

    monkeypatch.setattr(publication, "_publish", lost)
    with pytest.raises(OutputError, match="inspect"):
        frame_quorum.write_loaded_qp_bundle(source, _csv(tmp_path), tmp_path / "out")
    assert (tmp_path / "out" / "scenes.qp").read_bytes() == _QP
    assert {path.name for path in (tmp_path / "out").iterdir()} == {"scenes.qp", "audit.json"}


def test_cli_stdout_failure_preserves_publication(tmp_path: Path, monkeypatch) -> None:
    import frame_quorum.cli as cli

    source = tmp_path / "source.mkv"
    _video(source, (0, 40, 80, 120, 160, 200, 240))
    target = tmp_path / "out"
    args = build_parser().parse_args(
        ["native-load-qp", str(source), "--scene-csv", str(_csv(tmp_path)), "-o", str(target)]
    )

    class ShortOutput:
        def write(self, data):
            return len(data) - 1

        def flush(self):
            return None

    monkeypatch.setattr(cli.sys, "stdout", ShortOutput())
    with pytest.raises(OutputError):
        cli._handle_native_load_qp(args)
    assert (target / "scenes.qp").read_bytes() == _QP


def test_existing_target_and_racing_foreign_target_are_untouched(tmp_path: Path, monkeypatch) -> None:
    import frame_quorum.native_measurements as publication

    source = tmp_path / "source.mkv"
    _video(source, (0, 40, 80, 120, 160, 200, 240))
    csv_path = _csv(tmp_path)
    existing = tmp_path / "existing"
    existing.mkdir()
    (existing / "foreign.txt").write_bytes(b"foreign")
    with pytest.raises(OutputError):
        frame_quorum.write_loaded_qp_bundle(source, csv_path, existing)
    assert (existing / "foreign.txt").read_bytes() == b"foreign"
    original = publication._publish

    def race(stage, target):
        target.mkdir()
        (target / "foreign.txt").write_bytes(b"racer")
        original(stage, target)

    monkeypatch.setattr(publication, "_publish", race)
    target = tmp_path / "race"
    with pytest.raises(OutputError):
        frame_quorum.write_loaded_qp_bundle(source, csv_path, target)
    assert (target / "foreign.txt").read_bytes() == b"racer"
    assert not list(tmp_path.glob(".frame-quorum-measurements-*"))


def test_staged_close_failure_does_not_publish(tmp_path: Path, monkeypatch) -> None:
    import frame_quorum.native_measurements as publication

    source = tmp_path / "source.mkv"
    _video(source, (0, 40, 80, 120, 160, 200, 240))
    original = publication._managed_file

    @contextmanager
    def bad_close(path, owned):
        with original(path, owned) as handle:
            yield handle
        raise OSError("injected close failure")

    monkeypatch.setattr(publication, "_managed_file", bad_close)
    with pytest.raises(OutputError):
        frame_quorum.write_loaded_qp_bundle(source, _csv(tmp_path), tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_offline_example_uses_direct_decoder_oracle(capsys) -> None:
    runpy.run_path(str(Path(__file__).parents[1] / "examples" / "native_load_qp.py"), run_name="__main__")
    assert json.loads(capsys.readouterr().out) == {"frame_count": 7, "cut_count": 2}
