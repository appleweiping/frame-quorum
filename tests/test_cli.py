from __future__ import annotations

import json
from collections.abc import Callable
from io import StringIO
from pathlib import Path

import pytest
from PIL import Image

import frame_quorum.cli as cli_module
from frame_quorum.cli import main
from frame_quorum.demo import create_demo_sequence
from frame_quorum.errors import ConfigurationError, OutputError, ScanError


def test_scan_writes_json_to_stdout(
    image_factory: Callable[..., Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    image_factory("frame.png")
    assert main(["scan", str(tmp_path)]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["kind"] == "frame-quorum-scan"


def test_scan_writes_requested_file(
    image_factory: Callable[..., Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    image_factory("frame.png")
    output = tmp_path / "scan.json"
    assert main(["scan", str(tmp_path), "--output", str(output)]) == 0
    assert json.loads(output.read_text())["summary"]["frame_count"] == 1
    assert "Scanned 1 frames" in capsys.readouterr().out


def test_select_creates_manifest_and_contact_sheet(
    image_factory: Callable[..., Path], tmp_path: Path
) -> None:
    frames = tmp_path / "frames"
    for index in range(6):
        image_factory(f"{index}.png", pattern=index + 1, directory=frames)
    output = tmp_path / "result"
    status = main(["select", str(frames), "--output-dir", str(output), "--budget", "3"])
    assert status == 0
    assert (output / "manifest.json").is_file()
    with Image.open(output / "contact-sheet.png") as sheet:
        assert sheet.format == "PNG"


@pytest.mark.parametrize(
    ("option", "value", "message"),
    [
        ("--budget", str(1 << 63), "budget must be an integer"),
        ("--columns", str(1 << 63), "contact-sheet columns"),
        ("--thumbnail-width", str(10**500), "thumbnail width"),
    ],
)
def test_select_rejects_cli_integers_outside_signed_64_bits(
    image_factory: Callable[..., Path],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    option: str,
    value: str,
    message: str,
) -> None:
    frames = tmp_path / "frames"
    image_factory("frame.png", directory=frames)
    status = main(["select", str(frames), "--output-dir", str(tmp_path / "result"), option, value])
    assert status == 2
    assert message in capsys.readouterr().err


def test_select_accepts_filename_timestamp_rule(image_factory: Callable[..., Path], tmp_path: Path) -> None:
    frames = tmp_path / "frames"
    image_factory("frame_1000ms.png", pattern=1, directory=frames)
    output = tmp_path / "result"
    status = main(
        [
            "select",
            str(frames),
            "--output-dir",
            str(output),
            "--timestamp-mode",
            "filename",
            "--timestamp-regex",
            r"_(\d+)ms$",
            "--timestamp-unit",
            "milliseconds",
        ]
    )
    assert status == 0
    assert json.loads((output / "manifest.json").read_text())["frames"][0]["timestamp"] == 1.0


def test_select_rejects_output_inside_input(
    image_factory: Callable[..., Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    frames = tmp_path / "frames"
    image_factory("frame.png", directory=frames)
    assert main(["select", str(frames), "--output-dir", str(frames / "result")]) == 2
    assert "outside the input directory" in capsys.readouterr().err
    assert not (frames / "result").exists()


def test_scan_stdout_is_ascii_safe(
    image_factory: Callable[..., Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_factory("Straße.png")

    class AsciiStream(StringIO):
        @property
        def encoding(self) -> str:
            return "ascii"

        def write(self, value: str) -> int:
            value.encode("ascii")
            return super().write(value)

    stream = AsciiStream()
    monkeypatch.setattr(cli_module.sys, "stdout", stream)
    assert main(["scan", str(tmp_path)]) == 0
    assert json.loads(stream.getvalue())["frames"][0]["path"] == "Straße.png"


def test_select_status_is_safe_on_ascii_console(
    image_factory: Callable[..., Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frames = tmp_path / "frames"
    image_factory("frame.png", directory=frames)
    output = tmp_path / "结果"

    class AsciiStream(StringIO):
        encoding = "ascii"

        def write(self, value: str) -> int:
            value.encode("ascii")
            return super().write(value)

    stream = AsciiStream()
    monkeypatch.setattr(cli_module.sys, "stdout", stream)
    assert main(["select", str(frames), "--output-dir", str(output)]) == 0
    assert "\\u7ed3\\u679c" in stream.getvalue()


def test_error_status_is_safe_on_ascii_console(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class AsciiStream(StringIO):
        encoding = "ascii"

        def write(self, value: str) -> int:
            value.encode("ascii")
            return super().write(value)

    stream = AsciiStream()
    monkeypatch.setattr(cli_module.sys, "stderr", stream)
    assert main(["scan", str(tmp_path / "不存在")]) == 2
    assert "\\u4e0d\\u5b58\\u5728" in stream.getvalue()


def test_console_status_escapes_control_characters() -> None:
    stream = StringIO()
    cli_module._write_console("frame\nname\x1b[2J", stream=stream)
    assert stream.getvalue() == "frame\\nname\\x1b[2J\n"


def test_selection_render_failure_preserves_existing_bundle(
    image_factory: Callable[..., Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frames = tmp_path / "frames"
    image_factory("frame.png", directory=frames)
    output = tmp_path / "output"
    output.mkdir()
    manifest = output / "manifest.json"
    sheet = output / "contact-sheet.png"
    manifest.write_bytes(b"old manifest")
    sheet.write_bytes(b"old sheet")

    def fail_render(*args: object, **kwargs: object) -> None:
        raise ScanError("simulated render failure")

    monkeypatch.setattr(cli_module, "render_contact_sheet", fail_render)
    assert main(["select", str(frames), "--output-dir", str(output)]) == 2
    assert manifest.read_bytes() == b"old manifest"
    assert sheet.read_bytes() == b"old sheet"
    assert sorted(path.name for path in output.iterdir()) == ["contact-sheet.png", "manifest.json"]


def test_selection_rejects_directory_target_before_replacing_bundle(
    image_factory: Callable[..., Path], tmp_path: Path
) -> None:
    frames = tmp_path / "frames"
    image_factory("frame.png", directory=frames)
    output = tmp_path / "output"
    output.mkdir()
    manifest = output / "manifest.json"
    manifest.mkdir()
    sheet = output / "contact-sheet.png"
    sheet.write_bytes(b"old sheet")
    assert main(["select", str(frames), "--output-dir", str(output)]) == 2
    assert manifest.is_dir()
    assert sheet.read_bytes() == b"old sheet"


def test_bundle_commit_rolls_back_first_replacement(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    staging = tmp_path / "staging"
    output = tmp_path / "output"
    staging.mkdir()
    output.mkdir()
    staged_sheet = staging / "contact-sheet.png"
    staged_manifest = staging / "manifest.json"
    sheet = output / staged_sheet.name
    manifest = output / staged_manifest.name
    staged_sheet.write_bytes(b"new sheet")
    staged_manifest.write_bytes(b"new manifest")
    sheet.write_bytes(b"old sheet")
    manifest.write_bytes(b"old manifest")
    original_replace = Path.replace

    def fail_second_replace(source: Path, target: Path) -> Path:
        if source == staged_manifest:
            raise OSError("simulated locked manifest")
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", fail_second_replace)
    with pytest.raises(OutputError, match="prior artifacts were restored"):
        cli_module._commit_bundle(((staged_sheet, sheet), (staged_manifest, manifest)))
    assert sheet.read_bytes() == b"old sheet"
    assert manifest.read_bytes() == b"old manifest"


def test_expected_error_returns_two(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    status = main(["scan", str(tmp_path / "missing")])
    assert status == 2
    assert "does not exist" in capsys.readouterr().err


def test_demo_is_self_contained(tmp_path: Path) -> None:
    output = tmp_path / "demo"
    assert main(["demo", "--output-dir", str(output), "--budget", "5"]) == 0
    assert len(list((output / "frames").glob("*.png"))) == 18
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["summary"]["selected_count"] == 5


def test_demo_refuses_to_overwrite_without_force(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    output = tmp_path / "demo"
    assert main(["demo", "--output-dir", str(output)]) == 0
    assert main(["demo", "--output-dir", str(output)]) == 2
    assert "not empty" in capsys.readouterr().err


def test_demo_force_refuses_unrelated_files(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    output = tmp_path / "demo"
    frames = output / "frames"
    frames.mkdir(parents=True)
    (frames / "notes.txt").write_text("keep me")
    assert main(["demo", "--output-dir", str(output), "--force"]) == 2
    assert (frames / "notes.txt").read_text() == "keep me"
    assert "unrelated content" in capsys.readouterr().err


def test_demo_force_does_not_delete_lookalike_frame(tmp_path: Path) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    lookalike = frames / "frame_secretms.png"
    lookalike.write_text("keep me", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="unrelated content"):
        create_demo_sequence(frames, overwrite=True)
    assert lookalike.read_text(encoding="utf-8") == "keep me"


def test_demo_refuses_symlink_directory(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    sentinel = target / "frame_00000ms.png"
    sentinel.write_text("keep me", encoding="utf-8")
    link = tmp_path / "frames"
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"directory symlinks are unavailable: {error}")
    with pytest.raises(ConfigurationError, match="symbolic link"):
        create_demo_sequence(link, overwrite=True)
    assert sentinel.read_text(encoding="utf-8") == "keep me"


def test_demo_force_replaces_only_expected_frames(tmp_path: Path) -> None:
    frames = create_demo_sequence(tmp_path / "frames")
    expected = frames / "frame_00000ms.png"
    expected.write_bytes(b"stale")
    create_demo_sequence(frames, overwrite=True)
    with Image.open(expected) as image:
        assert image.format == "PNG"


def test_demo_reports_file_where_frame_directory_is_expected(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "demo"
    output.mkdir()
    (output / "frames").write_text("not a directory")
    assert main(["demo", "--output-dir", str(output)]) == 2
    assert "not a directory" in capsys.readouterr().err
