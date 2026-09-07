from __future__ import annotations

import json
import os
import subprocess
from dataclasses import replace
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

import frame_quorum.video as video_module
from frame_quorum.errors import ConfigurationError, OutputError, ScanError
from frame_quorum.video import (
    VideoExtractionConfig,
    VideoExtractionResult,
    extract_video_frames,
    ffmpeg_available,
)

_REAL_PROBE_FFMPEG_VERSION = video_module._probe_ffmpeg_version


def _video(tmp_path: Path) -> Path:
    path = tmp_path / "source.mp4"
    path.write_bytes(b"synthetic test placeholder")
    return path


@pytest.fixture(autouse=True)
def _stable_version(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(video_module, "_probe_ffmpeg_version", lambda _: "ffmpeg version test-1.0")


def _successful_ffmpeg(
    command: list[str], stage: Path, options: VideoExtractionConfig
) -> video_module._ProcessOutcome:
    pattern = Path(command[-1])
    for index in range(3):
        Image.new("RGB", (32, 24), (index * 50, 20, 90)).save(
            Path(str(pattern).replace("%09d", f"{index:09d}"))
        )
    total = sum(path.stat().st_size for path in stage.iterdir())
    return video_module._ProcessOutcome(total, "", False)


def test_extract_publishes_valid_sequence_atomically(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(video_module, "_execute_ffmpeg", _successful_ffmpeg)
    output = tmp_path / "frames"
    result = extract_video_frames(
        _video(tmp_path),
        output,
        VideoExtractionConfig(frame_rate=2.5, max_frames=10, timeout_seconds=2),
    )
    assert result.frame_count == 3
    assert len(list(output.glob("*.png"))) == 3
    metadata = json.loads((output / "extraction.json").read_text(encoding="utf-8"))
    assert metadata["sampling"]["frame_rate"] == 2.5
    assert metadata["result"]["frame_count"] == 3
    assert metadata["input"]["sha256"] == result.input_sha256
    assert metadata["ffmpeg"]["version"] == "ffmpeg version test-1.0"
    assert metadata["limits"]["max_output_bytes_scope"].startswith("generated PNG")
    assert metadata["limits"]["timeout_scope"] == "FFmpeg subprocess runtime only"
    png_bytes = sum(path.stat().st_size for path in output.glob("*.png"))
    assert result.total_output_bytes == png_bytes
    assert sum(path.stat().st_size for path in output.iterdir()) > result.total_output_bytes
    assert not list(tmp_path.glob(".frame-quorum-extract-*"))


def test_extract_refuses_existing_destination(tmp_path: Path) -> None:
    output = tmp_path / "frames"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")
    with pytest.raises(OutputError, match="must not already exist"):
        extract_video_frames(_video(tmp_path), output)
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_extract_rejects_supplied_symlinks_before_resolution_on_every_platform(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _video(tmp_path)
    output = tmp_path / "frames"
    original = Path.is_symlink
    marked: set[Path] = {source}

    def mark_supplied(path: Path) -> bool:
        return path in marked or original(path)

    monkeypatch.setattr(Path, "is_symlink", mark_supplied)
    with pytest.raises(ScanError, match="symbolic-link video input"):
        extract_video_frames(source, output)

    marked.clear()
    marked.add(output)
    with pytest.raises(OutputError, match="symbolic-link extraction output"):
        extract_video_frames(source, output)


def test_extract_rejects_real_dangling_input_and_output_symlinks(tmp_path: Path) -> None:
    dangling_input = tmp_path / "dangling-input.mp4"
    try:
        dangling_input.symlink_to(tmp_path / "missing-input.mp4")
    except OSError as error:
        pytest.skip(f"file symlinks are unavailable: {error}")
    with pytest.raises(ScanError, match="symbolic-link video input"):
        extract_video_frames(dangling_input, tmp_path / "frames")

    dangling_output = tmp_path / "dangling-output"
    try:
        dangling_output.symlink_to(tmp_path / "missing-output", target_is_directory=True)
    except OSError as error:
        pytest.skip(f"directory symlinks are unavailable: {error}")
    with pytest.raises(OutputError, match="symbolic-link extraction output"):
        extract_video_frames(_video(tmp_path), dangling_output)


def test_extract_rejects_input_replaced_while_ffmpeg_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _video(tmp_path)
    output = tmp_path / "frames"
    initial = source.stat()

    def replace_input(
        command: list[str], stage: Path, options: VideoExtractionConfig
    ) -> video_module._ProcessOutcome:
        outcome = _successful_ffmpeg(command, stage, options)
        source.write_bytes(source.read_bytes()[::-1])
        os.utime(source, ns=(initial.st_atime_ns, initial.st_mtime_ns))
        mutated = source.stat()
        assert (mutated.st_dev, mutated.st_ino, mutated.st_size, mutated.st_mtime_ns) == (
            initial.st_dev,
            initial.st_ino,
            initial.st_size,
            initial.st_mtime_ns,
        )
        return outcome

    monkeypatch.setattr(video_module, "_execute_ffmpeg", replace_input)
    with pytest.raises(ScanError, match="changed while FFmpeg extraction was running"):
        extract_video_frames(source, output)
    assert not output.exists()
    assert not list(tmp_path.glob(".frame-quorum-extract-*"))


def test_extract_rejects_input_that_becomes_a_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _video(tmp_path)
    output = tmp_path / "frames"
    original = Path.is_symlink
    source_checks = 0

    def changes_after_launch(path: Path) -> bool:
        nonlocal source_checks
        if path == source:
            source_checks += 1
            return source_checks > 1
        return original(path)

    monkeypatch.setattr(Path, "is_symlink", changes_after_launch)
    monkeypatch.setattr(video_module, "_execute_ffmpeg", _successful_ffmpeg)
    with pytest.raises(ScanError, match="became a symbolic link"):
        extract_video_frames(source, output)
    assert not output.exists()


def test_extract_rejects_missing_input(tmp_path: Path) -> None:
    with pytest.raises(ScanError, match="regular file"):
        extract_video_frames(tmp_path / "missing.mp4", tmp_path / "frames")
    with pytest.raises(ConfigurationError, match="config must be"):
        extract_video_frames(_video(tmp_path), tmp_path / "frames", config=0)  # type: ignore[arg-type]


def test_extract_wraps_parent_creation_error(tmp_path: Path) -> None:
    parent = tmp_path / "not-a-directory"
    parent.write_text("file", encoding="utf-8")
    with pytest.raises(OutputError, match="cannot create"):
        extract_video_frames(_video(tmp_path), parent / "frames")


@pytest.mark.parametrize(
    "config",
    [
        VideoExtractionConfig(frame_rate=0),
        VideoExtractionConfig(frame_rate=float("inf")),
        VideoExtractionConfig(max_frames=0),
        VideoExtractionConfig(max_frames=True),
        VideoExtractionConfig(max_frames=1_000_001),
        VideoExtractionConfig(max_output_bytes=0),
        VideoExtractionConfig(max_output_bytes=True),
        VideoExtractionConfig(max_output_bytes=(1 << 40) + 1),
        VideoExtractionConfig(timeout_seconds=0),
        VideoExtractionConfig(timeout_seconds=86_401),
        VideoExtractionConfig(executable="bad\nname"),
    ],
)
def test_extract_rejects_invalid_config(config: VideoExtractionConfig, tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError):
        extract_video_frames(_video(tmp_path), tmp_path / "frames", config)


def test_extract_wraps_missing_ffmpeg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(_: str) -> str:
        raise OutputError("FFmpeg executable was not found: ffmpeg")

    monkeypatch.setattr(video_module, "_probe_ffmpeg_version", missing)
    with pytest.raises(OutputError, match="not found"):
        extract_video_frames(_video(tmp_path), tmp_path / "frames")


def test_extract_wraps_timeout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def timeout(*_: object) -> video_module._ProcessOutcome:
        raise OutputError("FFmpeg exceeded the 1 second timeout")

    monkeypatch.setattr(video_module, "_execute_ffmpeg", timeout)
    with pytest.raises(OutputError, match="timeout"):
        extract_video_frames(
            _video(tmp_path),
            tmp_path / "frames",
            VideoExtractionConfig(timeout_seconds=1),
        )


def test_extract_reports_sanitized_ffmpeg_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*_: object) -> video_module._ProcessOutcome:
        raise OutputError("FFmpeg extraction failed with exit code 7: bad input")

    monkeypatch.setattr(video_module, "_execute_ffmpeg", fail)
    with pytest.raises(OutputError, match="exit code 7: bad input"):
        extract_video_frames(_video(tmp_path), tmp_path / "frames")


def test_extract_rejects_empty_and_invalid_ffmpeg_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_output(*_: object) -> video_module._ProcessOutcome:
        return video_module._ProcessOutcome(0, "", False)

    monkeypatch.setattr(video_module, "_execute_ffmpeg", no_output)
    with pytest.raises(OutputError, match="without producing"):
        extract_video_frames(_video(tmp_path), tmp_path / "empty")

    def invalid(command: list[str], stage: Path, _: object) -> video_module._ProcessOutcome:
        Path(str(command[-1]).replace("%09d", "000000000")).write_bytes(b"not a png")
        return video_module._ProcessOutcome(9, "", False)

    monkeypatch.setattr(video_module, "_execute_ffmpeg", invalid)
    with pytest.raises(OutputError, match="invalid image sequence"):
        extract_video_frames(_video(tmp_path), tmp_path / "invalid")


def test_extract_rejects_excess_and_nonregular_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def excess(command: list[str], stage: Path, _: object) -> video_module._ProcessOutcome:
        pattern = Path(command[-1])
        for index in range(2):
            Image.new("RGB", (8, 8)).save(Path(str(pattern).replace("%09d", f"{index:09d}")))
        return video_module._ProcessOutcome(sum(path.stat().st_size for path in stage.iterdir()), "", False)

    monkeypatch.setattr(video_module, "_execute_ffmpeg", excess)
    with pytest.raises(OutputError, match="more frames"):
        extract_video_frames(
            _video(tmp_path),
            tmp_path / "excess",
            VideoExtractionConfig(max_frames=1),
        )

    def unexpected(command: list[str], stage: Path, _: object) -> video_module._ProcessOutcome:
        pattern = Path(command[-1])
        Image.new("RGB", (8, 8)).save(Path(str(pattern).replace("%09d", "000000001")))
        return video_module._ProcessOutcome(sum(path.stat().st_size for path in stage.iterdir()), "", False)

    monkeypatch.setattr(video_module, "_execute_ffmpeg", unexpected)
    with pytest.raises(OutputError, match="contiguous sequence"):
        extract_video_frames(_video(tmp_path), tmp_path / "unexpected")

    def nonregular(command: list[str], stage: Path, _: object) -> video_module._ProcessOutcome:
        Path(str(command[-1]).replace("%09d", "000000000")).mkdir()
        return video_module._ProcessOutcome(0, "", False)

    monkeypatch.setattr(video_module, "_execute_ffmpeg", nonregular)
    with pytest.raises(OutputError, match="regular PNG"):
        extract_video_frames(_video(tmp_path), tmp_path / "nonregular")


def test_extraction_result_enforces_public_contract(tmp_path: Path) -> None:
    result = VideoExtractionResult(
        tmp_path / "frames", 1, 2.0, 10, 100, 1000, "sha256:" + "a" * 64, "ffmpeg test"
    )
    result.validate()
    invalid = [
        (replace(result, output_dir="frames"), "pathlib.Path"),  # type: ignore[arg-type]
        (replace(result, frame_count=0), "frame_count"),
        (replace(result, frame_rate=True), "frame_rate"),
        (replace(result, max_frames=1_000_001), "cannot exceed"),
        (replace(result, frame_count=11), "cannot exceed"),
        (replace(result, total_output_bytes=1001), "cannot exceed"),
        (replace(result, input_sha256="bad"), "sha256"),
    ]
    for record, message in invalid:
        with pytest.raises(ConfigurationError, match=message):
            record.validate()


def test_ffmpeg_availability_is_side_effect_free(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(video_module.shutil, "which", lambda executable: f"/bin/{executable}")
    assert ffmpeg_available()
    monkeypatch.setattr(video_module.shutil, "which", lambda executable: None)
    assert not ffmpeg_available("custom-ffmpeg")
    with pytest.raises(ConfigurationError):
        ffmpeg_available("bad\x00name")


class _FakeProcess:
    def __init__(self, returncode: int | None, diagnostic: bytes = b"") -> None:
        self.returncode = returncode
        self.stderr = BytesIO(diagnostic)
        self.stdout = BytesIO(diagnostic)
        self.terminated = False
        self.killed = False
        self.waited = False

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        self.waited = True
        if self.returncode is None:
            if timeout is not None and not self.terminated and not self.killed:
                raise subprocess.TimeoutExpired("ffmpeg", timeout)
            self.returncode = -9 if self.killed else -15
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = -15

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9


def test_execute_ffmpeg_bounds_stderr_and_waits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    process = _FakeProcess(0, b"x" * 20_000)
    monkeypatch.setattr(video_module.subprocess, "Popen", lambda *args, **kwargs: process)
    (tmp_path / "frame_000000000.png").write_bytes(b"png")
    outcome = video_module._execute_ffmpeg(["ffmpeg"], tmp_path, VideoExtractionConfig(max_output_bytes=100))
    assert outcome.total_output_bytes == 3
    assert outcome.stderr_truncated
    assert len(outcome.stderr_text) <= 530
    assert process.waited


def test_live_output_limit_stops_process_and_cleans_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _video(tmp_path)
    output = tmp_path / "frames"
    process = _FakeProcess(None)

    def launch(command: list[str], **_: object) -> _FakeProcess:
        Path(str(command[-1]).replace("%09d", "000000000")).write_bytes(b"x" * 11)
        return process

    monkeypatch.setattr(video_module.subprocess, "Popen", launch)
    with pytest.raises(OutputError, match="max_output_bytes during decoding"):
        extract_video_frames(
            source,
            output,
            VideoExtractionConfig(max_output_bytes=10, timeout_seconds=1),
        )
    assert process.terminated and process.waited
    assert not output.exists()
    assert not list(tmp_path.glob(".frame-quorum-extract-*"))


def test_execute_reports_exit_and_start_failures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    failed = _FakeProcess(7, b"bad\ninput")
    monkeypatch.setattr(video_module.subprocess, "Popen", lambda *args, **kwargs: failed)
    with pytest.raises(OutputError, match="exit code 7: bad input"):
        video_module._execute_ffmpeg(["ffmpeg"], tmp_path, VideoExtractionConfig())

    def missing(*_: object, **__: object) -> _FakeProcess:
        raise FileNotFoundError

    monkeypatch.setattr(video_module.subprocess, "Popen", missing)
    with pytest.raises(OutputError, match="not found"):
        video_module._execute_ffmpeg(["ffmpeg"], tmp_path, VideoExtractionConfig())


def test_probe_records_only_real_first_line(monkeypatch: pytest.MonkeyPatch) -> None:
    process = _FakeProcess(0, b"ffmpeg version test\nbuilt with flags\n")
    monkeypatch.setattr(video_module.subprocess, "Popen", lambda *args, **kwargs: process)
    assert _REAL_PROBE_FFMPEG_VERSION("ffmpeg") == "ffmpeg version test"


def test_probe_failure_and_timeout_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    failed = _FakeProcess(2)
    monkeypatch.setattr(video_module.subprocess, "Popen", lambda *args, **kwargs: failed)
    with pytest.raises(OutputError, match="query failed with exit code 2"):
        _REAL_PROBE_FFMPEG_VERSION("ffmpeg")

    waiting = _FakeProcess(None)
    monkeypatch.setattr(video_module.subprocess, "Popen", lambda *args, **kwargs: waiting)
    with pytest.raises(OutputError, match="query exceeded 10 seconds"):
        _REAL_PROBE_FFMPEG_VERSION("ffmpeg")
    assert waiting.terminated and waiting.waited


def test_execute_timeout_stops_running_process(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    process = _FakeProcess(None)
    monkeypatch.setattr(video_module.subprocess, "Popen", lambda *args, **kwargs: process)
    moments = iter((0.0, 2.0))
    monkeypatch.setattr(video_module.time, "monotonic", lambda: next(moments))
    with pytest.raises(OutputError, match="second timeout"):
        video_module._execute_ffmpeg(["ffmpeg"], tmp_path, VideoExtractionConfig(timeout_seconds=1))
    assert process.terminated and process.waited


def test_stop_process_escalates_to_kill(monkeypatch: pytest.MonkeyPatch) -> None:
    process = _FakeProcess(None)

    def stubborn_wait(timeout: float | None = None) -> int:
        process.waited = True
        if timeout is not None:
            raise subprocess.TimeoutExpired("ffmpeg", timeout)
        process.returncode = -9
        return -9

    monkeypatch.setattr(process, "wait", stubborn_wait)
    video_module._stop_process(process)  # type: ignore[arg-type]
    assert process.terminated and process.killed and process.waited


def test_hash_and_digest_contract_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ScanError, match="cannot hash input video"):
        video_module._snapshot_file(tmp_path / "missing")
    with pytest.raises(ConfigurationError, match="64 hex digits"):
        video_module._require_sha256("sha256:" + "z" * 64, "digest")

    def cannot_scan(_: object) -> object:
        raise OSError("denied")

    monkeypatch.setattr(video_module.os, "scandir", cannot_scan)
    with pytest.raises(OSError, match="denied"):
        video_module._directory_output_bytes(tmp_path, 1)


def test_bounded_collector_and_result_text_contract(tmp_path: Path) -> None:
    collector = video_module._BoundedCollector(4)
    collector.drain(None)
    collector.drain(BytesIO(b"abcdef"))
    assert collector.truncated
    assert "truncated" in collector.text()

    result = VideoExtractionResult(tmp_path, 1, 1.0, 1, 1, 1, "sha256:" + "0" * 64, "")
    with pytest.raises(ConfigurationError, match="FFmpeg version"):
        result.validate()
    with pytest.raises(ConfigurationError, match="cannot exceed"):
        replace(result, ffmpeg_version="ffmpeg", max_output_bytes=(1 << 40) + 1).validate()

    class BrokenStream(BytesIO):
        def read(self, size: int = -1) -> bytes:
            raise OSError("broken")

    collector.drain(BrokenStream())
    assert isinstance(collector.error, OSError)
