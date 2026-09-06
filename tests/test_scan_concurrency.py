"""Opt-in parallel scanning must change only speed, never observable output."""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Callable
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

import frame_quorum.scanner as scanner_module
from frame_quorum.benchmark import (
    BenchmarkConfig,
    benchmark_manifest,
    run_benchmark,
    source_content_digest,
)
from frame_quorum.cli import main
from frame_quorum.errors import ConfigurationError, ScanError
from frame_quorum.metrics import measure_image
from frame_quorum.models import AnimationConfig, ConcurrencyConfig, Frame, ScanConfig, SelectionConfig
from frame_quorum.scanner import scan_frames

# Only ever spent when the pool fails to run tasks concurrently, in which case a
# rendezvous can never complete and the test must fail instead of hanging.
_RENDEZVOUS_TIMEOUT = 60.0
_PARALLEL = ConcurrencyConfig(workers=4)
_GIF_ONLY = ScanConfig(extensions=(".gif",))


def _sequence(image_factory: Callable[..., Path], root: Path, count: int = 9) -> Path:
    directory = root / "frames"
    for index in range(count):
        image_factory(
            f"frame_{index:03d}.png",
            color=((index * 29) % 255, (index * 53) % 255, (index * 71) % 255),
            pattern=index + 1,
            directory=directory,
        )
    return directory


def _width_tagged_sequence(image_factory: Callable[..., Path], root: Path, count: int = 4) -> Path:
    """Write files whose distinct widths identify them inside a measurement hook."""

    directory = root / "frames"
    for index in range(count):
        image_factory(
            f"frame_{index:03d}.png",
            size=(100 + index, 80),
            pattern=index + 1,
            directory=directory,
        )
    return directory


def _animation(path: Path, count: int, *, size: tuple[int, int] = (64, 48)) -> Path:
    internal = []
    for position in range(count):
        image = Image.new("RGB", size, ((position * 47) % 256, 40, (200 - position * 30) % 256))
        ImageDraw.Draw(image).rectangle((position * 7, 6, position * 7 + 18, 26), fill=(240, 210, 30))
        internal.append(image)
    path.parent.mkdir(parents=True, exist_ok=True)
    internal[0].save(path, save_all=True, append_images=internal[1:], duration=120, loop=0)
    return path


def _canonical(frames: tuple[Frame, ...]) -> str:
    return json.dumps(
        [frame.serializable() for frame in frames],
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def test_parallel_scan_returns_records_identical_to_sequential(
    image_factory: Callable[..., Path], tmp_path: Path
) -> None:
    directory = _sequence(image_factory, tmp_path)
    config = ScanConfig(frame_rate=2.0)
    sequential = scan_frames(directory, config)
    parallel = scan_frames(directory, config, concurrency=_PARALLEL)
    assert parallel == sequential
    assert _canonical(parallel) == _canonical(sequential)
    assert [frame.index for frame in parallel] == list(range(9))


def test_explicit_single_worker_matches_an_absent_concurrency_config(
    image_factory: Callable[..., Path], tmp_path: Path
) -> None:
    directory = _sequence(image_factory, tmp_path, count=3)
    assert scan_frames(directory, concurrency=ConcurrencyConfig()) == scan_frames(directory)


def test_more_workers_than_files_still_scans_a_single_file(
    image_factory: Callable[..., Path],
) -> None:
    path = image_factory("only.png", size=(37, 29))
    frames = scan_frames(path, concurrency=ConcurrencyConfig(workers=64))
    assert (frames[0].width, frames[0].height) == (37, 29)
    assert frames[0].relative_path == "only.png"


def test_expanded_animations_keep_contiguous_indices_under_parallel_scanning(tmp_path: Path) -> None:
    _animation(tmp_path / "clip_a.gif", 3)
    _animation(tmp_path / "clip_b.gif", 4, size=(40, 32))
    _animation(tmp_path / "clip_c.gif", 2)
    animation = AnimationConfig()
    sequential = scan_frames(tmp_path, _GIF_ONLY, animation=animation)
    parallel = scan_frames(tmp_path, _GIF_ONLY, animation=animation, concurrency=_PARALLEL)
    assert parallel == sequential
    assert [frame.index for frame in parallel] == list(range(9))
    assert parallel[3].relative_path == "clip_b.gif#frame=0"
    assert parallel[3].source_frame_index == 0


def test_parallel_scan_preserves_the_checked_in_measured_record_fingerprint() -> None:
    repository = Path(__file__).parents[1]
    scan_config = ScanConfig()
    frames = scan_frames(repository / "examples" / "output" / "frames", scan_config, concurrency=_PARALLEL)
    result = run_benchmark(
        frames,
        SelectionConfig(budget=6),
        BenchmarkConfig(random_seed=1729, random_trials=8),
        scan_config=scan_config,
        source_digest=source_content_digest(frames),
    )
    expected = json.loads(
        (repository / "examples" / "benchmark" / "benchmark.json").read_text(encoding="utf-8")
    )
    actual = benchmark_manifest(result)
    expected["protocol"]["provenance"]["software"] = actual["protocol"]["provenance"]["software"]
    assert actual == expected


def test_workers_measure_several_files_at_the_same_time(
    image_factory: Callable[..., Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Evidence of real overlap that does not time an operation.

    Every task must reach the rendezvous before any of them may leave it, so a
    scan that measured one file at a time could not complete this at all.
    """

    workers = 4
    directory = _sequence(image_factory, tmp_path, count=workers)
    rendezvous = threading.Barrier(workers, timeout=_RENDEZVOUS_TIMEOUT)
    lock = threading.Lock()
    in_flight = 0
    peak = 0
    threads: set[str] = set()

    def instrumented(image: Image.Image) -> object:
        nonlocal in_flight, peak
        with lock:
            in_flight += 1
            peak = max(peak, in_flight)
            threads.add(threading.current_thread().name)
        try:
            rendezvous.wait()
            return measure_image(image)
        finally:
            with lock:
                in_flight -= 1

    monkeypatch.setattr(scanner_module, "measure_image", instrumented)
    frames = scan_frames(directory, concurrency=ConcurrencyConfig(workers=workers))
    assert len(frames) == workers
    assert peak == workers
    assert len(threads) == workers
    assert all(name.startswith("frame-quorum-scan") for name in threads)


def test_first_failure_in_path_order_is_reported_however_workers_finish(
    image_factory: Callable[..., Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A later file is forced to fail first; the earlier file must still be named."""

    directory = _width_tagged_sequence(image_factory, tmp_path)

    def failing(image: Image.Image) -> object:
        if image.width in {100, 103}:
            raise ValueError("unreadable pixel data")
        return measure_image(image)

    monkeypatch.setattr(scanner_module, "measure_image", failing)
    with pytest.raises(ScanError) as sequential_error:
        scan_frames(directory)

    started = threading.Barrier(4, timeout=_RENDEZVOUS_TIMEOUT)
    last_failed = threading.Event()

    def racing(image: Image.Image) -> object:
        started.wait()
        if image.width == 103:
            last_failed.set()
            raise ValueError("unreadable pixel data")
        if image.width == 100:
            if not last_failed.wait(timeout=_RENDEZVOUS_TIMEOUT):
                raise RuntimeError("the last file never failed")
            raise ValueError("unreadable pixel data")
        return measure_image(image)

    monkeypatch.setattr(scanner_module, "measure_image", racing)
    with pytest.raises(ScanError) as parallel_error:
        scan_frames(directory, concurrency=_PARALLEL)

    assert "frame_000.png" in str(sequential_error.value)
    assert str(parallel_error.value) == str(sequential_error.value)


def test_a_file_touched_by_one_worker_fails_the_post_read_identity_check(
    image_factory: Callable[..., Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = _width_tagged_sequence(image_factory, tmp_path)
    target = directory / "frame_002.png"
    mutating_threads: list[str] = []

    def touch_target(image: Image.Image) -> object:
        metrics = measure_image(image)
        if image.width == 102:
            moved = target.stat().st_mtime_ns + 2_000_000_000
            os.utime(target, ns=(moved, moved))
            mutating_threads.append(threading.current_thread().name)
        return metrics

    monkeypatch.setattr(scanner_module, "measure_image", touch_target)
    with pytest.raises(ScanError, match="changed while it was being scanned"):
        scan_frames(directory, concurrency=_PARALLEL)
    assert mutating_threads
    assert all(name.startswith("frame-quorum-scan") for name in mutating_threads)


@pytest.mark.parametrize("workers", [0, -1, 65, True, 1.5, "4", None])
def test_invalid_worker_counts_are_rejected(workers: object) -> None:
    with pytest.raises(ConfigurationError, match="concurrency workers"):
        ConcurrencyConfig(workers=workers).validate()  # type: ignore[arg-type]


def test_worker_count_bounds_are_reported_precisely() -> None:
    with pytest.raises(ConfigurationError, match="between 1 and"):
        ConcurrencyConfig(workers=0).validate()
    with pytest.raises(ConfigurationError, match="cannot exceed 64"):
        ConcurrencyConfig(workers=65).validate()


def test_scan_rejects_an_invalid_concurrency_config(
    image_factory: Callable[..., Path], tmp_path: Path
) -> None:
    directory = _sequence(image_factory, tmp_path, count=2)
    with pytest.raises(ConfigurationError, match="must be ConcurrencyConfig"):
        scan_frames(directory, concurrency=object())  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError, match="concurrency workers"):
        scan_frames(directory, concurrency=ConcurrencyConfig(workers=0))


def test_cli_worker_count_produces_a_byte_identical_manifest(
    image_factory: Callable[..., Path], tmp_path: Path
) -> None:
    directory = _sequence(image_factory, tmp_path, count=5)
    parallel = tmp_path / "parallel.json"
    sequential = tmp_path / "sequential.json"
    assert main(["scan", str(directory), "--workers", "4", "--output", str(parallel)]) == 0
    assert main(["scan", str(directory), "--output", str(sequential)]) == 0
    assert parallel.read_bytes() == sequential.read_bytes()
    assert "workers" not in json.loads(parallel.read_text(encoding="utf-8"))["scan_config"]


def test_cli_select_accepts_workers_and_matches_sequential_output(
    image_factory: Callable[..., Path], tmp_path: Path
) -> None:
    directory = _sequence(image_factory, tmp_path, count=6)
    parallel = tmp_path / "parallel-selection"
    sequential = tmp_path / "sequential-selection"
    assert main(["select", str(directory), "--output-dir", str(parallel), "--workers", "3"]) == 0
    assert main(["select", str(directory), "--output-dir", str(sequential)]) == 0
    assert (parallel / "manifest.json").read_bytes() == (sequential / "manifest.json").read_bytes()


def test_cli_rejects_a_non_positive_worker_count(
    image_factory: Callable[..., Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    directory = _sequence(image_factory, tmp_path, count=2)
    assert main(["scan", str(directory), "--workers", "0"]) == 2
    assert "concurrency workers" in capsys.readouterr().err
