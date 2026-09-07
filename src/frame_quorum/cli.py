"""Command-line interface for scanning, selecting, and generating a demo."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TextIO

from ._version import VERSION
from .benchmark import (
    BenchmarkConfig,
    benchmark_manifest,
    render_benchmark_svg,
    run_benchmark,
    source_content_digest,
)
from .contact_sheet import render_contact_sheet
from .demo import create_demo_sequence
from .errors import ConfigurationError, FrameQuorumError, OutputError
from .models import AnimationConfig, ConcurrencyConfig, Frame, ScanConfig, SelectionConfig
from .reporting import scan_manifest, selection_manifest, write_json
from .representation import (
    DEFAULT_COVERED_DISTANCE,
    MAX_REPORTED_GAPS,
    analyze_representation,
    budget_curve,
)
from .scanner import scan_frames
from .selector import select_frames
from .video import VideoExtractionConfig, extract_video_frames


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="frame-quorum",
        description="Explainable, content-aware key-frame selection for image sequences.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")
    commands = parser.add_subparsers(dest="command", required=True)

    scan = commands.add_parser("scan", help="measure a frame directory and emit JSON")
    scan.add_argument("input", type=Path)
    _add_scan_options(scan)
    _add_animation_options(scan)
    scan.add_argument("--output", "-o", type=Path, help="write JSON here instead of stdout")
    scan.set_defaults(handler=_handle_scan)

    select = commands.add_parser("select", help="select key frames and create a report")
    select.add_argument("input", type=Path)
    _add_scan_options(select)
    _add_animation_options(select)
    _add_selection_options(select)
    select.add_argument("--output-dir", "-o", type=Path, required=True)
    select.add_argument("--columns", type=int, default=3)
    select.add_argument("--thumbnail-width", type=int, default=320)
    select.set_defaults(handler=_handle_select)

    coverage = commands.add_parser(
        "coverage", help="measure how well a selection represents the frames it drops"
    )
    coverage.add_argument("input", type=Path)
    _add_scan_options(coverage)
    _add_animation_options(coverage)
    _add_selection_options(coverage)
    coverage.add_argument("--output", type=Path, help="write the JSON report here")
    coverage.add_argument(
        "--covered-distance",
        type=float,
        default=DEFAULT_COVERED_DISTANCE,
        help="content distance below which a dropped frame counts as represented",
    )
    coverage.add_argument(
        "--budgets",
        type=int,
        nargs="+",
        help="also re-select at these budgets and report what each represents",
    )
    coverage.set_defaults(handler=_handle_coverage)

    benchmark = commands.add_parser(
        "benchmark", help="compare selection against reproducible transparent baselines"
    )
    benchmark.add_argument("input", type=Path)
    _add_scan_options(benchmark)
    _add_selection_options(benchmark)
    benchmark.add_argument("--output-dir", "-o", type=Path, required=True)
    benchmark.add_argument("--random-seed", type=int, default=1729)
    benchmark.add_argument("--random-trials", type=int, default=8)
    benchmark.set_defaults(handler=_handle_benchmark)

    demo = commands.add_parser("demo", help="generate and process a synthetic sequence")
    demo.add_argument("--output-dir", "-o", type=Path, default=Path("frame-quorum-demo"))
    demo.add_argument("--force", action="store_true", help="replace the previously generated demo frames")
    _add_selection_options(demo, budget=6)
    demo.add_argument("--columns", type=int, default=3)
    demo.add_argument("--thumbnail-width", type=int, default=320)
    demo.set_defaults(handler=_handle_demo)

    extract = commands.add_parser("extract", help="extract a bounded image sequence with optional FFmpeg")
    extract.add_argument("input", type=Path)
    extract.add_argument("--output-dir", "-o", type=Path, required=True)
    extract.add_argument("--frame-rate", type=float, default=2.0)
    extract.add_argument("--max-frames", type=int, default=10_000)
    extract.add_argument(
        "--max-output-bytes",
        type=int,
        default=1_000_000_000,
        help="maximum total generated PNG bytes; extraction.json is excluded",
    )
    extract.add_argument(
        "--timeout", type=float, default=300.0, help="maximum FFmpeg subprocess runtime in seconds"
    )
    extract.add_argument("--ffmpeg", default="ffmpeg", help="FFmpeg executable name or path")
    extract.set_defaults(handler=_handle_extract)
    return parser


def _add_scan_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--recursive", action="store_true", help="include nested image directories")
    parser.add_argument(
        "--timestamp-mode",
        choices=("index", "filename", "mtime", "exif", "none"),
        default="index",
        help="timestamp source (default: index)",
    )
    parser.add_argument(
        "--frame-rate", type=float, default=1.0, help="frames per second for index timestamps"
    )
    parser.add_argument("--timestamp-regex", help="regex with first capture or named 'ts' group")
    parser.add_argument(
        "--timestamp-unit",
        choices=("seconds", "milliseconds", "microseconds"),
        default="seconds",
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        metavar="EXT",
        default=list(ScanConfig().extensions),
        help="image extensions to discover (default: %(default)s)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=ConcurrencyConfig().workers,
        help=(
            "threads that decode and measure files at once; output is identical "
            "at every value (default: %(default)s, sequential)"
        ),
    )


def _add_animation_options(parser: argparse.ArgumentParser) -> None:
    defaults = AnimationConfig()
    parser.add_argument(
        "--expand-animations",
        action="store_true",
        help="expand animated GIF, APNG, and WebP files into their internal frames",
    )
    parser.add_argument(
        "--max-animation-frames",
        type=int,
        default=defaults.max_frames,
        help="maximum internal frames accepted from one animated file",
    )
    parser.add_argument(
        "--max-animation-decoded-bytes",
        type=int,
        default=defaults.max_decoded_bytes,
        help="maximum estimated RGB bytes one expanded animated file may decode to",
    )


def _add_selection_options(parser: argparse.ArgumentParser, *, budget: int = 8) -> None:
    parser.add_argument("--budget", type=int, default=budget)
    parser.add_argument("--min-gap", type=float, default=0.0, help="minimum timestamp/index distance")
    parser.add_argument("--duplicate-threshold", type=float, default=0.035)
    parser.add_argument("--quality-weight", type=float, default=0.32)
    parser.add_argument("--change-weight", type=float, default=0.38)
    parser.add_argument("--coverage-weight", type=float, default=0.30)
    parser.add_argument("--no-endpoints", action="store_true", help="do not reserve sequence endpoints")


def _scan_config(args: argparse.Namespace) -> ScanConfig:
    return ScanConfig(
        recursive=args.recursive,
        timestamp_mode=args.timestamp_mode,
        frame_rate=args.frame_rate,
        timestamp_regex=args.timestamp_regex,
        timestamp_unit=args.timestamp_unit,
        extensions=tuple(args.extensions),
    )


def _concurrency_config(args: argparse.Namespace) -> ConcurrencyConfig:
    return ConcurrencyConfig(workers=args.workers)


def _animation_config(args: argparse.Namespace) -> AnimationConfig | None:
    if not args.expand_animations:
        return None
    return AnimationConfig(
        max_frames=args.max_animation_frames,
        max_decoded_bytes=args.max_animation_decoded_bytes,
    )


def _selection_config(args: argparse.Namespace) -> SelectionConfig:
    return SelectionConfig(
        budget=args.budget,
        min_gap=args.min_gap,
        duplicate_threshold=args.duplicate_threshold,
        quality_weight=args.quality_weight,
        change_weight=args.change_weight,
        coverage_weight=args.coverage_weight,
        keep_endpoints=not args.no_endpoints,
    )


def _handle_scan(args: argparse.Namespace) -> int:
    config = _scan_config(args)
    animation = _animation_config(args)
    frames = scan_frames(args.input, config, animation=animation, concurrency=_concurrency_config(args))
    rendered = write_json(
        scan_manifest(frames, config, animation=animation),
        args.output,
        ensure_ascii=args.output is None,
    )
    if args.output is None:
        sys.stdout.write(rendered)
    else:
        _write_console(f"Scanned {len(frames)} frames -> {args.output}")
    return 0


def _handle_select(args: argparse.Namespace) -> int:
    _require_output_outside_input(args.input, args.output_dir)
    scan_config = _scan_config(args)
    animation = _animation_config(args)
    frames = scan_frames(args.input, scan_config, animation=animation, concurrency=_concurrency_config(args))
    return _write_selection(
        frames,
        scan_config,
        _selection_config(args),
        args.output_dir,
        columns=args.columns,
        thumbnail_width=args.thumbnail_width,
        animation=animation,
    )


def _handle_coverage(args: argparse.Namespace) -> int:
    scan_config = _scan_config(args)
    animation = _animation_config(args)
    frames = scan_frames(args.input, scan_config, animation=animation, concurrency=_concurrency_config(args))
    selection_config = _selection_config(args)
    report = analyze_representation(
        select_frames(frames, selection_config),
        covered_distance=args.covered_distance,
    )
    payload: dict[str, object] = {"representation": report.as_dict()}
    print(
        f"kept {report.selected} of {report.selected + report.dropped}; "
        f"worst gap {report.worst_distance:.3f}, mean {report.mean_distance:.3f}, "
        f"{report.covered_fraction:.0%} of dropped frames within {args.covered_distance:.3f}"
    )
    for gap in report.gaps[:MAX_REPORTED_GAPS]:
        print(
            f"  frames {gap.start_index}-{gap.end_index} are represented no closer "
            f"than {gap.worst_distance:.3f}"
        )
    if args.budgets:
        curve = budget_curve(frames, selection_config, args.budgets, covered_distance=args.covered_distance)
        payload["budget_curve"] = curve.as_dict()
        print()
        print(f"{'budget':>7} {'kept':>5} {'worst':>7} {'mean':>7}")
        for point in curve.points:
            print(
                f"{point.budget:>7} {point.selected:>5} "
                f"{point.worst_distance:>7.3f} {point.mean_distance:>7.3f}"
            )
        knee = curve.knee()
        if knee is None:
            print(
                "  the worst gap was still falling at the largest budget tried, "
                "so the budget is still binding"
            )
        else:
            print(f"  the worst gap stops improving at a budget of {knee}")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        print(f"  wrote {args.output}")
    return 0


def _handle_benchmark(args: argparse.Namespace) -> int:
    _require_output_outside_input(args.input, args.output_dir)
    scan_config = _scan_config(args)
    selection_config = _selection_config(args)
    benchmark_config = BenchmarkConfig(
        random_seed=args.random_seed,
        random_trials=args.random_trials,
    )
    frames = scan_frames(args.input, scan_config, concurrency=_concurrency_config(args))
    result = run_benchmark(
        frames,
        selection_config,
        benchmark_config,
        scan_config=scan_config,
        source_digest=source_content_digest(frames),
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report = output_dir / "benchmark.json"
    chart = output_dir / "benchmark.svg"
    with TemporaryDirectory(prefix=".frame-quorum-benchmark-stage-", dir=output_dir) as staging:
        staging_dir = Path(staging)
        staged_report = staging_dir / report.name
        staged_chart = staging_dir / chart.name
        write_json(benchmark_manifest(result), staged_report)
        render_benchmark_svg(result, staged_chart)
        _commit_bundle(((staged_chart, chart), (staged_report, report)))
    _write_console(f"Benchmarked {len(frames)} frames across {len(result.runs)} runs")
    _write_console(f"Report: {report}")
    _write_console(f"Chart: {chart}")
    return 0


def _handle_demo(args: argparse.Namespace) -> int:
    output = args.output_dir.resolve()
    frames_dir = create_demo_sequence(output / "frames", overwrite=args.force)
    scan_config = ScanConfig(
        timestamp_mode="filename",
        timestamp_regex=r"_(?P<ts>\d+)ms$",
        timestamp_unit="milliseconds",
    )
    frames = scan_frames(frames_dir, scan_config)
    status = _write_selection(
        frames,
        scan_config,
        _selection_config(args),
        output,
        columns=args.columns,
        thumbnail_width=args.thumbnail_width,
    )
    _write_console(f"Demo frames: {frames_dir}")
    return status


def _handle_extract(args: argparse.Namespace) -> int:
    result = extract_video_frames(
        args.input,
        args.output_dir,
        VideoExtractionConfig(
            frame_rate=args.frame_rate,
            max_frames=args.max_frames,
            max_output_bytes=args.max_output_bytes,
            timeout_seconds=args.timeout,
            executable=args.ffmpeg,
        ),
    )
    _write_console(f"Extracted {result.frame_count} frames -> {result.output_dir}")
    return 0


def _write_selection(
    frames: tuple[Frame, ...],
    scan_config: ScanConfig,
    selection_config: SelectionConfig,
    output_dir: Path,
    *,
    columns: int,
    thumbnail_width: int,
    animation: AnimationConfig | None = None,
) -> int:
    result = select_frames(frames, selection_config)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = output_dir / "manifest.json"
    sheet = output_dir / "contact-sheet.png"
    with TemporaryDirectory(prefix=".frame-quorum-stage-", dir=output_dir) as staging:
        staging_dir = Path(staging)
        staged_manifest = staging_dir / manifest.name
        staged_sheet = staging_dir / sheet.name
        write_json(selection_manifest(result, scan_config, animation=animation), staged_manifest)
        render_contact_sheet(
            result,
            staged_sheet,
            columns=columns,
            thumbnail_width=thumbnail_width,
        )
        _commit_bundle(((staged_sheet, sheet), (staged_manifest, manifest)))
    _write_console(f"Selected {len(result.selected_indices)} of {len(frames)} frames")
    _write_console(f"Manifest: {manifest}")
    _write_console(f"Contact sheet: {sheet}")
    return 0


def _commit_bundle(files: tuple[tuple[Path, Path], ...]) -> None:
    backups: dict[Path, Path] = {}
    for _, target in files:
        if target.is_symlink() or (target.exists() and not target.is_file()):
            raise OutputError(f"report target must be a regular file or absent: {target}")
        if target.exists():
            backup = files[0][0].parent / f"{target.name}.previous"
            try:
                shutil.copy2(target, backup)
            except OSError as error:
                raise OutputError(f"cannot stage prior report artifact {target}: {error}") from error
            backups[target] = backup

    replaced: list[Path] = []
    try:
        for staged, target in files:
            staged.replace(target)
            replaced.append(target)
    except OSError as error:
        rollback_errors: list[str] = []
        for target in reversed(replaced):
            try:
                restore_from = backups.get(target)
                if restore_from is None:
                    target.unlink(missing_ok=True)
                else:
                    restore_from.replace(target)
            except OSError as rollback_error:
                rollback_errors.append(f"{target}: {rollback_error}")
        if rollback_errors:
            detail = "; ".join(rollback_errors)
            raise OutputError(f"report commit failed and rollback was incomplete: {detail}") from error
        raise OutputError(f"report commit failed; prior artifacts were restored: {error}") from error


def _require_output_outside_input(input_path: Path, output_dir: Path) -> None:
    supplied_input = input_path.expanduser()
    supplied_output = output_dir.expanduser()
    if supplied_input.is_symlink():
        raise ConfigurationError(f"input path must not be a symbolic link: {supplied_input}")
    if supplied_output.is_symlink():
        raise ConfigurationError(f"output directory must not be a symbolic link: {supplied_output}")
    resolved_input = supplied_input.resolve()
    resolved_output = supplied_output.resolve()
    if resolved_input.is_dir() and (
        resolved_output == resolved_input or resolved_input in resolved_output.parents
    ):
        raise ConfigurationError("output directory must be outside the input directory")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (FrameQuorumError, OSError) as error:
        _write_console(f"error: {error}", stream=sys.stderr)
        return 2


def _write_console(message: str, *, stream: TextIO | None = None) -> None:
    destination = stream or sys.stdout
    message = "".join(
        character if character.isprintable() else ascii(character)[1:-1] for character in message
    )
    encoding = getattr(destination, "encoding", None)
    if encoding:
        try:
            message.encode(encoding)
        except (LookupError, UnicodeEncodeError):
            message = message.encode("ascii", "backslashreplace").decode("ascii")
    try:
        destination.write(f"{message}\n")
    except UnicodeEncodeError:
        escaped = message.encode("ascii", "backslashreplace").decode("ascii")
        destination.write(f"{escaped}\n")


if __name__ == "__main__":
    raise SystemExit(main())
