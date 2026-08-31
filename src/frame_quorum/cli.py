"""Command-line interface for scanning, selecting, and generating a demo."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TextIO

from .contact_sheet import render_contact_sheet
from .demo import create_demo_sequence
from .errors import ConfigurationError, FrameQuorumError, OutputError
from .models import Frame, ScanConfig, SelectionConfig
from .reporting import scan_manifest, selection_manifest, write_json
from .scanner import scan_frames
from .selector import select_frames


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="frame-quorum",
        description="Explainable, content-aware key-frame selection for image sequences.",
    )
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")
    commands = parser.add_subparsers(dest="command", required=True)

    scan = commands.add_parser("scan", help="measure a frame directory and emit JSON")
    scan.add_argument("input", type=Path)
    _add_scan_options(scan)
    scan.add_argument("--output", "-o", type=Path, help="write JSON here instead of stdout")
    scan.set_defaults(handler=_handle_scan)

    select = commands.add_parser("select", help="select key frames and create a report")
    select.add_argument("input", type=Path)
    _add_scan_options(select)
    _add_selection_options(select)
    select.add_argument("--output-dir", "-o", type=Path, required=True)
    select.add_argument("--columns", type=int, default=3)
    select.add_argument("--thumbnail-width", type=int, default=320)
    select.set_defaults(handler=_handle_select)

    demo = commands.add_parser("demo", help="generate and process a synthetic sequence")
    demo.add_argument("--output-dir", "-o", type=Path, default=Path("frame-quorum-demo"))
    demo.add_argument("--force", action="store_true", help="replace the previously generated demo frames")
    _add_selection_options(demo, budget=6)
    demo.add_argument("--columns", type=int, default=3)
    demo.add_argument("--thumbnail-width", type=int, default=320)
    demo.set_defaults(handler=_handle_demo)
    return parser


def _add_scan_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--recursive", action="store_true", help="include nested image directories")
    parser.add_argument(
        "--timestamp-mode",
        choices=("index", "filename", "mtime", "none"),
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
    frames = scan_frames(args.input, config)
    rendered = write_json(
        scan_manifest(frames, config),
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
    frames = scan_frames(args.input, scan_config)
    return _write_selection(
        frames,
        scan_config,
        _selection_config(args),
        args.output_dir,
        columns=args.columns,
        thumbnail_width=args.thumbnail_width,
    )


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


def _write_selection(
    frames: tuple[Frame, ...],
    scan_config: ScanConfig,
    selection_config: SelectionConfig,
    output_dir: Path,
    *,
    columns: int,
    thumbnail_width: int,
) -> int:
    result = select_frames(frames, selection_config)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = output_dir / "manifest.json"
    sheet = output_dir / "contact-sheet.png"
    with TemporaryDirectory(prefix=".frame-quorum-stage-", dir=output_dir) as staging:
        staging_dir = Path(staging)
        staged_manifest = staging_dir / manifest.name
        staged_sheet = staging_dir / sheet.name
        write_json(selection_manifest(result, scan_config), staged_manifest)
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
                backup = backups.get(target)
                if backup is None:
                    target.unlink(missing_ok=True)
                else:
                    backup.replace(target)
            except OSError as rollback_error:
                rollback_errors.append(f"{target}: {rollback_error}")
        if rollback_errors:
            detail = "; ".join(rollback_errors)
            raise OutputError(f"report commit failed and rollback was incomplete: {detail}") from error
        raise OutputError(f"report commit failed; prior artifacts were restored: {error}") from error


def _require_output_outside_input(input_path: Path, output_dir: Path) -> None:
    resolved_input = input_path.expanduser().resolve()
    resolved_output = output_dir.expanduser().resolve()
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
