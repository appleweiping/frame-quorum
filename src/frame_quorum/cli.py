"""Command-line interface for scanning, selecting, and generating a demo."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from fractions import Fraction
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
from .editing import render_edl, render_scene_timecodes
from .errors import ConfigurationError, FrameQuorumError, OutputError
from .exports import render_detection_csv
from .models import AnimationConfig, ConcurrencyConfig, Frame, ScanConfig, SelectionConfig
from .native_histograms import (
    analyze_native_histograms,
    capture_native_histograms,
    read_native_histograms,
    write_native_histogram_replay,
    write_native_histograms,
)
from .native_measurements import (
    NativeMeasurementLimits,
    analyze_native_measurements,
    capture_native_measurements,
    read_native_measurements,
    write_native_measurements,
    write_native_replay,
)
from .native_pixel_changes import (
    PixelChangeDetectionConfig,
    analyze_native_pixel_changes,
    capture_native_pixel_changes,
    read_native_pixel_changes,
    write_native_pixel_change_replay,
    write_native_pixel_changes,
)
from .native_scenes import NativeSceneConfig, detect_native_scenes
from .native_splitting import NativeClip, NativeSplitConfig, split_native_video
from .native_video import NativeVideoConfig, NativeVideoStream
from .pixel_changes import PixelChangeConfig, PixelChangeLimits, PixelChangeWeights
from .pixel_histograms import HistogramDetectionConfig, PixelHistogramConfig, PixelHistogramLimits
from .reporting import scan_manifest, selection_manifest, write_json
from .representation import (
    DEFAULT_COVERED_DISTANCE,
    MAX_REPORTED_GAPS,
    analyze_representation,
    budget_curve,
)
from .scanner import scan_frames
from .scene_detection import DetectionConfig, detect_scenes
from .selector import select_frames
from .timecode import FrameRate
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

    native_scan = commands.add_parser("native-scan", help="stream exact-PTS video measurements as JSONL")
    _add_native_options(native_scan)
    native_scan.set_defaults(handler=_handle_native_scan)

    native_scenes = commands.add_parser("native-scenes", help="analyze exact-PTS video scenes as JSON")
    _add_native_options(native_scenes)
    _add_detector_options(native_scenes)
    native_scenes.set_defaults(handler=_handle_native_scenes)

    measure = commands.add_parser("native-measure", help="capture bounded replayable native measurements")
    _add_native_options(measure)
    _add_measurement_options(measure)
    measure.set_defaults(handler=_handle_native_measure)

    replay = commands.add_parser("native-replay", help="replay cached measurements without decoding media")
    replay.add_argument("input", type=Path)
    _add_detector_options(replay)
    _add_measurement_options(replay)
    replay.set_defaults(handler=_handle_native_replay)

    histogram_measure = commands.add_parser(
        "native-histogram-measure", help="capture full-RGB cell histograms and exact native coordinates"
    )
    _add_native_options(histogram_measure)
    _add_measurement_options(histogram_measure)
    _add_histogram_limits(histogram_measure)
    histogram_measure.add_argument("--bins", type=int, default=32, choices=(8, 16, 32, 64, 128, 256))
    histogram_measure.add_argument("--rows", type=int, default=2, choices=(1, 2))
    histogram_measure.add_argument("--columns", type=int, default=2, choices=(1, 2))
    histogram_measure.set_defaults(handler=_handle_histogram_measure)

    histogram_replay = commands.add_parser(
        "native-histogram-replay", help="replay stored RGB histogram thresholds without decoding"
    )
    histogram_replay.add_argument("input", type=Path)
    _add_measurement_options(histogram_replay)
    _add_histogram_limits(histogram_replay)
    histogram_replay.add_argument("--mode", choices=("global", "spatial"), default="spatial")
    histogram_replay.add_argument("--threshold", type=float, default=0.5)
    histogram_replay.add_argument("--min-scene-samples", type=int, default=1)
    histogram_replay.set_defaults(handler=_handle_histogram_replay)

    change_measure = commands.add_parser(
        "native-change-measure", help="capture exact full-pixel HSV and gradient changes"
    )
    _add_native_options(change_measure)
    _add_measurement_options(change_measure)
    _add_change_limits(change_measure)
    change_measure.add_argument("--edge-radius", type=int, choices=(1, 2, 3, 4), default=1)
    change_measure.set_defaults(handler=_handle_change_measure)

    change_replay = commands.add_parser(
        "native-change-replay", help="reweight cached pixel changes without decoding"
    )
    change_replay.add_argument("input", type=Path)
    _add_measurement_options(change_replay)
    _add_change_limits(change_replay)
    change_replay.add_argument("--detector", choices=("content", "adaptive"), default="content")
    change_replay.add_argument(
        "--weights", type=float, nargs=4, default=(1, 1, 1, 0), metavar=("H", "S", "V", "E")
    )
    change_replay.add_argument("--value-only", action="store_true")
    change_replay.add_argument("--threshold", type=float, default=0.30)
    change_replay.add_argument("--min-scene-samples", type=int, default=1)
    change_replay.add_argument("--window-radius", type=int, default=2)
    change_replay.add_argument("--adaptive-ratio", type=float, default=3.0)
    change_replay.add_argument("--min-content", type=float, default=0.15)
    change_replay.set_defaults(handler=_handle_change_replay)

    native_split = commands.add_parser("native-split", help="publish verified lossless video-only clips")
    native_split.add_argument("input", type=Path)
    native_split.add_argument("--output-dir", "-o", type=Path, required=True)
    native_split.add_argument(
        "--clip", nargs=2, type=_exact_seconds, action="append", required=True, metavar=("START", "END")
    )
    for name in NativeSplitConfig.__dataclass_fields__:
        native_split.add_argument(
            "--" + name.replace("_", "-"), type=int, default=getattr(NativeSplitConfig(), name)
        )
    native_split.set_defaults(handler=_handle_native_split)

    scenes = commands.add_parser("scenes", help="detect scene boundaries and export per-frame statistics")
    scenes.add_argument("input", type=Path)
    _add_scan_options(scenes)
    _add_animation_options(scenes)
    scenes.add_argument("--output-dir", "-o", type=Path, required=True)
    scenes.add_argument(
        "--detector", choices=("content", "luminance", "color", "adaptive", "threshold"), default="adaptive"
    )
    scenes.add_argument("--threshold", type=float, default=0.30, help="adjacent-distance threshold")
    scenes.add_argument("--min-scene-frames", type=int, default=1)
    scenes.add_argument("--window-radius", type=int, default=2)
    scenes.add_argument("--adaptive-ratio", type=float, default=3.0)
    scenes.add_argument("--min-content", type=float, default=0.15)
    scenes.add_argument("--dark-threshold", type=float, default=0.05)
    scenes.add_argument("--hysteresis", type=float, default=0.02)
    scenes.add_argument("--min-dark-frames", type=int, default=2)
    scenes.add_argument("--fade-bias", type=float, default=0.0)
    scenes.add_argument("--include-final-fade", action="store_true")
    scenes.add_argument(
        "--max-frames", type=int, default=1_000_000, help="maximum measured frames accepted by detection"
    )
    scenes.set_defaults(handler=_handle_scenes)
    scenes.add_argument(
        "--timecode-rate",
        help="exact CFR rate for timing exports, e.g. 30000/1001; overrides index timestamp rate",
    )
    scenes.add_argument("--drop-frame", action="store_true", help="use exact NTSC drop-frame labels")
    scenes.add_argument("--edl", action="store_true", help="also export a cuts-only single-reel video EDL")
    return parser


def _add_detector_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--detectors",
        nargs="+",
        default=["adaptive"],
        choices=("content", "luminance", "color", "adaptive", "threshold"),
    )
    parser.add_argument("--threshold", type=float, default=0.30)
    parser.add_argument("--min-scene-frames", type=int, default=1, help="per-detector sample minimum")
    parser.add_argument("--minimum-votes", type=int, default=1)
    parser.add_argument("--min-scene-samples", type=int, default=1, help="final ensemble sample minimum")
    parser.add_argument("--window-radius", type=int, default=2)
    parser.add_argument("--adaptive-ratio", type=float, default=3.0)
    parser.add_argument("--min-content", type=float, default=0.15)
    parser.add_argument("--dark-threshold", type=float, default=0.05)
    parser.add_argument("--hysteresis", type=float, default=0.02)
    parser.add_argument("--min-dark-frames", type=int, default=2)
    parser.add_argument("--fade-bias", type=float, default=0.0)
    parser.add_argument("--include-final-fade", action="store_true")


def _add_measurement_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-dir", "-o", type=Path, required=True)
    for name in NativeMeasurementLimits.__dataclass_fields__:
        parser.add_argument(
            "--" + name.replace("_", "-"), type=int, default=getattr(NativeMeasurementLimits(), name)
        )


def _add_histogram_limits(parser: argparse.ArgumentParser) -> None:
    for name in PixelHistogramLimits.__dataclass_fields__:
        parser.add_argument(
            "--" + name.replace("_", "-"), type=int, default=getattr(PixelHistogramLimits(), name)
        )


def _add_change_limits(parser: argparse.ArgumentParser) -> None:
    for name in PixelChangeLimits.__dataclass_fields__:
        parser.add_argument(
            "--" + name.replace("_", "-"), type=int, default=getattr(PixelChangeLimits(), name)
        )


def _add_native_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("input", type=Path)
    parser.add_argument("--start", type=_exact_seconds, help="inclusive exact presentation seconds")
    parser.add_argument("--end", type=_exact_seconds, help="exclusive exact presentation seconds")
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--video-stream", type=int, default=0, help="zero-based ordinal among video streams")
    parser.add_argument("--max-frames", type=int, default=10_000)
    parser.add_argument("--max-decoded-frames", type=int, default=100_000)
    parser.add_argument("--max-source-bytes", type=int, default=1_000_000_000)
    parser.add_argument("--max-frame-pixels", type=int, default=16_777_216)
    parser.add_argument("--max-total-pixels", type=int, default=1_000_000_000)


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


def _exact_seconds(text: str) -> Fraction:
    try:
        if len(text) > 128:
            raise ValueError("too long")
        # Fraction decimal parsing otherwise constructs 10**exponent before
        # NativeVideoConfig can reject the resulting out-of-range rational.
        # int accepts the same underscore/whitespace exponent spellings as
        # Fraction; checking only a digit regex would leave those unbounded.
        exponent = text.lower().rsplit("e", 1)
        if len(exponent) == 2 and abs(int(exponent[1])) > 128:
            raise ValueError("decimal exponent is too large")
        return Fraction(text)
    except (ValueError, ZeroDivisionError) as exc:
        raise argparse.ArgumentTypeError("time must be a bounded exact integer, decimal or fraction") from exc


def _native_config(args: argparse.Namespace) -> NativeVideoConfig:
    return NativeVideoConfig(
        video_stream=args.video_stream,
        start=args.start,
        end=args.end,
        frame_step=args.frame_step,
        max_frames=args.max_frames,
        max_decoded_frames=args.max_decoded_frames,
        max_source_bytes=args.max_source_bytes,
        max_frame_pixels=args.max_frame_pixels,
        max_total_pixels=args.max_total_pixels,
    )


def _handle_native_scenes(args: argparse.Namespace) -> int:
    result = detect_native_scenes(
        args.input,
        NativeSceneConfig(
            video=_native_config(args),
            detectors=_detector_configs(args),
            minimum_votes=args.minimum_votes,
            min_scene_samples=args.min_scene_samples,
        ),
    )
    sys.stdout.write(json.dumps(result.to_dict(), ensure_ascii=True, allow_nan=False, sort_keys=True) + "\n")
    return 0


def _detector_configs(args: argparse.Namespace) -> tuple[DetectionConfig, ...]:
    return tuple(
        DetectionConfig(
            detector=name,
            threshold=args.threshold,
            min_scene_frames=args.min_scene_frames,
            window_radius=args.window_radius,
            adaptive_ratio=args.adaptive_ratio,
            min_content=args.min_content,
            dark_threshold=args.dark_threshold,
            hysteresis=args.hysteresis,
            min_dark_frames=args.min_dark_frames,
            fade_bias=args.fade_bias,
            include_final_fade=args.include_final_fade,
        )
        for name in args.detectors
    )


def _measurement_limits(args: argparse.Namespace) -> NativeMeasurementLimits:
    return NativeMeasurementLimits(
        **{name: getattr(args, name) for name in NativeMeasurementLimits.__dataclass_fields__}
    )


def _handle_native_measure(args: argparse.Namespace) -> int:
    limits = _measurement_limits(args)
    measurements = capture_native_measurements(args.input, _native_config(args), limits=limits)
    path = write_native_measurements(measurements, args.output_dir, limits=limits)
    sys.stdout.write(
        json.dumps({"cache": str(path), "measurement_digest": measurements.digest}, ensure_ascii=True) + "\n"
    )
    return 0


def _pixel_limits(args: argparse.Namespace) -> PixelHistogramLimits:
    return PixelHistogramLimits(
        **{name: getattr(args, name) for name in PixelHistogramLimits.__dataclass_fields__}
    )


def _handle_histogram_measure(args: argparse.Namespace) -> int:
    limits, pixel_limits = _measurement_limits(args), _pixel_limits(args)
    data = capture_native_histograms(
        args.input,
        _native_config(args),
        histogram=PixelHistogramConfig(args.bins, args.rows, args.columns),
        limits=limits,
        pixel_limits=pixel_limits,
    )
    cache = write_native_histograms(data, args.output_dir, limits=limits, pixel_limits=pixel_limits)
    sys.stdout.write(
        json.dumps({"cache": str(cache), "measurement_digest": data.digest}, ensure_ascii=True) + "\n"
    )
    return 0


def _handle_histogram_replay(args: argparse.Namespace) -> int:
    limits, pixel_limits = _measurement_limits(args), _pixel_limits(args)
    config = HistogramDetectionConfig(args.mode, args.threshold, args.min_scene_samples)
    data = read_native_histograms(args.input, limits=limits, pixel_limits=pixel_limits)
    result = analyze_native_histograms(data, config, limits=limits, pixel_limits=pixel_limits)
    path = write_native_histogram_replay(result, args.output_dir, limits=limits, pixel_limits=pixel_limits)
    sys.stdout.write(
        json.dumps(
            {
                "output_dir": str(path),
                "measurement_digest": data.digest,
                "source_verified": False,
                "cut_positions": result.cut_positions,
            }
        )
        + "\n"
    )
    return 0


def _handle_native_replay(args: argparse.Namespace) -> int:
    limits = _measurement_limits(args)
    measurements = read_native_measurements(args.input, limits=limits)
    result = analyze_native_measurements(
        measurements,
        detectors=_detector_configs(args),
        minimum_votes=args.minimum_votes,
        min_scene_samples=args.min_scene_samples,
    )
    path = write_native_replay(result, args.output_dir, limits=limits)
    sys.stdout.write(
        json.dumps(
            {
                "output_dir": str(path),
                "measurement_digest": result.measurement_digest,
                "source_verified": False,
            },
            ensure_ascii=True,
        )
        + "\n"
    )
    return 0


def _change_pixel_limits(args: argparse.Namespace) -> PixelChangeLimits:
    return PixelChangeLimits(**{name: getattr(args, name) for name in PixelChangeLimits.__dataclass_fields__})


def _handle_change_measure(args: argparse.Namespace) -> int:
    limits, pixel_limits = _measurement_limits(args), _change_pixel_limits(args)
    data = capture_native_pixel_changes(
        args.input,
        _native_config(args),
        config=PixelChangeConfig(args.edge_radius),
        limits=limits,
        pixel_limits=pixel_limits,
    )
    cache = write_native_pixel_changes(data, args.output_dir, limits=limits, pixel_limits=pixel_limits)
    sys.stdout.write(json.dumps({"cache": str(cache), "measurement_digest": data.digest}) + "\n")
    return 0


def _handle_change_replay(args: argparse.Namespace) -> int:
    limits, pixel_limits = _measurement_limits(args), _change_pixel_limits(args)
    config = PixelChangeDetectionConfig(
        detector=args.detector,
        weights=PixelChangeWeights(*args.weights),
        value_only=args.value_only,
        threshold=args.threshold,
        min_scene_samples=args.min_scene_samples,
        window_radius=args.window_radius,
        adaptive_ratio=args.adaptive_ratio,
        min_content=args.min_content,
    )
    data = read_native_pixel_changes(args.input, limits=limits, pixel_limits=pixel_limits)
    result = analyze_native_pixel_changes(data, config, limits=limits, pixel_limits=pixel_limits)
    path = write_native_pixel_change_replay(result, args.output_dir, limits=limits, pixel_limits=pixel_limits)
    sys.stdout.write(
        json.dumps(
            {
                "output_dir": str(path),
                "measurement_digest": data.digest,
                "source_verified": False,
                "cut_positions": result.cut_positions,
            }
        )
        + "\n"
    )
    return 0


def _handle_native_scan(args: argparse.Namespace) -> int:
    config = _native_config(args)

    def emit(record_type: str, data: dict[str, object]) -> None:
        sys.stdout.write(
            json.dumps(
                {
                    "kind": "frame-quorum-native-scan",
                    "schema_version": 1,
                    "record_type": record_type,
                    **data,
                },
                ensure_ascii=True,
                allow_nan=False,
                sort_keys=True,
            )
            + "\n"
        )

    with NativeVideoStream(args.input, config) as stream:
        emit("source", {"metadata": stream.metadata.to_dict()})
        for frame in stream:
            emit("frame", {**frame.to_dict(), "metrics": frame.measure().serializable()})
    emit("summary", {"diagnostics": stream.diagnostics.to_dict()})
    return 0


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


def _handle_native_split(args: argparse.Namespace) -> int:
    config = NativeSplitConfig(
        **{name: getattr(args, name) for name in NativeSplitConfig.__dataclass_fields__}
    )
    result = split_native_video(
        args.input, args.output_dir, tuple(NativeClip(start, end) for start, end in args.clip), config
    )
    sys.stdout.write(json.dumps(result.to_dict(), sort_keys=True) + "\n")
    return 0


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


def _handle_scenes(args: argparse.Namespace) -> int:
    _require_output_outside_input(args.input, args.output_dir)
    rate = None if args.timecode_rate is None else FrameRate.parse(args.timecode_rate)
    if (args.drop_frame or args.edl) and rate is None:
        raise ConfigurationError("--drop-frame and --edl require --timecode-rate")
    if rate is not None:
        if args.timestamp_mode != "index":
            raise ConfigurationError("CFR timecode exports require index timestamps")
        args.frame_rate = float(rate.fraction)
    config = DetectionConfig(
        detector=args.detector,
        threshold=args.threshold,
        min_scene_frames=args.min_scene_frames,
        window_radius=args.window_radius,
        adaptive_ratio=args.adaptive_ratio,
        min_content=args.min_content,
        dark_threshold=args.dark_threshold,
        hysteresis=args.hysteresis,
        min_dark_frames=args.min_dark_frames,
        fade_bias=args.fade_bias,
        include_final_fade=args.include_final_fade,
        max_frames=args.max_frames,
    )
    config.validate()
    frames = scan_frames(
        args.input,
        _scan_config(args),
        animation=_animation_config(args),
        concurrency=_concurrency_config(args),
    )
    result = detect_scenes(frames, config)
    timing = None if rate is None else render_scene_timecodes(result.scenes, rate, drop_frame=args.drop_frame)
    edl = (
        render_edl(result.scenes, rate, drop_frame=args.drop_frame) if args.edl and rate is not None else None
    )
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report, statistics = output / "scenes.json", output / "statistics.csv"
    with TemporaryDirectory(prefix=".frame-quorum-scenes-stage-", dir=output) as staging:
        staged_report = Path(staging) / report.name
        staged_stats = Path(staging) / statistics.name
        write_json(result.serializable(), staged_report)
        staged_stats.write_text(render_detection_csv(result), encoding="utf-8", newline="")
        bundle = [(staged_stats, statistics)]
        for name, content in (("timecodes.csv", timing), ("scenes.edl", edl)):
            if content is not None:
                staged = Path(staging) / name
                staged.write_text(content, encoding="utf-8", newline="")
                bundle.append((staged, output / name))
        bundle.append((staged_report, report))
        _commit_bundle(tuple(bundle))
    _write_console(f"Detected {len(result.scenes)} scenes from {len(frames)} frames -> {report}")
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
