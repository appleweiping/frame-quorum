"""Explainable key-frame selection for image sequences."""

from ._version import VERSION
from .benchmark import (
    BenchmarkConfig,
    BenchmarkProvenance,
    BenchmarkResult,
    BenchmarkRun,
    EvaluationMetrics,
    benchmark_manifest,
    evaluate_selection,
    render_benchmark_svg,
    run_benchmark,
    source_content_digest,
)
from .detectors import DetectorName, Transition, detect_transitions
from .exports import render_decision_csv, write_decision_csv
from .models import (
    AnimationConfig,
    ConcurrencyConfig,
    Frame,
    FrameDecision,
    FrameMetrics,
    ScanConfig,
    SelectionConfig,
    SelectionResult,
)
from .scanner import scan_frames
from .scenes import SceneAnalysis, Shot, allocate_budget, analyze_scenes, detect_shots
from .selector import select_frames
from .video import VideoExtractionConfig, VideoExtractionResult, extract_video_frames, ffmpeg_available

__all__ = [
    "AnimationConfig",
    "BenchmarkConfig",
    "BenchmarkProvenance",
    "BenchmarkResult",
    "BenchmarkRun",
    "ConcurrencyConfig",
    "DetectorName",
    "EvaluationMetrics",
    "Frame",
    "FrameDecision",
    "FrameMetrics",
    "ScanConfig",
    "SceneAnalysis",
    "SelectionConfig",
    "SelectionResult",
    "Shot",
    "Transition",
    "VideoExtractionConfig",
    "VideoExtractionResult",
    "allocate_budget",
    "analyze_scenes",
    "benchmark_manifest",
    "detect_shots",
    "detect_transitions",
    "evaluate_selection",
    "extract_video_frames",
    "ffmpeg_available",
    "render_benchmark_svg",
    "render_decision_csv",
    "run_benchmark",
    "scan_frames",
    "select_frames",
    "source_content_digest",
    "write_decision_csv",
]

__version__ = VERSION
