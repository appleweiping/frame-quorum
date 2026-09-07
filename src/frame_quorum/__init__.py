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
    "EvaluationMetrics",
    "Frame",
    "FrameDecision",
    "FrameMetrics",
    "ScanConfig",
    "SceneAnalysis",
    "SelectionConfig",
    "SelectionResult",
    "Shot",
    "VideoExtractionConfig",
    "VideoExtractionResult",
    "allocate_budget",
    "analyze_scenes",
    "benchmark_manifest",
    "detect_shots",
    "evaluate_selection",
    "extract_video_frames",
    "ffmpeg_available",
    "render_benchmark_svg",
    "run_benchmark",
    "scan_frames",
    "select_frames",
    "source_content_digest",
]

__version__ = VERSION
