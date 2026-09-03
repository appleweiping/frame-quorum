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
from .models import Frame, FrameDecision, ScanConfig, SelectionConfig, SelectionResult
from .scanner import scan_frames
from .selector import select_frames
from .video import VideoExtractionConfig, VideoExtractionResult, extract_video_frames, ffmpeg_available

__all__ = [
    "BenchmarkConfig",
    "BenchmarkProvenance",
    "BenchmarkResult",
    "BenchmarkRun",
    "EvaluationMetrics",
    "Frame",
    "FrameDecision",
    "ScanConfig",
    "SelectionConfig",
    "SelectionResult",
    "VideoExtractionConfig",
    "VideoExtractionResult",
    "benchmark_manifest",
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
