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
from .editing import render_edl, render_scene_timecodes
from .exports import render_decision_csv, render_detection_csv, write_decision_csv
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
from .native_measurements import (
    NativeMeasurementLimits,
    NativeMeasurements,
    NativeReplayResult,
    analyze_native_measurements,
    capture_native_measurements,
    read_native_measurements,
    render_native_detection_csv,
    write_native_measurements,
    write_native_replay,
)
from .native_scenes import (
    NativeDetectorStatistic,
    NativeScene,
    NativeSceneConfig,
    NativeSceneResult,
    NativeSceneSample,
    NativeSceneStatistic,
    detect_native_scenes,
)
from .native_splitting import (
    NativeClip,
    NativeSplitConfig,
    NativeSplitResult,
    native_scene_clips,
    split_native_video,
)
from .native_video import (
    NativeVideoConfig,
    NativeVideoDiagnostics,
    NativeVideoFrame,
    NativeVideoMetadata,
    NativeVideoStatus,
    NativeVideoStream,
)
from .scanner import scan_frames
from .scene_detection import (
    DetectionConfig,
    DetectionResult,
    FrameStatistic,
    SceneDetectorName,
    detect_scenes,
)
from .scenes import SceneAnalysis, Shot, allocate_budget, analyze_scenes, detect_shots
from .selector import select_frames
from .timecode import FrameRate, FrameTimecode
from .video import VideoExtractionConfig, VideoExtractionResult, extract_video_frames, ffmpeg_available

__all__ = [
    "AnimationConfig",
    "BenchmarkConfig",
    "BenchmarkProvenance",
    "BenchmarkResult",
    "BenchmarkRun",
    "ConcurrencyConfig",
    "DetectionConfig",
    "DetectionResult",
    "DetectorName",
    "EvaluationMetrics",
    "Frame",
    "FrameDecision",
    "FrameMetrics",
    "FrameRate",
    "FrameStatistic",
    "FrameTimecode",
    "NativeClip",
    "NativeDetectorStatistic",
    "NativeMeasurementLimits",
    "NativeMeasurements",
    "NativeReplayResult",
    "NativeScene",
    "NativeSceneConfig",
    "NativeSceneResult",
    "NativeSceneSample",
    "NativeSceneStatistic",
    "NativeSplitConfig",
    "NativeSplitResult",
    "NativeVideoConfig",
    "NativeVideoDiagnostics",
    "NativeVideoFrame",
    "NativeVideoMetadata",
    "NativeVideoStatus",
    "NativeVideoStream",
    "ScanConfig",
    "SceneAnalysis",
    "SceneDetectorName",
    "SelectionConfig",
    "SelectionResult",
    "Shot",
    "Transition",
    "VideoExtractionConfig",
    "VideoExtractionResult",
    "allocate_budget",
    "analyze_native_measurements",
    "analyze_scenes",
    "benchmark_manifest",
    "capture_native_measurements",
    "detect_native_scenes",
    "detect_scenes",
    "detect_shots",
    "detect_transitions",
    "evaluate_selection",
    "extract_video_frames",
    "ffmpeg_available",
    "native_scene_clips",
    "read_native_measurements",
    "render_benchmark_svg",
    "render_decision_csv",
    "render_detection_csv",
    "render_edl",
    "render_native_detection_csv",
    "render_scene_timecodes",
    "run_benchmark",
    "scan_frames",
    "select_frames",
    "source_content_digest",
    "split_native_video",
    "write_decision_csv",
    "write_native_measurements",
    "write_native_replay",
]

__version__ = VERSION
