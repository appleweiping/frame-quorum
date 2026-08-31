"""Explainable key-frame selection for image sequences."""

from .models import Frame, FrameDecision, ScanConfig, SelectionConfig, SelectionResult
from .scanner import scan_frames
from .selector import select_frames

__all__ = [
    "Frame",
    "FrameDecision",
    "ScanConfig",
    "SelectionConfig",
    "SelectionResult",
    "scan_frames",
    "select_frames",
]

__version__ = "0.1.0"
