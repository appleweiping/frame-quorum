"""Typed records shared by scanning, selection, and reporting."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .errors import ConfigurationError


@dataclass(frozen=True, slots=True)
class ScanConfig:
    """Controls image discovery and optional timestamp extraction."""

    recursive: bool = False
    timestamp_mode: str = "index"
    frame_rate: float = 1.0
    timestamp_regex: str | None = None
    timestamp_unit: str = "seconds"
    extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")

    def validate(self) -> None:
        if not isinstance(self.recursive, bool):
            raise ConfigurationError("recursive must be a boolean")
        if not isinstance(self.timestamp_mode, str) or self.timestamp_mode not in {
            "index",
            "filename",
            "mtime",
            "none",
        }:
            raise ConfigurationError("timestamp_mode must be index, filename, mtime, or none")
        if (
            isinstance(self.frame_rate, bool)
            or not isinstance(self.frame_rate, (int, float))
            or not _is_finite_number(self.frame_rate)
            or self.frame_rate <= 0
        ):
            raise ConfigurationError("frame_rate must be greater than zero")
        if not isinstance(self.timestamp_unit, str) or self.timestamp_unit not in {
            "seconds",
            "milliseconds",
            "microseconds",
        }:
            raise ConfigurationError("timestamp_unit must be seconds, milliseconds, or microseconds")
        if self.timestamp_regex is not None and (
            not isinstance(self.timestamp_regex, str) or not self.timestamp_regex
        ):
            raise ConfigurationError("timestamp_regex must be a non-empty string or null")
        if self.timestamp_mode == "filename" and self.timestamp_regex is None:
            raise ConfigurationError("timestamp_regex is required when timestamp_mode is filename")
        if (
            isinstance(self.extensions, (str, bytes))
            or not isinstance(self.extensions, Sequence)
            or not self.extensions
            or any(not _valid_extension(extension) for extension in self.extensions)
        ):
            raise ConfigurationError("extensions must be a non-empty sequence of file extensions")


@dataclass(frozen=True, slots=True)
class SelectionConfig:
    """Weights and constraints used by the deterministic selector."""

    budget: int = 8
    min_gap: float = 0.0
    duplicate_threshold: float = 0.035
    quality_weight: float = 0.32
    change_weight: float = 0.38
    coverage_weight: float = 0.30
    keep_endpoints: bool = True

    def validate(self) -> None:
        if isinstance(self.budget, bool) or not isinstance(self.budget, int) or self.budget < 1:
            raise ConfigurationError("budget must be at least one")
        if (
            isinstance(self.min_gap, bool)
            or not isinstance(self.min_gap, (int, float))
            or not _is_finite_number(self.min_gap)
            or self.min_gap < 0
        ):
            raise ConfigurationError("min_gap cannot be negative")
        if (
            isinstance(self.duplicate_threshold, bool)
            or not isinstance(self.duplicate_threshold, (int, float))
            or not _is_finite_number(self.duplicate_threshold)
            or not 0 <= self.duplicate_threshold <= 1
        ):
            raise ConfigurationError("duplicate_threshold must be between zero and one")
        weights = (self.quality_weight, self.change_weight, self.coverage_weight)
        if any(
            isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not _is_finite_number(weight)
            or weight < 0
            for weight in weights
        ):
            raise ConfigurationError("selection weights cannot be negative")
        weight_sum = sum(weights)
        if not _is_finite_number(weight_sum) or weight_sum <= 0:
            raise ConfigurationError("at least one selection weight must be positive")
        if not isinstance(self.keep_endpoints, bool):
            raise ConfigurationError("keep_endpoints must be a boolean")


@dataclass(frozen=True, slots=True)
class FrameMetrics:
    """Content measurements extracted without external model inference."""

    perceptual_hash: int
    luminance: float
    entropy: float
    sharpness: float
    colorfulness: float
    mean_red: float
    mean_green: float
    mean_blue: float

    def serializable(self) -> dict[str, float | str]:
        values: dict[str, float | str] = asdict(self)
        values["perceptual_hash"] = f"{self.perceptual_hash:016x}"
        return values


@dataclass(frozen=True, slots=True)
class Frame:
    """One scanned image and its immutable measurements."""

    index: int
    path: Path
    relative_path: str
    timestamp: float | None
    width: int
    height: int
    byte_size: int
    metrics: FrameMetrics

    @property
    def time_coordinate(self) -> float:
        """Return timestamp when present, otherwise the stable sequence index."""

        return self.timestamp if self.timestamp is not None else float(self.index)

    def serializable(self) -> dict[str, Any]:
        _require_safe_text(self.relative_path, "frame relative path")
        return {
            "index": self.index,
            "path": self.relative_path,
            "timestamp": self.timestamp,
            "width": self.width,
            "height": self.height,
            "byte_size": self.byte_size,
            "metrics": self.metrics.serializable(),
        }


@dataclass(frozen=True, slots=True)
class ScoreBreakdown:
    """Normalized factors recorded for an auditable decision."""

    quality: float = 0.0
    change: float = 0.0
    coverage: float = 0.0
    utility: float = 0.0

    def serializable(self) -> dict[str, float]:
        return {key: round(value, 6) for key, value in asdict(self).items()}


@dataclass(frozen=True, slots=True)
class FrameDecision:
    """Selection outcome and plain-language rationale for one frame."""

    index: int
    path: str
    selected: bool
    rank: int | None
    reason_code: str
    reason: str
    nearest_selected_index: int | None = None
    scores: ScoreBreakdown = field(default_factory=ScoreBreakdown)

    def serializable(self) -> dict[str, Any]:
        data = asdict(self)
        data["scores"] = self.scores.serializable()
        return data


@dataclass(frozen=True, slots=True)
class SelectionResult:
    """Complete result including selections and every rejected frame."""

    frames: tuple[Frame, ...]
    selected_indices: tuple[int, ...]
    decisions: tuple[FrameDecision, ...]
    config: SelectionConfig

    @property
    def selected_frames(self) -> tuple[Frame, ...]:
        selected = set(self.selected_indices)
        return tuple(frame for frame in self.frames if frame.index in selected)

    def decision_for(self, index: int) -> FrameDecision:
        for decision in self.decisions:
            if decision.index == index:
                return decision
        raise KeyError(index)


def _require_safe_text(value: object, label: str) -> None:
    if not isinstance(value, str) or not value:
        raise ConfigurationError(f"{label} must be a non-empty string")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ConfigurationError(f"{label} must contain valid Unicode scalar values") from exc
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ConfigurationError(f"{label} must not contain control characters")


def _valid_extension(value: object) -> bool:
    if not isinstance(value, str) or not value or value != value.strip():
        return False
    suffix = value[1:] if value.startswith(".") else value
    return bool(suffix) and suffix.isascii() and suffix.isalnum()


def _is_finite_number(value: object) -> bool:
    try:
        return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)
    except (OverflowError, TypeError, ValueError):
        return False
