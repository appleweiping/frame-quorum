"""Typed records shared by scanning, selection, and reporting."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .errors import ConfigurationError

_MAX_SIGNED_64 = (1 << 63) - 1
_MIN_SIGNED_64 = -(1 << 63)
_MAX_UNSIGNED_64 = (1 << 64) - 1
_MAX_ANIMATION_FRAMES = 100_000
_MAX_ANIMATION_DECODED_BYTES = 1 << 40
_MAX_SCAN_WORKERS = 64


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
            "exif",
            "none",
        }:
            raise ConfigurationError("timestamp_mode must be index, filename, mtime, exif, or none")
        if not _is_stable_json_number(self.frame_rate) or self.frame_rate <= 0:
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
class AnimationConfig:
    """Opt-in limits for expanding an animated image into its internal frames.

    Passing an instance to ``scan_frames`` enables expansion. Both limits apply
    to one animated container: ``max_decoded_bytes`` bounds the RGB bytes the
    whole container would decode to, estimated before any internal frame is
    read. Exceeding either limit raises ``ScanError`` instead of silently
    truncating the animation.
    """

    max_frames: int = 64
    max_decoded_bytes: int = 268_435_456

    def validate(self) -> None:
        _require_int64(self.max_frames, "animation max_frames", minimum=1)
        if self.max_frames > _MAX_ANIMATION_FRAMES:
            raise ConfigurationError(f"animation max_frames cannot exceed {_MAX_ANIMATION_FRAMES}")
        _require_int64(self.max_decoded_bytes, "animation max_decoded_bytes", minimum=1)
        if self.max_decoded_bytes > _MAX_ANIMATION_DECODED_BYTES:
            raise ConfigurationError(
                f"animation max_decoded_bytes cannot exceed {_MAX_ANIMATION_DECODED_BYTES}"
            )


@dataclass(frozen=True, slots=True)
class ConcurrencyConfig:
    """Opt-in worker count for measuring several files at once.

    ``workers=1`` is the default and scans strictly sequentially, so callers who
    supply nothing, or an explicit single worker, keep today's behaviour. Larger
    values only change how fast the files are read: discovery order, indices,
    metrics, and error messages are identical at every worker count, so the
    count is an execution detail and is deliberately absent from manifests.

    The count is caller-supplied rather than derived from the host because each
    worker holds one fully decoded image, making peak memory a multiple of the
    worker count. A machine-derived default would make that multiple, and the
    thread count, vary by host for an unchanged command.
    """

    workers: int = 1

    def validate(self) -> None:
        _require_int64(self.workers, "concurrency workers", minimum=1)
        if self.workers > _MAX_SCAN_WORKERS:
            raise ConfigurationError(f"concurrency workers cannot exceed {_MAX_SCAN_WORKERS}")


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
        _require_int64(self.budget, "budget", minimum=1)
        if not _is_stable_json_number(self.min_gap) or self.min_gap < 0:
            raise ConfigurationError("min_gap cannot be negative")
        if not _is_stable_json_number(self.duplicate_threshold) or not 0 <= self.duplicate_threshold <= 1:
            raise ConfigurationError("duplicate_threshold must be between zero and one")
        weights = (self.quality_weight, self.change_weight, self.coverage_weight)
        if any(not _is_stable_json_number(weight) or weight < 0 for weight in weights):
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

    def validate(self) -> None:
        """Reject values outside the stable metrics contract."""

        _require_bounded_integer(
            self.perceptual_hash,
            "perceptual hash",
            minimum=0,
            maximum=_MAX_UNSIGNED_64,
        )
        normalized = (
            self.luminance,
            self.entropy,
            self.sharpness,
            self.colorfulness,
            self.mean_red,
            self.mean_green,
            self.mean_blue,
        )
        if any(not _is_finite_number(value) or not 0 <= value <= 1 for value in normalized):
            raise ConfigurationError("frame metrics must be finite values between zero and one")

    def serializable(self) -> dict[str, float | str]:
        self.validate()
        values: dict[str, float | str] = asdict(self)
        values["perceptual_hash"] = f"{self.perceptual_hash:016x}"
        return values


@dataclass(frozen=True, slots=True)
class Frame:
    """One scanned image and its immutable measurements.

    An expanded animated container contributes one record per internal frame.
    Those records share ``path`` and ``byte_size`` with the container and are
    told apart by ``source_frame_index`` and a ``name#frame=N`` relative path.
    ``source_frame_index`` is ``None`` for an ordinary single-image file.
    """

    index: int
    path: Path
    relative_path: str
    timestamp: float | None
    width: int
    height: int
    byte_size: int
    metrics: FrameMetrics
    source_frame_index: int | None = None

    def validate(self) -> None:
        """Validate a directly constructed frame before it enters an operation."""

        _require_int64(self.index, "frame index")
        if not isinstance(self.path, Path):
            raise ConfigurationError("frame path must be a pathlib.Path")
        _require_safe_text(self.relative_path, "frame relative path")
        if self.timestamp is not None and not _is_stable_json_number(self.timestamp):
            raise ConfigurationError("frame timestamp must be a finite number or null")
        _require_int64(self.width, "frame width", minimum=1)
        _require_int64(self.height, "frame height", minimum=1)
        _require_int64(self.byte_size, "frame byte size")
        if not isinstance(self.metrics, FrameMetrics):
            raise ConfigurationError("frame metrics must be FrameMetrics")
        self.metrics.validate()
        if self.source_frame_index is not None:
            _require_int64(self.source_frame_index, "frame source frame index")

    @property
    def time_coordinate(self) -> float | int:
        """Return timestamp when present, otherwise the stable sequence index."""

        if self.timestamp is not None:
            if not _is_stable_json_number(self.timestamp):
                raise ConfigurationError("frame timestamp must be a finite number or null")
            return float(self.timestamp)
        _require_int64(self.index, "frame index")
        return self.index

    def serializable(self) -> dict[str, Any]:
        self.validate()
        data: dict[str, Any] = {
            "index": self.index,
            "path": self.relative_path,
            "timestamp": self.timestamp,
            "width": self.width,
            "height": self.height,
            "byte_size": self.byte_size,
            "metrics": self.metrics.serializable(),
        }
        if self.source_frame_index is not None:
            data["source_frame_index"] = self.source_frame_index
        return data


@dataclass(frozen=True, slots=True)
class ScoreBreakdown:
    """Normalized factors recorded for an auditable decision."""

    quality: float = 0.0
    change: float = 0.0
    coverage: float = 0.0
    utility: float = 0.0

    def validate(self) -> None:
        """Ensure every published normalized score is finite and bounded."""

        if any(not _is_finite_number(value) or not 0 <= value <= 1 for value in asdict(self).values()):
            raise ConfigurationError("decision scores must be finite values between zero and one")

    def serializable(self) -> dict[str, float]:
        self.validate()
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

    def validate(self) -> None:
        """Validate a decision created outside the selector."""

        _require_int64(self.index, "decision frame index")
        _require_safe_text(self.path, "decision path")
        if not isinstance(self.selected, bool):
            raise ConfigurationError("decision selected must be a boolean")
        if self.rank is not None:
            _require_int64(self.rank, "decision rank", minimum=1)
        if self.selected != (self.rank is not None):
            raise ConfigurationError(
                "selected decisions require a rank and rejected decisions must not have one"
            )
        _require_safe_text(self.reason_code, "decision reason code")
        _require_safe_text(self.reason, "decision reason")
        if self.nearest_selected_index is not None:
            _require_int64(self.nearest_selected_index, "nearest selected frame index")
        if not isinstance(self.scores, ScoreBreakdown):
            raise ConfigurationError("decision scores must be ScoreBreakdown")
        self.scores.validate()

    def serializable(self) -> dict[str, Any]:
        self.validate()
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

    def validate(self) -> None:
        """Validate cross-record invariants for directly constructed results."""

        if not isinstance(self.config, SelectionConfig):
            raise ConfigurationError("result config must be SelectionConfig")
        self.config.validate()

        if not isinstance(self.frames, tuple) or any(not isinstance(frame, Frame) for frame in self.frames):
            raise ConfigurationError("result frames must be a tuple of Frame objects")
        for frame in self.frames:
            frame.validate()
        frame_indices = tuple(frame.index for frame in self.frames)
        if len(frame_indices) != len(set(frame_indices)):
            raise ConfigurationError("result frame indices must be unique")
        if frame_indices != tuple(sorted(frame_indices)):
            raise ConfigurationError("result frames must be ordered by index")

        if not isinstance(self.selected_indices, tuple):
            raise ConfigurationError("selected indices must be a tuple")
        for index in self.selected_indices:
            _require_int64(index, "selected frame index")
        if len(self.selected_indices) != len(set(self.selected_indices)):
            raise ConfigurationError("selected frame indices must be unique")
        if self.selected_indices != tuple(sorted(self.selected_indices)):
            raise ConfigurationError("selected frame indices must be ordered")
        if not set(self.selected_indices).issubset(frame_indices):
            raise ConfigurationError("selected frame indices must reference result frames")
        if len(self.selected_indices) > self.config.budget:
            raise ConfigurationError("selected frame count must not exceed the selection budget")

        if not isinstance(self.decisions, tuple) or any(
            not isinstance(decision, FrameDecision) for decision in self.decisions
        ):
            raise ConfigurationError("result decisions must be a tuple of FrameDecision objects")
        for decision in self.decisions:
            decision.validate()
        decision_indices = tuple(decision.index for decision in self.decisions)
        if decision_indices != frame_indices:
            raise ConfigurationError("result decisions must correspond to frames in index order")
        if any(
            decision.path != frame.relative_path
            for frame, decision in zip(self.frames, self.decisions, strict=True)
        ):
            raise ConfigurationError("decision paths must match their corresponding frame paths")
        selected = set(self.selected_indices)
        if any(decision.selected != (decision.index in selected) for decision in self.decisions):
            raise ConfigurationError("decision selected states must match selected frame indices")
        ranks = sorted(decision.rank for decision in self.decisions if decision.rank is not None)
        if ranks != list(range(1, len(selected) + 1)):
            raise ConfigurationError("selected decision ranks must be unique and contiguous")
        if any(
            decision.nearest_selected_index is not None and decision.nearest_selected_index not in selected
            for decision in self.decisions
        ):
            raise ConfigurationError("nearest selected indices must reference selected frames")

    @property
    def selected_frames(self) -> tuple[Frame, ...]:
        self.validate()
        selected = set(self.selected_indices)
        return tuple(frame for frame in self.frames if frame.index in selected)

    def decision_for(self, index: int) -> FrameDecision:
        self.validate()
        _require_int64(index, "decision lookup index")
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


def _is_stable_json_number(value: object) -> bool:
    """Accept finite floats and signed-64 integer spellings for continuous values."""

    if not _is_finite_number(value):
        return False
    return not isinstance(value, int) or _MIN_SIGNED_64 <= value <= _MAX_SIGNED_64


def _require_int64(value: object, label: str, *, minimum: int = 0) -> None:
    _require_bounded_integer(value, label, minimum=minimum, maximum=_MAX_SIGNED_64)


def _require_bounded_integer(value: object, label: str, *, minimum: int, maximum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ConfigurationError(f"{label} must be an integer between {minimum} and {maximum}")
