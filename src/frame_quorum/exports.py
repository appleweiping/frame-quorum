"""Stable text exports for interoperability with editing tools."""

from __future__ import annotations

import csv
import io
from pathlib import Path

from .errors import ConfigurationError
from .models import SelectionResult
from .scene_detection import DetectionResult


def render_detection_csv(result: DetectionResult) -> str:
    """Render all detector statistics, including suppressed boundary candidates."""

    if not isinstance(result, DetectionResult):
        raise ConfigurationError("result must be a DetectionResult")
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(
        (
            "position",
            "frame_index",
            "timestamp",
            "content_score",
            "luminance",
            "detector_score",
            "candidate",
            "accepted",
            "reason",
        )
    )
    for item in result.statistics:
        writer.writerow(
            (
                item.position,
                item.frame_index,
                item.timestamp,
                item.content_score,
                item.luminance,
                item.detector_score,
                str(item.candidate).lower(),
                str(item.accepted).lower(),
                item.reason,
            )
        )
    return output.getvalue()


def render_decision_csv(result: SelectionResult) -> str:
    """Render one row per frame, including selection reason and score fields."""

    if not isinstance(result, SelectionResult):
        raise ConfigurationError("result must be a SelectionResult")
    result.validate()
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(
        (
            "index",
            "path",
            "selected",
            "rank",
            "reason_code",
            "nearest_selected_index",
            "quality",
            "change",
            "coverage",
            "utility",
        )
    )
    for decision in result.decisions:
        writer.writerow(
            (
                decision.index,
                decision.path,
                str(decision.selected).lower(),
                decision.rank or "",
                decision.reason_code,
                decision.nearest_selected_index or "",
                f"{decision.scores.quality:.8f}",
                f"{decision.scores.change:.8f}",
                f"{decision.scores.coverage:.8f}",
                f"{decision.scores.utility:.8f}",
            )
        )
    return output.getvalue()


def write_decision_csv(result: SelectionResult, destination: str | Path) -> Path:
    """Atomically publish the decision CSV and return its path."""

    path = Path(destination)
    if not str(path):
        raise ConfigurationError("destination must not be empty")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(render_decision_csv(result), encoding="utf-8", newline="")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


__all__ = ["render_decision_csv", "render_detection_csv", "write_decision_csv"]
