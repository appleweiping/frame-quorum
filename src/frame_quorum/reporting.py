"""Stable JSON reports for scans and selections."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .models import Frame, ScanConfig, SelectionResult

SCHEMA_VERSION = "1.0"


def scan_manifest(frames: tuple[Frame, ...], config: ScanConfig) -> dict[str, Any]:
    """Build a JSON-serializable scan report."""

    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "frame-quorum-scan",
        "summary": {
            "frame_count": len(frames),
            "timestamped_count": sum(frame.timestamp is not None for frame in frames),
            "total_bytes": sum(frame.byte_size for frame in frames),
        },
        "scan_config": _serializable_scan_config(config),
        "frames": [frame.serializable() for frame in frames],
    }


def selection_manifest(result: SelectionResult, scan_config: ScanConfig) -> dict[str, Any]:
    """Build a complete selection report, including rejected frames."""

    selected = set(result.selected_indices)
    decisions = {decision.index: decision for decision in result.decisions}
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "frame-quorum-selection",
        "summary": {
            "frame_count": len(result.frames),
            "selected_count": len(result.selected_indices),
            "rejected_count": len(result.frames) - len(result.selected_indices),
            "selected_indices": list(result.selected_indices),
        },
        "scan_config": _serializable_scan_config(scan_config),
        "selection_config": asdict(result.config),
        "frames": [
            {
                **frame.serializable(),
                "decision": decisions[frame.index].serializable(),
            }
            for frame in result.frames
        ],
        "selected": [frame.serializable() for frame in result.frames if frame.index in selected],
    }


def write_json(
    data: dict[str, Any],
    destination: str | Path | None,
    *,
    ensure_ascii: bool = False,
) -> str:
    """Serialize consistently and optionally write to disk."""

    rendered = (
        json.dumps(
            data,
            indent=2,
            sort_keys=False,
            ensure_ascii=ensure_ascii,
            allow_nan=False,
        )
        + "\n"
    )
    if destination is not None:
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered, encoding="utf-8", newline="\n")
    return rendered


def _serializable_scan_config(config: ScanConfig) -> dict[str, Any]:
    data = asdict(config)
    data["extensions"] = list(config.extensions)
    return data
