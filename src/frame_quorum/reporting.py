"""Stable JSON reports for scans and selections."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .errors import ConfigurationError
from .models import (
    _MAX_SIGNED_64,
    AnimationConfig,
    Frame,
    ScanConfig,
    SelectionResult,
    _require_bounded_integer,
    _require_int64,
)

SCHEMA_VERSION = "1.0"


def scan_manifest(
    frames: tuple[Frame, ...],
    config: ScanConfig,
    *,
    animation: AnimationConfig | None = None,
) -> dict[str, Any]:
    """Build a JSON-serializable scan report."""

    _validate_scan_inputs(frames, config)
    total_bytes = sum(frame.byte_size for frame in frames)
    _require_int64(total_bytes, "manifest total byte size")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "kind": "frame-quorum-scan",
        "summary": {
            "frame_count": len(frames),
            "timestamped_count": sum(frame.timestamp is not None for frame in frames),
            "total_bytes": total_bytes,
        },
        "scan_config": _serializable_scan_config(config),
        "frames": [frame.serializable() for frame in frames],
    }
    return _with_animation_config(manifest, animation)


def selection_manifest(
    result: SelectionResult,
    scan_config: ScanConfig,
    *,
    animation: AnimationConfig | None = None,
) -> dict[str, Any]:
    """Build a complete selection report, including rejected frames."""

    if not isinstance(result, SelectionResult):
        raise ConfigurationError("result must be a SelectionResult")
    if not isinstance(scan_config, ScanConfig):
        raise ConfigurationError("scan_config must be ScanConfig")
    result.validate()
    scan_config.validate()
    selected = set(result.selected_indices)
    decisions = {decision.index: decision for decision in result.decisions}
    manifest = {
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
    return _with_animation_config(manifest, animation)


def write_json(
    data: dict[str, Any],
    destination: str | Path | None,
    *,
    ensure_ascii: bool = False,
) -> str:
    """Serialize consistently and optionally write to disk."""

    _validate_json_integers(data)
    try:
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
    except (OverflowError, RecursionError, TypeError, ValueError) as error:
        raise ConfigurationError("report data must be finite and JSON-serializable") from error
    if destination is not None:
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered, encoding="utf-8", newline="\n")
    return rendered


def _with_animation_config(manifest: dict[str, Any], animation: AnimationConfig | None) -> dict[str, Any]:
    """Record expansion limits only when animated-image expansion was requested."""

    if animation is None:
        return manifest
    if not isinstance(animation, AnimationConfig):
        raise ConfigurationError("animation must be AnimationConfig or None")
    animation.validate()
    manifest["animation_config"] = asdict(animation)
    return manifest


def _serializable_scan_config(config: ScanConfig) -> dict[str, Any]:
    data = asdict(config)
    data["extensions"] = list(config.extensions)
    return data


def _validate_scan_inputs(frames: object, config: object) -> None:
    if not isinstance(config, ScanConfig):
        raise ConfigurationError("config must be ScanConfig")
    config.validate()
    if not isinstance(frames, tuple) or any(not isinstance(frame, Frame) for frame in frames):
        raise ConfigurationError("frames must be a tuple of Frame objects")
    for frame in frames:
        frame.validate()
    indices = tuple(frame.index for frame in frames)
    if len(indices) != len(set(indices)):
        raise ConfigurationError("manifest frame indices must be unique")
    if indices != tuple(sorted(indices)):
        raise ConfigurationError("manifest frames must be ordered by index")


def _validate_json_integers(value: object) -> None:
    pending = [value]
    seen_containers: set[int] = set()
    while pending:
        current = pending.pop()
        if isinstance(current, bool):
            continue
        if isinstance(current, int):
            _require_bounded_integer(
                current,
                "JSON integer",
                minimum=-(1 << 63),
                maximum=_MAX_SIGNED_64,
            )
            continue
        if isinstance(current, (dict, list, tuple)):
            identity = id(current)
            if identity in seen_containers:
                continue
            seen_containers.add(identity)
            if isinstance(current, dict):
                pending.extend(current.keys())
                pending.extend(current.values())
            else:
                pending.extend(current)
