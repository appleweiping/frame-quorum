"""Bounded encoder QP instructions from complete native scene observations."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import ConfigurationError, OutputError
from .native_measurements import _bundle
from .native_scenes import NativeScene, NativeSceneConfig, NativeSceneResult, NativeSceneStatistic
from .native_video import (
    NativeVideoConfig,
    NativeVideoDiagnostics,
    NativeVideoMetadata,
    NativeVideoStatus,
    _integer,
)

_MAX_OUTPUT_BYTES = 32 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class NativeQPResult:
    output_path: Path
    frame_count: int
    cut_count: int
    qp_bytes: int
    total_output_bytes: int
    qp_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_path": self.output_path.as_posix(),
            "frame_count": self.frame_count,
            "cut_count": self.cut_count,
            "qp_bytes": self.qp_bytes,
            "total_output_bytes": self.total_output_bytes,
            "qp_sha256": self.qp_sha256,
            "encoder_input_verified": False,
        }


def _lines(cuts: tuple[int, ...]) -> Iterator[bytes]:
    yield b"0 I -1\n"
    for position in cuts:
        yield f"{position} I -1\n".encode("ascii")


def _admit(result: NativeSceneResult) -> tuple[int, tuple[int, ...], int]:
    if type(result) is not NativeSceneResult:
        raise ConfigurationError("QP export requires an exact NativeSceneResult")
    # A frozen dataclass can still be forged through object.__setattr__.
    # Check nested exact types before the outer result traverses their fields.
    if type(result.config) is not NativeSceneConfig or type(result.config.video) is not NativeVideoConfig:
        raise ConfigurationError("QP export requires exact native scene configuration")
    if (
        type(result.diagnostics) is not NativeVideoDiagnostics
        or type(result.metadata) is not NativeVideoMetadata
    ):
        raise ConfigurationError("QP export requires exact native diagnostics and metadata")
    result.config.video.__post_init__()
    result.config.__post_init__()
    result.diagnostics.__post_init__()
    result.metadata.__post_init__()
    if (
        type(result.statistics) is not tuple
        or type(result.scenes) is not tuple
        or len(result.statistics) > result.config.video.max_frames
        or len(result.scenes) > len(result.statistics)
    ):
        raise ConfigurationError("QP export requires immutable native statistics and scenes")
    for row in result.statistics:
        if type(row) is not NativeSceneStatistic:
            raise ConfigurationError("QP export requires exact native statistics")
        row.__post_init__()
        row.sample.__post_init__()
        for detector in row.detectors:
            detector.__post_init__()
    for scene in result.scenes:
        if type(scene) is not NativeScene:
            raise ConfigurationError("QP export requires exact native scenes")
        scene.__post_init__()
    result.__post_init__()
    video = result.config.video
    diagnostics = result.diagnostics
    if video.start is not None or video.end is not None or video.frame_step != 1:
        raise ConfigurationError("QP export requires unwindowed, unsampled full-source analysis")
    if diagnostics.status is not NativeVideoStatus.EOF or diagnostics.generation != 0:
        raise ConfigurationError("QP export requires complete generation-zero EOF")
    frame_count = len(result.statistics)
    if (
        not frame_count
        or diagnostics.decoded_frames != diagnostics.returned_frames
        or (diagnostics.returned_frames != frame_count)
    ):
        raise ConfigurationError("QP export requires every decoded source frame")
    if any(
        row.sample.decode_index != position or row.sample.sample_index != position
        for position, row in enumerate(result.statistics)
    ):
        raise ConfigurationError("QP export requires exact zero-based decoded frame ordinals")
    cuts = result.cut_positions
    previous = 0
    for position in cuts:
        if not previous < position < frame_count:
            raise ConfigurationError("QP cut ordinal is duplicate or outside the decoded source")
        previous = position
    return frame_count, cuts, video.video_stream


def write_native_qp_bundle(
    result: NativeSceneResult,
    output_dir: str | Path,
    *,
    max_output_bytes: int = _MAX_OUTPUT_BYTES,
) -> NativeQPResult:
    """Publish x264/x265-style I-frame requests for an unchanged decoded frame sequence.

    The caller-supplied scene result is structurally checked, not authenticated
    against media. This function neither invokes nor verifies an encoder.
    """
    _integer(max_output_bytes, "max_output_bytes", 1, _MAX_OUTPUT_BYTES)
    frame_count, cuts, stream_index = _admit(result)
    size = 0
    digest = hashlib.sha256()
    for line in _lines(cuts):
        size += len(line)
        if size > max_output_bytes:
            raise OutputError("QP output exceeds total byte limit")
        digest.update(line)
    audit = {
        "kind": "frame-quorum-native-qp",
        "schema_version": 1,
        "qp_format": "x264-x265-qpfile-v1",
        "video_stream": stream_index,
        "frame_count": frame_count,
        "cut_count": len(cuts),
        "qp_bytes": size,
        "qp_sha256": digest.hexdigest(),
        "source_content_authenticated": False,
        "encoder_input_verified": False,
    }
    audit_bytes = (json.dumps(audit, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "ascii"
    )
    total = size + len(audit_bytes)
    if total > max_output_bytes:
        raise OutputError("QP bundle including audit exceeds total byte limit")
    target = _bundle(
        output_dir,
        (("scenes.qp", _lines(cuts)), ("audit.json", iter((audit_bytes,)))),
        max_output_bytes,
        expected_files=(
            ("scenes.qp", size, digest.hexdigest()),
            ("audit.json", len(audit_bytes), hashlib.sha256(audit_bytes).hexdigest()),
        ),
    )
    return NativeQPResult(target / "scenes.qp", frame_count, len(cuts), size, total, digest.hexdigest())


__all__ = ["NativeQPResult", "write_native_qp_bundle"]
