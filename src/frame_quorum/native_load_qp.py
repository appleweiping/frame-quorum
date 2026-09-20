"""Complete native-frame CSV import to bounded encoder QP instructions."""

from __future__ import annotations

import hashlib
import json
import stat
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .errors import ConfigurationError, OutputError, ScanError
from .native_load_fcpxml import load_native_scene_csv
from .native_measurements import _bundle
from .native_qp import _lines
from .native_splitting import _target_path
from .native_video import NativeVideoConfig, NativeVideoStatus, NativeVideoStream, _integer

_MAX_FRAMES = 100_000
_MAX_INPUT_BYTES = 8 * 1024 * 1024
_MAX_SCENES = 10_000
_MAX_OUTPUT_BYTES = 32 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class LoadedQPResult:
    output_path: Path
    frame_count: int
    cut_count: int
    csv_sha256: str
    qp_bytes: int
    total_output_bytes: int
    qp_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_path": self.output_path.as_posix(),
            "frame_count": self.frame_count,
            "cut_count": self.cut_count,
            "csv_sha256": self.csv_sha256,
            "qp_bytes": self.qp_bytes,
            "total_output_bytes": self.total_output_bytes,
            "qp_sha256": self.qp_sha256,
            "detector_performed": False,
            "source_content_authenticated": False,
            "encoder_input_verified": False,
        }


def _complete_video(path: str | Path, limits: NativeVideoConfig) -> int:
    count = 0
    decode = replace(limits, max_frames=limits.max_frames + 1)
    with NativeVideoStream(path, decode) as stream:
        metadata = stream.metadata
        opened_identity = stream._captured_identity()
        for frame in stream:
            if count >= limits.max_frames:
                raise ConfigurationError("loaded QP video exceeds configured frame limit")
            if frame.decode_index != count or frame.sample_index != count or frame.generation != 0:
                raise ScanError("loaded QP requires contiguous generation-zero frame ordinals")
            if frame.width != metadata.width or frame.height != metadata.height:
                raise ScanError("loaded QP video dimensions changed during decoding")
            count += 1
            del frame
    diagnostics = stream.diagnostics
    if (
        count == 0
        or diagnostics.status is not NativeVideoStatus.EOF
        or not diagnostics.closed
        or diagnostics.cleanup_errors
        or diagnostics.generation != 0
        or diagnostics.decoded_frames != count
        or diagnostics.returned_frames != count
    ):
        raise ScanError("loaded QP requires complete error-free source EOF")
    try:
        current = stream.path.lstat()
    except OSError as error:
        raise ScanError("loaded QP source changed during scan") from error
    if (
        not stat.S_ISREG(current.st_mode)
        or (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns) != opened_identity
    ):
        raise ScanError("loaded QP source changed during scan")
    return count


def write_loaded_qp_bundle(
    video_path: str | Path,
    csv_path: str | Path,
    output_dir: str | Path,
    *,
    video_limits: NativeVideoConfig | None = None,
    max_input_bytes: int = _MAX_INPUT_BYTES,
    max_scenes: int = _MAX_SCENES,
    max_output_bytes: int = _MAX_OUTPUT_BYTES,
) -> LoadedQPResult:
    """Publish CSV-requested I-frame ordinals without running a detector or encoder."""
    limits = NativeVideoConfig() if video_limits is None else video_limits
    if type(limits) is not NativeVideoConfig:
        raise ConfigurationError("video_limits must be NativeVideoConfig")
    limits.__post_init__()
    if (
        limits.start is not None
        or limits.end is not None
        or limits.frame_step != 1
        or limits.video_stream != 0
        or limits.max_frames > _MAX_FRAMES
        or limits.max_decoded_frames <= limits.max_frames
    ):
        raise ConfigurationError("loaded QP requires bounded complete stream-zero video")
    _integer(max_input_bytes, "max_input_bytes", 1, _MAX_INPUT_BYTES)
    _integer(max_scenes, "max_scenes", 1, _MAX_SCENES)
    _integer(max_output_bytes, "max_output_bytes", 1, _MAX_OUTPUT_BYTES)
    loaded = load_native_scene_csv(csv_path, max_input_bytes=max_input_bytes, max_scenes=max_scenes)
    _target_path(output_dir)  # Read-only preflight; _bundle rechecks publication.
    frame_count = _complete_video(video_path, limits)
    if loaded.start_ordinals[-1] >= frame_count:
        raise ConfigurationError("scene CSV cut is outside the decoded video")
    cuts = loaded.start_ordinals[1:]
    size = 0
    digest = hashlib.sha256()
    for line in _lines(cuts):
        size += len(line)
        if size > max_output_bytes:
            raise OutputError("loaded QP output exceeds total byte limit")
        digest.update(line)
    qp_sha256 = digest.hexdigest()
    audit = {
        "kind": "frame-quorum-loaded-qp",
        "schema_version": 1,
        "qp_format": "x264-x265-qpfile-v1",
        "video_stream": 0,
        "frame_count": frame_count,
        "cut_count": len(cuts),
        "csv_bytes": loaded.csv_bytes,
        "csv_sha256": loaded.csv_sha256,
        "qp_bytes": size,
        "qp_sha256": qp_sha256,
        "detector_performed": False,
        "source_content_authenticated": False,
        "encoder_input_verified": False,
    }
    audit_bytes = (json.dumps(audit, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "ascii"
    )
    total = size + len(audit_bytes)
    if total > max_output_bytes:
        raise OutputError("loaded QP bundle including audit exceeds total byte limit")
    target = _bundle(
        output_dir,
        (("scenes.qp", _lines(cuts)), ("audit.json", iter((audit_bytes,)))),
        max_output_bytes,
        expected_files=(
            ("scenes.qp", size, qp_sha256),
            ("audit.json", len(audit_bytes), hashlib.sha256(audit_bytes).hexdigest()),
        ),
    )
    return LoadedQPResult(
        target / "scenes.qp", frame_count, len(cuts), loaded.csv_sha256, size, total, qp_sha256
    )


__all__ = ["LoadedQPResult", "write_loaded_qp_bundle"]
