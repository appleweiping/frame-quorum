"""Two-pass, bounded still export from an exact declared composite timeline.

The declaration supplies coordinates, not proof of physical media duration or
cryptographic source identity. Every returned sample is compared across passes.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from tempfile import mkdtemp
from typing import Any

from PIL import __version__ as pillow_version

from .errors import ConfigurationError, OutputError, ScanError
from .native_concat import NativeConcatClip, NativeConcatFrame, NativeConcatStream
from .native_concat import _identity as _source_identity
from .native_concat_scenes import (
    NativeConcatSceneConfig,
    NativeConcatSceneResult,
    NativeConcatSceneSample,
    _scenes,
    _statistics,
    _window,
)
from .native_scene_images import (
    _RESAMPLING,
    NativeSceneImageConfig,
    _dimensions,
    _encode_image,
    _managed_image,
    _positions,
    _reconcile,
    _Slot,
)
from .native_scenes import _measure_native_sample
from .native_splitting import (
    _ByteBudget,
    _cleanup,
    _hash_file,
    _managed_file,
    _publish,
    _target_path,
    _Writer,
)
from .native_splitting import (
    _identity as _stage_identity,
)
from .native_video import _integer, _load_av, _pair


@dataclass(frozen=True, slots=True)
class NativeConcatSceneImageResult:
    output_dir: Path
    scene_count: int
    image_count: int
    unique_sample_count: int
    total_output_bytes: int
    manifest_sha256: str
    timeline_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.output_dir, Path):
            raise ConfigurationError("output_dir must be a pathlib.Path")
        _integer(self.scene_count, "scene_count", 1, 1000)
        _integer(self.image_count, "image_count", self.scene_count, 10_000)
        _integer(self.unique_sample_count, "unique_sample_count", self.scene_count, self.image_count)
        _integer(self.total_output_bytes, "total_output_bytes", 1, 1_000_000_000)
        if self.image_count % self.scene_count or self.image_count // self.scene_count > 100:
            raise ConfigurationError("each scene needs the same bounded image slot count")
        for name in ("manifest_sha256", "timeline_digest"):
            digest = getattr(self, name)
            if (
                type(digest) is not str
                or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)
            ):
                raise ConfigurationError(f"{name} must be a lowercase SHA-256 digest")

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_dir": self.output_dir.as_posix(),
            "scene_count": self.scene_count,
            "image_count": self.image_count,
            "unique_sample_count": self.unique_sample_count,
            "total_output_bytes": self.total_output_bytes,
            "manifest_sha256": self.manifest_sha256,
            "timeline_digest": self.timeline_digest,
        }


@dataclass(frozen=True, slots=True)
class _Observed:
    sample: NativeConcatSceneSample
    width: int
    height: int
    rgb_sha256: str


def _sample(frame: NativeConcatFrame) -> NativeConcatSceneSample:
    return NativeConcatSceneSample(
        frame.timeline.digest,
        frame.clip_index,
        frame.activation,
        frame.generation,
        frame.sample_index,
        frame.presentation_time,
        _measure_native_sample(frame.native),
    )


def _identities(clips: tuple[NativeConcatClip, ...]) -> dict[Path, tuple[int, int, int, int]]:
    result: dict[Path, tuple[int, int, int, int]] = {}
    for clip in clips:
        path = Path(clip.path)
        if path not in result:
            result[path] = _source_identity(path)
    return result


def _check_identities(expected: dict[Path, tuple[int, int, int, int]]) -> None:
    for path, identity in expected.items():
        if _source_identity(path) != identity:
            raise ScanError("composite scene image source changed between passes")


def _capture(
    stream: NativeConcatStream, config: NativeConcatSceneConfig
) -> tuple[NativeConcatSceneResult, tuple[_Observed, ...]]:
    _, end, _ = _window(config, stream.timeline)
    records: list[_Observed] = []
    with stream:
        metadata = stream.metadata
        for frame in stream:
            records.append(
                _Observed(
                    _sample(frame),
                    frame.native.width,
                    frame.native.height,
                    hashlib.sha256(frame.rgb).hexdigest(),
                )
            )
            del frame
    diagnostics = stream.diagnostics
    if (
        diagnostics.status not in ("intervals_exhausted", "range_end")
        or not diagnostics.closed
        or diagnostics.cleanup_errors
    ):
        raise ScanError("incomplete composite scene image analysis")
    samples = tuple(row.sample for row in records)
    rows = _statistics(samples, config)
    scenes = _scenes(rows, stream.timeline, end, config.video.limits.max_span_parts)
    # The ordinary public result validates the whole partition, per-source
    # budgets, native ordinals, range completion and detector replay.
    result = NativeConcatSceneResult(config, stream.timeline, metadata, diagnostics, rows, scenes)
    if not records:
        raise ScanError("composite scene images require observed samples")
    return result, tuple(records)


def _plan(
    result: NativeConcatSceneResult, observed: tuple[_Observed, ...], config: NativeSceneImageConfig
) -> tuple[_Slot, ...]:
    if (
        len(result.scenes) > config.max_scenes
        or len(result.scenes) * config.images_per_scene > config.max_images
    ):
        raise ConfigurationError("composite scene image slot count exceeds configured limit")
    slots: list[_Slot] = []
    pixels = 0
    for scene in result.scenes:
        for index, position in enumerate(
            _positions(
                scene.start_position, scene.end_position, config.images_per_scene, config.sample_margin
            )
        ):
            source = observed[position]
            width, height = _dimensions(source.width, source.height, config)
            pixels += width * height
            if pixels > config.max_total_pixels or pixels > config.max_verification_pixels:
                raise ConfigurationError("composite scene image pixel work exceeds configured limit")
            slots.append(_Slot(scene.ordinal, index, position, width, height))
    return tuple(slots)


def _matches(frame: NativeConcatFrame, original: _Observed) -> bool:
    sample, native = original.sample, original.sample.native
    return (
        frame.timeline.digest == sample.timeline_digest
        and (
            frame.clip_index,
            frame.activation,
            frame.generation,
            frame.sample_index,
            frame.presentation_time,
        )
        == (
            sample.clip_index,
            sample.activation,
            sample.generation,
            sample.sample_index,
            sample.presentation_time,
        )
        and (
            frame.native.pts,
            frame.native.time_base,
            frame.native.decode_index,
            frame.native.sample_index,
            frame.native.generation,
        )
        == (native.pts, native.time_base, native.decode_index, native.sample_index, native.generation)
        and (frame.native.width, frame.native.height) == (original.width, original.height)
        and hashlib.sha256(frame.rgb).hexdigest() == original.rgb_sha256
    )


def _save_slot(
    frame: NativeConcatFrame,
    original: _Observed,
    slot: _Slot,
    stage: Path,
    config: NativeSceneImageConfig,
    budget: _ByteBudget,
    owned: dict[Path, tuple[int, int]],
) -> dict[str, Any]:
    extension = "png" if config.image_format == "png" else "jpg"
    path = stage / f"scene-{slot.scene_ordinal + 1:06d}-image-{slot.image_index + 1:06d}.{extension}"
    with _managed_image(frame.image()) as image:
        if image.size == (slot.width, slot.height):
            transformed, decoded, size, file_hash = _encode_image(image, path, config, budget, owned)
        else:
            with _managed_image(
                image.resize((slot.width, slot.height), _RESAMPLING[config.interpolation])
            ) as resized:
                transformed, decoded, size, file_hash = _encode_image(resized, path, config, budget, owned)
    native = frame.native
    return {
        "scene_ordinal": slot.scene_ordinal,
        "image_index": slot.image_index,
        "sample_index": frame.sample_index,
        "presentation_time": _pair(frame.presentation_time),
        "clip_index": frame.clip_index,
        "activation": frame.activation,
        "generation": frame.generation,
        "native_decode_index": native.decode_index,
        "native_sample_index": native.sample_index,
        "native_generation": native.generation,
        "native_pts": native.pts,
        "native_time_base": _pair(native.time_base),
        "source_width": native.width,
        "source_height": native.height,
        "width": slot.width,
        "height": slot.height,
        "source_rgb_sha256": original.rgb_sha256,
        "transformed_rgb_sha256": transformed,
        "decoded_rgb_sha256": decoded,
        "file": path.name,
        "bytes": size,
        "sha256": file_hash,
    }


def export_native_concat_scene_images(
    clips: tuple[NativeConcatClip, ...],
    output_dir: str | Path,
    scene_config: NativeConcatSceneConfig | None = None,
    image_config: NativeSceneImageConfig | None = None,
) -> NativeConcatSceneImageResult:
    """Publish verified PNG/JPEG slots from two complete composite decode passes."""
    if scene_config is not None and type(scene_config) is not NativeConcatSceneConfig:
        raise ConfigurationError("scene_config must be NativeConcatSceneConfig")
    if image_config is not None and type(image_config) is not NativeSceneImageConfig:
        raise ConfigurationError("image_config must be NativeSceneImageConfig")
    scenes_config = NativeConcatSceneConfig() if scene_config is None else replace(scene_config)
    config = NativeSceneImageConfig() if image_config is None else replace(image_config)
    scenes_config.__post_init__()
    config.__post_init__()
    # Timeline and window validation precede any source acquisition or staging.
    first = NativeConcatStream(clips, scenes_config.video)
    _window(scenes_config, first.timeline)
    if config.images_per_scene > config.max_images:
        raise ConfigurationError("composite scene image slots exceed configured limit")
    target = _target_path(output_dir)
    budget = _ByteBudget(config.max_output_bytes)
    stage: Path | None = None
    stage_identity: tuple[int, int] | None = None
    owned: dict[Path, tuple[int, int]] = {}
    publication_attempted = False
    try:
        identities = _identities(first.timeline.clips)
        scenes, observed = _capture(first, scenes_config)
        _check_identities(identities)
        slots = _plan(scenes, observed, config)
        av = _load_av()
        replay = NativeConcatStream(clips, scenes_config.video)
        if replay.timeline.digest != scenes.timeline.digest:
            raise ScanError("composite scene image replay timeline differs from analysis")
        rows: list[dict[str, Any]] = []
        slot_index = 0
        count = 0
        with replay:
            if replay.metadata != scenes.metadata:
                raise ScanError("composite scene image replay metadata differs from analysis")
            _check_identities(identities)
            stage = Path(mkdtemp(prefix=".frame-quorum-concat-scene-images-", dir=target.parent))
            stage_identity = _stage_identity(stage.lstat())
            for frame in replay:
                if count >= len(observed) or not _matches(frame, observed[count]):
                    raise ScanError("composite scene image replay sample differs from analysis")
                while slot_index < len(slots) and slots[slot_index].position == count:
                    rows.append(
                        _save_slot(frame, observed[count], slots[slot_index], stage, config, budget, owned)
                    )
                    slot_index += 1
                count += 1
                del frame
        if count != len(observed) or slot_index != len(slots) or replay.diagnostics != scenes.diagnostics:
            raise ScanError("composite scene image replay completion differs from analysis")
        if replay.metadata != scenes.metadata:
            raise ScanError("composite scene image final replay metadata differs from analysis")
        _check_identities(identities)
        unique_count = len({slot.position for slot in slots})
        pixels = sum(slot.width * slot.height for slot in slots)
        document = {
            "kind": "frame-quorum-native-concat-scene-images",
            "schema_version": 1,
            "selection": "observed-composite-samples-v1",
            "coverage": "declared_only",
            "source_content_authenticated": False,
            "timeline_digest": scenes.timeline.digest,
            "scene_config": scenes_config.to_dict(),
            "image_config": config.to_dict(),
            "detection_diagnostics": scenes.diagnostics.to_dict(),
            "replay_diagnostics": replay.diagnostics.to_dict(),
            "sample_count": len(observed),
            "scene_count": len(scenes.scenes),
            "image_count": len(slots),
            "unique_sample_count": unique_count,
            "scenes": [scene.to_dict() for scene in scenes.scenes],
            "images": rows,
            "transformed_pixels": pixels,
            "verification_pixels": pixels,
            "decode_passes": 2,
            "pillow_version": pillow_version,
            "pyav_version": av.__version__,
            "native_libraries": {key: list(value) for key, value in av.library_versions.items()},
            "jpeg_encoding": {"subsampling": 0, "progressive": False, "optimize": False},
        }
        with _managed_file(stage / "manifest.json", owned) as handle:
            writer = _Writer(handle, budget, config.max_manifest_bytes)
            for chunk in json.JSONEncoder(
                ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":")
            ).iterencode(document):
                writer.write(chunk.encode("utf-8"))
            writer.write(b"\n")
        result = NativeConcatSceneImageResult(
            target,
            len(scenes.scenes),
            len(slots),
            unique_count,
            budget.total,
            _hash_file(stage / "manifest.json", config.max_manifest_bytes),
            scenes.timeline.digest,
        )
        expected = {stage / row["file"]: (row["bytes"], row["sha256"]) for row in rows}
        expected[stage / "manifest.json"] = (writer.size, result.manifest_sha256)
        _reconcile(stage, stage_identity, owned, expected, budget)
        _check_identities(identities)
        publication_attempted = True
        _publish(stage, target)
        stage = None
        return result
    except BaseException as primary:
        if publication_attempted:
            primary.add_note(
                f"Publication did not acknowledge success; inspect destination {target}. "
                "A moved destination is never removed by failure cleanup."
            )
        if stage is not None:
            if stage_identity is None:
                primary.add_note(f"Staging identity unavailable; inspect {stage}; not removed.")
            else:
                _cleanup(stage, primary, owned, stage_identity)
        if isinstance(primary, (ConfigurationError, ScanError, OutputError)) or not isinstance(
            primary, Exception
        ):
            raise
        message = (
            f"composite scene image publication did not acknowledge success; inspect {target}"
            if publication_attempted
            else "composite scene image export failed before publication"
        )
        failure = OutputError(message)
        if publication_attempted:
            failure.add_note(
                f"Publication did not acknowledge success; inspect destination {target}. "
                "A moved destination is never removed by failure cleanup."
            )
        raise failure from primary


__all__ = ["NativeConcatSceneImageResult", "export_native_concat_scene_images"]
