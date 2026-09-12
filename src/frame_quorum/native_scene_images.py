"""Bounded observed-sample scene images, verified across two native decode passes.

Selection is by actual sampled positions, never FPS-derived seek time. Encoded
images and a provenance manifest are published together without replacement.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from tempfile import mkdtemp
from typing import Any, BinaryIO, cast

from PIL import Image
from PIL import __version__ as pillow_version

from .errors import ConfigurationError, OutputError, ScanError
from .native_scenes import (
    NativeSceneConfig,
    NativeSceneResult,
    NativeSceneSample,
    _analyze_native_samples,
    _measure_native_sample,
)
from .native_splitting import (
    _ByteBudget,
    _cleanup,
    _close_all,
    _hash_file,
    _identity,
    _managed_file,
    _publish,
    _target_path,
    _Writer,
)
from .native_video import (
    NativeVideoFrame,
    NativeVideoStream,
    _fraction,
    _integer,
    _load_av,
    _pair,
    _source_path,
)

_RESAMPLING = {
    "nearest": Image.Resampling.NEAREST,
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
    "lanczos": Image.Resampling.LANCZOS,
}


@dataclass(frozen=True, slots=True)
class NativeSceneImageConfig:
    images_per_scene: int = 3
    sample_margin: int = 1
    image_format: str = "png"
    png_compression: int = 6
    jpeg_quality: int = 95
    width: int | None = None
    height: int | None = None
    scale: Fraction | None = None
    interpolation: str = "bicubic"
    max_scenes: int = 1000
    max_images: int = 10_000
    max_image_pixels: int = 16_777_216
    max_total_pixels: int = 1_000_000_000
    max_verification_pixels: int = 1_000_000_000
    max_image_bytes: int = 64 * 1024 * 1024
    max_output_bytes: int = 1_000_000_000
    max_manifest_bytes: int = 16 * 1024 * 1024

    def __post_init__(self) -> None:
        for name, minimum, maximum in (
            ("images_per_scene", 1, 100),
            ("sample_margin", 0, 1_000_000),
            ("png_compression", 0, 9),
            ("jpeg_quality", 0, 100),
            ("max_scenes", 1, 1000),
            ("max_images", 1, 10_000),
            ("max_image_pixels", 1, 16_777_216),
            ("max_total_pixels", 1, 1_000_000_000),
            ("max_verification_pixels", 1, 1_000_000_000),
            ("max_image_bytes", 1, 64 * 1024 * 1024),
            ("max_output_bytes", 1, 1_000_000_000),
            ("max_manifest_bytes", 1, 16 * 1024 * 1024),
        ):
            _integer(getattr(self, name), name, minimum, maximum)
        if type(self.image_format) is not str or self.image_format not in ("png", "jpeg"):
            raise ConfigurationError("image_format must be png or jpeg")
        if type(self.interpolation) is not str or self.interpolation not in _RESAMPLING:
            raise ConfigurationError("interpolation must be nearest, bilinear, bicubic or lanczos")
        for name in ("width", "height"):
            if getattr(self, name) is not None:
                _integer(getattr(self, name), name, 1, self.max_image_pixels)
        if self.scale is not None:
            object.__setattr__(self, "scale", _fraction(self.scale, "scale", positive=True))
            if self.width is not None or self.height is not None:
                raise ConfigurationError("scale is mutually exclusive with width/height")
        if (
            self.width is not None
            and self.height is not None
            and self.width * self.height > self.max_image_pixels
        ):
            raise ConfigurationError("width/height exceed output image pixel limit")
        if self.images_per_scene > self.max_images:
            raise ConfigurationError("images_per_scene exceeds total image limit")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["scale"] = _pair(self.scale)
        return result


@dataclass(frozen=True, slots=True)
class NativeSceneImageResult:
    output_dir: Path
    scene_count: int
    image_count: int
    unique_sample_count: int
    total_output_bytes: int
    manifest_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.output_dir, Path):
            raise ConfigurationError("output_dir must be a pathlib.Path")
        _integer(self.scene_count, "scene_count", 1, 1000)
        _integer(self.image_count, "image_count", self.scene_count, 10_000)
        _integer(self.unique_sample_count, "unique_sample_count", self.scene_count, self.image_count)
        _integer(self.total_output_bytes, "total_output_bytes", 1, 1_000_000_000)
        if self.image_count % self.scene_count:
            raise ConfigurationError("each scene must have the same number of image slots")
        if self.image_count // self.scene_count > 100:
            raise ConfigurationError("image count exceeds per-scene ceiling")
        if (
            type(self.manifest_sha256) is not str
            or len(self.manifest_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.manifest_sha256)
        ):
            raise ConfigurationError("manifest_sha256 must be a SHA-256 hex digest")

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "output_dir": self.output_dir.as_posix()}


@dataclass(frozen=True, slots=True)
class _Observed:
    sample: NativeSceneSample
    width: int
    height: int
    rgb_sha256: str


@dataclass(frozen=True, slots=True)
class _Slot:
    scene_ordinal: int
    image_index: int
    position: int
    width: int
    height: int


def _positions(start: int, end: int, count: int, margin: int) -> tuple[int, ...]:
    """Internal planner over a validated nonempty half-open sample interval."""
    padding = min(margin, (end - start - 1) // 2)
    first, last = start + padding, end - 1 - padding
    if count == 1:
        return ((first + last) // 2,)
    return tuple(first + index * (last - first) // (count - 1) for index in range(count))


def _dimensions(width: int, height: int, config: NativeSceneImageConfig) -> tuple[int, int]:
    if config.width is not None and config.height is not None:
        output_width, output_height = config.width, config.height
    elif config.width is not None:
        output_width, output_height = config.width, height * config.width // width
    elif config.height is not None:
        output_width, output_height = width * config.height // height, config.height
    elif config.scale is not None:
        output_width, output_height = int(width * config.scale), int(height * config.scale)
    else:
        output_width, output_height = width, height
    if output_width < 1 or output_height < 1 or output_width * output_height > config.max_image_pixels:
        raise ConfigurationError("resized image is empty or exceeds output image pixel limit")
    return output_width, output_height


def _plan(
    scenes: NativeSceneResult, observed: tuple[_Observed, ...], config: NativeSceneImageConfig
) -> tuple[_Slot, ...]:
    if not scenes.scenes:
        raise ScanError("native scene images require at least one observed sample")
    if len(scenes.scenes) > config.max_scenes:
        raise ConfigurationError("native scene images exceed scene count limit")
    if len(scenes.scenes) * config.images_per_scene > config.max_images:
        raise ConfigurationError("native scene images exceed total image count limit")
    result = []
    pixels = 0
    for scene in scenes.scenes:
        positions = _positions(
            scene.start_position, scene.end_position, config.images_per_scene, config.sample_margin
        )
        for index, position in enumerate(positions):
            source = observed[position]
            width, height = _dimensions(source.width, source.height, config)
            pixels += width * height
            if pixels > config.max_total_pixels:
                raise ConfigurationError("native scene images exceed aggregate transformed pixel limit")
            if pixels > config.max_verification_pixels:
                raise ConfigurationError("native scene images exceed aggregate verification pixel limit")
            result.append(_Slot(scene.ordinal, index, position, width, height))
    return tuple(result)


def _fingerprint(path: Path) -> tuple[int, int, int, int]:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise ScanError("native scene image source must remain a regular nonsymlink file")
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns


def _check_source(path: Path, expected: tuple[int, int, int, int]) -> None:
    if _fingerprint(path) != expected:
        raise ScanError("native scene image source changed between detection and replay")


def _capture(path: Path, config: NativeSceneConfig) -> tuple[NativeSceneResult, tuple[_Observed, ...]]:
    observed = []
    with NativeVideoStream(path, config.video) as stream:
        metadata = stream.metadata
        for frame in stream:
            observed.append(
                _Observed(
                    _measure_native_sample(frame),
                    frame.width,
                    frame.height,
                    hashlib.sha256(frame.rgb).hexdigest(),
                )
            )
            del frame
    records = tuple(observed)
    result = _analyze_native_samples(
        config, metadata, stream.diagnostics, tuple(row.sample for row in records)
    )
    return result, records


def _matches(frame: NativeVideoFrame, original: _Observed) -> bool:
    sample = original.sample
    return (
        (frame.pts, frame.time_base, frame.decode_index, frame.sample_index, frame.generation)
        == (sample.pts, sample.time_base, sample.decode_index, sample.sample_index, sample.generation)
        and (frame.width, frame.height) == (original.width, original.height)
        and hashlib.sha256(frame.rgb).hexdigest() == original.rgb_sha256
    )


@contextmanager
def _managed_image(image: Image.Image) -> Iterator[Image.Image]:
    primary: BaseException | None = None
    try:
        yield image
    except BaseException as error:
        primary = error
        raise
    finally:
        _close_all((("image.close", image),), primary)


def _verify_image(path: Path, config: NativeSceneImageConfig, size: tuple[int, int], digest: str) -> str:
    with _managed_file(path) as handle, _managed_image(Image.open(handle)) as image:
        if image.format != config.image_format.upper() or image.mode != "RGB" or image.size != size:
            raise OutputError("native scene image format, mode or dimensions verification failed")
        image.load()
        decoded = hashlib.sha256(image.tobytes()).hexdigest()
        if config.image_format == "png" and decoded != digest:
            raise OutputError("native scene PNG RGB verification failed")
        return decoded


def _encode_image(
    image: Image.Image,
    path: Path,
    config: NativeSceneImageConfig,
    budget: _ByteBudget,
    owned: dict[Path, tuple[int, int]],
) -> tuple[str, str, int, str]:
    digest = hashlib.sha256(image.tobytes()).hexdigest()
    parameters = (
        {"compress_level": config.png_compression, "optimize": False}
        if config.image_format == "png"
        else {"quality": config.jpeg_quality, "subsampling": 0, "progressive": False, "optimize": False}
    )
    with _managed_file(path, owned) as handle:
        writer = _Writer(handle, budget, config.max_image_bytes)
        image.save(cast(BinaryIO, writer), format=config.image_format.upper(), **parameters)
        if writer.failed:
            raise OutputError("native scene image encoder swallowed an output writer failure")
    # Hashing checks encoded bytes again before asking Pillow to decode them.
    file_hash = _hash_file(path, config.max_image_bytes)
    decoded = _verify_image(path, config, image.size, digest)
    return digest, decoded, writer.size, file_hash


def _save_slot(
    frame: NativeVideoFrame,
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
    return {
        "scene_ordinal": slot.scene_ordinal,
        "image_index": slot.image_index,
        "source_sample_index": frame.sample_index,
        "source_decode_index": frame.decode_index,
        "source_generation": frame.generation,
        "source_pts": frame.pts,
        "source_time_base": _pair(frame.time_base),
        "source_time": _pair(frame.presentation_time),
        "source_width": frame.width,
        "source_height": frame.height,
        "width": slot.width,
        "height": slot.height,
        "source_rgb_sha256": original.rgb_sha256,
        "transformed_rgb_sha256": transformed,
        "decoded_rgb_sha256": decoded,
        "file": path.name,
        "bytes": size,
        "sha256": file_hash,
    }


def _staged_digest(path: Path, identity: tuple[int, int], expected_size: int) -> str:
    """Reconcile the actual owned file, without claiming hostile-FS atomicity."""
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or _identity(before) != identity or before.st_size != expected_size:
        raise OutputError("staged scene image file identity or size changed")
    digest = hashlib.sha256()
    with _managed_file(path) as handle:
        opened = os.fstat(handle.fileno())
        if _identity(opened) != identity or opened.st_size != expected_size:
            raise OutputError("staged scene image file changed while opening")
        total = 0
        while chunk := handle.read(min(1024 * 1024, expected_size - total + 1)):
            total += len(chunk)
            if total > expected_size:
                raise OutputError("staged scene image file grew during reconciliation")
            digest.update(chunk)
        after = os.fstat(handle.fileno())
        current = path.lstat()
        if (
            total != expected_size
            or not stat.S_ISREG(current.st_mode)
            or _identity(current) != identity
            or current.st_size != expected_size
            or current.st_mtime_ns != before.st_mtime_ns
            or after.st_mtime_ns != before.st_mtime_ns
            or after.st_size != expected_size
        ):
            raise OutputError("staged scene image file changed during reconciliation")
    return digest.hexdigest()


def _reconcile(
    stage: Path,
    identity: tuple[int, int],
    owned: dict[Path, tuple[int, int]],
    expected: dict[Path, tuple[int, str]],
    budget: _ByteBudget,
) -> None:
    before = stage.lstat()
    if not stat.S_ISDIR(before.st_mode) or _identity(before) != identity:
        raise OutputError("staged scene image directory identity changed")
    if expected.keys() != owned.keys():
        raise OutputError("staged scene image inventory differs from owned files")
    count = 0
    for path in stage.iterdir():
        count += 1
        if count > len(owned) or path not in owned:
            raise OutputError("staged scene image directory contains an unowned entry")
    if count != len(owned):
        raise OutputError("staged scene image directory is missing an owned file")
    total = 0
    for path, (size, digest) in expected.items():
        total += size
        if total > budget.maximum or _staged_digest(path, owned[path], size) != digest:
            raise OutputError("staged scene image byte budget or content hash changed")
    after = stage.lstat()
    if total != budget.total or not stat.S_ISDIR(after.st_mode) or _identity(after) != identity:
        raise OutputError("staged scene image aggregate bytes or directory identity changed")


def export_native_scene_images(
    source: str | Path,
    output_dir: str | Path,
    scene_config: NativeSceneConfig | None = None,
    image_config: NativeSceneImageConfig | None = None,
) -> NativeSceneImageResult:
    """Detect, replay and publish PNG/JPEG slots from observed scene samples.

    Each pass has the supplied native limits. EOF does not invent a terminal
    duration; count-limited observations remain explicitly count-limited.
    """
    scenes_config = NativeSceneConfig() if scene_config is None else scene_config
    config = NativeSceneImageConfig() if image_config is None else image_config
    if type(scenes_config) is not NativeSceneConfig or type(config) is not NativeSceneImageConfig:
        raise ConfigurationError("configs must be NativeSceneConfig and NativeSceneImageConfig")
    scenes_config.__post_init__()
    scenes_config.video.__post_init__()
    config.__post_init__()
    target = _target_path(output_dir)
    path = _source_path(source)
    budget = _ByteBudget(config.max_output_bytes)
    stage: Path | None = None
    stage_identity: tuple[int, int] | None = None
    owned: dict[Path, tuple[int, int]] = {}
    publication_attempted = False
    try:
        fingerprint = _fingerprint(path)
        scenes, observed = _capture(path, scenes_config)
        _check_source(path, fingerprint)
        slots = _plan(scenes, observed, config)
        av = _load_av()
        rows = []
        slot_index = 0
        count = 0
        with NativeVideoStream(path, scenes_config.video) as replay:
            if replay.metadata != scenes.metadata:
                raise ScanError("native scene image replay metadata differs from detection")
            _check_source(path, fingerprint)
            stage = Path(mkdtemp(prefix=".frame-quorum-scene-images-", dir=target.parent))
            stage_identity = _identity(stage.lstat())
            for frame in replay:
                if count >= len(observed) or not _matches(frame, observed[count]):
                    raise ScanError("native scene image replay sample differs from detection")
                while slot_index < len(slots) and slots[slot_index].position == count:
                    rows.append(
                        _save_slot(frame, observed[count], slots[slot_index], stage, config, budget, owned)
                    )
                    slot_index += 1
                count += 1
                del frame
        if count != len(observed) or slot_index != len(slots) or replay.diagnostics != scenes.diagnostics:
            raise ScanError("native scene image replay completion differs from detection")
        if replay.metadata != scenes.metadata:
            raise ScanError("native scene image replay final metadata differs from detection")
        _check_source(path, fingerprint)
        unique_count = len({slot.position for slot in slots})
        document = {
            "kind": "frame-quorum-native-scene-images",
            "schema_version": 1,
            "selection": "observed-sample-endpoints-v1",
            "source": scenes.metadata.to_dict(),
            "scene_config": scenes_config.to_dict(),
            "image_config": config.to_dict(),
            "detection_diagnostics": scenes.diagnostics.to_dict(),
            "replay_diagnostics": replay.diagnostics.to_dict(),
            "decode_passes": 2,
            "verified_observed_sample_count": len(observed),
            "scene_count": len(scenes.scenes),
            "image_count": len(slots),
            "unique_sample_count": unique_count,
            "scenes": [scene.to_dict() for scene in scenes.scenes],
            "images": rows,
            "transformed_pixels": sum(slot.width * slot.height for slot in slots),
            "verification_pixels": sum(slot.width * slot.height for slot in slots),
            "pillow_version": pillow_version,
            "pyav_version": av.__version__,
            "native_libraries": {key: list(value) for key, value in av.library_versions.items()},
            "jpeg_encoding": {"subsampling": 0, "progressive": False, "optimize": False},
        }
        with _managed_file(stage / "manifest.json", owned) as handle:
            writer = _Writer(handle, budget, config.max_manifest_bytes)
            for chunk in json.JSONEncoder(ensure_ascii=True, allow_nan=False, sort_keys=True).iterencode(
                document
            ):
                writer.write(chunk.encode("utf-8"))
        result = NativeSceneImageResult(
            target,
            len(scenes.scenes),
            len(slots),
            unique_count,
            budget.total,
            _hash_file(stage / "manifest.json", config.max_manifest_bytes),
        )
        expected = {stage / row["file"]: (row["bytes"], row["sha256"]) for row in rows}
        expected[stage / "manifest.json"] = (writer.size, result.manifest_sha256)
        _reconcile(stage, stage_identity, owned, expected, budget)
        _check_source(path, fingerprint)
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
            f"native scene image publication did not acknowledge success; inspect {target}"
            if publication_attempted
            else "native scene image export failed before publication"
        )
        raise OutputError(message) from primary


__all__ = ["NativeSceneImageConfig", "NativeSceneImageResult", "export_native_scene_images"]
