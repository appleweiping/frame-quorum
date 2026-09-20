# Composite video scene stills

`export_native_concat_scene_images` detects scene boundaries across an exact
caller-declared sequence of local clips and publishes real PNG or JPEG stills.
Install `frame-quorum[video]` for PyAV. This Python API has no composite CLI or
clip-JSON parser. Run `python examples/native_concat_scene_images.py` for a
fully offline, generated mixed-clock example.

```python
from fractions import Fraction
from frame_quorum import (
    DetectionConfig,
    NativeConcatClip,
    NativeConcatSceneConfig,
    NativeSceneImageConfig,
    export_native_concat_scene_images,
)

clips = (
    NativeConcatClip("first.mkv", start=5, end=Fraction(27, 5)),
    NativeConcatClip("second.nut", start=1, end=Fraction(13, 10)),
)
scenes = NativeConcatSceneConfig(
    detectors=(DetectionConfig(detector="luminance", threshold=0.5),),
)
images = NativeSceneImageConfig(images_per_scene=3, sample_margin=1, image_format="png")
result = export_native_concat_scene_images(clips, "scene-stills", scenes, images)
print(result.to_dict())
```

The output must be a **new** directory in an existing parent. It contains only
`manifest.json` and `scene-000001-image-000001.png`-style files. Image names
never include source paths. The manifest is canonical ASCII-safe UTF-8 JSON:
sorted compact keys, exact rational pairs, and one final newline. It records
every still's composite sample index/time, occurrence, native PTS/time base,
native decode/sample index, dimensions, RGB digests, encoded size and file
hash. `NativeConcatSceneImageResult` reports the directory, scene and image
counts, unique sampled positions, aggregate output bytes, manifest SHA-256 and
timeline digest.

Selection divides each **observed sample partition** `[start, end)` into
equally spaced slots after a symmetric sample-count margin. A short scene can
repeat the same sample in multiple image files. Global stride is applied by
the composite decoder before scene detection or image-slot selection. Native
indices restart for each occurrence; repeated paths are separate occurrences.
A scene with samples but zero time duration still has stills. Declared spans
can include a physically unobserved interval; that interval receives no
invented images. A completely unobserved range is rejected.

The first decode pass retains bounded measurements, native identity and an RGB
hash for **every** returned sample. It constructs the same validated scene
partition as `detect_native_concat_scenes`. After selecting image slots, the
second sequential pass compares every returned frame's composite/native
coordinates, dimensions and RGB hash, not just selected frames. Each encoded
image is reopened to verify format, size, mode and decoded pixels. PNG decoded
RGB must match exactly; JPEG is lossy and instead records its decoded digest.
Per-pass source decoding uses the existing independent concat work limits;
image transformations, verification pixels and written bytes use the existing
`NativeSceneImageConfig` ceilings. No image corpus is held in memory, no FPS
seek is used, and no final-frame duration is guessed.

Source files must be local, regular and nonsymlink. Each unique source path's
device/inode/size/mtime identity is rechecked across both passes, and the
decoder's mutation checks remain active. This is **not** cryptographic source
authentication or a defense against adversarial swap-and-restore. The manifest
says `source_content_authenticated: false` and `coverage: "declared_only"`:
clip endpoints are declaration coordinates, not observed physical durations.
Neither a declared span nor this export proves continuous source coverage.

An owned stage directory is reconciled for exact inventory, identity, size,
hash and byte budget, then atomically published without replacing an existing
destination on Windows/Linux. If publication succeeds but its acknowledgement
fails, the complete destination is retained and the error asks for manual
inspection. Foreign/racing targets are never removed. There is no network
access, model download, background encoder or mutable global cache.

This closes declared-composition **scene stills** only. Composite CLI, HTML,
video/audio split, cache/replay and broader algorithm/backend parity remain
outside this API. For declaration, completion and budget details, see
[composite scenes](native-concat-scenes.md) and
[single-source scene stills](native-scene-images.md).
