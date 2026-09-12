# Native observed-sample scene images

`export_native_scene_images` detects scenes, exports PNG/JPEG stills from actual
observed samples, and publishes one new directory containing the images and
`manifest.json`. It uses the existing optional PyAV decoder and Pillow; there is
no new dependency, external encoder process, model download or network input.

Run `python examples/native_scene_images.py` for a fully offline demonstration:
it generates a lossless variable-frame-rate source, selects four stills across
two scenes, and independently checks sample times, hashes and resized PNG pixels.
The example removes only its own temporary directory on exit. Use the CLI below
with your own source and a new output directory to retain the images.

```python
from frame_quorum import (
    NativeSceneConfig,
    NativeSceneImageConfig,
    NativeVideoConfig,
    export_native_scene_images,
)

result = export_native_scene_images(
    "input.mkv",
    "new-scene-images",  # Must not exist; its parent must already exist.
    NativeSceneConfig(video=NativeVideoConfig(max_frames=10_000)),
    NativeSceneImageConfig(images_per_scene=3, sample_margin=1, width=640),
)
print(result.to_dict())
```

The existing [native input and resource contract](native-video.md) and
[scene detection semantics](native-scenes.md) apply. The function accepts a
source and detection configuration, not an arbitrary caller-supplied scene
result as proof that the source contains those scenes. No observed samples is
an error; there is no empty successful image bundle.

## Command line

```console
frame-quorum native-scene-images input.mkv --output-dir new-scene-images --images-per-scene 3 --sample-margin 1 --width 640
frame-quorum native-scene-images input.mkv --output-dir new-jpeg-images --image-format jpeg --jpeg-quality 90 --scale 1/2
```

`INPUT` and `--output-dir`/`-o` are required. The output directory must be new and
its parent must exist. There is no `--force`, output-directory creation/merge,
external scene-report import or separate preliminary detection invocation.
The command calls the verified two-pass exporter once and prints one sorted,
ASCII-safe JSON result line only after the bundle is published.

Native range, stream and detector arguments have the same meanings/defaults as
`native-scenes`. In particular the CLI defaults to the adaptive detector, while
the Python API's default `NativeSceneConfig` uses its existing content detector.
`--detectors` accepts the existing ensemble choices. `--frame-step` sampling and
nonzero video-stream ordinals are supported; no media origin, FPS or invented
final endpoint is required. Unknown/count-limited scene tails remain explicit
in the manifest, not converted into a claim of complete-source observation.

Image arguments map directly to the API: `--images-per-scene`, `--sample-margin`,
`--image-format {png,jpeg}`, `--png-compression`, `--jpeg-quality`, `--width`,
`--height`, `--scale`, and
`--interpolation {nearest,bilinear,bicubic,lanczos}`. Scale accepts bounded exact
integer, decimal and fraction strings (for example `2`, `0.5`, `1/2`), not an
intermediate float. Nonpositive/out-of-range values and oversized exponent/text
inputs are rejected before decoding or staging. Scale cannot be combined with
width/height; width and height may be supplied together.

The independent pixel budgets have deliberately distinct flags:

| CLI flag | Meaning |
|---|---|
| `--max-frame-pixels` | Pixels per decoded source frame |
| `--max-total-pixels` | Aggregate decoded pixels **per native pass**, unchanged from other native commands |
| `--max-image-pixels` | Pixels per transformed output image |
| `--max-image-total-pixels` | Aggregate transformed image pixels, mapping to `NativeSceneImageConfig.max_total_pixels` |
| `--max-verification-pixels` | Aggregate image verification pixels |

Other output limits are `--max-scenes`, `--max-images`, `--max-image-bytes`,
`--max-output-bytes` (including the manifest), and `--max-manifest-bytes`.
Defaults and ceilings are listed below. Existing native input/decode limits
remain available and unchanged; image-slot limits are not source-frame limits.

Malformed arguments and expected configuration/source/output failures exit with
status 2. Before publication, failure cleanup follows the ownership policy below.
If writing or flushing the success JSON to stdout fails after publication, the command can
return status 2 while the completed bundle remains on disk: inspect the requested
output directory. A console error never rolls back a published image bundle.
Short writes or invalid write counts are failures; stdout may contain a partial
JSON result in that case and is not an atomic publication channel.
Control exceptions retain the existing CLI policy rather than becoming a fake
success result.

## Verification record

The final library and CLI passed the complete Windows Python 3.11.2 suite:
**1,999 passed, three existing symlink-privilege skips**, 182.91 seconds,
**98.1197%** combined statement/branch coverage. Independent real Linux Python
3.12.3 verification passed **2,002 tests with no skips**, 96.20 seconds,
**98.1531%** coverage. Both retained the original 95% gate and promoted
RuntimeWarning and ResourceWarning to errors. All 151 delivery-file hashes were
unchanged across each full run. PyAV 18.1.0 and Pillow 12.3.0 were used; the Linux
environment was rebuilt from existing hash-checked offline cache wheels.

All **324 new cases** are included: 203 image-core, 34 real-codec image, and
87 CLI cases. The new image module covers all 306 statements and 116 branches,
with no exclusions. A separate Linux review ran the 87 CLI tests plus nine
independent write/control probes: **96 passed**. These nine are extra reviewer
probes, not additional repository tests.

Retained failures include missing API/command REDs, stale staged-image hashes
after a same-pixel re-encode, and incomplete stdout acknowledgement (seven RED
failures before the write-count/flush correction). The Linux review's first
probe collection failed because its own helper import was wrong; corrected
probes and the full run then passed. Vanished temporary Linux environments were
replaced by a separate persistent cache environment, not by relaxing tests.

The full runs precede only final README/changelog/acceptance documentation and
the standalone generated example, not changes to library or test code. The
example independently checks its exact PNG pixels and native times. Package,
installed-example and hosted checks are separate gates on the final tree.
This is synthetic correctness/resource evidence, not calibrated real-video
accuracy, throughput or whole-reference parity.

## Sample selection, not guessed seek times

`images_per_scene` defaults to 3, with `sample_margin=1`. Margins count returned
samples after configured range/stride filtering, not seconds, native decode
positions, global frame numbers or nominal-FPS frames.

For each nonempty half-open scene sample interval `[s,e)`, let
`m=min(sample_margin,(e-s-1)//2)`, `a=s+m`, and `b=e-1-m`.

- One image selects `floor((a+b)/2)`, the lower midpoint on a tie.
- For `k>1`, image slot `j` selects `a+floor(j*(b-a)/(k-1))`, for `j=0..k-1`.

Exactly `k` images are written per scene. Short scenes intentionally repeat
samples: the default selects relative positions `[1,3,5]` from seven samples,
`[1,1,2]` from four, `[0,0,1]` from two, and `[0,0,0]` from one. Fixed output
filenames use one-based scene/slot numbers, for example
`scene-000001-image-000001.png`. Manifest ordinals and source sample indices
remain zero-based. `unique_sample_count` distinguishes selected samples from
the total number of image slots; repeated rows retain the same source identity.

Equal PTS values do not collapse distinct samples. Raw integer PTS, exact
rational time base and derived presentation time remain distinct fields, along
with sample/decode/generation indices. Native indices are not advertised as
global source frame numbers.

Unknown scene ends stay null. EOF does not imply a known final-frame duration;
frame/decode limits remain explicit count-limited observations. A configured
end becomes the scene end only under the existing observed `range_end` policy.
Stills require neither a caller-supplied terminal end nor FPS interpolation.
A successful bundle means every planned image of the observed scene partition
was exported, not that the entire source was observed.

## Resize and encoding

The default format is `png`, with `png_compression=6` (integer 0 through 9).
PNG output is fully decoded and its RGB digest must match the transformed RGB
snapshot exactly. `image_format="jpeg"` uses `jpeg_quality=95` (integer 0 through
100), subsampling 0, progressive false and optimization false. JPEG quality 100
is still not a lossless guarantee: verification checks full decoding, RGB mode,
format and dimensions, recording the decoded RGB digest separately.

Optional integer `width`/`height` specify exact pixels. A single dimension
preserves stored-pixel aspect ratio using integer-floor arithmetic; two
dimensions specify an exact size and may change aspect ratio. Alternatively,
`scale` accepts a positive exact `Fraction` or integer, with bounded int64
numerator/denominator. Scale and width/height are mutually exclusive. Both scaled
dimensions are floored; zero-sized or oversized results are rejected before
resizing. Fractions are serialized as numerator/denominator pairs, never floats.

`interpolation` is one of Pillow's `nearest`, `bilinear`, `bicubic` (default), or
`lanczos`. Unchanged dimensions bypass resizing. Source EXIF/ICC data is not
copied. There is no additional SAR correction, crop, orientation transform,
color-management pipeline or promise of OpenCV-equivalent interpolation pixels.
Encoded hashes can differ with codec/library versions; versions are recorded.

## Two-pass verification and bounded output

The first pass retains native scene measurements plus each returned frame's
dimensions and RGB SHA-256, not the RGB snapshots. After offline scene analysis,
one sequential replay checks every observed sample, including unselected ones:
raw PTS, rational base, all indices, dimensions and RGB digest must match.
Metadata and complete closed diagnostics must also agree. Selected images are
encoded and verified incrementally; at most the current source, transformed and
verification images are needed, not a complete retained scene's RGB images.

Each native pass separately uses the supplied `NativeVideoConfig` limits.
Maximum native decode work is therefore twice the configured per-pass frame and
pixel limits, plus the separately bounded output transformation/verification
work. This is not a single decoder budget silently reset between passes.
Measurement storage remains O(samples * detectors), with O(samples) additional
dimension/digest records and O(image slots) output records.

The following defaults are also hard ceilings; callers may lower them:

| Limit | Ceiling |
|---|---:|
| Scenes | 1,000 |
| Images per scene | 100 |
| Total image slots | 10,000 |
| Pixels per output image | 16,777,216 |
| Aggregate transformed pixels, counting repeated slots | 1,000,000,000 |
| Aggregate verification pixels, counting repeated slots | 1,000,000,000 |
| Encoded bytes per image | 67,108,864 |
| Total output bytes, including manifest | 1,000,000,000 |
| Manifest bytes | 16,777,216 |

`sample_margin` is a strict integer from 0 to 1,000,000. Numeric options reject
booleans, floats and strings. Width/height and their resulting pixel product are
bounded. Output counts and pixel totals are checked before replay/staging.
Encoder writes go through a bounded file writer, not an unbounded `BytesIO`.
These Python/native-boundary controls do not promise hard native CPU, RSS or
wall-clock isolation; an encoder can allocate internal buffers before a write.

## Manifest and publication

The manifest retains detector/image configurations, source metadata, unchanged
scene endpoints, both pass diagnostics and decoder/Pillow/native-library
versions. Every slot records source indices, raw PTS/rational times, original and
output dimensions, source/transformed/decoded RGB digests, encoded size and file
SHA-256. The result contains scene/image/unique-sample counts, total bytes and
the manifest's SHA-256. Exact large integers require a lossless integer-aware
JSON reader; binary64-only consumers must not silently round PTS provenance.

Source fingerprints are checked across the operation, and each decoder keeps
its existing open-file checks. Agreement establishes matched observed decoded
samples, not authenticated media origin, complete container-byte identity, or
protection against every hostile concurrent filesystem race.

Output uses the existing owned staging and atomic no-replace directory
publication primitives on Windows and Linux. Before publication, the owned
inventory, file/directory identities, actual byte sizes and hashes (including
the manifest) are reconciled again. Files, Pillow images and decoders are closed
before successful publication. Existing output is never overwritten.

Errors and control exceptions do not return a partial-success result. Cleanup
removes only known owned staging files; substituted/unknown entries are left for
inspection with a cleanup error. If publication moved the directory but did not
acknowledge success, the error directs the caller to inspect the destination;
failure cleanup never deletes the moved destination to simulate rollback.

## Reference gap remains open

This is an original observed-sample subset of the frozen
[PySceneDetect per-scene image workflow](https://github.com/Breakthrough/PySceneDetect/blob/24953b0bf76af17c450bc143d330eea48fc5e276/scenedetect/output/image.py).
It does not reproduce the reference's seconds-based target selection and seeks,
frame/seconds/timecode margin forms, OpenCV interpolation or encoded bytes.
WebP, filename templates, threaded image output, SAR/crop controls, HTML
overview, broader source/backend support and cross-backend image interoperability
remain open. This addition does not close whole-reference parity.
