# Full-pixel HSV and gradient changes

This independent measurement compares corresponding RGB pixels, not difference
hashes, mean colors or histogram marginals. Two pixels changing from red/cyan
to yellow/blue retain identical per-channel histograms but have circular hue
change `1/3`. A 4 x 4 checkerboard changing to vertical stripes retains every
2 x 2 cell's histogram but changes value by `1/2` and gradient by `3/8`.

```bash
frame-quorum native-change-measure local.mkv --edge-radius 1 --output-dir change-cache
frame-quorum native-change-replay change-cache/pixel-changes.fqm.jsonl \
  --weights 1 1 1 0 --threshold 0.3 --output-dir hsv-review
frame-quorum native-change-replay change-cache/pixel-changes.fqm.jsonl \
  --weights 0 0 0 1 --detector adaptive --window-radius 2 \
  --adaptive-ratio 3 --min-content 0.15 --output-dir edge-review
python examples/native_pixel_changes.py
```

Capture needs the installed optional `[video]` extra. Replay needs only the base
package: it does not import PyAV, open the recorded path or require the original
video. Output directories must be new and have existing parents. Cache output
is `pixel-changes.fqm.jsonl`; review output is `replay.json` plus `statistics.csv`.
CLI stdout follows successful publication; a stdout failure does not roll it back.

## Exact, original numerical definition

`PixelChangeConfig(edge_radius=1)` admits an integer radius from 1 to 4. There is
no resize, alignment, registration or pixel sampling. Each frame is converted to
8-bit RGB; alpha is discarded, not composited. ICC profiles and EXIF orientation
are not applied. Native decoding supplies the selected stream's decoded RGB;
codec/color conversion differences remain part of the acquisition environment.
**Adjacent selected frames must have identical actual width and height.**

For one pixel `(R,G,B)`, let `V=max(R,G,B)` and `C=V-min(R,G,B)`. If `C=0`,
`H=S=0`. Otherwise `S=floor(255*C/V)`, and the hue sector is:

```text
R is maximum: (G-B) modulo (6*C)
otherwise G is maximum: B-R+2*C
otherwise: R-G+4*C
H = floor(256*sector/C)  # integer circle [0,1535]
```

The stated branch order also defines ties. Given two pixels, their hue evidence
is `min(abs(Ha-Hb),1536-abs(Ha-Hb))*min(Sa,Sb)`. The minimum-saturation factor
suppresses undefined hue in gray/black pixels; it is fixed, not a replay option.
Saturation and value evidence are `abs(Sa-Sb)` and `abs(Va-Vb)`.

For radius `r`, define the forward gradient at `(x,y)` as:

```text
G(x,y) = abs(V(min(x+r,W-1),y)-V(x,y))
       + abs(V(x,min(y+r,H-1))-V(x,y))
```

The right/bottom boundary clamps to the final pixel. Gradient evidence is
`abs(Ga-Gb)`, not an orientation comparison or Canny edge map. Sum all four
integer evidences across the `P=W*H` pixel pairs. `PixelChange` stores these exact
`hue_sum`, `saturation_sum`, `value_sum`, `edge_sum` plus dimensions/configuration.
Its four normalized components divide respectively by
`P*768*255`, `P*255`, `P*255`, `P*510`, and are in `[0,1]`.

```python
from frame_quorum import PixelChangeWeights, measure_pixel_change

# left/right are borrowed Pillow images; they remain open and unchanged.
evidence = measure_pixel_change(left, right)
score = evidence.score(PixelChangeWeights(hue=1, saturation=1, value=1, edges=0))
```

Weights are finite built-in numbers in `[0,1_000_000]`, not booleans, with at least
one positive value. Scaling weights by their maximum before summation avoids
overflow/underflow normalization failures, including all-subnormal weights.
`PixelChangeDetectionConfig(value_only=True)` uses the exact V component,
regardless of the otherwise valid supplied weights. V is maximum RGB channel,
**not** mean brightness, luminance or perceptual luma.

## Capture once, replay sufficient evidence

`capture_native_pixel_changes(path, video, config=..., on_sample=..., limits=...,
pixel_limits=...)` shares native decoding/selection, summary measurement,
source fingerprint/full SHA256 checks and callback cleanup with existing native
measurement capture. Callbacks receive frozen `NativePixelChangeSample` values,
not borrowed images. They must return `None`; awaitables are not supported.
Trusted callbacks may have external side effects, which cannot be rolled back.

The first sample's `change` is **None**, never an invented comparison against
black. Each later sample describes the immediately preceding *selected* sample.
With stride, this spans skipped frames and does not localize a cut inside that
gap. `NativePixelChangeMeasurements` binds all original range/stride/stream
configuration, source digest and historical path label, producer diagnostics,
exact rational native PTS/index/generation coordinates, actual dimensions,
acquisition radius and the four sums. The first CSV row displays zero component
and weighted scores for convenience; the cache preserves absent evidence.

`analyze_native_pixel_changes` can change weights, V-only mode, threshold,
minimum scene samples and adaptive radius/ratio/absolute floor without decoding.
It cannot change acquisition radius, color/edge definitions, resolution, stride
or selected source/range. Those require new pixels and a new capture. No function
upgrades old summary or histogram measurements into this evidence.

Content accepts scores `>= threshold`, except the first sample is never a cut.
Adaptive reuses the existing centered-window kernel: a full `2*r+1` window is
required, excludes the initial dummy pair score from windows, compares the target
with its neighbors' mean, and requires both ratio and absolute content floor.
The shared policy uses a `1e-12` near-zero floor and ratio cap `1_000_000`.
It applies prior-scene and final-tail sample minimums deterministically; no
wall-clock minimum, online callback or flash-merging mode is implied. Reports
include all rejected candidates and reasons. Exact native PTS are retained;
unknown final endpoints stay unknown.

`NativePixelChangeReplayResult` recomputes scores and decisions from its typed
integer records on construction. This detects inconsistent reports, **not forged
measurements**. The wire digest is integrity/provenance bookkeeping, not media
authentication. Every replay explicitly reports `source_verified=false` and
`execution=cached_pixel_changes`; native splitting rejects this result type.
Neither capture-time hashes nor historical diagnostics certify the current file.

## Independent wire and resource boundaries

The new canonical JSONL schema uses kind `frame-quorum-native-pixel-changes`,
schema version 1 and measurement version `frame-quorum-circular-hsv-gradient-v1`.
Header, per-sample records and SHA256 completion footer use the existing strict
reader: exact keys, ASCII canonical bytes, no duplicate keys/nonfinite numbers,
39-digit numeric token limits checked before conversion, depth 16, bounded lines,
bounded sample count/file size, no trailing records. Rational coordinates retain
the original native integer/time-base bounds. Old summary/histogram wire bytes
and algorithms are unchanged. This is not a generic extensible metric registry.

`PixelChangeLimits` defaults:

| Admission | Default | Compiled ceiling / exact accounting |
| --- | --- | --- |
| `max_measurement_pixels` | 50,000,000 | 1,000,000,000; standalone pair charges `2P`; native charges initial snapshot `P`, then `2P` for each later pair: `(2*n-1)*P`, empty 0 |
| `max_pair_rgb_bytes` | 128 MiB | 512 MiB; two packed RGB snapshots charge `6P` |

Native admission checks `6*video.max_frame_pixels` against pair capacity before
decoding, and charges each pair before creating its Pillow summary image or
running its pixel kernel. Default native maximum 16,777,216 pixels requires
96 MiB for the two packed snapshots. The importer checks the same complete-work
formula in the header before parsing sample records. A standalone image may
have at most 67,108,864 pixels and must also satisfy the smaller configured limits.

The kernel retains only previous/current RGB snapshots and scalar accumulators,
not HSV/gradient maps or pixel-object lists. Four integer sums are retained per
sample. Conversion/summary scratch images, decoder buffers, callbacks, caller
images and Python/native allocation overhead are outside the packed-byte limit.
The work count measures snapshot/pair pixel visits, not exact RGB byte reads:
gradient neighbor reads, HSV arithmetic and old summaries add bounded constant
factors. This is neither a hard RSS limit nor a wall-time guarantee. Native
source byte, all decoded frame/pixel, stream/range and returned sample limits
remain separately enforced; skipped frames still consume decode limits.

`NativeMeasurementLimits` additionally defaults to 100,000 samples, 64 MiB cache,
16 KiB line and 128 MiB total output (compiled ceilings 1,000,000 samples,
256 MiB cache, 64 KiB line, 512 MiB output). Reports materialize bounded per-sample
tables; large inputs can fail configured persistence limits after capture.

Publication reuses owned staging files and no-replace directory rename on
supported Windows/Linux filesystems. Existing targets and unknown/replaced
staging entries are never deleted. Unsupported no-replace operations fail closed.
Primary/control exception precedence and cleanup residue reporting follow the
existing native bundle contract. Published-but-unacknowledged output may remain;
inspect the destination before retrying. Atomic visibility is not file/directory
fsync or power-loss durability. Local codecs are trusted native code, not a sandbox.

## Verification and remaining work

Independent tests cover primary/non-primary hues, wraparound, gray suppression,
all radii, border gradients, histogram-identical color/layout changes, extreme
weights, admission-before-allocation, malformed/rechecksummed wire, canonical old
goldens and result consistency. Real FFV1 VFR tests reopen every tiny generated
frame independently and compare its RGB and rational PTS with prescribed pixels;
expected sums are calculated independently of the implementation kernel.
A fresh process forbids all `av` imports while replaying a source-deleted cache.

The frozen reference's [content detector](https://github.com/Breakthrough/PySceneDetect/blob/24953b0bf76af17c450bc143d330eea48fc5e276/scenedetect/detectors/content_detector.py)
provides weighted HSV and Canny-based edge controls, and its
[adaptive detector](https://github.com/Breakthrough/PySceneDetect/blob/24953b0bf76af17c450bc143d330eea48fc5e276/scenedetect/detectors/adaptive_detector.py)
adds local contrast filtering. This implementation is independently authored,
with explicitly different numerical definitions, not a numerical reproduction.
Canny/dilation, online detection, flash merging, motion compensation, learned
models, calibrated real-video accuracy/throughput evaluation and the broader
[whole-repository ledger](parity-detection.md) remain open.
