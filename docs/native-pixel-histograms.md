# Full-pixel RGB histograms and offline scene tuning

The existing content detector uses a difference hash, mean RGB and luminance.
Those summaries can miss distribution changes: a 16 x 16 image with an upper
black half and lower gray-254 half has the same content distance as a uniform
gray-127 image: **zero**. Their complete RGB histograms are disjoint. This
module measures that additional evidence instead of inferring it from summaries.

```bash
frame-quorum native-histogram-measure local.mkv --bins 32 --rows 2 --columns 2 \
  --output-dir histogram-cache
frame-quorum native-histogram-replay histogram-cache/histograms.fqm.jsonl \
  --mode global --threshold 0.5 --output-dir global-review
frame-quorum native-histogram-replay histogram-cache/histograms.fqm.jsonl \
  --mode spatial --threshold 0.5 --min-scene-samples 2 --output-dir spatial-review
python examples/native_pixel_histograms.py
```

Capture requires the installed optional `[video]` extra. Replay uses the base
package without opening any recorded media path or importing PyAV; it works after
the source has been removed. Both output directories must be new, with existing
parents. Capture publishes `histograms.fqm.jsonl`; replay publishes `replay.json`
and `statistics.csv`. CLI stdout is a short summary **after** publication; a
stdout failure does not remove published files.

## Measurement definition

`PixelHistogramConfig(bins=32, rows=2, columns=2)` specifies a fixed acquisition
representation. Bins may be 8, 16, 32, 64, 128 or 256; rows and columns may each
be 1 or 2. There is no implicit resizing, thumbnailing, subsampling or spatial
alignment in this histogram measurement. All pixels in each selected native
RGB snapshot participate once.

For width `W`, height `H`, cell `(r, c)` includes the half-open rectangle:

```text
[floor(c*W/columns), floor((c+1)*W/columns))
    x [floor(r*H/rows), floor((r+1)*H/rows))
```

Empty cells are rejected. Each channel value `v` enters bin
`floor(v*bins/256)`. `PixelHistogram(config, width, height, counts)` stores exact
integers in row-major cell order, then R/G/B channel order, then increasing bin
order. Every channel's count sum must equal its own cell area. Changing frame
dimensions is supported when all cells remain nonempty; each frame retains its
own measured dimensions rather than borrowing stream metadata.

`measure_pixel_histogram(image, config=None, *, limits=None)` also accepts a
caller-owned Pillow image. It converts to eight-bit RGB without modifying or
closing the caller's image. Alpha is discarded, not composited; embedded ICC
profiles and EXIF orientation are not applied by this function. Native capture
uses the decoder's already converted RGB snapshot. This is not a color-managed
or high-dynamic-range measurement contract.
Conversion/crop results that alias a borrowed image are rejected before ownership
is assumed. Custom Pillow plugins still execute trusted in-process code; callers
must not concurrently mutate images while measurement runs.

## Distances and scene decisions

`histogram_distance(left, right, *, mode="spatial")` requires identical bins/grid
configurations, but not identical image dimensions. Each distribution is divided
by its own positive pixel area. Total variation is half the sum of absolute
differences between normalized bin frequencies, producing a score in `[0, 1]`.

- `global` merges all cells before normalization and averages the R/G/B total
  variations. Reordering pixels without changing global channel counts has score 0.
- `spatial` averages the total variations of all corresponding cell/channel
  distributions with equal cell weights. For odd dimensions, a larger cell does
  not receive more weight. Swapping black/white upper and lower halves gives
  global score 0 and spatial score 1 in a two-row grid.

The numeric distance uses finite floating-point normalization and `math.fsum`;
raw counts and native time coordinates remain exact. Independent tests include
scores 0, 1/2 and 1 and nonuniform channel changes. Histograms are per-channel
marginals, not joint RGB distributions. They discard arrangement within a cell;
spatial mode can react to camera motion or object movement and does not identify
semantic scene changes by itself.

`HistogramDetectionConfig(mode="spatial", threshold=0.5, min_scene_samples=1)`
emits a candidate before sample `i > 0` when its score is **greater than or equal
to** the threshold. Consequently threshold 0 also makes identical adjacent
frames candidates. The first score is zero and never a candidate. The existing
sample-minimum policy checks both the previous scene and remaining tail;
lengths count supplied samples, not elapsed time or discarded original frames.
Exact native PTS partitioning is shared with the existing native scene engine.
Unknown final times stay unknown; no nominal-FPS tail is invented.

```python
from frame_quorum import (
    HistogramDetectionConfig,
    PixelHistogramConfig,
    analyze_native_histograms,
    capture_native_histograms,
    read_native_histograms,
    write_native_histograms,
    write_native_histogram_replay,
)

captured = capture_native_histograms("local.mkv", histogram=PixelHistogramConfig(bins=32, rows=2, columns=2))
cache = write_native_histograms(captured, "histogram-cache")
restored = read_native_histograms(cache)
result = analyze_native_histograms(restored, HistogramDetectionConfig(mode="global", threshold=0.4))
write_native_histogram_replay(result, "global-review")
assert result.source_verified is False
print(result.cut_times)
```

Capture's optional synchronous `on_sample` callback receives an immutable
`NativeHistogramSample(sample, histogram)`, not an accepted scene decision.
It must return None, not a value or awaitable. Callback/measurement failures
close owned resources and do not return a successful partial capture.

## Independent wire identity and provenance

`NativeHistogramMeasurements(base, config, histograms)` composes the unchanged
`NativeMeasurements` source/options/PTS/stride/summary records with required
histograms. One decode acquires both representations. The source is hashed
before and after the pass using the existing bounded source-identity checks.
These detect ordinary observed mutation, not hostile swap-and-restore or an
untrusted decoder.

The canonical ASCII JSONL format has distinct identifiers:

```json
{
  "kind": "frame-quorum-native-histograms",
  "schema_version": 1,
  "measurement_version": "frame-quorum-rgb-cell-histogram-v1"
}
```

The full header also contains `histogram_config`, `sample_count`, and the exact
old-format `base` header. Each following record contains `sample` and
`histogram` (`width`, `height`, `counts`). The completion record binds all preceding
header/sample bytes with SHA256 and verifies the count. The original path label
is preserved verbatim, not reinterpreted as a path to open.

Missing/unknown fields, noncanonical encodings, duplicate keys, malformed numeric
tokens, unreduced rationals, incorrect count slot lengths or sums, conflicting
configurations, inconsistent observed pixel work, truncation and trailing bytes
are rejected. Selected frame areas plus at least one pixel per discarded decoded
frame must fit the historical decoded pixel total. This consistency constraint
does not certify imported metadata or the honesty of those observations.

The old fixed `FrameMetrics` and summary-cache bytes/schema are unchanged.
Neither reader automatically upgrades the other's kind. Old cache records do
not contain the missing pixel counts. Changing bins, grid, source range or stride
requires recapture; only global/spatial mode, threshold and minimum scene length
are replay options. There is no arbitrary metric registry or object import hook.

`NativeHistogramReplayResult` is a separate result with
`execution="cached_histograms"` and `source_verified=false`, never a fresh
`NativeSceneResult`. The existing `native_scene_clips` rejects it. The stored
source hash is consistency provenance, not authentication or confirmation of a
current source file. This increment does not expand the splitting trust boundary.
The result constructor recomputes scores from its actual stored histograms and
checks threshold/minimum decisions, rejecting fabricated but threshold-consistent
scores. This bounded additional distance pass validates internal mathematics,
not the authenticity of the imported histograms or their original source.

## Work, allocation and publication limits

`PixelHistogramLimits` defaults to 8,000,000 total count slots and 100,000,000
measurement pixels, with compiled ceilings of 16,000,000 and 1,000,000,000.
Before decoding or parsing sample records, the declared
`video.max_frames * rows * columns * 3 * bins` must fit the count-slot budget.
Even a short observed file cannot bypass this declared-capacity admission; lower
`max_frames` when requesting a larger grid/bin configuration.

Every selected frame is charged to the aggregate measurement pixel budget before
allocating its Pillow image. Native decode limits separately cover all decoded
frames/pixels, including discarded range/stride frames. Histogram counting uses
fixed-size Pillow RGB histograms per cell, not a Python object per pixel. The
standalone image API applies the same pixel/count budgets per call.

The existing `NativeMeasurementLimits` controls cache bytes, maximum line length,
sample count and total output bytes. The shared reader checks file size before
reading, bounds each line, and limits numeric conversion and JSON nesting before
materialization. A line must still fit its limit when a large histogram is used.
Limits are strict bounded integers; booleans and nonfinite settings fail.

Storage/analysis is bounded offline `O(samples * cells * channels * bins)`, plus
full-pixel measurement work and the existing summary calculation. RGB conversion,
cell crops, parsed JSON and the output dictionary can temporarily coexist with
retained records. These are logical admission/work bounds, **not** a hard RSS,
native codec sandbox or CPU deadline. No frames are silently dropped to fit a
budget, and no learned model, codec binary or network source is downloaded.

Publication reuses the reviewed fixed-name staging and ownership-aware cleanup
primitives, with Windows no-replace rename and Linux `renameat2(RENAME_NOREPLACE)`.
Unsupported platforms/filesystems fail closed. Only still-owned artifacts may be
cleaned; unknown or replaced paths are preserved with residue diagnostics.
Control exceptions are not masked by ordinary cleanup errors. After an uncertain
publication acknowledgment, inspect the destination before retrying. Atomic
visibility is not fsync/power-loss durability. Existing targets are never erased
to simulate rollback.

## Reference boundary

The frozen reference exposes a Y-channel histogram detector with configurable
bins, threshold and scene separation, frame processing, correlation statistics
and a CLI command. This module instead measures RGB cell distributions and total
variation, with independently specified numerical behavior. It does not claim
equivalence to that correlation detector, whose helper documentation and default
normalization call should not be assumed to imply identical probability scaling.
[Frozen public implementation](https://github.com/Breakthrough/PySceneDetect/blob/24953b0bf76af17c450bc143d330eea48fc5e276/scenedetect/detectors/histogram_detector.py).

Corresponding-pixel circular HSV and forward-gradient evidence now has its own
[separate capture/cache/replay workflow](native-pixel-changes.md); it cannot be
reconstructed from histogram marginals. Canny-based edges, learned detectors,
arbitrary plugins, online detection, time-based scene minima, joint color
distributions, labeled-video calibration and broader whole-repository parity
remain open. This histogram module closes a pixel-distribution workflow, not
those additional gaps.
