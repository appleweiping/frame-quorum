# Scene detection and diagnostics

`detect_scenes(frames, DetectionConfig(...))` segments the ordered measurements
returned by `scan_frames`. The result contains a contiguous, half-open partition,
the cut frame indices, and one diagnostic for every input frame. Existing
`detect_shots`, `analyze_scenes`, and `detect_transitions` keep their previous
contracts; the new workflow adds independent detector policies and stricter
first/final-scene bounds.

```python
from frame_quorum import DetectionConfig, detect_scenes, scan_frames

frames = scan_frames("./frames")
result = detect_scenes(
    frames,
    DetectionConfig(
        detector="adaptive",
        window_radius=2,
        adaptive_ratio=3.0,
        min_content=0.15,
        min_scene_frames=5,
    ),
)
print(result.cut_indices)
print(result.serializable())
```

## Algorithm semantics

| Detector | Evidence | Threshold policy |
| --- | --- | --- |
| `content` | 0.65 × difference-hash distance + 0.25 × RGB-mean distance + 0.10 × luminance distance | Distance ≥ `threshold` |
| `color` | Euclidean RGB-mean distance / √3 | Distance ≥ `threshold` |
| `luminance` | Absolute difference of adjacent mean luminances | Distance ≥ `threshold` |
| `adaptive` | Current content distance divided by local background distance | Ratio ≥ `adaptive_ratio` **and** content distance ≥ `min_content` |
| `threshold` | Absolute mean luminance through a dark interval and subsequent bright return | Completed fade with ≥ `min_dark_frames` dark samples |

All distances and luminances are in `[0, 1]`. The three distance modes are useful
for different changes, but a luminance difference alone does not track a fade.
The threshold detector maintains the state of a fade across several frames.

### Centered adaptive contrast

For a possible cut before position `i`, let `d[i]` be the content distance
between positions `i - 1` and `i`. The baseline is the mean of `d[i-r]` through
`d[i+r]`, excluding `d[i]`, where `r = window_radius`. Distance `d[0]` is absent
because there is no preceding frame. A score therefore requires `i >= r+1`
and `i+r < n`. Border scores are `null`, rather than estimates from a smaller
window. These cuts are deliberately not inferred; use a distance detector when
border cuts must be evaluated.

Ratios use `max(baseline, 1e-12)` as their denominator and are capped at `1e6`,
so zero-background transitions remain JSON-serializable. The separate
`min_content` floor suppresses large ratios produced by negligible differences.
The centered window requires future evidence, so this is an offline detector.
The score calculation uses a sliding sum with constant work per new position.

A smooth grayscale sequence `[0, .05, .1, .15, .8, .85, .9, .95, 1]` has a
content jump of `.2275` at position 4 and background distance `.0175`. With
radius 2 its contrast ratio is 13, giving a cut at 4. This hand-computed fixture
is executable in `tests/test_scene_detection.py`. Adjacent flash edges can raise
each other's baseline; flash rejection is not guaranteed for every radius or
threshold. No camera-motion or semantic classifier is implied.

### Fade-through-dark state

A fade can begin only after a frame with luminance strictly above
`dark_threshold + hysteresis`. An initial black leader alone cannot create a
cut. A frame at or below `dark_threshold` starts the dark interval. Samples
between the dark and release thresholds keep the interval open but do not
count toward `min_dark_frames`.

A bright return strictly above the release threshold confirms the interval
`[start, return)`. If enough samples reached darkness, the candidate boundary is:

```text
start + floor((return - start) * (1 + fade_bias) / 2)
```

Bias `-1` places it at the first dark frame, `0` at the midpoint, and `+1` at
the returning bright frame. A one-frame dark flash is suppressed by the default
minimum of two truly dark samples. An incomplete fade at EOF produces no cut
unless `include_final_fade=True`, which proposes its first dark frame. Final
scene-length checks still apply.

This detects transitions through darkness; it does not claim to detect every
cross-dissolve, white flash, camera exposure change, or scene with very dark
content. Its thresholds need calibration to the input material.

## Scene boundaries and resource limits

Indices must be contiguous and increasing, although the first index may be
nonzero. Timestamps may all be absent, or all present and non-decreasing. Scene
end indices are exclusive. All cuts use the first frame of the following scene.
Every accepted boundary leaves at least `min_scene_frames` before it and in the
remaining final scene. Rejected candidates report `short_previous_scene` or
`short_final_scene`. An input shorter than the minimum stays one unsplit scene.

All lengths count **supplied frames**, not original video frames. For a sequence
extracted at 2 FPS, 10 supplied frames represent about five seconds. Variable
frame spacing is retained as timestamps in statistics; it does not change a
frame-count minimum into a duration minimum.

Analysis takes `O(n)` time and `O(n)` result memory. `max_frames` is configurable
up to a hard maximum of 1,000,000; excess input is rejected before score arrays
are allocated. Window radius is bounded to 10,000. The result retains one
statistic per input frame. The analysis cap applies after scanning in the CLI;
it does not bound the image discovery/decoding step. Optional FFmpeg extraction
has its separate frame, byte, and runtime limits.

## CLI and interoperable statistics

```bash
frame-quorum scenes ./frames --output-dir ./scene-report \
  --detector adaptive --window-radius 2 --adaptive-ratio 3 \
  --min-content 0.15 --min-scene-frames 5

frame-quorum scenes ./frames --output-dir ./fade-report \
  --detector threshold --dark-threshold 0.05 --hysteresis 0.02 \
  --min-dark-frames 2 --fade-bias 0 --include-final-fade

python examples/detect_scenes.py
```

The command accepts the same discovery, timestamp, animation, and worker flags
as `scan`. It writes `scenes.json` (schema version 1) and `statistics.csv` as a
staged bundle outside the input directory. The CSV includes position, original
frame index, timestamp, raw content score, luminance, detector score, candidate,
accepted, and reason. Empty score cells correspond to JSON `null`; numeric zero
is preserved. Candidate locations in a fade may precede its confirming bright
frame because detection operates on the full sequence. The public
`render_detection_csv(result)` API emits the same table.

Statistics can be plotted or compared between parameters. They are currently
write-only: importing a CSV, registering arbitrary metrics, or replaying a new
detector directly from a cached statistics file is not implemented.
