# Exact-PTS native scene workflow

`detect_native_scenes` connects the [local native decoder](native-video.md) to
all five [scene detection algorithms](scene-detection.md). It produces scenes
and per-sample evidence without extracting image files, assigning fake image
paths, converting VFR timestamps to floats, or assuming a constant frame rate.

```python
from fractions import Fraction
from frame_quorum import (
    DetectionConfig,
    NativeSceneConfig,
    NativeVideoConfig,
    detect_native_scenes,
)

result = detect_native_scenes(
    "local.mkv",
    NativeSceneConfig(
        video=NativeVideoConfig(
            start=Fraction(5),
            end=Fraction(57, 10),
            frame_step=2,
            max_frames=5_000,
            max_decoded_frames=50_000,
        ),
        detectors=(
            DetectionConfig(detector="adaptive", window_radius=2),
            DetectionConfig(detector="luminance", threshold=0.3),
        ),
        minimum_votes=1,
        min_scene_samples=2,
    ),
)
print(result.cut_positions, result.cut_times)
print(result.to_dict())
```

The public frozen records are `NativeSceneConfig`, `NativeSceneSample`,
`NativeDetectorStatistic`, `NativeSceneStatistic`, `NativeScene`, and
`NativeSceneResult`. Collections are immutable tuples and configurations reject
unknown detector types, duplicate types, invalid numeric/boolean settings, and
excess work before opening the decoder. Direct result construction checks
structural consistency: sample order/stride, configured interval, generation,
counts, voting decisions, scene partition and exact endpoints. These checks are
not authenticated evidence that a caller-supplied measurement came from a video.

## Timing and sample coordinates

Each statistic retains the actual integer PTS, rational time base, generation,
generation-local decode index, and selected-sample index. The workflow opens
one decode generation, numbered zero; a start-bound seek can decode earlier
keyframes before returning its first selected sample. It never promises a global
video frame number. `frame_step` counts eligible decoded frames, not seconds.

The sample index and scene positions are zero-based in the supplied sequence.
Each scene covers `[start_position, end_position)`. Its `sample_count` is the
number of selected samples, not the number of original/decoded frames or a
duration. A cut at position `k` means **before selected sample `k`**, at that
sample's exact presentation time. With sampling, the actual edit could have
occurred anywhere since the preceding selected sample: this workflow does not
localize an unobserved original frame.

`start_time` is the first selected sample time, not necessarily the requested
start. `last_sample_time` is the last selected sample's PTS-based time, not the
end of its presentation duration. `end_time` is:

- the next scene's first selected PTS for `end_reason="cut"`;
- the configured exclusive end only if the decoder actually observed
  `range_end`, for `end_reason="requested_end"`;
- `None` for `end_reason="unknown"` otherwise, including true EOF and count
  limits, even when a later requested end or nominal FPS is available.

An empty selected interval returns no scenes or statistics, with its actual
termination diagnostics. This differs intentionally from the legacy image
`detect_scenes`, which requires a nonempty input. Equal native timestamps are
allowed, so a positive-sample-count scene can have equal start and cut times.
No division by elapsed time, synthetic endpoint or positive-duration promise
is made. JSON expresses times as `{numerator, denominator}`, never rounded floats.
Consumers must preserve integer precision; PTS multiplied by time base can exceed
signed 64-bit numerator range. This is not a CFR EDL/export conversion.

## Shared algorithms and ensemble provenance

Native and image-domain detection share the same pure measurement kernels,
including adaptive summation order, centered-window borders, fade bias,
hysteresis, initial black-leader suppression and minimum first/final scene
length. A single detector with `min_scene_samples=1` produces the same cuts,
scores and per-detector reasons as legacy `detect_scenes` on the same ordered
measurements. No algorithm is duplicated or renamed as a new implementation.

Up to five **unique detector types** may be configured. The supplied order is
retained in each statistic. Multiple parameter variants of one type are not
supported in this API. At each selected position:

1. Each detector computes its raw `candidate` and `score`.
2. Its own `min_scene_frames` filter is applied, exactly as in the image API.
   The result is `qualified`, with a detector-specific reason. Here that legacy
   field still counts supplied samples, not original video frames.
3. Only **qualified** candidates vote. `minimum_votes=1` is union; requiring all
   configured detectors is exact-position consensus. There is no fuzzy
   timestamp clustering or tolerance window.
4. Positions meeting the vote quorum are considered in increasing sample order.
   The additional `min_scene_samples` filter uses the previous **aggregate** cut
   and the remaining selected tail. Rejected positions do not advance it.

The aggregate reason is `sequence_start`, `insufficient_votes`,
`short_previous_scene`, `short_final_scene`, or `quorum`. Raw and qualified
provenance remain available even when another detector or aggregate filter
rejects a boundary. With different detector scene minima, a raw two-detector
candidate can therefore have only one vote. There are no nondeterministic ties.

Adaptive detection needs future measurements; completed fades may place a cut
before the bright sample that confirmed it. Decisions describe the bounded
**supplied sequence**, including its minimum-tail policy. In particular,
`include_final_fade=True` can propose a tail fade even when decoding stopped at
a count limit rather than true EOF. Inspect diagnostics and `final_fade` evidence;
do not call that an observed complete-source transition.

## Work, retention and lifecycle

Decoding is incremental and the coordinator discards each RGB snapshot after
measuring it. It retains measurements, exact coordinates, and output diagnostics;
it does **not** retain the preceding RGB frames or create an image directory.
The detection phase is offline, with `O(n × d)` time and record memory for `n`
selected samples and `d` configured detectors. Adaptive and final-scene policies
are not presented as online `O(1)` memory algorithms. Result serialization also
allocates a complete report.

The compiled ceiling is **1,000,000 sample × detector cells**. Admission checks
`video.max_frames × len(detectors)`, not an optimistic estimate of the eventual
sample count. Each detector's `max_frames` must also admit `video.max_frames`.
The scene minimum is bounded to 1,000,000 samples. Defaults retain at most
10,000 samples for one detector. Native byte, frame, pixel and decode limits
still apply independently. This accounting bounds Python records/work, not
their exact byte footprint, native allocator RSS, codec probing or CPU time.
The native backend's local-file restrictions and sandbox limitations are
unchanged. No network or paid model inference is involved.

The operation owns its `NativeVideoStream` context. It closes the decoder before
offline detection and before returning a result. Failures in measurement or an
optional `on_sample(sample)` callback leave that context; Python exceptions and
interrupts propagate after cleanup. The synchronous callback sees an immutable
RGB-free `NativeSceneSample` while decoding is active, and must return `None`.
It reports sample progress, **not** a final scene decision. Async callbacks are
not supported; returned coroutine objects are closed and rejected. Caller
side effects are not rolled back, and caller-retained records consume memory.

Count limits return a valid bounded analysis, explicitly marked `frame_limit`
or `decode_limit`. Decode, pixel/source-limit, measurement and callback failures
do not return a successful partial result. A native cleanup failure is still a
failure; there is no successful result claiming an open decoder was closed.
This convenience function does not expose the owned decoder for cleanup retry;
use `NativeVideoStream` directly if manual resource-recovery control is needed.

## CLI and generated demonstration

```bash
frame-quorum native-scenes local.mkv \
  --detectors adaptive luminance --minimum-votes 2 \
  --min-scene-samples 2 --start 5 --end 57/10 --max-frames 5000

frame-quorum native-scenes local.mkv --detectors threshold \
  --min-dark-frames 2 --fade-bias 0

python examples/native_scenes.py
```

`native-scenes` writes one JSON report to stdout with
`kind="frame-quorum-native-scenes"` and `schema_version=1`, after successful
analysis. No report is emitted for an analysis failure; output-device failure
can still interrupt report writing. Native time arguments use the same bounded
exact decimal/integer/fraction parser as `native-scan`. CLI detector parameters
are shared across the selected types; the Python API supports individual
parameters. `--min-scene-frames` is the per-detector minimum and
`--min-scene-samples` the aggregate minimum.

The executable demonstration generates a local lossless FFV1 VFR clip with
nonzero PTS and a four-sample dark interval. A midpoint fade cut is at the actual
sample time **527/100 seconds**, rather than an FPS-derived or interpolated
time. The temporary video is deleted after decoding closes; no external media,
FFmpeg executable or model is needed beyond the optional PyAV dependency.

Tests compare all five policies against the existing offline implementation on
hand-authored measurements; real generated videos independently check direct
PyAV measurements, manual cuts/fades, VFR sampling, native coordinate mapping,
limits/EOF, empty intervals, exact CLI output and callback cleanup. Codec
compatibility beyond the fixtures is not implied. Cached-statistics imports,
live sources, arbitrary detector plugins, duration-based minima, audio,
clip/image export and complete PySceneDetect repository parity remain open.
