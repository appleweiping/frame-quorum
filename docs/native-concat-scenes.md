# Scene analysis across declared video clips

`detect_native_concat_scenes` owns real sequential video decoding and runs the
five representation-level scene detectors on one ordered measurement sequence.
Adaptive neighborhoods and fade-through-dark context continue across source
boundaries. A boundary alone does **not** force a cut. Original native PTS and
time bases remain separate from the exact composite presentation coordinate.

Install the existing `frame-quorum[video]` extra to decode video. There is no new
dependency, model, network request, automatic duration guess or media upload.

```python
from fractions import Fraction
from frame_quorum import (
    DetectionConfig,
    NativeConcatClip,
    NativeConcatConfig,
    NativeConcatSceneConfig,
    detect_native_concat_scenes,
)

clips = (
    NativeConcatClip("first.mkv", start=5, end=Fraction(26, 5)),
    NativeConcatClip("second.nut", start=1, end=Fraction(6, 5)),
)
config = NativeConcatSceneConfig(
    video=NativeConcatConfig(frame_step=2),
    detectors=(
        DetectionConfig(detector="adaptive", window_radius=1),
        DetectionConfig(detector="luminance", threshold=0.3),
    ),
    minimum_votes=2,
    min_scene_samples=2,
)
result = detect_native_concat_scenes(clips, config)
for row in result.statistics:
    print(
        row.sample.presentation_time,
        row.sample.native.pts,
        row.sample.native.time_base,
        row.votes,
        row.accepted,
        row.reason,
    )
print(result.cut_positions, result.cut_times)
document = result.to_dict()
```

Run `python examples/native_concat_scenes.py` for an entirely generated, offline
Matroska/NUT example with different source epochs, clocks and nominal rates. It
checks every native timestamp, the cross-source cut, explicit declared tail and
closed decoder ownership, also under `python -O`. Temporary media are removed.
This entry point is a Python API; the existing `native-scenes` CLI continues to
be single-source.

## Evidence and coordinates

The immutable configuration admits one to five **distinct** `DetectionConfig`
values: `content`, `color`, `luminance`, `adaptive`, and `threshold`. Their existing
arithmetic and individual minimum-length rules are unchanged. A separate quorum
accepts a qualified boundary only when enough detectors vote and both aggregate
scene-length constraints hold. Lengths count supplied samples, not original
video frames or seconds. Global stride is applied by the concat decoder before
measurement; detector context therefore consists of selected samples only.

Each `NativeConcatSceneSample` contains the timeline digest, occurrence index,
activation, composite generation, contiguous global sample index and exact
composite presentation time. Its nested `NativeSceneSample` retains the original
PTS, time base, native decode/sample indices, native generation and measured
`FrameMetrics`. It contains no image, RGB buffer or invented global native PTS.
Native indices restart with a child activation and must not be equated to the
global sample index. Repeated paths remain distinct source occurrences. Video
selector zero is not necessarily container stream index zero.

Every occurrence is probed once (activation zero). Playback observations have
activation one and generation zero. For C declared occurrences and K occurrences
intersecting the requested positive range, successful analysis has C+K total
activations, including playback of an intersecting physically empty source.

`NativeConcatSceneStatistic` retains content distance, ordered raw/qualified
detector evidence, votes and the aggregate decision/reason. Its public value is
checked against bounded measurement-only recomputation when a result is built
or serialized. That detects inconsistent records; it does not re-open sources
or establish that externally supplied measurements match source pixels.

## Declared endpoints, gaps and equal times

Clips use the [exact declared concat map](native-concat.md). Scene partitions
are half-open in **sample positions**. A scene starts at its first observation;
the next cut supplies an interior endpoint. The final endpoint is exactly the
requested composite end or the full declared timeline duration when omitted.
Its reason is `declared_end`, never an estimated EOF/frame duration.

No scene is invented for an unobserved initial gap or a range with no samples.
Native equal timestamps remain valid. Two cuts at the same time can enclose a
nonempty sample partition with a zero-duration time interval and no source span
pieces; such scenes are not silently collapsed. Positive scene intervals map to
algebraic `NativeConcatSpan` pieces, including declared but unobserved time.

The result document identifies itself as `frame-quorum-native-concat-scenes`,
schema version 1, execution `offline_declared_concat`, with
`source_verified: false` and `coverage: "declared_only"`. Neither a declared
span nor successful codec EOF proves continuous physical media coverage. There
is no persistent cache/parser, media hashing or content-verified export in this
profile. Original single-source partial-prefix and endpoint contracts are not
changed.

## Completion, budgets and validation

A result requires closed ownership, no seeks/generation changes, no historical
cleanup error and terminal `intervals_exhausted` or `range_end`. Frame, decode,
RGB, pixel, source-byte or activation exhaustion produces **no successful partial
result**. Even a prefix that happens to include all physical frames remains
incomplete without a clean completion observation. There is no extra EOF peek
at a configured count limit. End positions beyond the timeline are rejected,
not clipped. Start equal to timeline duration with omitted end is the one
permitted configured empty range; it still probes all sources.

Result validation binds the terminal status and cursor to the declared range,
requires zero decode work for probe-only occurrences, and checks global stride
against cumulative child RGB positions. Playback must have observed native EOF
or range completion; probe-only completion retains the native `closed` state.
Returned/RGB counters must remain below their stop-on-frame ceilings. Decode
and pixel counters may equal their ceilings when a range-end sentinel consumed
the final allowance. Known RGB dimensions bound observed pixel work without
assuming that non-RGB preroll or sentinel frames have those same dimensions.

Existing aggregate/per-occurrence decoder budgets remain active. Before decoder
or callback acquisition this profile additionally checks:

- `max_frames` fits every detector's capacity;
- `max_frames * detector_count <= 1_000_000`;
- `max_frames * occurrence_count <= 1_000_000`;
- `max_activations >= C + K`.

Each scene obeys the existing `max_span_parts` limit without truncation. Total
positive span pieces obey both the fixed one-million ceiling and the disjoint
partition bound `scene_count + occurrence_count - 1` (no scenes means no pieces).
Work is O(samples × detectors + scenes × occurrences); retained measurement and
evidence memory is O(samples × detectors), without prior RGB frames. These are
count/work bounds, not a hard RSS limit or a hostile-codec/native CPU deadline.

New records use exact immutable tuples and exact admitted scalar/record types,
including rejection of booleans as integers. Nested frozen records are rechecked
and detached when constructing a result. Timelines are validated at result
boundaries rather than resolved once per sample. All scene span records in one
result must share one timeline object with the same declared geometry; this
prevents repeated validation of independently supplied per-span manifests.
Result serialization is detached, bounded and does not decode/replay video.

## Synchronous callbacks and failed cleanup

`on_sample` is an optional trusted synchronous observer of detached scalar
measurements. It must return `None`, not a value, awaitable or task. A returned
fresh native coroutine is closed before the callback contract failure is
raised. No task is created. The RGB-bearing local frame is discarded before the
observer is called. Observer side effects are not transactional or rolled back;
the callback is not a final scene notification.

Acquisition, measurement, observer, result-building and cleanup failures return
no partial result. Native primary-control priority and the selected exception's
identity are retained. Every post-stream-construction failure exposes the same
owner as `error.native_concat_cleanup`, including failures in context entry:

```python
try:
    result = detect_native_concat_scenes(clips, config)
except BaseException as error:
    owner = getattr(error, "native_concat_cleanup", None)
    if owner is not None:
        owner.close()  # Retry on this same owning thread; may fail again.
        print(owner.diagnostics.closed, owner.diagnostics.cleanup_errors)
    raise
```

An unsuccessful close keeps failed resources reachable through that explicit
owner. Retry does not decode, acquire another child, replay a callback or erase
historical cleanup diagnostics. A closed owner may also accompany an ordinary
failure. A different thread may not take over cleanup. Callers should preserve
the original exception when handling a further retry failure.

The attachment guarantee applies to normal attribute-capable exceptions and a
trusted runtime. Hostile exception attribute hooks, process termination and
allocation failure preventing retention of the capability are outside it. An
application requiring explicit ownership independent of this convenience API
should retain a `NativeConcatStream` itself. No general safe cancellation or
unconditional no-resource-loss guarantee is made across those failures.

## Verification scope

The dedicated suite combines handwritten detector/quorum/span oracles, exact
native identity and cleanup tests, forged-record admission, and real generated
lossless Matroska/NUT video. The arithmetic oracle uses independent direct
neighbor sums and bright-interval fade enumeration; real solid-gray metrics are
computed from direct decoded bytes. These finite tests are not scene-detection
accuracy benchmarks on a human-labelled video corpus. Pixel/histogram online
pipelines, automatic clip duration discovery, composite cache/replay, composite
still/audio/video export and CLI integration remain separate work.
