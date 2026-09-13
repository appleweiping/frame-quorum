# Exact declared native video composition

`NativeConcatStream` actually decodes multiple local sources with at most one
owned child decoder. It exposes an exact composite timeline and preserves each
frame's original RGB, PTS and time base. This is the **caller-declared clip**
profile, not an automatic-duration playlist, media encoder or scene detector.
It requires the existing optional `frame-quorum[video]` dependency; importing the
package and constructing a timeline do not import PyAV or open video handles.

```python
from fractions import Fraction
from frame_quorum import NativeConcatClip, NativeConcatConfig, NativeConcatStream

clips = (
    NativeConcatClip("first.mkv", start=5, end=Fraction(53, 10)),
    NativeConcatClip("second.nut", start=1, end=Fraction(6, 5)),
)
with NativeConcatStream(clips, NativeConcatConfig(frame_step=2)) as stream:
    for frame in stream:
        print(frame.clip_index, frame.presentation_time, frame.native.pts, frame.native.time_base)
    stream.seek(Fraction(3, 10))  # The seam belongs to the right-hand source.
    frame = next(stream)
    parts = stream.map_span(Fraction(1, 10), Fraction(2, 5))
    print(stream.diagnostics.to_dict())
```

Run `python examples/native_concat.py` for a generated, offline, lossless
two-source demonstration with an independent complete-RGB/timestamp oracle.
The example writes only inside an automatically removed temporary directory.

## What the time axis means

Every clip declares an absolute source-native half-open interval `[a_i, b_i)`.
Negative and nonzero origins are valid. The immutable offsets are
`O_0 = 0`, `O_(i+1) = O_i + b_i - a_i`. A selected frame with native time `t`
has composite time `T = O_i + t - a_i`. For the example the second source starts
at `3/10`, regardless of either source's nominal frame rate or duration metadata.

Only exact integers and `Fraction` values are accepted. Integers mean seconds,
not frame numbers. Floats, booleans, NaN, implicit FPS interpolation and
microsecond quantization are not accepted. Native endpoints retain the existing
signed-int64 numerator/positive-int64 denominator contract. Derived composite
coordinates allow signed 127-bit numerators and positive int64 denominators.
An otherwise finite sum with an oversized denominator fails explicitly. Every
actual inverse-mapped native bound and seek tick must also fit the native limits.

Declared endpoints are **not measured last-frame durations**. A sample exactly
at `b_i` is excluded. Early EOF leaves an unsampled gap; an interval containing
no samples still occupies its declared positive length. No end discovery,
metadata over/underestimate, seek or failure rewrites the immutable offsets.
Repeated paths are distinct ordered source occurrences, with separate per-source
budgets. Duplicate PTS within a source are preserved; decreasing PTS fail.

The stream freezes a detached `NativeConcatTimeline`, including ordered resolved
paths, exact rational endpoints, selected video stream indexes and offsets. Its
SHA-256 digest binds the canonical, sorted-key UTF-8 JSON manifest, not file bytes
or authenticity. It is path-dependent: moving identical media changes this
manifest identity. `timeline` and `config` are read-only public properties.

## Real sources, not fabricated frame provenance

Entering probes **every** source serially, closes each probe before opening the
next, and rejects a missing/corrupt/no-video source or incompatible dimensions
before playback. Playback reopens only the required source. Supported local
container signatures, single-thread native codec configuration and denied
secondary/network I/O are inherited unchanged from [native video](native-video.md).
Missing rate, duration and start metadata are acceptable because they are not
used to fabricate timestamps. In-range frame dimensions must match frozen source
metadata, including on frames discarded by global stride.

The expected `(device, inode, size, mtime_ns)` is captured for every occurrence.
Device/inode must be stable, exact integer identities (inode zero is unavailable).
The path identity is checked before and after every probe/playback/seek activation
and reconciled with the native child's captured opened-file identity. Frozen
metadata is compared on each playback activation. This detects ordinary path
replacement and metadata changes; it is **not** a content hash, hostile-filesystem
atomicity proof or defense against a privileged actor restoring all stat fields.
The existing native stream also checks source identity while decoding.

`NativeConcatFrame` is a separate immutable type. `native` is the exact detached
`NativeVideoFrame` supplied by the child; its PTS, time base, native decode/sample
indexes and generation are not overwritten with global coordinates. `clip_index`
identifies the occurrence. `activation` is that occurrence's zero-based lifetime
open ordinal, including probes (ordinary first playback is activation 1).
`generation` is the composite seek count and `sample_index` is the lifetime
composite output-allocation ordinal. `rgb` shares the immutable owned native byte
string. `image()` creates an independently owned Pillow image that callers close.

`frame.to_dict()` includes the new manifest digest and native timing provenance,
not the RGB bytes. Existing native frame, scene, cache and export wire formats
are unchanged. Do not pass a composite frame off as a single-source native scene.

## Iteration, seek and span mapping

Use one active `with` context and one owning thread. Setup, iteration, seek,
reset and close reject reentrancy. Breaking a loop alone does not release native
resources; context exit or explicit `close()` does. A stream cannot be re-entered.

The native children run at unit step. Global `frame_step=K` selects eligible
in-window ordinal 0, K, 2K across source boundaries, not separately per source.
It resets its phase only on composite seek. Thus even a globally discarded frame
has native RGB work, which is charged independently of emitted composite frames.

`seek(start, end=None)` validates the complete range, inverse mapping, tick and
remaining activation/work/seek limits before touching an active child. Invalid
preflight is nonmutating. A valid seek closes the old child, increases generation,
resets the stride phase and opens the selected source using backward keyframe
seek plus exact forward filtering. A later source-identity or native-open failure
is terminal, not a resumable preflight rejection. Seeking into an unsampled gap
can traverse empty eligible selections under the same finite source/work budgets.
The exact end-of-timeline seek closes and completes without opening a source.
`reset()` means `seek(0)` and never replenishes lifetime budgets.

Logical `range_end` and `intervals_exhausted` allow another admitted seek. Manual
close, errors, interruption, unknown cleanup and budget-limited states do not
restart decoding. A failed cleanup retains its child reference; `close()` may be
retried, but no second child can be acquired while the previous reference remains.
Every owned native resource receives a close attempt. Original KeyboardInterrupt
or SystemExit identity wins over another cleanup control; if the original error
is ordinary, a cleanup control still propagates. Cleanup names remain diagnostic
history after a successful retry.

`map_span(start, end)` works before or after decoding and returns immutable
`NativeConcatSpan` half-open intersections in source order. It requires
`0 <= start <= end <= declared duration`; equal endpoints return `()` and an
out-of-range span fails rather than clipping silently. Each span records the
manifest digest, occurrence/path, composite range and inverse-mapped **absolute
native** `local_start`/`local_end`. These are not directly usable as muxer-relative
`ffmpeg -ss` arguments. Mapping includes declared gaps and is labeled
`coverage: declared_only`, not proof that every instant has a frame.

## Lifetime admission and diagnostics

Limits may be lowered or raised only to the compiled hard ceilings below. All are
positive exact integers. Source tuples are fully admitted and snapshotted before
native acquisition: 1..128 exact `NativeConcatClip` values, local paths at most
4096 UTF-8 bytes after resolution, and a canonical manifest at most 1 MiB.

| `NativeConcatLimits` field | Default | Hard maximum |
| --- | ---: | ---: |
| `max_sources` | 32 | 128 |
| `max_source_bytes` | 1,000,000,000 | 2^40 |
| `max_manifest_source_bytes` | 4,000,000,000 | 2^40 |
| `max_opened_source_bytes` | 16,000,000,000 | 2^40 |
| `max_activations` | 256 | 10,000 |
| `max_seeks` | 128 | 10,000 |
| `max_decoded_frames`, `max_source_decoded_frames` | 100,000 | 10,000,000 |
| `max_rgb_frames`, `max_source_rgb_frames` | 100,000 | 1,000,000 |
| `max_frames` | 10,000 | 1,000,000 |
| `max_frame_pixels` | 16,777,216 | 67,108,864 |
| `max_total_pixels`, `max_source_total_pixels` | 1,000,000,000 | 2^40 |
| `max_span_parts` | 32 | 128 |

Probe, playback and seek opens share the same activation and conservative
whole-file byte charges. `max_opened_source_bytes` is not an actual I/O byte
meter. Per-occurrence and aggregate decoded/RGB/pixel balances are passed to each
new child as the minimum remaining limit; exhausted budgets are never converted
into an invalid zero-limit child config. Repeated paths/seeks do not reset them.

`decoded_frames` and `decoded_pixels_observed` include preroll, range-boundary and
stride-discarded native frames. `owned_rgb_frames` counts successful native RGB
materializations even when global stride discards them. `returned_frames` counts
composite output-allocation **attempts**, including an attempt that fails before
delivery. Allocation failure does not undo completed work. Pixel rejection may
observe one overshoot frame, inherited from native decoding; observed pixels are
reported accurately and no over-budget composite row is emitted.

Every successful or exceptional child open/next attempt has a final accounting
boundary; closing does not double count. Failure at that boundary is terminal:
logical StopIteration cannot hide an ordinary accounting failure. Per-occurrence
count tuples have O(number of sources) size; no activation history or prior RGB
frames are retained by the stream. Retaining yielded frames is the caller's
responsibility. Native parsing/codec work before the Python frame boundary is
not a hard CPU, wall-time or RSS guarantee; use process isolation for untrusted
codec execution and external cancellation requirements.

Diagnostics expose `new`, `open`, `intervals_exhausted`, `range_end`, `frame_limit`,
`decode_limit`, `rgb_limit`, `pixel_limit`, `source_limit`, `activation_limit`,
`closed`, `error` or `interrupted`. Rejected seek budgets leave the current status
unchanged. `limit_scope` distinguishes `source` and `total`; `last_child_status`
independently records the most recently observed native state (including `eof`
and `range_end`). An already-terminal budget reason survives ordinary exception
unwinding through context exit, including the original pixel-limit error. A new
body control changes the status to `interrupted`; cleanup failures and controls
retain the exception-priority and ownership rules above. Ordinary exceptions are
still propagated, not suppressed by retaining the earlier budget reason.
Only clean logical completion says the requested **declared
intervals** were processed. A limit never proves source EOF, complete media
coverage or a physical final-sample duration.

## Scope still open

This core is original real multi-source composition, not complete repository
parity. Automatic estimated-duration discovery, explicitly revisioned lazy maps,
alternate backends/live/file-like sources, composite-aware detector/cache/image
pipelines, scene-span video/audio export, CLI and external interoperability
remain separate required work. Fixed declarations deliberately do not imitate
an automatic last-observed-position-plus-nominal-frame seam estimate.

The [whole-reference ledger](parity-detection.md) retains those gaps. Tests use
independent exact piecewise arithmetic and direct PyAV decoding of generated
lossless videos, including every RGB byte, source clock and occurrence; they are
not a labeled real-world accuracy or broad codec performance benchmark.
