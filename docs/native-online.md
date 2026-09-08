# Bounded online pixel-change detection

`NativePixelChangeStream` runs the original corresponding-pixel circular HSV
and forward-gradient measurement while decoding a local video. Unlike capture
followed by replay, it delivers final decisions before reaching the end of the
input and does not retain a full measurement, statistic or scene table.

It requires the optional `video` dependencies. It does not download media,
models or executables. The [pixel definition](native-pixel-changes.md) is
unchanged; this is not numerical compatibility with another detector.

## Consume final decisions

```python
from frame_quorum import (
    NativePixelChangeStream,
    NativePixelChangeUpdate,
    PixelChangeDetectionConfig,
)

with NativePixelChangeStream(
    "input.mkv",
    detection=PixelChangeDetectionConfig(
        detector="adaptive",
        value_only=True,
        window_radius=2,
        min_scene_samples=4,
    ),
) as stream:
    for event in stream:
        if isinstance(event, NativePixelChangeUpdate):
            if event.statistic.accepted:
                print(event.sample.sample.presentation_time, event.closed_scene)
        else:
            print(event.diagnostics, event.final_scene)
```

The stream accepts existing `NativeVideoConfig`, `PixelChangeConfig`,
`PixelChangeDetectionConfig` and `PixelChangeLimits`, plus `NativeOnlineLimits`.
Configuration properties are read-only and each configuration is immutable.
It has one owner thread, one context lifetime, no seek, no background producer,
no asynchronous callback and no automatic retry. Calling `next()` drives all
work. Pausing consumption does not schedule further decode or measurement.
Use a context or explicit `close()`; `break` alone does not close the source.

One `NativePixelChangeUpdate` is delivered per selected sample, in sample-index
order. It contains the original coordinates and pair evidence, its final
`PixelChangeStatistic`, the latest observed sample index and exact time, and
an optional `closed_scene`. Only accepted cuts close scenes. The one successful
`NativePixelChangeEnd` contains closed native diagnostics and the final scene,
or `None` for empty input. No RGB is exposed by these events.

These are result containers, not certificates for arbitrary manually supplied
history. Constructing an event does not prove its pixels came from a source or
that all preceding decisions were correct.

## Confirmation delay and numerical identity

Let `m` be `min_scene_samples`; let `r` be `window_radius` in adaptive mode and
zero in content mode. The uniform confirmation lag is `L = max(r, m - 1)`.
Sample `i` is normally finalized once sample `i + L` has been observed. The
reported cut still refers to sample `i`, not to the later confirming frame.

This delay supplies both the adaptive right-hand window and enough samples
after a candidate cut to guarantee the minimum final scene. Results are not
provisional and are never retracted. Initial samples without a full adaptive
window have an absent score. Pair zero remains absent pixel evidence with
weighted score zero, never a fabricated comparison.

The shared rolling ring preserves the pre-existing `math.fsum` initialization
and three-term sliding update order, including exclusion of distance zero.
Shared single-point minimum-scene decisions preserve short-previous-scene
precedence over short-final-scene. Weight normalization, V-only values and all
four integer sums are identical to offline pixel replay. This does not add
flash merging or change the original sample-count policy into a time minimum.

## Exact time and completion

PTS, rational time base, sample/decode indices and generation zero are retained.
No FPS interpolation is used. Repeated PTS can represent distinct samples;
scenes are half-open **sample** partitions, not a promise of positive time
duration. Stride-based boundaries cover selected samples, not unobserved frames.
Changing actual frame dimensions is rejected by the shared measurement engine.

| Exit | Output and meaning |
| --- | --- |
| EOF | Close native resources, drain pending decisions against the final observed count, emit End. Incomplete final adaptive windows stay absent. |
| Requested range reached | Same drain; final scene has the configured exact `end` and `requested_end` reason. |
| Frame/decode count budget | Same drain for the observed prefix, with the explicit native limit status. No extra frame is requested to distinguish a coincident EOF. |
| EOF before a requested end | It remains EOF; the requested end is not invented as a video endpoint. |
| Close/cancel | Drop unconfirmed pending records and final-scene state; no successful End and no fake EOF finalization. |
| Decode/measurement/output error | Attempt resource cleanup and propagate failure; already delivered events remain a prefix, not a complete report. |

The final endpoint at EOF or a count limit is `None`; metadata duration or a
nominal frame interval is not substituted. `NativeVideoDiagnostics.closed`
must be true, with no cleanup errors, before a successful End is built.

`cancel()` only sets a thread-safe request flag. The owner checks it before
decode, after return/measurement and before event delivery. A native `next()`
can itself scan multiple unselected frames. Cancellation cannot interrupt that
call, arbitrary codec work, or a blocking sink; there is no hard deadline.
Cancellation arriving after End was delivered does not undo completion.

Native closure and retained-state release are both attempted. An active control exception is retained over
ordinary cleanup errors; a genuine cleanup control exception is not converted
to an ordinary domain error. Unknown native closure remains visible in native
diagnostics and explicit close can retry. External-body exceptions also close
the stream. Reusing a failed stream is unsupported.

## Resource accounting

Two RGB snapshots still require `6P` bytes for frame area `P`, admitted against
`PixelChangeLimits.max_pair_rgb_bytes` before opening the decoder. Measurement
work charges first acquisition `P` and every later pair `2P`, or `(2N - 1)P`
for `N > 0`; empty input charges zero. These counts are complete-frame work
units, not a count of every neighbor read in the gradient calculation.

`NativeOnlineLimits.max_buffered_samples` defaults to 4096, with a compiled
ceiling of 65,536. Its conservative admission formula is:

```text
(2r + 1 adaptive score slots, or zero for content)
+ (L + 1 pending measurement/statistic slots)
+ 8 fixed and overlapping work/output/scene-coordinate slots
```

The fixed reserve covers the current frame, measured record, constructed
update, start/last/latest coordinates and terminal scene/End overlap. There is
no separate ready queue. It intentionally overcounts aliases rather than
claiming zero memory while an event is being constructed. Diagnostics report
the peak score-ring plus pending slots and that fixed reserve. This is a
logical retained-object bound, not a Python byte/RSS estimate or a bound on
objects the caller chooses to retain after delivery.

Original source-byte, decoded-frame, returned-frame and observed-pixel limits
still apply. There is no unlimited-stream mode. Native probing, image/codec
allocations and CPU are not an OS sandbox. Very large existing offline window
or scene-minimum configurations may be rejected by this online buffer ceiling;
the old offline API remains unchanged.

## JSONL and atomic file output

```bash
frame-quorum native-change-stream input.mkv --detector adaptive --value-only
frame-quorum native-change-stream input.mkv --min-scene-samples 4 --output-dir new-report
```

Without an output directory, canonical ASCII JSONL is written and flushed one
line at a time. With a new directory, `events.jsonl` is staged and published
without replacing an existing target. The corresponding Python APIs are
`iter_native_pixel_change_jsonl(stream)` for a fresh active stream and
`write_native_pixel_change_stream(new_stream, output_dir)`, which owns its
stream context throughout publication. Do not mix direct event consumption
with the JSONL iterator: missing rows are rejected, not signed off by a footer.

The independent header kind is `frame-quorum-native-pixel-change-events`,
schema version 1. It binds video/measurement/detection settings, limits,
metadata, lag and `source_verified=false`. Sample rows contain exact rational
coordinates and final statistics. The End line includes a SHA-256 checksum of
all preceding canonical lines, their newlines included. This detects accidental
transcript changes; it is not a signature or source-content authentication.

`max_line_bytes` defaults to 16 KiB (compiled ceiling 64 KiB);
`max_output_bytes` defaults to 256 MiB (compiled ceiling 1 GiB). These two limits
apply to JSONL output, not in-process event objects. Header, every sample,
complete End/footer and every newline are charged before writing that line.
A last-line budget failure must not write a successful footer. Missing footer
means incomplete output, even if native decoding itself already finished.

Stdout is not transactional: a failure can leave a valid prefix, and a failed
write/flush acknowledgement can leave uncertain delivered bytes. File output
uses the existing owned-file/inode cleanup and no-replace directory publication;
unknown staging entries are preserved. After an attempted rename with unknown
acknowledgement, inspect the named destination before retrying. Publication is
atomic visibility where supported, not a file/directory-fsync durability promise.

## Provenance and open scope

This execution is `online_native_pixels`, not cached replay. It intentionally
does not perform two whole-source SHA passes before/after decoding. The native
reader checks observed file identity during reads, but this is not a complete
source certificate or protection against adversarial swap-and-restore.
`source_verified` remains false. Events/End are not `NativeSceneResult` and
cannot be passed to `native_scene_clips`; the splitting boundary is unchanged.

Old summary, histogram and pixel-change cache schemas remain unchanged. This
increment does not include online fades/quorum/histograms, Canny/dilation,
flash merging, time-duration minimums, live devices/network media, resumable
checkpoints, automatic splitting, learned detectors or whole-reference parity.
The generated [offline example](../examples/native_online.py) uses tiny
lossless VFR frames and requires no external dataset.
