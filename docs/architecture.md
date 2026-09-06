# Architecture

Frame Quorum separates image I/O, deterministic measurement, constrained selection, and presentation so that
each boundary can be inspected and tested independently.

```text
directory / image
       │
       ▼
 scanner ── timestamp policy
       │
       ▼
 immutable Frame records ── visual metrics
       │
       ▼
 selector ── constraints ── score breakdowns ── decisions
       │
       ├──────────────► JSON manifest
       ├──────────────► contact sheet
       └──────────────► benchmark baselines ── JSON + SVG
```

## Modules

- `scanner.py` discovers supported files, establishes natural ordering, corrects EXIF orientation, optionally
  expands animated containers, optionally decodes several files at once, and creates immutable `Frame` records.
  Decode errors stop the scan with a path-specific message.
- `timestamps.py` owns all conversions from index, filename, modification time, or EXIF capture time to a
  floating-point coordinate.
- `metrics.py` implements bounded-resolution Pillow measurements and the composite content distance.
- `selector.py` validates the sequence, applies hard constraints, chooses frames, and explains every decision.
- `reporting.py` is the JSON schema boundary. It converts hashes to fixed-width hexadecimal strings and rounds
  score fields.
- `contact_sheet.py` is a presentation adapter. It does not participate in scoring.
- `demo.py` generates deterministic synthetic data used for the public example.
- `benchmark.py` owns transparent baselines, label-free metrics, aggregation, raw-source and measured-record digests,
  runtime/scan provenance, and SVG output.
- `video.py` is an optional subprocess boundary for monitored FFmpeg extraction; FFmpeg is not imported or bundled.
- `cli.py` maps commands and flags to the public Python API.

## Measurement model

Images are converted to RGB, bounded to 128×128 while retaining aspect ratio, and also represented in grayscale.
The measurements intentionally avoid learned weights:

- Difference hash compares adjacent pixels in a 9×8 grayscale reduction and yields 64 structural bits.
- Entropy is Shannon entropy of the 256-bin grayscale histogram divided by the theoretical maximum of 8 bits.
- Sharpness is the mean Pillow `FIND_EDGES` response after removing the one-pixel filter border, scaled and clamped.
- Colorfulness combines the mean and standard deviation of red-green and yellow-blue opponent channels.
- RGB and luminance means retain information that a gradient-only hash loses on flat fields.

Content distance is `0.65 × hash + 0.25 × RGB + 0.10 × luminance`. All terms are normalized to `[0, 1]`.
The constants are explicit baseline policy, not learned estimates.

## EXIF capture time

The `exif` timestamp policy is one more strategy behind the same `timestamp_for` boundary; the scanner, selector,
and manifests are unchanged. It re-opens the file to read the root IFD and the Exif sub-IFD (`0x8769`) and never
decodes pixels.

Precedence is `DateTimeOriginal`, then `DateTimeDigitized`, then `DateTime`: most specific record of the capture
event first, the software write time last. Skipping and rejecting are deliberately different outcomes. A tag that
is absent or holds a placeholder made only of zeros, colons, and spaces was never recorded, so the next tag is
consulted. A tag that is populated but unreadable stops the scan, because continuing would silently answer a
question about one tag using a different one.

EXIF datetimes are naive wall clock. The matching offset tag (`OffsetTimeOriginal`, `OffsetTimeDigitized`,
`OffsetTime`) is applied when it holds a `+HH:MM` or `-HH:MM` value, is rejected when it holds anything else, and
otherwise leaves the value read as UTC. Selection depends only on differences between coordinates, so a single
unknown offset shared by a sequence is harmless; a sequence mixing zones is not, and the selector's existing
non-decreasing-timestamp invariant reports it. Sub-second tags are not read, so the policy resolves to one second.

## Animated container expansion

One input file normally yields one `Frame`. Passing an `AnimationConfig` to `scan_frames` lets an animated GIF,
APNG, or WebP container yield one `Frame` per internal frame instead. Expansion is default-deny: without that
config an animated file is still measured as its first frame only, and the default extension set still excludes
`.gif`.

Both limits are checked from the container header before any internal frame is decoded. A container above
`max_frames`, or whose `frames × width × height × 3` decoded RGB estimate exceeds `max_decoded_bytes`, raises
`ScanError`; expansion never returns a truncated prefix of an animation.

Expanded records keep the container's `path` and `byte_size` and add two identity fields: `source_frame_index`
holds the position inside the container, and `relative_path` becomes `name#frame=N`. Both travel through
decisions, manifests, and the contact sheet, and the renderer seeks to `source_frame_index` before re-reading
pixels, so a selected internal frame is re-verified against the exact image that was measured. Frame indices
stay contiguous across a mixed directory of expanded and single-image files.

Because one container is opened for pixels and read again for metadata, the scanner snapshots the file's size,
modification time, device, and inode before decoding and re-checks them afterwards. A container replaced during
its own scan produces a `ScanError` instead of a record mixing measurements from two files.

## Parallel scanning

A scan is sequential unless the caller passes a `ConcurrencyConfig`, and the worker count is an execution choice
rather than a measurement input: it never reaches a manifest, and every observable output is identical at every
worker count.

Two properties make that hold. First, a worker only ever runs `_measure_file`, which snapshots one file, decodes
it, and measures its internal frames. That step touches no shared state and depends on no other file's result, so
it is safe to run in any order. Second, everything order-dependent stays on the calling thread. `_build_frames`
walks the discovered paths in order, assigns indices from the running frame count, applies the timestamp policy,
and performs the post-read identity check, so indices stay contiguous even across files that expand into
different numbers of frames.

Failure reporting is order-stable for the same reason. A parallel scan submits one future per discovered path and
then consumes those futures in discovery order, so `Future.result()` re-raises the exception belonging to the
earliest failing path no matter which worker failed first. Remaining futures are cancelled, keeping the
sequential promise that a scan stops instead of reading the rest of a directory after a bad file. A parallel scan
may already have decoded some later files; decoding writes nothing, so that is not observable.

Threads rather than processes, because the bottleneck is Pillow's decode and resample. Those run in C and release
the GIL, while the Python-level work per file is bounded by the 128x128 metric sample. Processes would add
interpreter start-up, would have to pickle every record back across a boundary, and would multiply peak memory by
the worker count on every platform including Windows spawn; threads keep one address space, one warning-filter
installation, and one error path.

The decompression-bomb escalation moved with this change. `warnings` filters are process-global, so entering
`catch_warnings` once per file would let one worker restore the filters while another worker is still decoding.
The escalation is now installed once, on the calling thread, and stays in force for every decode the scan
performs.

The per-file integrity rules are unchanged. The size, modification time, device, and inode snapshot is taken by
the worker before decoding and compared on the calling thread after the timestamp policy has read whatever
metadata it needs, so the window a concurrently replaced file must survive still spans both the pixel read and
the metadata read.

The count is caller-supplied and bounded to 64. A default derived from the host's CPU count would leave output
identical but would make thread count and peak memory — one decoded image per worker — depend on the machine, so
the default stays one worker and parallelism is opt-in.

## Selection invariants

`select_frames` guarantees:

1. the selected count never exceeds the budget;
2. selected indices are unique and returned in chronological order;
3. timestamps are either present for all frames or none and cannot decrease;
4. every input frame receives exactly one decision;
5. selected frames respect minimum gap and near-duplicate constraints;
6. the same records and configuration produce the same result.

Public integer fields are bounded before sorting, arithmetic, formatting, or JSON encoding. Frame indices,
selection indices, ranks, dimensions, byte sizes, budgets, and renderer sizing arguments fit a non-negative signed
64-bit integer; fields such as dimensions, ranks, and budgets are additionally nonzero. The difference hash is the
single unsigned 64-bit exception. A manifest's aggregate byte count must remain within the signed 64-bit range.
Continuous settings and timestamps accept finite floats or signed-64 integer spellings. Operation boundaries
revalidate documented field bounds and result relationships because callers can construct frozen records directly.

Endpoint preservation is best-effort. The beginning is reserved first. The end is reserved only if it satisfies
constraints against earlier reservations. A one-frame budget therefore chooses the first frame, which is explicit
in its `endpoint_start` reason.

After reservations, greedy scoring is dynamic because coverage depends on the already selected timestamps. Quality
and change are intrinsic. Coverage is twice the nearest selected time distance divided by the whole sequence span,
clamped to one. This makes the middle of a large uncovered interval competitive without forcing fixed time bins.

## Decision semantics

Selected frames use `endpoint_start`, `endpoint_end`, or `highest_utility`. Rejected frames distinguish
`near_duplicate`, `min_gap`, `budget_exhausted`, and `constraint_limited`. Reasons are human-readable summaries;
`reason_code` and score fields are the stable automation surface.

Rank means greedy selection order, not chronological position. The output's `selected_indices` is chronological.

## Complexity

Scanning is linear in frame count and bounded metric pixels. Optional workers divide the wall-clock decode cost
without changing that bound, and bound peak decode memory to one image per worker. The selector computes the
sequence span once, then greedy selection compares candidates only with already selected frames. Its runtime is
approximately `O(n × budget²)` with a small user-controlled budget, and record memory is `O(n)`. The
contact-sheet output canvas is capped at 50 million pixels. Source decoding and EXIF transposition can still use
memory proportional to an input image's full resolution, so OS-level limits remain necessary for untrusted media.

## Extension points

Potential optional adapters should produce normalized measurements without changing core records or making network
access implicit. Examples include semantic embeddings, optical-flow change, or domain-specific quality signals.
Any adapter must expose its provenance and parameters in the manifest so decisions remain reproducible.

Video decoding remains outside the core. The optional adapter emits an image sequence and timestamp rule so the
scanner and selector remain unchanged. The FFmpeg adapter implements this boundary by staging a
new PNG directory, enforcing frame, live generated-PNG byte, and FFmpeg subprocess-runtime caps, continuously draining
bounded diagnostics, verifying the input snapshot before and after decoding, validating every output through the
scanner, and publishing only a complete sequence. The subprocess timeout excludes hashing, version probing, and
validation. The extraction manifest records the input SHA-256, decoder version, effective arguments, limits,
byte-accounting scope, and result; codec-specific behavior remains external.

## Evaluation boundary

The reproducible benchmark reuses production record validation and hard constraints, then changes only candidate
ordering. Time-uniform sampling uses actual frame time coordinates, change peaks test local visual-transition
preference, and a
portable hash-ranked random baseline estimates a declared seed distribution. Reports contain every trial as well as
means and population standard deviations; they never select a best random trial.

Metrics intentionally require no labels. They quantify intrinsic quality, nearest-selected content and temporal
coverage, transition proximity, and selected-set non-redundancy. They cannot determine whether a selected image is a
human-important event or improves a downstream VLM. External annotated datasets belong above this layer and must
publish their label protocol, license, splits, and decoder provenance.
