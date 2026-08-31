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
       └──────────────► contact sheet
```

## Modules

- `scanner.py` discovers supported files, establishes natural ordering, corrects EXIF orientation, and creates
  immutable `Frame` records. Decode errors stop the scan with a path-specific message.
- `timestamps.py` owns all conversions from index, filename, or modification time to a floating-point coordinate.
- `metrics.py` implements bounded-resolution Pillow measurements and the composite content distance.
- `selector.py` validates the sequence, applies hard constraints, chooses frames, and explains every decision.
- `reporting.py` is the JSON schema boundary. It converts hashes to fixed-width hexadecimal strings and rounds
  score fields.
- `contact_sheet.py` is a presentation adapter. It does not participate in scoring.
- `demo.py` generates deterministic synthetic data used for the public example.
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

## Selection invariants

`select_frames` guarantees:

1. the selected count never exceeds the budget;
2. selected indices are unique and returned in chronological order;
3. timestamps are either present for all frames or none and cannot decrease;
4. every input frame receives exactly one decision;
5. selected frames respect minimum gap and near-duplicate constraints;
6. the same records and configuration produce the same result.

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

Scanning is linear in frame count and bounded metric pixels. The selector computes the sequence span once, then
greedy selection compares candidates only with already selected frames. Its runtime is approximately
`O(n × budget²)` with a small user-controlled budget, and record memory is `O(n)`. The contact-sheet output canvas
is capped at 50 million pixels. Source decoding and EXIF transposition can still use memory proportional to an
input image's full resolution, so OS-level limits remain necessary for untrusted media.

## Extension points

Potential optional adapters should produce normalized measurements without changing core records or making network
access implicit. Examples include semantic embeddings, optical-flow change, or domain-specific quality signals.
Any adapter must expose its provenance and parameters in the manifest so decisions remain reproducible.

Video decoding is intentionally outside the core. A future adapter may emit an image sequence and timestamp rule;
the scanner and selector then remain unchanged.
