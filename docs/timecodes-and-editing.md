# Exact timecodes and editing export

Frame Quorum distinguishes three coordinates:

- A frame number is an integer index on a declared constant-frame-rate grid.
- Elapsed time is `frame_number / rate`, retained as an exact `Fraction`.
- A SMPTE-style label is a frame counter. At NTSC rates its nominal clock is
  not identical to elapsed time. Drop-frame skips labels, not image frames.

```python
from frame_quorum import FrameRate, FrameTimecode, Shot, render_edl

rate = FrameRate.parse("30000/1001")
cut = FrameTimecode(1800, rate, drop_frame=True)
assert cut.smpte() == "00:01:00;02"
assert str(cut.seconds) == "3003/50"
assert cut.timestamp() == "00:01:00.060"
assert FrameTimecode.parse_smpte(cut.smpte(), rate) == cut

edl = render_edl(
    [Shot(0, 0, 1800), Shot(1, 1800, 3600)],
    rate,
    drop_frame=True,
    reel="CAMERA1",
    title="Scene assembly",
)
print(edl)
```

Rates are exact integer/fraction/decimal strings or `Fraction` values. A binary
float is rejected. `"29.97"` means exactly `2997/100`, **not** `30000/1001`;
use the latter for NTSC. Rational rates in `(0, 1000]` are supported for elapsed
time. Integer rates and exact `24000/1001`, `30000/1001`, `60000/1001`,
`120000/1001` support text frame counters. Drop-frame labels support exactly
`30000/1001` and `60000/1001`. At these rates, the first two or four labels of
each minute are omitted, except every tenth minute. Parsing rejects omitted
labels. The APIs do not implement binary SMPTE ancillary-data encoding.

`from_seconds`, `from_timestamp`, `from_pts` and `rescale` accept explicit
`rounding="floor"`, `"ceil"` or `"nearest"` (default). Nearest ties go forward.
`from_timestamp` parses elapsed `HH:MM:SS[.fraction]`, while `parse_smpte`
parses counter labels. Formatting timestamps rounds once using integer
arithmetic, including carries into the next second/minute. Precision is 0–9
decimal places. `shift_frames` preserves the exact rate and label mode.

`from_pts(pts, Fraction(1, 90000), rate, origin_pts=...)` explicitly quantizes
a presentation timestamp onto a CFR grid. It is **not** a VFR decoder and
does not preserve sub-frame presentation timing; retain the original PTS for
that purpose. Index bounds are nonnegative signed-64 values; negative shifts
and rescaling that would exceed these bounds fail. Timecode hours do not
wrap silently. Explicit `smpte(wrap_24_hours=True)` is available for display.

## Scene workflow

```bash
frame-quorum scenes frames/ --output-dir reports/scenes \
  --detector adaptive --timecode-rate 30000/1001 --drop-frame --edl
```

`--timecode-rate` sets the exact CFR export grid and overrides the scanner's
floating index-timestamp rate for consistent diagnostic timestamps. It requires
`--timestamp-mode index`; filenames, EXIF and file times are not assumed CFR.
For sampled video, this rate refers to the **sample sequence**, not the source
video's native frame numbers. Do not apply sampled-frame EDL bounds to the
original video without an explicit coordinate mapping. Expanded animations
are likewise interpreted on this declared grid, not their encoded durations.

The output bundle contains `scenes.json`, `statistics.csv` and
`timecodes.csv`; `--edl` adds `scenes.edl`. All requested outputs are prepared
before the report bundle is committed. Invalid rates/editor settings do not
replace existing reports. Optional files from an earlier run are not deleted
when an export option is omitted; consumers should use the files requested by
their current command. The timing CSV records exact rational seconds,
rounded elapsed timestamps and counter labels alongside integer coordinates.

`render_edl` writes a bounded, single-reel, video-only, cuts-only CMX-style EDL:

- All out-points are exclusive. Source scenes may contain gaps; the record
  timeline assembles their durations contiguously.
- `source_start` and `record_start` are frame offsets, not seconds.
- Nominal 24/25/30 FPS only; up to 999 scenes, one 1–8-character ASCII reel,
  one printable-ASCII title. Coordinates reaching 24 label-hours are rejected.
- No audio, dissolves, speed changes, source filename resolution, multitrack
  layout or automatic rate conversion. It does not cut media files itself.

The half-open timing CSV allows up to 100,000 ordered nonoverlapping scenes.
Both exporters validate scene ordinals, durations and coordinate types before
returning text. Strings are assembled in memory within the explicit scene cap.

## Verification and standards boundary

Tests use explicit minute/hour/day vectors and a separate counter enumerator
that walks legal labels through eleven minutes for both drop-frame rates.
They also check rational rescaling, rounding carries, invalid/missing labels,
malformed inputs, source gaps, exclusive out-points and the full CLI workflow.
These checks establish the documented subset, not universal editor support.

A separate OpenTimelineIO `cmx_3600` reader was run against a generated 25 FPS
EDL: it reconstructed both disjoint source ranges (100–150 and 250–275 frames)
and the contiguous 75-frame record duration. This is an independently executed
interchange smoke test, not a claim that every editor or EDL feature was tested.

The [FFmpeg timecode API](https://ffmpeg.org/doxygen/trunk/timecode_8h.html)
documents the distinction between frame numbers, rational rates, drop-frame
adjustment, textual labels and binary packing. The
[OpenTimelineIO adapter documentation](https://opentimelineio.readthedocs.io/en/v0.15/tutorials/adapters.html)
identifies CMX3600 as a separate editorial interchange format and points to
its full specification. Frame Quorum implements only the subset above;
native decoding, VFR timelines and the other editor formats remain open work.
