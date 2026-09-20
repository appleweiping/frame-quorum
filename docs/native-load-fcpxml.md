# Import a scene-start CSV into FCPXML

`native-load-fcpxml` turns an edited PySceneDetect-style `list-scenes` CSV into
one video-only FCPXML 1.9 timeline without running a scene detector:

```bash
frame-quorum native-load-fcpxml source.nut --scene-csv scenes.csv \
  --frame-rate 25 --final-end 7/25 --output-dir new-editor-xml
python -I examples/native_load_fcpxml.py
```

The output directory must not exist. A successful bundle contains exactly
`scenes.fcpxml` and `audit.json`. The XML references an absolute source `file:`
URI and **can reveal local path names** when shared. No media or CSV is copied.
The sidecar binds the exact CSV bytes read by SHA-256, the XML byte count and
digest, scene/frame counts, declared rate and final endpoint. Its
`source_verified=false`, `source_content_authenticated=false` and
`editor_import_verified=false` flags are intentional: the source is decoded
once to check timing, but its content is not durably authenticated and no
Final Cut Pro import has been performed.

## Input and timing contract

`load_native_scene_csv` accepts bounded UTF-8 CSV (optional BOM; LF or CRLF;
RFC-4180 quotes). It requires unique `Scene Number` and `Start Frame`
columns, with any other columns ignored. It also accepts the optional leading
`Timecode List:` row from PySceneDetect's default `list-scenes` output, or
the empty leading row of a no-cut list. `Scene Number` is contiguous `1..N`;
`Start Frame` is strictly increasing one-based decimal, starting at `1`.
Thus CSV starts `1,3,6` become decoded ordinals `0,2,5`. Stale values in
`End Frame`, timecode or other columns do not override those starts. Invalid
or ambiguous rows are rejected, not sorted, rounded or deduplicated. Input
has an 8 MiB hard ceiling and at most 10,000 scene rows.

The composite reads a local regular nonsymlink video stream zero from frame
zero to actual EOF, without a detector. It checks contiguous decoded ordinals,
stable dimensions and exact CFR presentation times
`first_pts + ordinal / frame_rate` using rational arithmetic. VFR, duplicate
or decreasing PTS, sampling, seeks, windows, count-limit termination, empty
media and CSV cuts outside the observed frames fail before output staging.
The default accepted frame limit is 10,000; the hard maximum is 100,000.
`--final-end` must equal `first_pts + decoded_frame_count / frame_rate`.
That exclusive last-frame endpoint is a **caller declaration of frame
duration**, not an additional timestamp observed at EOF. A nonzero initial
PTS is supported; editor source coordinates subtract it exactly.

Python callers can parse CSV separately or run the complete composition:

```python
from fractions import Fraction
from frame_quorum import load_native_scene_csv, write_loaded_fcpxml_bundle

starts = load_native_scene_csv("scenes.csv")
print(starts.start_ordinals)
published = write_loaded_fcpxml_bundle(
    "source.nut",
    "scenes.csv",
    "new-editor-xml",
    frame_rate=Fraction(25),
    final_end=Fraction(7, 25),
)
```

The FCPXML uses contiguous half-open source spans and cumulative record
offsets, with the same XML renderer and frame-lattice checks as
[`native-fcpxml`](fcpxml-export.md). It does not create a `NativeSceneResult`:
CSV decisions are imported user input, not fabricated detector statistics.
The source can change after the scan, and an editor can later relink to other
media; neither is certified by this export.

## Failure and reference limits

Malformed/oversized CSV or invalid options fail before video decoding or
output staging. Decode, cadence and endpoint failures also fail before staging.
The bundle writer reconciles staged XML and audit lengths/digests before a
no-replace publication. Short writes, close failures and a racing target must
not publish a partial bundle or delete a foreign directory. If publication
may have succeeded despite a lost acknowledgment, inspect the reported
target before retrying. A stdout write/flush failure does not roll back an
already complete bundle.

The frozen PySceneDetect
[`load-scenes` command](https://github.com/Breakthrough/PySceneDetect/blob/24953b0bf76af17c450bc143d330eea48fc5e276/docs/cli.rst#L444-L475)
is the user-facing workflow category. This profile only imports one-based
`Start Frame` CSV into FCPXML. It does not implement `Start Timecode`, other
columns, locale-encoded CSV, reference sorting/postprocessing, arbitrary
chained outputs, VFR conversion, FCP7, FCPXML DTD validation, real editor
import, audio, or whole-reference parity. The generated example compares
independent decoded CFR ordinals and pixels with exact XML intervals; it is
not a broad codec or editor compatibility test.
