# Exact cuts-only OTIO export

This increment passed the local Windows and Linux gates recorded below. Hosted
checks remain separate; the independent-reader observations describe the actual
interoperability subset and one retained third-party rewrite limitation.

## API and native workflow

```python
from fractions import Fraction
from pathlib import Path
from frame_quorum import OTIOCut, OTIOMedia, OTIOExportConfig, write_otio_bundle

cuts = (OTIOCut(Fraction(4), Fraction(6)), OTIOCut(Fraction(10), Fraction(11)))
media = OTIOMedia(Path("source.nut").absolute(), origin=Fraction(0))
result = write_otio_bundle(cuts, media, Path("new-editor-report"), OTIOExportConfig())
```

`render_otio(cuts, media, config=None)` returns ASCII JSON text ending in one
newline. `write_otio_bundle` returns `OTIOExportResult` after publishing
`scenes.otio` and `audit.json`; the audit contains OTIO SHA256, file length,
clip/track counts, tick rate, duration and unverified source declarations.
Result objects describe this operation, not arbitrary data validation proofs.

```bash
frame-quorum native-otio source.nut --media-origin 5 --final-end 57/10 \
  --detectors luminance --output-dir new-editor-report
python examples/native_otio.py
```

The CLI decodes with the existing native scene workflow before converting its
scene intervals. `--media-origin` is required; `--final-end` is needed when the
final endpoint is unknown. `--available-start` and `--available-end` must be
supplied together, in original native coordinates. `--include-audio` merely
declares a matching synchronous track; the exporter does not inspect audio.
Nondefault `--video-stream` is rejected. OTIO ExternalReference has no selected
media stream ordinal: restricting this adapter to ordinal zero still does not
certify how every editor chooses tracks in multi-video-stream media. The pure
API likewise exports a declared reference, not relinking or playback behavior.

`OTIOCut` coordinates and `OTIOMedia.origin` accept exact integers/Fractions,
with reduced numerator magnitude and denominator at most signed int64 maximum.
Floats and booleans are rejected. The media path must be an absolute local
`Path`; protocols/UNC/control characters are rejected. It is encoded as an
escaped file URI but neither resolved nor opened by the pure exporter. Missing
media can therefore be described for later relinking. Names/title are at most
128 Unicode scalar characters; paths at most 4096. Arbitrary metadata trees
are not accepted. Cuts are an exact tuple of `OTIOCut` objects, not lazy inputs
that could execute unbounded callbacks while a report is being written.

## Timing and declaration contract

The exporter writes original standard OTIO JSON containing one video track and,
only when explicitly requested, a corresponding caller-declared audio track.
It exports editorial references, not media, playback or proof that a source has
audio. Media coordinates are the supplied exact native time minus an explicitly
supplied `OTIOMedia.origin`. There is no inferred average-FPS conversion or
editor zero-point. Unknown media availability stays `available_range: null`;
explicit availability must contain every selected source interval.

All time coordinates use a common positive integer rate: the bounded least
common multiple of required reduced denominators. The rate cannot exceed
1,000,000,000; source starts/ends, durations and accumulated record coordinates
must fit signed integer magnitude 2**53-1. These integer pairs are exactly
representable in OTIO's binary64 value/rate fields. This does not promise exact
arbitrary floating-point operations by an editor after loading. Unrepresentable
coordinates fail; no quantization, average-FPS tail extrapolation or rounding is
performed. Source gaps are omitted in the contiguous cuts-only record track.

The public pure API requires immutable, ordered, nonoverlapping positive-duration cuts,
an absolute local media path, bounded safe names and a bounded configuration.
It does not open the media. The native adapter preserves `native_scene_clips`
requirements: complete unsampled analysis and an explicit validated final end
when the last endpoint is unknown. Export objects are not authenticated source
certificates and do not widen any native splitting input type.

At most 10,000 source clips, two tracks and 64 MiB of output are supported.
Individual clips are encoded separately. The bundle writer preflights complete
output length, including the audit document and both final newlines, before
creating staging. It reuses the existing no-replace owned-directory publication
and cleanup rules; there is no fsync crash-durability guarantee. Returned text
necessarily retains the bounded serialized document; these are logical work and
byte limits, not a Python/native RSS sandbox.

Reference gap: frozen PySceneDetect's `write_scene_list_otio` exports actual
timelines, whereas our previous optional `cmx_3600` reader smoke only validated
EDL output. This separate implementation received the independent OTIO reader
and native-pixel/PTS checks recorded below for its bounded interchange subset.
Concatenated decoding, FCP, transitions, resampling, arbitrary multitrack editing
and broad editor certification remain separate whole-repository work.

## Independent reader observations

The standard [serialized schema](https://github.com/AcademySoftwareFoundation/OpenTimelineIO/blob/main/docs/tutorials/otio-serialized-schema.md)
defines the timeline/track/clip/reference structure. The official
[RationalTime implementation](https://github.com/AcademySoftwareFoundation/OpenTimelineIO/blob/main/src/opentime/rationalTime.h)
uses binary64 `value` and `rate`; integer pairs do not make subsequent floating
computations or third-party serialization universally lossless.

On 2026-09-08 the standalone acceptance script
[`tests/verify_otio_interop.py`](../tests/verify_otio_interop.py) ran through actual
OpenTimelineIO **0.18.1** core deserialization (not our JSON reader). It used
Windows Python **3.11.2**, executable
`D:/Company/frame-quorum/.venv/Scripts/python.exe`, with `-B -I` and the existing
trusted package root `D:/uvcache/archive-v0/37SOyVtSIkJt7q_8`. No package was
installed or downloaded. All 62 hash-bearing cached `RECORD` entries matched
their declared SHA256 and lengths. `RECORD` SHA256 was
`e4929a1cfb2dd3490ef137689dc89fb227aa2b2a0c009e78372064a7026538db`;
`METADATA` SHA256 was
`2c0e5a9cf38f8bceace8a2d176df24e2957775118be58def53bd021232002cf8`.

- All **nine first-read vectors passed** with exact integer-pair fields:
  eight hand-specified range/media configurations and one actual eight-frame
  FFV1 VFR workflow. They check negative times plus explicit origin, both integer
  boundaries, null/explicit availability, URI escaping, video/audio tracks,
  exact exclusive ends and cumulative duration. Native RGB and PTS were also
  independently re-decoded; its two scenes have record durations 27/100 and
  43/100 seconds, totaling 7/10.
- **Eight core writer/reader roundtrips matched. One did not.** At the negative
  extreme our JSON integer `-9007199254740991` was initially read exactly.
  OTIO's writer then emitted `-9007199254740991.0`; its own 0.18.1 reader read
  that decimal as `-9007199254740990.0`. Original timing metadata still retained
  its integer. The first unconditional roundtrip verification consequently
  failed. The final script records this exact known third-party discrepancy
  with both complete wire strings and digests, rather than skipping it or
  counting it as a successful roundtrip. Other discrepancies still fail.

The accepted coordinate ceiling remains unchanged: our integer JSON wire and
this reader's **first read** retain the exact bounded integer pair. Do not infer
that repeated OTIO/editor rewrites are lossless over that entire range. For
such workflows retain the original export/audit and check the rewritten ranges.
This acceptance is one identified reader build, not all OTIO versions/editors.

Current development evidence: API RED collection failed before the module
existed; the first implementation had seven failures because the shared text
validator incorrectly rejected the allowed empty automatic cut name. Correcting
that adapter produced 22 passing initial cases. The later new/native and old
EDL/cache focus ran **324 passing cases**, with all new exporter statements and
branches covered; these targeted checks are not a substitute for full gates.

## Final local verification

The frozen Windows suite completed with **1675 passed and three existing
symlink-privilege skips**; its 1678-case JUnit records zero failures/errors and
452.930 seconds. On 2026-09-12, Linux Python **3.12.3** passed **1678 tests with
no skips** in 120.87 seconds (JUnit 120.749 seconds). RuntimeWarning and
ResourceWarning were errors. The new 48 cases are included in both full suites.
Source, tests, examples, benchmark inputs and locked configuration hashes were
identical before and after the Linux run.

The original **95%** statement/branch gate is unchanged. Windows coverage is
**98.0185%** (6275/6373 statements and 2085/2156 branches); Linux is **98.0537%**
(6276/6373 and 2087/2156). The exporter covers all **192 statements and 62
branches**, with no new coverage exclusions.

The first Linux attempt retained **1677 passes and one failure**: an isolated
`python -I` child could not import the package from a PYTHONPATH-only test
environment. Installing the already source-verified wheel offline into that
environment corrected the setup; no production code or test was changed, and
the entire suite was repeated. All nine test/runtime packages came from existing
cached wheels, with hashes checked against the unchanged frozen requirements.
The failed attempt remains separate from final acceptance.

Ruff lint/format (108 files), strict Mypy (34 source modules), Bandit, the offline
frozen 64-package lock and whitespace checks pass. The sdist-to-wheel build,
strict Twine and wheel-content gates pass. All 35 package files and 135 sdist
source entries match the current source; isolated installed package files also
match. The native VFR example passes under `python -B -I`. A separate installed
wheel recheck of OpenTimelineIO 0.18.1 on 2026-09-12 again produced nine exact
first reads, eight matching rewrites and the same one retained negative-extreme
rewrite discrepancy. Package checks are repeated after documentation updates;
local validation does not imply a hosted check or broader editor certification.
