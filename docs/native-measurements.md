# Measure once, inspect thresholds without decoding again

`native-measure` captures all eight fixed `FrameMetrics` fields with exact native
PTS and a bounded historical source/configuration identity. `native-replay`
recomputes scene decisions from those measurements. It does not load PyAV or
open the recorded video path, and remains usable after the source is removed.
Capture requires the optional `[video]` extra; replay uses the base installation
without importing the optional decoder.

```bash
frame-quorum native-measure local.mkv --output-dir measurements
frame-quorum native-replay measurements/measurements.fqm.jsonl \
  --detectors adaptive luminance --minimum-votes 2 \
  --adaptive-ratio 2.5 --output-dir review-01
frame-quorum native-replay measurements/measurements.fqm.jsonl \
  --detectors threshold --dark-threshold 0.08 --output-dir review-02
```

Both destinations must be new directories under existing parents. Capture
publishes `measurements.fqm.jsonl`; replay publishes `replay.json` and
`statistics.csv`. Standard output is a small summary after publication. A stdout
failure does not undo published files. Existing `native-scenes` behavior and its
schema are unchanged; no implicit cache lookup or source substitution was added.

## Python API and replayable changes

```python
from frame_quorum import (
    DetectionConfig,
    capture_native_measurements,
    write_native_measurements,
    read_native_measurements,
    analyze_native_measurements,
    write_native_replay,
)

captured = capture_native_measurements("local.mkv")
cache_path = write_native_measurements(captured, "measurements")
loaded = read_native_measurements(cache_path)
replay = analyze_native_measurements(
    loaded,
    detectors=(DetectionConfig(detector="luminance", threshold=0.25),),
    min_scene_samples=2,
)
write_native_replay(replay, "review-01")
assert replay.source_verified is False
print(replay.analysis.cut_times)
```

Every parameter of the existing five `DetectionConfig` policies can change:
distance threshold, adaptive ratio/window/content floor, fade threshold,
hysteresis/dark sample minimum/bias/final-fade policy, per-detector scene minimum,
and detector selection. Quorum and aggregate minimum may change too. New limits
must still admit the original `video.max_frames`; the existing 1,000,000
sample-times-detector ceiling is checked before analysis.

Stream ordinal, selected interval, sampling stride, original decode budgets,
sample coordinates and measurement algorithm are historical identity, not replay
options. No missing frame or RGB image can be reconstructed. Missing metrics,
unknown fields, and unknown measurement versions are rejected, not filled with
defaults. Pixel HSV, edge maps, histograms, learned-model features and future
measurement algorithms require an explicitly supported new schema, not a guessed
conversion from the current RGB-summary metrics. Capture and replay use the same
original scene kernel as `detect_native_scenes`; no detector was reimplemented.

## Provenance is not source authentication

The cache binds all original video options, metadata, closed diagnostics, exact
PTS/time-base pairs, generation-local decode/sample indices, complete metrics,
measurement algorithm ID `frame-quorum-rgb-summary-v1`, and producer Pillow
version. The producer version is historical provenance, not a reason to rerun
measurements using the currently installed Pillow.

Capture hashes the **whole** regular local source before and after its one
measurement decode, under `video.max_source_bytes`, and compares SHA256 plus
observed device/inode/size/mtime identity. The two extra sequential hash passes
read at most twice that configured source size (plus one EOF probe per pass).
This detects ordinary observed changes, not malicious swap-and-restore, a
compromised decoder, or dishonest in-memory records. Source paths in imported
data are labels and are never opened by replay. No signature/authentication or
claim about the current file at that path is supplied by a matching cache hash.
The original path label is retained separately from the host's `Path` view and
preserved verbatim in cache/replay JSON, including POSIX filenames containing a
backslash. Use `source_path_label` for historical text, not the host-interpreted
`analysis.metadata.path`. Stream metadata describes the historical stream; raw
summary samples do not retain frame dimensions, so it cannot prove every decoded
frame shared those dimensions. Count/pixel lower and upper bounds are checked,
but imported diagnostics are not independently certified observations.

`NativeReplayResult` is a separate wrapper, not a `NativeSceneResult` subtype.
Its JSON has `kind="frame-quorum-native-replay"`,
`execution="cached_measurements"`, and `source_verified=false`. Its `analysis`
contains historical metadata and decode diagnostics, **not** a newly performed
decode. `native_scene_clips(replay)` rejects this wrapper. Explicitly extracting
`replay.analysis` does not authenticate media either: all existing unsampled,
complete-termination and exact-tail requirements remain, and actual splitting
still independently decodes the supplied source. No splitting trust boundary was
expanded. Incomplete/count-limited inputs and unknown final times stay marked as
such; neither an FPS-derived endpoint nor an unobserved final fade is invented.

## Wire format, diagnostics and resource boundaries

The canonical ASCII JSONL format has one header, exactly `sample_count` sample
records, and one completion record. Keys are sorted, separators compact, lines
end in a single LF, and floating-point values are finite. The completion SHA256
covers the exact header/sample bytes, not itself. It is the measurement digest
shown in replay provenance. Truncation, trailing data, duplicate/unknown keys,
noncanonical encoding, unreduced rationals, count/stride/PTS contradictions and
checksum changes fail. No pickle, imports, codec execution or generic object
hooks occur during cache parsing. Native PTS may be negative or equal; they must
not decrease. Rational times never pass through float or nominal FPS conversion.
This is the project's versioned JSON encoding, not RFC 8785 compatibility.
Metric integer and float spellings retain their admitted types (for example
`0` versus `0.0`); their distinct encoded bytes produce distinct digests.
Before parsing, structural nesting is limited to 16 levels. Integer tokens are
limited to 39 digits before conversion (sufficient for the 127-bit derived-time
numerator); float tokens are limited to 64 characters, and nonfinite constants
are rejected. Ordinary integer/rational fields then enforce their narrower
schema bounds. This does not rely on the process-wide integer digit safeguard.

`NativeMeasurementLimits` defaults to 64 MiB cache bytes, 16 KiB per line,
100,000 samples and 128 MiB total output bytes. Compiled ceilings are 256 MiB,
64 KiB, 1,000,000 samples and 512 MiB respectively. All settings are strict bounded
integers. Files are size-checked before loading; each read is line-limited and
the declared sample count is admitted before constructing the sample list.
Records are parsed incrementally, but replay retains the full supplied sequence:
this is offline `O(samples × detectors)` work/record memory, not O(1) streaming.
Serialization of the result's dictionary also allocates those bounded records;
output byte limits do not impose a hard Python/native RSS or CPU deadline.

CSV has one row per sample and configured detector, including suppressed
candidates, qualified votes, aggregate decisions and exact rational coordinates.
An absent adaptive score is empty and a real zero remains numeric zero. CSV is
an inspection view, not the complete replayable cache or an upstream CSV-format
compatibility claim. `render_native_detection_csv` is byte-bounded; directory
export streams rows under the aggregate output ceiling.

## Publication and failure behavior

Publication reuses the native splitter's reviewed ownership-aware primitives:
exclusive staging, fixed filenames, Windows no-replace rename or Linux
`renameat2(RENAME_NOREPLACE)`, and fail-closed unsupported platforms/filesystems.
Only still-owned file identities are cleaned. Unknown/replaced files and failed
cleanup leave explicit inspectable residue, not broad recursive deletion.
Ordinary cleanup failure does not mask a control exception. If publication may
have succeeded before acknowledgment failed, inspect the target before retrying;
the destination is never deleted to simulate rollback. Atomic visibility is not
file/directory fsync or power-loss durability. Hostile filesystem races, native
codec sandboxing and remote sources remain outside this local workflow contract.

The offline generated example `examples/native_measurement_replay.py` verifies
that two different thresholds produce manually expected cuts from one captured
VFR source after that source is deleted.
