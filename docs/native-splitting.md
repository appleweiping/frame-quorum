# Exact-PTS native video splitting

`split_native_video(source, output_dir, clips, config)` transcodes explicitly
requested `NativeClip(start, end)` intervals into actual local video files.
Intervals are sorted, non-overlapping, start-inclusive/end-exclusive **absolute
native presentation seconds**, expressed as exact integers or `Fraction`s.
Every frame in an interval is decoded and encoded; no sampling, inferred frame
rate or keyframe-aligned codec copy is used. Empty intervals and duplicate or
decreasing selected timestamps are rejected, not silently repaired.

```python
from fractions import Fraction
from frame_quorum import NativeClip, NativeSplitConfig, split_native_video

result = split_native_video(
    "local.mkv",
    "new-clips",
    (NativeClip(5, Fraction(1001, 100)), NativeClip(Fraction(1001, 100), 15)),
    NativeSplitConfig(max_frames=5000, max_output_bytes=100_000_000),
)
print(result.to_dict())
```

Equivalent CLI and an offline generated VFR demonstration:

```bash
frame-quorum native-split local.mkv --output-dir new-clips \
  --clip 5 1001/100 --clip 1001/100 15 --max-frames 5000
python examples/native_splitting.py
```

The parent directory must exist; `new-clips` must not. The CLI accepts repeated
exact rational/decimal bounds and the configuration fields below as kebab-case
options (for example `--max-verification-pixels`). It has no frame-stride or FPS
option and does not load arbitrary codec arguments. Its JSON summary is written
**after** publication: a broken stdout pipe does not undo the published clips.

## Output and timing contract

The fixed output is FFV1 `bgr0` video in a NUT container (`clip-000000.nut`, ...).
There is no audio, subtitle, attachment, metadata preservation or arbitrary
codec/format option. NUT was selected to preserve exact rational presentation
times, including 1001/30000-second spacing that a millisecond container clock
cannot represent exactly. It is a lossless decoded-RGB workflow, not a promise
to preserve the source's original YUV, HDR, alpha or compressed packet bytes.

The first selected frame in each clip has output presentation time zero.
For every later selected frame:

`output_pts * output_time_base = source_pts * source_time_base - first_selected_time`

Rebasing uses the first selected frame, **not** the requested interval start.
The start can fall between native frames. Output integer PTS/time bases can
differ from the source representation; the rational presentation value must
match exactly. The encoder uses a bounded rational tick; nonrepresentable times
or ticks outside signed 64-bit range fail before that frame is encoded. There
is no float rounding or fallback to constant-rate timestamps.

After encoding, every output clip is reopened and its complete frame count,
exact presentation times, dimensions and RGB SHA-256 values are checked against
the selected source frames. Extra, missing, reordered or changed output frames
prevent publication. A bounded `manifest.json` records the exact source/output
PTS mapping, requested intervals, original RGB hashes and actual output bytes.
These hashes detect changes; they are not signatures or writer authentication.
Container duration and the final frame's display duration are not asserted to
equal the requested interval length: the decoder does not provide a validated
exclusive duration for the source's final frame.

`native_scene_clips(result, final_end=...)` bridges existing scene results. It
requires an unsampled, successfully completed `NativeSceneResult` and refuses
frame/decode-limit truncation. An unknown final endpoint requires an explicit
exact bound greater than the final sample time. It never adds `1/FPS` to guess
an endpoint. Detected boundaries are still detector decisions, not ground truth.
The result's metadata is not an authenticated source identity: splitting always
decodes the actual supplied local file independently.

## Bounded ownership and publication

The source retains `NativeVideoStream`'s local regular-file, fixed-demuxer,
secondary-open rejection and observed-mutation checks. A narrow NUT signature
is supported so these generated clips can use that same decoder. This is not
arbitrary-container, network, codec-sandbox or audio support.

`NativeSplitConfig` bounds clip count, retained expected-frame/hash records,
source decoding work, source bytes, frame pixels, aggregate decoded pixels,
serialized manifest bytes and total output-file bytes. Output writes check
growth before writing. Native probing/codec buffers and trusted-library CPU
allocations remain outside a hard RSS or time guarantee. The operation keeps
bounded per-frame metadata and the current RGB snapshot, not a list of full
source images. Verification has its own bounded pass over the selected frames.

| Configuration | Default | Accounting |
| --- | ---: | --- |
| `video_stream` | 0 | Video-stream ordinal, not the container's global stream index |
| `max_clips` | 100 | Nonempty requested intervals; compiled ceiling 1000 |
| `max_frames` | 10,000 | Selected source records/hashes across every clip; ceiling 100,000 |
| `max_decoded_frames` | 100,000 | One sequential source pass from the beginning; ceiling 1,000,000 |
| `max_source_bytes` | 1,000,000,000 | Source regular-file length |
| `max_frame_pixels` | 16,777,216 | Per-frame dimensions, both passes |
| `max_source_pixels` | 1,000,000,000 | All decoded source pixels, including gaps and the end-boundary frame |
| `max_verification_pixels` | 1,000,000,000 | Aggregate second-pass decoded pixels, not reset per clip |
| `max_output_bytes` | 1,000,000,000 | Sum of every clip and manifest file's maximum written extent |
| `max_manifest_bytes` | 16 MiB | Serialized JSON bytes; ceiling 64 MiB |

Each verification pass permits at most its expected frame count plus one so an
extra decoded frame is detected; across clips the ceiling is selected frames
plus clip count. A native frame's actual dimensions are observable only after
native decoding: the first violating frame can already have been allocated but
is rejected before its RGB conversion. Source count-limit termination is never
accepted as EOF, even if the cap happens to equal the file's true frame count.
Allow work for EOF/end-boundary confirmation. Input-proportional metadata is
`O(selected frames + clips)` and verification adds bounded manifest rows; the
JSON byte cap is checked while streaming serialization, not a hard Python-object
memory cap. The encoder tick denominator is at most 1,000,000,000; each rebased
PTS must be a representable signed 64-bit nonnegative integer in that tick.

All files are written into an exclusively created sibling staging directory.
The target's parent must already exist. No existing destination is overwritten,
including an empty directory or a destination created by a competing publisher.
Publication uses Windows no-replace rename or Linux `renameat2` with
`RENAME_NOREPLACE`; unsupported platforms/filesystems fail closed. It is atomic
visibility of this directory, not a filesystem power-loss durability guarantee.
Files and directories are not fsynced. If publication returns an error without
acknowledging success, the diagnostic tells the caller to inspect the target;
failure cleanup never removes a potentially published destination.
No external executable, shell helper or codec download is used.

Failure attempts all owned resource closes and removes only newly created
staging files whose recorded device/inode identities are unchanged. Unknown or
replaced entries and replacement staging directories are preserved and reported.
Existing destinations are never cleanup targets. Ordinary
cleanup errors do not hide an active interruption; residual stage paths are
reported when cleanup cannot finish. Genuine cleanup control exceptions are
not swallowed. Concurrent hostile filesystem mutation and abandoned process
cleanup are outside this same-user local workflow's guarantees.

## Reference scope

The frozen [PySceneDetect CLI splitting contract](https://github.com/Breakthrough/PySceneDetect/blob/24953b0bf76af17c450bc143d330eea48fc5e276/docs/cli.rst)
includes FFmpeg encoding, less-precise codec copy, mkvmerge, configurable codec
arguments and audio/subtitle mapping. This implementation uses its own bounded
native lossless-video contract; it does not claim those modes or complete
reference compatibility. [PyAV container APIs](https://pyav.basswood.io/docs/stable/api/container.html)
provide the native encode/mux lifecycle. No upstream implementation was copied.

Full-repository parity, other output containers/codecs, audio preservation,
stream copy, editor interchange and external video compatibility/throughput
benchmarks remain open in [the repository audit](parity-detection.md).
