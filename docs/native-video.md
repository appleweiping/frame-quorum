# Native local-video streams

`NativeVideoStream` incrementally decodes one local video with optional PyAV.
Each returned snapshot owns RGB bytes and keeps the native integer presentation
timestamp (PTS) and rational time base. It does not extract a temporary image
directory, interpolate timestamps from FPS, or download media or models.

Install the optional decoder and run a fully offline generated-video example:

```bash
python -m pip install -e ".[video]"
python examples/native_pts.py
```

The development/CI lock pins PyAV 18.1.0. Native codec versions depend on the
installed wheel/build; they are not the version of an external FFmpeg executable.
Image-directory APIs remain usable without importing or installing PyAV.

## Exact timing and ownership

```python
from fractions import Fraction
from frame_quorum import NativeVideoConfig, NativeVideoStream

config = NativeVideoConfig(
    start=Fraction(5),
    end=Fraction(53, 10),
    frame_step=1,
    max_frames=100,
    max_decoded_frames=1_000,
)
with NativeVideoStream("local.mkv", config) as stream:
    print(stream.metadata.to_dict())
    for frame in stream:
        assert frame.presentation_time == frame.pts * frame.time_base
        print(frame.to_dict(), frame.measure().serializable())
print(stream.diagnostics.to_dict())
```

The interval is start-inclusive and end-exclusive in **absolute native
presentation seconds**. A video beginning at PTS 5000 with a time base of 1/1000
starts at 5 seconds, not at zero. API bounds accept exact `Fraction` or integer
values, not floats. Missing PTS/time base and decreasing presentation time fail;
equal timestamps are allowed. Stream `average_rate` and `base_rate` are optional
descriptive metadata, never the source of a frame timestamp.

The immutable `NativeVideoFrame` contains detached RGB `bytes`. `frame.image()`
creates a fresh caller-owned Pillow image; close it, normally with `with`.
`frame.measure()` creates and closes that image internally and reuses the normal
visual measurements. Those measurements are floating-point summaries, not an
exact-timestamp conversion into the existing image-file `Frame` schema. No
implicit native-video-to-CFR scene/EDL conversion is performed. Use an explicit
quantization policy when bridging to [CFR timecodes](timecodes-and-editing.md).

`to_dict()` emits PTS integers and rational numerator/denominator objects, without
RGB bytes. Consumers must preserve integer precision: native PTS can exceed the
exact range of JavaScript's binary64 number, and a product's rational numerator
can exceed signed 64 bits. Python's integer-preserving JSON decoder is suitable.

## Seeking, sampling, and indices

`stream.seek(Fraction(6), end=Fraction(7))` closes the old decoder and opens a new
decode generation on the same file snapshot. The requested start is floored into
stream-time-base units for a backward keyframe seek, then decoded frames are
filtered against the original exact bound. This follows the
[PyAV seek contract](https://pyav.basswood.io/docs/stable/api/container.html).
Unsupported native seeking fails explicitly. The API does not estimate a VFR
frame number from FPS.

Within the active context, `seek(None)` replays from the beginning. Automatic EOF
or range closure allows another seek when lifetime budgets remain. Explicit
`close()`, a failed stream, or leaving the context prohibits seeking. Seeking
does not reset any work/output budget or returned-frame count.

Three coordinates deliberately have different meanings:

| Attribute | Meaning |
| --- | --- |
| `generation` | Zero initially, incremented for each seek/replay |
| `decode_index` | Zero-based within that generation, including pre-range and stride-discarded frames |
| `sample_index` | Zero-based lifetime count of returned snapshots |

None promises a stable global frame index after a keyframe seek. `frame_step=2`
returns the first, third, fifth, ... eligible in-range frame in each generation.
Skipped frames are still decoded, timestamp-validated and counted, but not
converted into retained RGB snapshots. An end-boundary frame is consumed and
counted to establish that the interval ended, but is not returned.

## File-only boundary

Only regular local files with Matroska/WebM, AVI, or `ftyp`-stamped MP4/MOV headers
are accepted. The header selects a fixed demuxer; broad format autodetection is
not used. URLs, protocol paths, UNC paths, symbolic-link inputs, HLS/DASH/concat
playlists, raw streams and unrecognized headers are refused. This is deliberately
narrower than everything PyAV/FFmpeg can decode.

The already-open file is passed through a bounded reader. Secondary file/protocol
opens are denied through
[PyAV's `io_open` callback](https://pyav.basswood.io/docs/stable/api/_globals.html),
the native protocol whitelist is restricted to `file`, and MOV external data
references/absolute paths are disabled. See the primary
[FFmpeg protocol controls](https://ffmpeg.org/ffmpeg-protocols.html) and
[MOV demuxer controls](https://ffmpeg.org/ffmpeg-formats.html).
Tests exercise the real callback against a local HLS fixture, separately from
the product's earlier playlist rejection. No network server or media download is
needed for this test.

Size, modification time, device and inode are compared before native reads and
across replay. An observed change stops the stream. This is not a content hash,
authenticated source, adversarial filesystem-race defense, or proof that an
external writer cannot change bytes between checks. Do not concurrently modify
the input.

## Limits and lifecycle

| Configuration | Default | Accounting |
| --- | ---: | --- |
| `max_source_bytes` | 1,000,000,000 | Initial regular-file length, not cumulative bytes reread across seeks |
| `max_frames` | 10,000 | Returned RGB snapshots across the entire stream lifetime |
| `max_decoded_frames` | 100,000 | Native frames delivered to Python, including discarded frames and replay |
| `max_frame_pixels` | 16,777,216 | Header dimensions and each delivered decoded frame, before RGB conversion |
| `max_total_pixels` | 1,000,000,000 | Sum of valid-dimension decoded-frame pixels across the lifetime |

All settings also have strict compiled ceilings; booleans are not accepted as
integer settings. A video-stream selector is zero-based among video streams;
metadata's native stream index may differ when the container also holds audio.
Only the selected video stream is decoded.

Iteration keeps no list of prior frames. Each retained snapshot costs three bytes
per pixel plus record overhead; callers accumulating snapshots own that growing
memory. Native probing, decoder/reference-frame buffers, image conversion and
codec CPU work are outside this accounting. Header probing can allocate before
Python sees dimensions; total-pixel limits are checked after a native frame is
decoded, before RGB conversion. The violating frame has already cost work and
`decoded_pixels_observed` can exceed the configured total by that frame. An
invalid/oversized-dimension frame is counted as decoded but not added to that
pixel sum. These settings are **not a hard process-memory or runtime sandbox**.
Codec internal buffering/probing does not count as Python-delivered frames.

The context owns the native container, iterator and file on its entering thread.
Cross-thread or reentrant reading is refused. Breaking a loop alone leaves the
context live: leave `with` or explicitly call `close()` for early cleanup.
EOF, interval end, and count limits close automatically. The last allowed output
is returned without decoding an extra frame merely to probe EOF, so a count-limit
status must not be interpreted as complete-source consumption.

Diagnostics distinguish `eof`, `range_end`, `frame_limit`, `decode_limit`,
`pixel_limit`, `source_limit`, `closed`, `interrupted`, and `error` (plus initial
`new`/live `open`). Count limits are normal bounded termination; source/pixel or
decode errors raise. Python interruption triggers cleanup when control returns
from native code; this API does not provide a hard read timeout or preempt a
stuck native call.

Cleanup attempts every resource even when an earlier close fails. A failed
resource reference is retained, `closed` stays false, status becomes error or
interrupted, and `cleanup_errors` records stable resource names. An ordinary
cleanup failure raises `ScanError` unless another setup/decode/context-body
exception is already being propagated; that primary exception is preserved.
Genuine interrupts are never silently swallowed. Keep the stream object and
retry `close()` if cleanup fails. After a successful retry, `closed` becomes true
but failure status/history remain available. There is no automatic finalizer
guarantee for abandoned stream objects.

## Streaming CLI

```bash
frame-quorum native-scan local.mkv --start 5 --end 53/10 --max-frames 100
```

`native-scan` writes JSON Lines to stdout: a `source` metadata record, zero or
more `frame` records with exact PTS and visual measurements, then one `summary`
record with final diagnostics. Each record has `kind: frame-quorum-native-scan`,
`schema_version: 1`, and `record_type`. Start/end text can be an integer, decimal
or fraction; decimal text is converted directly to a rational value.
Time text is limited to 128 characters and decimal exponent magnitude to 128
before constructing a rational; the API's rational bounds still apply afterward.

A failure can leave a useful **partial prefix** on stdout, returns a nonzero exit
code, and does not emit a successful summary. Consumers must require the terminal
summary and inspect its status; a summary with `frame_limit` is not EOF. Output
does not include RGB images, an authenticated input digest or native library
provenance. The API's metadata similarly is not a complete container inventory.

## Verification and remaining scope

Tests generate local lossless FFV1 CFR and nonzero-PTS VFR clips with independent
hand-authored PTS/color expectations, verify direct native decode, and exercise
MPEG-4 interframe keyframe seek, replay, interval/stride behavior, audio-only and
corrupt containers, early close and file deletion. Controlled decoders cover
missing/decreasing timestamps, limits, observed file mutation, failed setup,
reentrancy, and each resource's close failure/retry. No downloaded fixture or
trained model is used.

This backend is not yet a streaming scene-manager integration, audio processor,
clip exporter, live-source reader, GPU decoder, or codec compatibility benchmark.
The [whole-repository audit](parity-detection.md) keeps those gaps explicit.
