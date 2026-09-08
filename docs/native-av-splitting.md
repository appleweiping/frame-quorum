# Exact video and PCM16 scene clips

This increment adds a separate `split_native_av(source, output_dir, clips,
config)` contract. It does not change `split_native_video`, its video-only
manifest, or any previous measurement cache. `clips` is the existing bounded,
ordered, non-overlapping tuple of exact half-open `NativeClip` intervals.

```python
from fractions import Fraction
from frame_quorum import NativeAVSplitConfig, NativeClip, split_native_av

result = split_native_av(
    "local.nut",
    "new-clips",
    (NativeClip(Fraction(5), Fraction(51, 10)),),
    NativeAVSplitConfig(video_stream=0, audio_stream=0),
)
print(result.to_dict())
```

The equivalent CLI is `frame-quorum native-av-split local.nut --output-dir
new-clips --clip 5 51/10`. Repeat `--clip START END` for ordered intervals;
all config fields have explicit dashed integer flags. The generated
[offline example](../examples/native_av_splitting.py) checks actual stereo PCM,
VFR frames and sub-audio-sample track offsets without downloading source media.

## Admitted media and time

Select one video stream and one audio stream explicitly by their type-relative
ordinals. The source is a regular local file accepted by the existing fixed
demuxer/secondary-open rules. Audio must decode to signed 16-bit packed or
planar PCM, mono or stereo, with a constant sample rate in 8000..192000 Hz.
This initial implementation checks for a little-endian host before opening
media; other host byte orders are rejected, not silently hashed as little-endian.
No resampling, float conversion, missing-timestamp synthesis, discontinuity
repair, subtitle, attachment, metadata or compressed-packet preservation is
performed. Every requested clip must contain both video frames and audio
samples; an empty selected track fails the complete operation.

NUT can expose its original one/two audio channels without a named speaker
layout. Such unspecified one/two-channel layouts retain their exact channel
order; this is not proof of a spatial speaker assignment. Other explicitly
named two-channel layouts are rejected rather than remapped.

Let `A` be the first actually decoded source audio sample time and `Fs` its
sample rate. All later decoded audio blocks must lie on the continuous lattice
`A + n/Fs`. For each clip the common output epoch is

`E = A + floor((clip.start - A) * Fs) / Fs`.

Each output video frame and audio sample preserves exactly `source_time - E`.
Neither track is independently shifted to zero. Only source video frames and
audio sample **start times** in `[clip.start, clip.end)` are selected. The first
sample at an exclusive audio endpoint is excluded, even inside a decoded block.
An audio sample beginning before the endpoint can have its support interval
extend beyond it by less than one sample period; it is not synthesized or
fractionally resampled. `E` can precede the requested start by less than `1/Fs`;
this is an explicit initial timeline gap, not inserted silent/audio/video data.
An originally later-starting track retains its larger original leading gap.
This validates only the actually consumed prefix through the requested range;
unused trailing audio is not decoded or certified as continuous.

The output is FFV1 `bgr0` plus `pcm_s16le` in NUT. NUT negotiates the audio clock
to `1/Fs`; assigning an arbitrary finer audio time base can silently quantize
offsets. A preliminary independent PyAV 18.1 memory probe demonstrated that
fact. With clip start `1001/30000`, source audio anchor zero and 48000 Hz, the
chosen epoch is `1601/48000`: real reopened video time `1/80000` and audio time
`1/48000` preserved their relative offset and original PCM bytes exactly.
This small feasibility observation is not the completed implementation gate.

## Lifecycle and bounds

Two bounded source readers (video and audio) share the same observed source
fingerprint and retain at most one next frame/block each. Work is charged to
both readers. The audio handle and original path identity are checked again
before closure, including when native input buffering caused no further reads.
This does not detect adversarial swap-and-restore or authenticate a source.
Work includes discarded gaps; their separate reads are not a claim
of one shared demux pass. A clip encoder consumes the two inputs in presentation
order. The fixed codecs must return one immediate, timestamp-preserving packet
per input and no delayed flush packet; unsupported buffering/reordering fails.
At most one observed packet is admitted to the mux call at a time, with a strict
packet-byte ceiling. Native encode result allocation, probing, mux internals and
stuck native calls are not hard RSS or wall-time guarantees.

`NativeAVSplitConfig` preserves the existing video limits and adds:

| Field | Default | Accounting |
| --- | ---: | --- |
| `audio_stream` | 0 | Audio-stream ordinal, distinct from global stream index |
| `max_streams` | 32 | Source container stream entries admitted during setup |
| `max_audio_blocks` | 100000 | Decoded source audio blocks, including discarded intervals |
| `max_audio_block_samples` | 1048576 | Per-channel samples in any decoded source/verification block |
| `max_source_audio_samples` | 48000000 | Source per-channel sample positions, including discarded gaps |
| `max_audio_samples` | 48000000 | Selected per-channel samples across all clips |
| `max_verification_audio_samples` | 48000000 | Independent reopened output per-channel samples |
| `max_verification_audio_blocks` | 100000 | Independent reopened output blocks across all clips |
| `max_pending_packet_bytes` | 64 MiB | Each observed immediate encoded packet before muxing |

Each value has a compiled ceiling and rejects booleans/non-integers. Actual
native audio blocks are inspected after decode but before copying their PCM
planes; a violating block has already cost native allocation/work. Mono/stereo
means canonical PCM bytes cost two/four times the per-channel sample count.
Selected slices are admitted against the remaining selected-sample budget before
copying them. A block cap prevents another decode attempt, including an EOF
probe; confirming a complete output therefore needs unused block allowance.
The one next source block, its selected slice, a next-block transition and
native input/output audio frames can coexist. Block caps bound these Python
PCM buffers; they do not mean only one PCM allocation exists at every instant.
Video expected rows are bounded by the existing selected-frame cap. Audio
verification uses a running canonical interleaved PCM hash, sample count and
time-lattice checks, not retention of all source PCM samples or dependence on
the encoder's block segmentation.

Every output is independently reopened: all video frame RGB/PTS/dimensions and
all audio PCM bytes/times/rate/channels/count must agree before publication.
The new manifest records requested bounds, common epochs, original coordinates,
actual track counts and verification evidence. It is not source authentication,
a claim about the last video frame's display duration, or whole-repository
parity. Existing file fingerprints detect observed mutations, not adversarial
swap-and-restore.

No existing destination is overwritten. Reuse the reviewed owned-file identity
cleanup and Windows/Linux no-replace publication behavior. Ordinary cleanup
failures cannot hide active control exceptions. Unknown entries/replacements
remain for inspection; an unacknowledged publication requires inspecting the
destination before retrying. Atomic visibility is not fsync durability.

## Implementation verification

Baseline: signed `e9aeb7ea91955b1c7f131f23ef180ab696144044`. All source/tests were
frozen during the final full runs on 2026-09-08; neither dependencies, versions,
coverage exclusions nor the original 95% coverage gate were changed.

| Local platform | Full result | Statement/branch coverage | Time |
| --- | --- | ---: | ---: |
| Windows, Python 3.11.2 | 1627 passed, 3 existing symlink-permission skips | 97.95% | 263.40 s |
| WSL/Linux, Python 3.12.3 | 1630 passed, no skips | 97.99% | 262.65 s |

Both used PyAV 18.1.0, Pillow 12.3.0 and pytest 9.1.1, with `RuntimeWarning` and
`ResourceWarning` treated as errors. Linux used a task-specific temporary venv
and the unchanged frozen requirements export, installing with SHA256 checking
in the same shell lifetime as the tests. This executes actual NUT codecs and
Linux no-replace publication, not mocked platform checks. `native_av.py`
reached 100% statement/branch coverage; `native_av_splitting.py` reached 98.81%
on both platforms, with two defensive guards left uncovered.

The 213 new cases comprise 163 unit and 50 actual native/integration/example
cases. Independent interval membership, PCM byte ordering, exact sample-grid
arithmetic and actual reopened RGB/PTS/PCM oracles cover stereo/mono, source
blocks crossing cut boundaries, nonzero and unequal track starts, 44.1/48/192
kHz against non-sample-aligned NTSC bounds, a selected second real audio track,
and explicit EOF/range-tail handling. Failure cases exercise every limit,
missing/float/discontinuous audio, lost output acknowledgements, competing
publication, unknown staging entries, exported-buffer release and primary
control exceptions with secondary cleanup failures. All prior video-only
regressions and three fixed cache-wire goldens still pass.

A non-editable wheel installed into a fresh scoped environment ran the actual
[generated stereo/VFR example](../examples/native_av_splitting.py) under Python
`-I`, without a source import path. It independently checked 2 clips, 4 video
frames and 6406 stereo sample positions. All 34 package files matched source,
wheel and installed copies byte-for-byte. Wheel SHA256:
`897395402e61376cd1f058a66a0337a8811f84401c14d3092dfc73ccb85df698`.
Ruff lint/format (102 files), strict Mypy (33 modules), configured Bandit,
frozen-lock checks, strict Twine/wheel contents, documentation links and
source/sdist comparisons also passed. These are local checks, not hosted CI,
arbitrary codecs/containers or a broad external interoperability corpus.

Development failures are preserved as evidence. The published baseline first
failed collection because the new API did not exist. Initial real NUT tests
exposed unspecified one/two-channel layout names; those names are now admitted
without inventing a speaker assignment. The first source-mutation fixture
chose an endpoint before the video cursor had ended and failed its own ordering
assertion. The corrected `[5, 128/25)` fixture proved a real issue: changing the
source after video RANGE_END and the final buffered audio read was accepted
before the final identity check. It now fails the whole operation. The check
covers the observed prefix and final identity, not unconsumed trailing audio.

One empty-first-interval test initially expected the later-clip error wording;
its assertion was corrected without changing production behavior. A mistyped
focused node selector ran no tests and is not a gate. The first offline wheel
dependency install lacked cached PyAV and therefore could not run the example;
the successful scoped install used the exact existing Pillow/PyAV versions and
hashes. No system environment, model or media download was introduced.
