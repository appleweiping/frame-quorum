# Whole-repository reference audit: scene detection

Audit date: 2026-09-07. Reference:
[PySceneDetect at `24953b0bf76af17c450bc143d330eea48fc5e276`](https://github.com/Breakthrough/PySceneDetect/tree/24953b0bf76af17c450bc143d330eea48fc5e276),
committed 2026-08-28. The GitHub tree API returned 202 blobs, including 81 Python
files and 28 blobs under `tests/`, with `truncated=false`. These counts freeze
the examined snapshot, not a quality score. Frame Quorum baseline for this work
is `eaca750` (v0.4.0). Implementation in this change was authored independently.

This audit keeps the **entire reference repository** as the target. The work
below closes part of the detection workflow. It does not establish entire-repo
parity, and neither passing tests nor broad capability labels close the
remaining differences.

## Current detector and pipeline comparison

| Reference capability and source | Frame Quorum state | Work still required |
| --- | --- | --- |
| [Content detector](https://github.com/Breakthrough/PySceneDetect/blob/24953b0bf76af17c450bc143d330eea48fc5e276/scenedetect/detectors/content_detector.py): weighted pixel HSV change and edge controls | Composite summaries plus original full-pixel HSV/gradient sums, bounded native capture, weighted/V-only cache replay and synchronous online final decisions | Canny/dilation and flash-merging alternatives, calibrated resolution/accuracy benchmarks; numerical definitions differ |
| [Adaptive detector](https://github.com/Breakthrough/PySceneDetect/blob/24953b0bf76af17c450bc143d330eea48fc5e276/scenedetect/detectors/adaptive_detector.py): centered local change contrast with absolute content floor | Shared bounded rolling arithmetic, explicit window borders/minimum-tail confirmation, offline diagnostics and exact-PTS online pixel events | Motion-rich labeled evaluations, time-duration minimums, comparison across resolutions and broader online detector coordination |
| [Threshold detector](https://github.com/Breakthrough/PySceneDetect/blob/24953b0bf76af17c450bc143d330eea48fc5e276/scenedetect/detectors/threshold_detector.py): fade-in/fade-out, bias and final-fade policy | Absolute-luminance fade state, hysteresis, actual-dark-sample count, bias, explicit supplied-tail policy and generated native VFR fade evidence | Real-video labeled fade benchmark and online detector interface; intensity definitions differ |
| [Hash detector](https://github.com/Breakthrough/PySceneDetect/blob/24953b0bf76af17c450bc143d330eea48fc5e276/scenedetect/detectors/hash_detector.py) and [histogram detector](https://github.com/Breakthrough/PySceneDetect/blob/24953b0bf76af17c450bc143d330eea48fc5e276/scenedetect/detectors/histogram_detector.py) | Difference hash participates in composite distance; full-pixel RGB cell histograms now provide separate global/spatial total-variation detection and native cache/replay | Dedicated hash configuration, calibrated histogram accuracy corpus and time-based/online integration; RGB total variation is not the reference Y-channel correlation algorithm |
| [TransNet V2 source](https://github.com/Breakthrough/PySceneDetect/blob/24953b0bf76af17c450bc143d330eea48fc5e276/scenedetect/detectors/transnet_v2.py) | No learned detector | Assess reference integration/support level and implement optional model lifecycle, batching, resource controls and a verified model evaluation; no silent model download |
| [SceneManager](https://github.com/Breakthrough/PySceneDetect/blob/24953b0bf76af17c450bc143d330eea48fc5e276/scenedetect/scene_manager.py): processing, scenes, images, callbacks | Offline image/native scene partitions, five-detector kernels, exact-PTS callbacks, qualified-candidate union/quorum and bounded online pixel decision events | Past-boundary RGB callbacks, per-scene image export, crop/downscale/interpolation controls, richer detector/plugin lifecycle and interoperability |
| [StatsManager](https://github.com/Breakthrough/PySceneDetect/blob/24953b0bf76af17c450bc143d330eea48fc5e276/scenedetect/stats_manager.py): metric registry, frame metrics, CSV export/tuning; CSV loading is explicitly deprecated in the frozen source | Fixed-schema diagnostics plus bounded canonical native measurement capture/import and same-kernel threshold replay, explicit cached provenance and native multi-detector CSV | Arbitrary metric registry, broader schema interoperability and tuning accuracy/performance corpus; no compatibility claim for the deprecated CSV loader |
| [Video backends](https://github.com/Breakthrough/PySceneDetect/blob/24953b0bf76af17c450bc143d330eea48fc5e276/docs/cli/backends.rst): OpenCV/PyAV/MoviePy; native PTS timing on supported VFR backends | Pillow images/animations, bounded FFmpeg extraction, incremental local PyAV video and fixed PCM16 decoding with exact PTS | Multi-file `VideoStreamConcat`, broader backend/codec compatibility evidence, general container inventory, file-like/network/live input contracts |
| [Timecode APIs](https://github.com/Breakthrough/PySceneDetect/blob/24953b0bf76af17c450bc143d330eea48fc5e276/scenedetect/common.py): `Timecode` / `FrameTimecode` conversion | Rational CFR timecodes, explicit PTS quantization, text drop/non-drop counters, native rational VFR scenes with unknown tails preserved | Broader interoperability and explicit native scene-to-editor conversion; binary SMPTE packing is not a frozen-reference public capability |
| [Video splitting](https://github.com/Breakthrough/PySceneDetect/blob/24953b0bf76af17c450bc143d330eea48fc5e276/scenedetect/output/video.py): FFmpeg and mkvmerge paths | Video-only FFV1/NUT plus separate exact FFV1/PCM16 NUT clips, common audio-grid epoch, complete RGB/PTS/PCM/sample verification and no-replace publication | General audio format/channel/resampling policies, copy/mkvmerge modes, broader codecs/containers, explicit subtitle policy and external compatibility/throughput corpus |
| [CLI output commands](https://github.com/Breakthrough/PySceneDetect/blob/24953b0bf76af17c450bc143d330eea48fc5e276/scenedetect/_cli/commands.py): scene lists, images, HTML, QP/keyframes, EDL, FCP, OTIO | Image/native JSON scene reports, CSV stats, selected-frame CSV/contact sheet, exact CFR timing CSV, cuts-only EDL and native PTS measurement JSONL | Broader editor formats/conformance, image export per scene, HTML overview and keyframe encoder interchange |
| [CLI/configuration](https://github.com/Breakthrough/PySceneDetect/tree/24953b0bf76af17c450bc143d330eea48fc5e276/scenedetect/_cli) | Explicit argparse commands, structured errors, atomic report bundle | Config file precedence, chained detector/output workflow and backend feature reporting |
| [Reference tests and release tests](https://github.com/Breakthrough/PySceneDetect/tree/24953b0bf76af17c450bc143d330eea48fc5e276/tests) | Cross-platform Python CI, 95% coverage gate, analytic and real-Pillow regression inputs | Labeled video scene corpus, precision/recall and timestamp tolerances, native-codec matrix, decoder/format compatibility and performance benchmarks |

## Remaining whole-reference public workflow audit (2026-09-08)

This follow-up read freezes the same commit and compares actual public symbols,
not directory or test counts. The following correct and expand the ledger;
they do not reduce the whole-repository target or claim completed parity.

- [`VideoStreamConcat` / `SourceSpan` / `map_span`](https://github.com/Breakthrough/PySceneDetect/blob/24953b0bf76af17c450bc143d330eea48fc5e276/scenedetect/backends/concat.py)
  form a real multi-source timeline. The reference opens one source at a time,
  remaps offsets using actual observed ends, checks dimensions and maps scene
  spans back to source intervals. Frame Quorum has no equivalent composition.
- The real timecode implementation is `common.py`; `frame_timecode.py` is a
  deprecated compatibility shim. The frozen public API does not supply binary
  SMPTE packing, so that earlier row was inaccurate as a reference gap. It
  could be an adjacent feature, not evidence of a missing frozen capability.
- [`output/__init__.py`](https://github.com/Breakthrough/PySceneDetect/blob/24953b0bf76af17c450bc143d330eea48fc5e276/scenedetect/output/__init__.py)
  contains CSV/HTML, EDL, Final Cut and OTIO exporters. Our existing optional
  OTIO `cmx_3600` **reader smoke test** checks EDL interchange only: it is not an
  OTIO exporter. FCP7/FCPX, OTIO audio/video timelines, HTML scene overview,
  QP/Aegisub keyframes and per-scene image output remain open.
- [`output/image.py`](https://github.com/Breakthrough/PySceneDetect/blob/24953b0bf76af17c450bc143d330eea48fc5e276/scenedetect/output/image.py)
  provides per-scene image counts/margins, resize/interpolation, templates and
  encoding parameters. A selected-frame contact sheet does not implement this
  native per-scene export chain. SceneManager crop/downscale/plugin lifecycle,
  CLI config precedence/chaining, `load-scenes`, short-scene drop/last merge
  and backend/live input variants likewise remain distinct capabilities.
- Frozen threshold detection includes a `CEILING` bright-fade mode as well as
  dark fades. Hash detection uses configurable DCT perceptual hashing, not our
  fixed difference hash; histogram detection uses Y-channel correlation, not
  RGB marginal/cell total variation. Weighted HSV/Canny/dilation and flash
  MERGE/SUPPRESS policy are not numerically equivalent to our circular-hue and
  forward-gradient definitions. These algorithm/parameter gaps remain open.
- The TransNetV2 source is an optional ONNX prototype with model-path/provider
  and bounded batch handling. Its docstring mentions CLI use, but the frozen
  CLI registration/config files do not register it. We do not count that
  statement as an implemented reference CLI workflow, nor implement a learned
  model without a deliberate lifecycle and evaluation contract.
- The frozen FFmpeg splitter allows codec arguments and stream-copy paths,
  but also appends `-sn`. Optional subtitle-map text alone is not evidence of
  default subtitle preservation. Our new [fixed audio/video split](native-av-splitting.md)
  is a real narrow dual-track chain, not arbitrary-format or subtitle parity.

## Evidence for this implementation increment

`tests/test_scene_detection.py` provides manually computed scene expectations,
not comparisons with the implementation's own internal helper. Cases include
smooth motion plus a large cut, a finite zero baseline, insufficient adaptive
windows, an adjacent flash pair, dark flashes, initial darkness, threshold
hysteresis, repeated fades, all bias endpoints, incomplete final fades, first
and last scene minimums, all five detector choices, invalid/oversized settings,
mixed/decreasing timestamps, invalid indices, CSV null/zero preservation, real
image decoding through the CLI, and report staging failure preserving prior
artifacts. `examples/detect_scenes.py` runs as a test without network access or
repository writes. Detailed semantics and limits are in
[Scene detection and diagnostics](scene-detection.md).

The original selector's label-free representation benchmarks remain useful for
key-frame selection; they are not scene-detection precision/recall evidence.
Follow-up implementation now adds `timecode.py` and `editing.py`: exact rational
CFR coordinates, explicit PTS quantization, non-drop/drop text labels, timing CSV
and a bounded cuts-only single-reel EDL. See [timecode/export contracts](timecodes-and-editing.md).
The corresponding timecode/editor rows are partial, not complete: native VFR
snapshots and scenes now retain exact timing, but native editor integration, binary
SMPTE, other editor formats and broad editor integration remain open.

Every open row above requires implementation and external-behavior evidence
before the whole-repository objective can be marked complete.

Local verification of this increment (Windows, Python 3.11.2):

- Full regression: 412 passed; 3 existing tests skipped because Windows did not
  grant symlink-creation privilege. Total branch-aware coverage: 95.71%, exceeding
  the repository's 95% gate.
- New `scene_detection.py`: 100% statement/branch coverage. The 39 focused tests
  include the CLI, CSV, and executable example checks described above.
- Ruff lint/format, strict Mypy, and Bandit passed.

These are local checks, not results from the remote CI matrix or a labeled-video
accuracy comparison.

## Native decoder increment verification

The native backend adds an original, separate local-file decoder contract; it
does not assert compatibility with every reference backend or turn an offline
detector into a streaming scene manager. See [native video](native-video.md) for
PTS, seek-generation, lifetime-budget, source and native-allocation boundaries.

Local verification on 2026-09-07 used Windows build 26200, Python 3.11.2 and PyAV
18.1.0. The wheel reports libavformat 62.12.102 and libavcodec 62.28.102.

- Full regression: **570 passed, 3 skipped**, including 88 native API/integration
  cases. The same three pre-existing Windows symlink-privilege tests were skipped;
  no native integration test was skipped. Branch-aware total coverage: **96.61%**
  with the unchanged 95% gate; `native_video.py`: **99.46%**.
- Generated FFV1 CFR/VFR clips have hand-authored native PTS/color expectations;
  an MPEG-4 interframe fixture checks actual keyframe-backward seek and forward
  interval filtering. Controlled failures cover missing/decreasing PTS, each
  resource close/retry, observed source change and pre-materialization CLI bounds.
- Ruff lint/format, strict Mypy (23 source modules), Bandit, source/wheel build,
  strict Twine metadata checks, wheel-content checks and frozen-lock verification
  passed. Resource warnings were escalated during the full test run.

The CI configuration installs the video extra across the existing Linux/Windows
matrix. Those remote matrix results, a broader codec/platform compatibility
corpus, and scene-detection accuracy/performance are **not** established by the
local checks above. Open capability rows remain open.

## Native scene coordinator increment verification

The increment from published baseline `132ec11` connects native decoding to the
five existing detection policies with exact rational PTS, immutable measurement
records, explicit selected-sample partitions, qualified-candidate union/quorum,
synchronous sample callbacks and a JSON CLI. Shared measurement-only kernels
preserve the image API; this remains bounded **offline** detection, not an
online scene manager. See [native scene contracts](native-scenes.md).

Local verification on 2026-09-07 used Windows, Python 3.11.2 and PyAV 18.1.0:

- Full regression with resource warnings escalated: **713 passed, 3 skipped**.
  The three skips remain the pre-existing Windows symlink-privilege cases;
  no native scene integration test was skipped. The unchanged 95% branch-aware
  gate passed at **96.93%** total. `native_scenes.py` and the refactored
  `scene_detection.py` both reached **100% statement/branch coverage**.
- The 143 new native-scene cases include all five policies at multiple scene
  minima, exact sample/quorum provenance and constructor contradictions,
  callback exceptions/interrupts/invalid coroutine returns, work admission,
  empty intervals and count limits. Eighteen genuine PyAV cases cover manual
  lossless VFR cuts/fades, direct decoded-measurement comparison, native seek
  indices/stride, unknown EOF tails, callback file cleanup, CLI and offline demo.
- Ruff lint/format, strict Mypy (24 source modules), Bandit, source/wheel build,
  strict Twine metadata, wheel contents, frozen-lock and whitespace checks passed.

These are local checks, not remote matrix results, a representative labeled-video
accuracy benchmark, broader codec certification or whole-repository parity.
Native scene-to-editor conversion, online boundary delivery, richer
detectors/plugins and the other open capability rows remain open. A subsequent
increment adds the separately documented native measurement replay workflow.

## Native splitting increment verification

The increment from published baseline `7231022` adds real FFV1/NUT video-only
clips, not just export plans. The separate [splitting contract](native-splitting.md)
defines exact native `[start, end)` frame inclusion, first-selected-frame
rebasing, independently decoded output verification, complete-directory
publication, per-operation limits and preserved cleanup diagnostics.

Local verification on 2026-09-07 used locked PyAV 18.1.0 and Pillow 12.3.0:

- Windows, Python 3.11.2: **866 passed, 3 skipped**; the skips remain the three
  pre-existing symlink-privilege cases. Total branch-aware coverage **97.10%**;
  `native_splitting.py` **98.42%**. Resource warnings were escalated.
- WSL Ubuntu/Linux, Python 3.12.3, isolated temporary venv: **869 passed**, no
  skips; total coverage **97.15%**. This executes the actual Linux
  `renameat2(RENAME_NOREPLACE)` path as well as real native codecs, not a mocked
  platform result. The unchanged 95% gate passes on both systems.
- The **153 new cases** include exact nonzero VFR/submillisecond intervals,
  complete direct-PyAV RGB and rational-PTS oracles, one-frame clips, gap/empty
  intervals, deliberate audio exclusion, unknown/sampled/truncated scene bounds,
  every configured limit, altered output rejection, competing publication and
  cleanup identity/interrupt/ordinary-error combinations. The generated offline
  example independently checks two clips and six frames.
- Ruff lint/format, strict Mypy (25 modules), Bandit, source/wheel build, strict
  Twine metadata, wheel contents, frozen-lock and whitespace checks passed.

These results do not establish arbitrary codec/container, audio/subtitle,
stream-copy/mkvmerge, crash-durable storage, native-code sandboxing, broad
external video/editor compatibility or whole-repository reference parity.
No new dependency or upstream implementation code was added.

## Native measurement replay increment verification

The increment from published baseline `c4999c0` adds a complete, fixed-schema
native measurement cache and threshold-tuning workflow, not an importer for the
image-statistics CSV. It reuses the existing five-detector kernel and preserves
the fresh-decoding API. See [measurement/replay contracts](native-measurements.md)
for source/configuration identity, exact coordinates, canonical ingress limits,
historical provenance and ownership-aware directory publication.

Local verification on 2026-09-07 used locked PyAV 18.1.0 and Pillow 12.3.0:

- Windows, Python 3.11.2: **1027 passed, 3 skipped** in 523.04 seconds; the skips
  remain the three pre-existing symlink-privilege cases. Total branch-aware
  coverage **97.36%**; `native_measurements.py` **99.65%**.
- WSL Ubuntu/Linux, Python 3.12.3: **1030 passed**, no skips, in 215.84 seconds;
  total coverage **97.41%**. This uses an isolated temporary venv populated from
  the frozen lock export with package hash verification, real native codecs and
  the actual Linux no-replace publication path. Both full runs escalate resource
  warnings and retain the unchanged 95% coverage gate.
- The **161 new cases**, including **12 real PyAV integration cases**, cover
  hand-computed cuts/fades/quorum, independent source SHA256 and direct decoded
  measurements, all five policies against fresh decoding, exact VFR/stride/range
  identity and threshold replay after source deletion. A fresh subprocess rejects
  any optional decoder import while successfully replaying the cache. The
  generated offline example verifies two independently specified thresholds.
- Ingress cases include recomputed-checksum semantic corruption, duplicate and
  unknown fields, truncation, noncanonical encodings, portable path-label text,
  bounded numeric conversion with the process-wide digit guard disabled, exact
  large rationals, contradictory diagnostics, pre-open special-file rejection,
  competing publication and cleanup identity/control-exception failures.
- Ruff lint/format, strict Mypy (26 source modules), Bandit, source/wheel build,
  strict Twine metadata, wheel contents, frozen-lock and whitespace checks passed.

Cached results explicitly report `source_verified=false`: neither consistency
checks nor stored hashes authenticate a current source. No native splitting
trust boundary, version or dependency was changed. Arbitrary metric registries,
HSV/edge/histogram or learned measurements, online processing, a labeled-video
tuning corpus, broader interoperability and the other whole-repository gaps
remain open. These local checks are not remote CI or whole-repository parity.

## Full-pixel histogram increment verification

The increment from published baseline `801a2b7` adds full-resolution RGB cell
histograms, global/spatial total-variation scores, native capture, a separate
versioned canonical cache, threshold replay, JSON/CSV reports and CLI commands.
It reuses the existing native collector, decision policy, exact scene partition
and ownership-aware publication paths. The frozen reference's histogram detector
uses Y-channel correlation: this is an independently authored RGB measurement
contract, not numerical equivalence. See [pixel histogram contracts](native-pixel-histograms.md).

Local verification on 2026-09-07 used locked PyAV 18.1.0 and Pillow 12.3.0:

- Windows, Python 3.11.2: **1185 passed, 3 skipped** in 131.67 seconds; the skips
  remain the three pre-existing symlink-privilege cases. Total branch-aware
  coverage **97.57%**.
- WSL Ubuntu/Linux, Python 3.12.3: **1188 passed**, no skips, in 159.30 seconds;
  total coverage **97.62%**. The isolated temporary environment used a frozen
  lock export with package hash verification. Both full runs escalated resource
  warnings and retained the unchanged 95% gate. `pixel_histograms.py`,
  `native_histograms.py` and the refactored `native_scenes.py` each reached
  **100% statement/branch coverage**.
- The **158 new cases**, including **15 real PyAV integration cases**, cover
  independent per-pixel bin counts, odd-sized cell areas, all supported bin/grid
  combinations, RGB marginal-versus-layout changes, exact nonzero VFR PTS,
  equal summary metrics with different histograms, range/stride admission and
  replay after source deletion. A fresh subprocess forbids decoder imports
  while successfully replaying the cache; the offline example checks global
  and spatial results against independently specified cuts.
- Wire/resource/ownership cases cover recomputed-checksum corruption,
  pre-parser count and pixel admission, malformed input, result-score
  consistency, borrowed image aliases, old/new schema rejection, competing
  publication, lost acknowledgment and primary/cleanup control exceptions.
  The old cache serializer remains byte-identical to a fixed fixture produced
  by the signed baseline: 2874 bytes, SHA256
  `7f717a62f64de44bf75e2cb0909b604e13be6bbd4ba2da617cbede40231f1e9f`.
- Ruff lint/format, strict Mypy (28 source modules), Bandit, source/wheel build,
  strict Twine metadata, wheel contents, frozen-lock and whitespace checks passed.

The two failed development test fixtures were corrected before these full gates:
one incorrectly expected one scene for a hand-authored sequence with a cut; the
other exceeded its helper's eight-frame cap before reaching the intended large
pixel-budget assertion. Neither fix changed production behavior or a threshold.

Cached histogram results retain `source_verified=false` and cannot be passed as
fresh scene results to native splitting. No old measurement schema, splitting
trust boundary, dependency or version was changed. This increment does not add
HSV/edge/learned measurements, online detection, arbitrary spatial grids, an
accuracy-calibrated video corpus or reference-algorithm equivalence. The remaining
whole-repository capability rows stay open; these local gates are not remote CI
or proof of complete parity.

## Full-pixel HSV/gradient increment

The increment from signed `b5925bdb` adds independent complete-pixel evidence,
not an inference from summaries or histogram marginals. Four exact integer sums
enable weighted/V-only content and existing adaptive policy replay without
decoding. It adds a separately versioned strict cache and native CLI/example,
reusing collection, parsing, policy, scene partition and owned publication cores.
See [the numerical, resource and provenance contracts](native-pixel-changes.md).

Local verification on 2026-09-08 used locked PyAV 18.1.0 and Pillow 12.3.0:

- Windows Python 3.11.2: **1312 passed, 3 skipped** in 403.70 seconds, total
  branch-aware coverage **97.77%**. Skips are the same three pre-existing
  Windows symlink-privilege cases.
- WSL/Linux Python 3.12.3: **1315 passed**, no skips, in 461.34 seconds, total
  coverage **97.81%**. Both full runs escalated `RuntimeWarning` and
  `ResourceWarning`; the unchanged coverage gate remains 95%.
- Both new modules reached **100% statement and branch coverage** on both
  platforms. The **127 new tests**, including **10 real PyAV cases**, cover
  hand-calculated hues, wraparound, gray suppression, all four radii, clamped
  gradients, histogram-identical pixel changes, subnormal/large weights, exact
  native VFR/stride/range coordinates, admission, corruption, ownership and
  publication. A separate read-only review checked all four sums against an
  independent Fraction HSV and two-dimensional gradient oracle on 80 small
  image pairs; it also checked exchange symmetry and subnormal weights.
- Summary and histogram golden bytes remain unchanged. Their independent
  fixture digests are respectively
  `7f717a62f64de44bf75e2cb0909b604e13be6bbd4ba2da617cbede40231f1e9f`
  (2874-byte complete summary cache) and
  `da8ebbf429f09f95d6b028e0ebf30112965089c7cb13896b234739873e017d05`
  (7912-byte histogram header/records before completion footer).
- An offline-installed, non-editable wheel ran the actual generated-video
  example under Python `-I`. All 31 package files matched source, wheel and
  installed copies byte-for-byte. A fresh `-I` child forbade every `av` import
  and replayed that captured cache after source deletion, obtaining exact cuts
  `511/100` and `527/100`. Wheel SHA256:
  `1c4d954a433f649c38d70bc30e83f1456e35d33830b4fbe635186cd446e9aed1`.
- Whole-repository Ruff lint/format, strict Mypy (30 source modules), Bandit,
  wheel build, strict Twine metadata, wheel contents, frozen-lock/whitespace
  checks and 35 local documentation links passed.

Development evidence is retained: the first module tests failed before their
implementations existed. One subsequent coverage-focused run had 124 passes
and one fixture failure because the hand-authored golden fixture used `mean_r`
instead of the existing `mean_red` field; that fixture was corrected without
changing production arithmetic or acceptance thresholds. An initially created,
hash-installed WSL `/tmp` environment was absent on a later invocation; the
successful full run instead created, hash-installed and tested its fresh scoped
environment inside one shell lifetime. No system configuration was changed and
the cross-invocation disappearance's cause was not inferred.

Old summary/histogram source algorithms and bytes, dependencies/version and
native splitting trust are unchanged. This does not close online, learned,
calibrated accuracy, interop/editor/backend or broader whole-reference repository
gaps. These local gates are not hosted CI or proof of complete parity.

## Bounded online pixel detection increment

Baseline `2538a06a7bcb8c9f4ec6e7397f31c7bf92e77260`; frozen reference unchanged.
The reference's real detector interface is
[`scenedetect/detector.py`](https://github.com/Breakthrough/PySceneDetect/blob/24953b0bf76af17c450bc143d330eea48fc5e276/scenedetect/detector.py),
not the deprecated `scene_detector.py` shim. Its per-frame processing, EOF
post-processing and delayed-event bound establish the online workflow gap.
The frozen adaptive detector buffers `2r+1` scores and returns a past center
timecode; its "two-pass" description does not mean whole-video materialization.
SceneManager has a bounded decode queue but also retains its cutting list;
that queue alone is not evidence of bounded total result memory.

The original [online implementation](native-online.md) now shares the existing
two-RGB measurement, exact rolling arithmetic, minimum-scene decision and native
endpoint kernels. It confirms final per-sample statistics at a conservative
`max(r, m-1)` horizon, preserving the existing minimum-tail semantics without
retractions. It has no full-source hash prerequisite, complete sample table or
background producer. Source-unverified event/End types and an independent JSONL
schema do not change prior cache bytes or become native splitting input.

Tests include independently specified content/adaptive cuts, the frozen former
rolling fsum operation sequence, finite buffer peaks, cooperative cancellation,
resource failure/control precedence, final-line byte admission, checksum and
no-replace output. Generated FFV1 VFR cases are independently reopened to verify
every RGB frame and exact PTS; hue/gradient integer oracles and offline replay
comparisons check evidence, scores and complete scene partitions separately.
Initial red collection failed because the online module did not yet exist.

Final local evidence, with source/tests frozen during the full runs:

- Windows Python 3.11.2: **1414 passed, 3 existing symlink-permission skips**,
  202.57 seconds; statement/branch coverage **97.85%**.
- Linux Python 3.12.3: **1417 passed, no skips**, 187.74 seconds;
  statement/branch coverage **97.88%**. Both used Pillow 12.3.0, PyAV 18.1.0
  and pytest 9.1.1 and retained the original 95% coverage gate. The 102 new
  tests comprise 89 unit and 13 real-video/integration/example cases.
- The new online module has **98.70%** statement/branch coverage on both
  platforms. No coverage exclusions or production limit relaxations were added.
  Independent review covered the delayed decision horizon, bounded retained
  state, source closure before draining, cleanup/control precedence and output
  publication; these checks do not certify arbitrarily hand-constructed events.
- The former rolling algorithm's arithmetic order matches a frozen independent
  3001-score corpus across four radii. All three previous wire formats remain
  unchanged. In addition to the two preceding golden fixtures, the complete
  3617-byte pixel-change cache independently generated by the signed baseline
  wheel has SHA256
  `b863f789598b2517db4eb0bd63896e860f7d63c72990c2cd1e76050b7eba62e8`.
- A fresh, non-editable wheel environment ran the actual generated VFR example
  under Python `-I`, returning cuts `259/50` and `541/100` with observation
  watermarks `527/100` and `553/100`, then a source-unverified EOF End. Its
  32 package files matched source, wheel and installed files byte-for-byte;
  the package was imported from that environment, not `src`. Wheel SHA256:
  `a150c758760bcd4fc4f31d40063ba70e0edb4f3d2fb4e72c83aff87b9fb67e16`.
- Whole-repository Ruff lint/format (96 files), strict Mypy (31 modules),
  configured Bandit, frozen lock, whitespace, local documentation links,
  source/wheel/sdist byte comparisons, strict Twine and wheel-contents checks
  passed. These are local gates, not hosted CI results.

Setup failures are retained rather than counted as tests: the first Linux
inline shell launch exited 2 before any test output; a task-owned Bash script
then created, hash-installed and tested a fresh scoped environment in one
shell lifetime. The launch failure's cause was not asserted. The first offline
isolated-wheel install could not resolve the absent cached PyAV wheel; the
successful install used the exact existing lock's Pillow/PyAV versions and
SHA256 hashes, without changing dependencies or the system environment.
An initial Bandit invocation without the repository configuration reported
existing exclusions; the actual configured gate above passed.

Online fades, quorum/histogram coordination, live input, time-duration minimums,
flash merging, richer backend/editor interoperability, learned detection and
calibrated real-video accuracy/performance remain separate whole-repository gaps.

## Exact dual-track native splitting increment

From signed `e9aeb7ea91955b1c7f131f23ef180ab696144044`, a separate API/CLI now
implements actual fixed FFV1 + PCM16 / NUT clips with a common source-audio-grid
epoch. It independently reopens complete selected video and audio output,
validates original half-open coordinates, runs strict source/selected/verify
budgets and publishes only the complete verified directory. The old video-only
API and cache schemas retain their previous meanings. See the
[audio/video contract and full evidence](native-av-splitting.md).

Local Windows full: **1627 passed, 3 existing symlink-permission skips**, 263.40 s,
97.95% statement/branch coverage. Local Linux full: **1630 passed, no skips**,
262.65 s, 97.99%. All **213 new cases** ran on both systems, including 50 genuine
native/integration/example cases, exact stereo/VFR bytes and timestamps,
non-sample-aligned NTSC cuts at several sample rates, second-track selection,
observed buffered-tail source mutation, and cleanup/publication failures.
The 95% gate and warning escalation remain unchanged. Installed-wheel execution
and bytewise package/source comparisons passed; no source code was copied.

This closes a narrow actual dual-track splitting chain, not arbitrary audio
formats, channels, resampling, subtitles, attachments, metadata, stream-copy,
mkvmerge, arbitrary-container exports, live/concatenated sources or durable
fsync storage. The other rows remain open, especially reference-algorithm
variants, per-scene image/editor workflows and calibrated real-world accuracy
and compatibility evidence. Local test results are not whole-repository parity.
