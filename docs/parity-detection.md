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
| [Content detector](https://github.com/Breakthrough/PySceneDetect/blob/24953b0bf76af17c450bc143d330eea48fc5e276/scenedetect/detectors/content_detector.py): weighted pixel HSV change and edge controls | Composite difference hash/RGB mean/luminance distances, independent color/luminance modes | Pixel change representation, configurable weights/edge contribution, resolution/accuracy benchmarks; the current composite is not numerically equivalent |
| [Adaptive detector](https://github.com/Breakthrough/PySceneDetect/blob/24953b0bf76af17c450bc143d330eea48fc5e276/scenedetect/detectors/adaptive_detector.py): centered local change contrast with absolute content floor | Centered rolling ratio, content floor, explicit window-border policy, minimum scene checks, per-sample diagnostics and native/image CLI | Online decision buffering, motion-rich labeled evaluations and comparison across resolutions |
| [Threshold detector](https://github.com/Breakthrough/PySceneDetect/blob/24953b0bf76af17c450bc143d330eea48fc5e276/scenedetect/detectors/threshold_detector.py): fade-in/fade-out, bias and final-fade policy | Absolute-luminance fade state, hysteresis, actual-dark-sample count, bias, explicit supplied-tail policy and generated native VFR fade evidence | Real-video labeled fade benchmark and online detector interface; intensity definitions differ |
| [Hash detector](https://github.com/Breakthrough/PySceneDetect/blob/24953b0bf76af17c450bc143d330eea48fc5e276/scenedetect/detectors/hash_detector.py) and [histogram detector](https://github.com/Breakthrough/PySceneDetect/blob/24953b0bf76af17c450bc143d330eea48fc5e276/scenedetect/detectors/histogram_detector.py) | Difference hash participates in composite distance; no standalone configurable hash or histogram detector | Dedicated hash configuration and image histograms with their own calibrated detectors |
| [TransNet V2 source](https://github.com/Breakthrough/PySceneDetect/blob/24953b0bf76af17c450bc143d330eea48fc5e276/scenedetect/detectors/transnet_v2.py) | No learned detector | Assess reference integration/support level and implement optional model lifecycle, batching, resource controls and a verified model evaluation; no silent model download |
| [SceneManager](https://github.com/Breakthrough/PySceneDetect/blob/24953b0bf76af17c450bc143d330eea48fc5e276/scenedetect/scene_manager.py): processing, scenes, images, callbacks | Offline typed image/native scene partitions, shared five-detector kernels, exact-PTS native decoder coordinator, bounded measurement callbacks, qualified-candidate union/quorum and detailed provenance | Online boundary callbacks, per-scene image export, richer detector/plugin lifecycle and interoperability |
| [StatsManager](https://github.com/Breakthrough/PySceneDetect/blob/24953b0bf76af17c450bc143d330eea48fc5e276/scenedetect/stats_manager.py): metric registry, frame metrics, CSV load/save | Write-only fixed-schema JSON and CSV diagnostics including suppressed candidates | Statistics import, schema/metric registry, corruption handling and recomputation from cached measurements |
| [Video backends](https://github.com/Breakthrough/PySceneDetect/blob/24953b0bf76af17c450bc143d330eea48fc5e276/docs/cli/backends.rst): OpenCV/PyAV/MoviePy; native PTS timing on supported VFR backends | Pillow images/animations, bounded FFmpeg extraction, incremental PyAV local-file decoding with exact PTS, seek/replay, generation-local indexing and native offline scene integration | Broader backend/codec compatibility evidence, audio/container inventory and live input contracts |
| [Timecode APIs](https://github.com/Breakthrough/PySceneDetect/tree/24953b0bf76af17c450bc143d330eea48fc5e276/scenedetect): frame/time coordinate conversion | Rational CFR timecodes, explicit PTS quantization, text drop/non-drop counters, native rational VFR scenes with unknown tails preserved | Binary SMPTE, broader interoperability and explicit native scene-to-editor conversion |
| [Video splitting](https://github.com/Breakthrough/PySceneDetect/blob/24953b0bf76af17c450bc143d330eea48fc5e276/scenedetect/video_splitter.py): FFmpeg and mkvmerge paths | FFmpeg extraction only | Scene clip splitting, copy/re-encode modes, audio preservation, precise boundary verification and interruption cleanup |
| [CLI output commands](https://github.com/Breakthrough/PySceneDetect/blob/24953b0bf76af17c450bc143d330eea48fc5e276/scenedetect/_cli/commands.py): scene lists, images, HTML, QP/keyframes, EDL, FCP, OTIO | Image/native JSON scene reports, CSV stats, selected-frame CSV/contact sheet, exact CFR timing CSV, cuts-only EDL and native PTS measurement JSONL | Broader editor formats/conformance, image export per scene, HTML overview and keyframe encoder interchange |
| [CLI/configuration](https://github.com/Breakthrough/PySceneDetect/tree/24953b0bf76af17c450bc143d330eea48fc5e276/scenedetect/_cli) | Explicit argparse commands, structured errors, atomic report bundle | Config file precedence, chained detector/output workflow and backend feature reporting |
| [Reference tests and release tests](https://github.com/Breakthrough/PySceneDetect/tree/24953b0bf76af17c450bc143d330eea48fc5e276/tests) | Cross-platform Python CI, 95% coverage gate, analytic and real-Pillow regression inputs | Labeled video scene corpus, precision/recall and timestamp tolerances, native-codec matrix, decoder/format compatibility and performance benchmarks |

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
Native scene-to-editor conversion, online boundary delivery, cached-statistics
replay, richer detectors/plugins and the other open capability rows remain open.
