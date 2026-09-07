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
| [Adaptive detector](https://github.com/Breakthrough/PySceneDetect/blob/24953b0bf76af17c450bc143d330eea48fc5e276/scenedetect/detectors/adaptive_detector.py): centered local change contrast with absolute content floor | Implemented centered rolling ratio, content floor, explicit window-border policy, minimum scene checks, per-frame diagnostics and CLI | Online frame interface, decoder buffering integration, motion-rich labeled evaluations and comparison across resolutions |
| [Threshold detector](https://github.com/Breakthrough/PySceneDetect/blob/24953b0bf76af17c450bc143d330eea48fc5e276/scenedetect/detectors/threshold_detector.py): fade-in/fade-out, bias and final-fade policy | Implemented absolute-luminance fade state, hysteresis, actual-dark-sample count, bias and explicit EOF behavior | Video-level fade benchmark and stream interface; detector intensity definitions differ |
| [Hash detector](https://github.com/Breakthrough/PySceneDetect/blob/24953b0bf76af17c450bc143d330eea48fc5e276/scenedetect/detectors/hash_detector.py) and [histogram detector](https://github.com/Breakthrough/PySceneDetect/blob/24953b0bf76af17c450bc143d330eea48fc5e276/scenedetect/detectors/histogram_detector.py) | Difference hash participates in composite distance; no standalone configurable hash or histogram detector | Dedicated hash configuration and image histograms with their own calibrated detectors |
| [TransNet V2 source](https://github.com/Breakthrough/PySceneDetect/blob/24953b0bf76af17c450bc143d330eea48fc5e276/scenedetect/detectors/transnet_v2.py) | No learned detector | Assess reference integration/support level and implement optional model lifecycle, batching, resource controls and a verified model evaluation; no silent model download |
| [SceneManager](https://github.com/Breakthrough/PySceneDetect/blob/24953b0bf76af17c450bc143d330eea48fc5e276/scenedetect/scene_manager.py): processing, scenes, images, callbacks | Offline typed scene partition, detailed detector statistics, key-frame selection/contact sheet, `scenes` CLI | Native incremental processing, callbacks, combining multiple detectors, seek/start/end/skip controls and per-scene image export |
| [StatsManager](https://github.com/Breakthrough/PySceneDetect/blob/24953b0bf76af17c450bc143d330eea48fc5e276/scenedetect/stats_manager.py): metric registry, frame metrics, CSV load/save | Write-only fixed-schema JSON and CSV diagnostics including suppressed candidates | Statistics import, schema/metric registry, corruption handling and recomputation from cached measurements |
| [Video backends](https://github.com/Breakthrough/PySceneDetect/blob/24953b0bf76af17c450bc143d330eea48fc5e276/docs/cli/backends.rst): OpenCV/PyAV/MoviePy; native PTS timing on supported VFR backends | Pillow images/animations and bounded optional FFmpeg extraction to a sampled image directory | Native decoder interfaces, seek/replay, audio/container metadata, PTS-preserving VFR behavior and live input contracts |
| [Timecode APIs](https://github.com/Breakthrough/PySceneDetect/tree/24953b0bf76af17c450bc143d330eea48fc5e276/scenedetect): frame/time coordinate conversion | Integer frame indices plus floating timestamps from index/name/EXIF/mtime | Rational frame-rate/timebase, timecode parsing/formatting/arithmetic and interoperability at non-integer FPS |
| [Video splitting](https://github.com/Breakthrough/PySceneDetect/blob/24953b0bf76af17c450bc143d330eea48fc5e276/scenedetect/video_splitter.py): FFmpeg and mkvmerge paths | FFmpeg extraction only | Scene clip splitting, copy/re-encode modes, audio preservation, precise boundary verification and interruption cleanup |
| [CLI output commands](https://github.com/Breakthrough/PySceneDetect/blob/24953b0bf76af17c450bc143d330eea48fc5e276/scenedetect/_cli/commands.py): scene lists, images, HTML, QP/keyframes, EDL, FCP, OTIO | JSON scene report, CSV stats, selected-frame CSV and contact sheet | Editor formats, timing conformance, image export per scene, HTML overview, keyframe encoder interchange |
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
timing, binary SMPTE, other editor formats and broad editor integration remain open.

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
