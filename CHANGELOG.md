# Changelog

All notable changes are documented here. The project follows semantic versioning.

## [Unreleased]

### Verified native video splitting

- Add actual FFV1/NUT local video-only clip transcoding, exact half-open PTS
  intervals, first-selected-frame rebasing and complete RGB/PTS/count verification.
- Add unsampled native-scene interval conversion with explicit unknown-tail
  handling, bounded output manifests, fixed-format `native-split` CLI and an
  offline generated-video demonstration.
- Bound selected hash records, decode/verification work and all output bytes;
  publish new directories without replacement on supported Windows/Linux paths.
  Cleanup tracks created file identities and preserves interruptions and residue.

### Native scene coordination

- Connect native PTS-preserving decoding to all five shared scene detectors,
  with exact-position vote provenance and half-open selected-sample scenes.
- Add immutable measurements, bounded offline analysis, explicit unknown final
  endpoints, synchronous progress callbacks and the `native-scenes` JSON CLI.
- Add generated VFR cut/fade demonstrations and independent coordinate,
  algorithm-equivalence, callback cleanup and strict-result regression tests.

### Native video

- Optional incremental PyAV local-file streams preserving integer PTS and
  rational time bases, owned RGB snapshots, exact presentation windows and
  keyframe seek/replay with explicit generation-local indices.
- Lifetime source/frame/pixel limits, strict single-owner lifecycle, denied
  secondary protocol opens and honest cleanup-failure diagnostics with retry.
- `native-scan` streaming measurement JSONL, an offline generated-video example,
  real CFR/VFR/interframe codec tests and optional video coverage in CI.

### Exact timing and editing

- Rational frame-rate and timecode coordinates with explicit rounding, NTSC
  drop-frame counter parsing/formatting, elapsed timestamps and PTS quantization.
- Exact scene-timing CSV and cuts-only single-reel EDL exports in the scene CLI;
  bounded metadata and exclusive out-points with no silent day wrapping.

### Added

- Centered adaptive scene detection with an absolute content floor, finite contrast
  ratios and explicit incomplete-window behavior.
- Stateful threshold fade detection with hysteresis, minimum dark samples, bias,
  final-fade policy and first/final scene length checks.
- `detect_scenes`, typed detection settings/results/statistics, and the `scenes`
  CLI producing staged scene JSON and per-frame CSV diagnostics.
- Executable synthetic cut/fade examples and a pinned whole-repository reference
  gap audit covering detector, decoder, timecode, splitting and export surfaces.

## [0.4.0] - 2026-09-07

### Added

- Added content, luminance, and color transition detectors with explicit thresholds and minimum-run controls.
- Added stable CSV decision exports for downstream review and reproducible selection reports.

## [0.3.0] - 2026-09-07

### Added

- Added deterministic shot-boundary analysis based on representation distance,
  plus per-shot frame-budget allocation with explicit short-budget behavior.
  The API reports boundaries and allocation diagnostics without claiming
  semantic scene understanding.

## [0.2.0] - 2026-09-07

### Added

- `frame-quorum coverage`: measure how well a selection represents the frames it dropped. The
  manifest already said why each frame was kept or rejected, which answers whether a decision
  was defensible and not whether the result missed anything.
- Representation error is the distance from each dropped frame to the nearest kept one, in the
  same content metric the selector uses to detect duplicates. A second notion of "similar" would
  let a selection look well represented under one measure while duplicates were rejected under
  another, and the disagreement would be invisible.
- Consecutive under-represented frames are reported as one gap, since a missed event appears as
  a stretch of neighbours and listing them separately describes one absence many times.
- `--budgets` re-selects at several budgets and reports what each represents. The budget is the
  one setting chosen with no basis; a worst gap that keeps falling says it is binding, and one
  that flattens says the extra frames are being spent on moments already covered. A curve still
  improving at its largest budget reports no knee rather than naming the last point.
- The curve reports frames kept rather than frames requested, so a budget past the length of the
  sequence does not look like a plateau.
- `frame_quorum.representation` as a Python API: `analyze_representation` and `budget_curve`.

- Add a deterministic experiment runner comparing Frame Quorum with time-uniform, change-peak, and repeated seeded-random
  baselines under shared hard constraints.
- Add normalized quality, content coverage, temporal coverage, transition coverage, non-redundancy, and explicitly
  non-semantic balanced metrics with machine-readable JSON and dependency-free SVG output.
- Add a checked-in reproducibility fixture, selection-only performance protocol, research limitations, and quality and
  runtime regression coverage.
- Add an optional bounded FFmpeg adapter that stages, validates, and atomically publishes a new PNG sequence without
  a shell or mandatory video dependency.
- Add opt-in animated-image expansion. `AnimationConfig` and `--expand-animations` turn one animated GIF, APNG, or
  WebP container into one measured frame per internal frame, bounded by explicit frame-count and decoded-byte limits
  that refuse an oversized container instead of truncating it. Expanded frames carry a `source_frame_index` and a
  `name#frame=N` path so every decision traces back to its container and position, and the contact sheet re-reads the
  exact internal frame it reports.
- Add opt-in parallel scanning. `ConcurrencyConfig` and `--workers` decode and measure several files at once while
  every observable output stays identical: discovery order, frame indices, all metrics, manifest bytes, and the
  benchmark's measured-record fingerprint are unchanged at any worker count, and the count is deliberately absent
  from manifests because it is an execution detail. Only measurement overlaps; indices, timestamps, and the
  post-read size/mtime/device/inode check stay on the calling thread in discovery order, and futures are consumed
  in that order so the earliest failing path is reported with the same message no matter which worker failed
  first. The default stays one worker because each worker holds one decoded image, so a host-derived default would
  make thread count and peak memory machine-dependent.
- Add a `--extensions` flag so the CLI can admit file types outside the default discovery set, such as `.gif`.
- Add an opt-in `exif` timestamp policy that reads capture time from `DateTimeOriginal`, then `DateTimeDigitized`,
  then `DateTime`, applying the matching EXIF 2.31 UTC offset tag when one is recorded and reading an undeclared
  zone as UTC. Unset placeholder tags are skipped; populated but unreadable datetimes and offsets are rejected
  rather than replaced by a weaker source.

### Changed

- CI and tagged releases now consume the frozen dependency lock with pinned automation actions;
  publishing requires successful cross-platform tests and CodeQL, uses reproducible archive
  timestamps, emits checksums and provenance, and refuses to replace an existing release asset.
- The bundled benchmark now records the current package version, and its documented verification
  normalizes only declared runtime provenance while comparing every algorithmic JSON field and the
  rendered SVG exactly.

### Fixed

- Representation and budget-curve records now validate direct construction and
  `dataclasses.replace()`, defensively snapshot bounded collections, and reject infinite budget
  iterables before unbounded materialization.
- Representation analysis revalidates a supplied selection before reading any of its fields, so a
  forged or forcibly mutated result fails with the documented configuration error.
- Validate identifier, count, dimension, and integer-valued continuous inputs against explicit signed or unsigned
  64-bit bounds before selection, rendering, formatting, or JSON serialization; finite floating-point spellings of
  continuous settings retain their original domain limits.
- Convert invalid and unbounded report data into `ConfigurationError` instead of exposing Python integer-string
  conversion errors.
- Reject directly constructed results whose selected count exceeds their budget or whose decision paths disagree
  with the corresponding frame paths.
- Reject symbolic-link scan inputs and nested image links before resolution, including dangling links, and apply the
  same raw-path rule to optional video extraction inputs and outputs.
- Verify video input identity, size, modification time, and SHA-256 before and after FFmpeg decoding; extraction now
  fails without publishing output when the input changes concurrently.
- Verify image identity, size, and modification time before and after each file is scanned, so a file replaced between
  its pixel and metadata reads fails instead of producing a record that mixes two files.

## [0.1.0] - 2026-08-31

### Added

- Natural-order scanning for common image formats with explicit timestamp policies.
- Difference-hash, luminance, entropy, sharpness, and colorfulness measurements.
- Deterministic budgeted selection with endpoint, spacing, duplicate, quality, change, and coverage policies.
- Per-frame JSON decisions and a labeled PNG contact sheet.
- Offline synthetic demo, public Python API, CLI, architecture documentation, tests, and CI.
