# Changelog

All notable changes are documented here. The project follows semantic versioning.

## [Unreleased]

### Added

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
- Add a `--extensions` flag so the CLI can admit file types outside the default discovery set, such as `.gif`.
- Add an opt-in `exif` timestamp policy that reads capture time from `DateTimeOriginal`, then `DateTimeDigitized`,
  then `DateTime`, applying the matching EXIF 2.31 UTC offset tag when one is recorded and reading an undeclared
  zone as UTC. Unset placeholder tags are skipped; populated but unreadable datetimes and offsets are rejected
  rather than replaced by a weaker source.

### Fixed

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
