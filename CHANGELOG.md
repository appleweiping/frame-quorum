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

## [0.1.0] - 2026-08-31

### Added

- Natural-order scanning for common image formats with explicit timestamp policies.
- Difference-hash, luminance, entropy, sharpness, and colorfulness measurements.
- Deterministic budgeted selection with endpoint, spacing, duplicate, quality, change, and coverage policies.
- Per-frame JSON decisions and a labeled PNG contact sheet.
- Offline synthetic demo, public Python API, CLI, architecture documentation, tests, and CI.
