# Changelog

All notable changes are documented here. The project follows semantic versioning.

## [Unreleased]

### Fixed

- Validate identifier, count, dimension, and integer-valued continuous inputs against explicit signed or unsigned
  64-bit bounds before selection, rendering, formatting, or JSON serialization; finite floating-point spellings of
  continuous settings retain their original domain limits.
- Convert invalid and unbounded report data into `ConfigurationError` instead of exposing Python integer-string
  conversion errors.
- Reject directly constructed results whose selected count exceeds their budget or whose decision paths disagree
  with the corresponding frame paths.

## [0.1.0] - 2026-08-31

### Added

- Natural-order scanning for common image formats with explicit timestamp policies.
- Difference-hash, luminance, entropy, sharpness, and colorfulness measurements.
- Deterministic budgeted selection with endpoint, spacing, duplicate, quality, change, and coverage policies.
- Per-frame JSON decisions and a labeled PNG contact sheet.
- Offline synthetic demo, public Python API, CLI, architecture documentation, tests, and CI.
