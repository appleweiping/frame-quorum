# Frame Quorum

[![CI](https://github.com/appleweiping/frame-quorum/actions/workflows/ci.yml/badge.svg)](https://github.com/appleweiping/frame-quorum/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776AB)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-58d6a9)](LICENSE)

Frame Quorum selects a compact, explainable set of key frames from an image sequence. It favors meaningful
visual change, clear and information-rich images, and coverage across time while enforcing a frame budget,
minimum spacing, and near-duplicate suppression.

It works directly on image directories. There is no mandatory video decoder, model download, GPU, network
request, or hidden inference step. Every selected and rejected frame receives a machine-readable reason.

![Contact sheet produced by the demo](examples/output/contact-sheet.png)

## Why use it?

- **Auditable selection:** the manifest records quality, change, coverage, total utility, rank, and reason.
- **Content-aware:** a structural difference hash is combined with mean color and luminance, avoiding the
  common failure where differently colored flat frames appear identical.
- **Constraint-aware:** budget, minimum time gap, endpoint preservation, and duplicate threshold are explicit.
- **Portable:** Python 3.11+ and Pillow are sufficient; input can be PNG, JPEG, WebP, BMP, or TIFF.
- **Reproducible:** discovery, scoring, tie-breaking, JSON layout, and the included demo are deterministic.
- **Bounded presentation:** metric sampling and contact-sheet canvas size have explicit memory guards.

## Quick start

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"

frame-quorum demo --output-dir demo-output
```

The demo creates 18 synthetic frames, `manifest.json`, and `contact-sheet.png`. It uses no downloaded data.
Recreate the repository's checked-in example exactly with:

```bash
python examples/create_demo.py
```

## Commands

### Inspect an image sequence

```bash
frame-quorum scan ./frames --output scan.json
```

`scan` discovers images in natural filename order (`frame_2` before `frame_10`), reads EXIF orientation,
extracts content measurements, and emits JSON. Add `--recursive` to include nested directories.

### Select key frames

```bash
frame-quorum select ./frames \
  --output-dir ./selection \
  --budget 8 \
  --min-gap 1.5 \
  --duplicate-threshold 0.035
```

Output consists of:

- `manifest.json`: every input frame, measurement, score, decision, and selected-frame summary;
- `contact-sheet.png`: the selected source pixels with index, rank, utility, and timestamp labels.

Endpoint preservation is enabled by default. Pass `--no-endpoints` when the first and last images are not
semantically meaningful. `--min-gap` uses seconds when timestamps exist and sequence indices otherwise.

### Extract timestamps from filenames

The default timestamp is `index / frame_rate`. For files such as `camera_004250ms.png`:

```bash
frame-quorum select ./frames --output-dir ./selection \
  --timestamp-mode filename \
  --timestamp-regex '_(?P<ts>\d+)ms$' \
  --timestamp-unit milliseconds
```

The regex may contain a named `ts` group or use its first capture group. Other policies are `mtime` and
`none`. A filename mismatch is an error rather than a silently invented timestamp.

## How selection works

Each image is measured once:

1. A 64-bit horizontal difference hash captures low-frequency structure.
2. Mean RGB and luminance complement that hash for flat or low-texture scenes.
3. Grayscale entropy measures information variety.
4. Interior edge energy estimates sharpness.
5. Chroma spread estimates colorfulness.

The selector reserves admissible endpoints, then greedily fills the remaining budget. At each step it scores
an eligible frame using three normalized terms:

```text
utility = quality_weight × quality
        + change_weight  × change_from_previous
        + coverage_weight × distance_from_selected_times
```

Weights are normalized by their sum. A candidate is ineligible when it is closer than `min_gap` to an already
selected frame or its combined content distance is at or below `duplicate_threshold`. Equal scores are resolved
by content change and then sequence index, making repeated runs stable.

This is an interpretable heuristic, not semantic understanding. See [architecture and algorithm boundaries](docs/architecture.md).

## Python API

```python
from frame_quorum import ScanConfig, SelectionConfig, scan_frames, select_frames

frames = scan_frames("frames", ScanConfig(frame_rate=2.0))
result = select_frames(frames, SelectionConfig(budget=6, min_gap=1.0))

for frame in result.selected_frames:
    decision = result.decision_for(frame.index)
    print(frame.relative_path, decision.reason)
```

The records are frozen dataclasses. The selector never modifies the source images or copies them into its output.

## Manifest contract

Manifests declare `schema_version: "1.0"` and `kind`. Floating-point score fields are rounded to six decimal
places. Paths are relative to the scanned input root. The selection manifest includes both a concise `selected`
list and the complete frame list with decisions, allowing consumers to audit rejection as well as inclusion.

Minor releases may add fields. Removing or changing field meaning requires a schema-version change.

## Development

```bash
python -m pip install -e ".[dev]"
python -m ruff check .
python -m ruff format --check .
python -m pytest --cov
```

Tests synthesize their own images and make no network requests. CI runs the full suite on Linux with Python 3.11
through 3.14, plus Windows with Python 3.12. The coverage floor is 94% with branch coverage enabled.

## Limitations

- A perceptual change is not necessarily an important event; the tool does not recognize people or objects.
- Sharp text, overlays, camera flashes, and cuts can receive high change or quality scores.
- Animated image formats are treated as one image file, not expanded into their internal frames.
- EXIF timestamps are not inferred. Use an explicit filename rule, modification time, or indexed frame rate.
- Very large directories are scanned sequentially. Images are downsampled for metrics, but file decoding still
  occurs once per frame.
- The contact sheet reads selected source images again to preserve visual quality and rejects files whose
  dimensions, size, or measured content changed after scanning.
- Filename timestamp regular expressions are caller-supplied Python regular expressions; do not accept an
  arbitrary pattern from an untrusted tenant without an external execution timeout.

These boundaries are deliberate: the baseline stays inspectable, offline, and inexpensive. Semantic embeddings
can be added later behind an optional adapter without changing the manifest's decision model.

## Privacy and security

Frame Quorum is local-only and does not transmit images. Treat manifests as potentially sensitive because file
names, dimensions, timestamps, and image statistics may reveal information. Review [SECURITY.md](SECURITY.md)
before processing untrusted or confidential files. Selection output must be outside the scanned input directory,
so generated reports cannot become inputs on a later run.

## License

[MIT](LICENSE)
