# Cuts-only FCPXML 1.9 export

This is a first, deliberately narrow Final Cut Pro interchange profile: one
caller-declared local video asset, one video-only project timeline, and ordered
nonoverlapping cuts. It does not render media, copy audio, perform relinking, or
certify import/playback in Final Cut Pro. In particular, `source_verified: false`
and `editor_import_verified: false` in `audit.json` are intentional.

## Pure API

```python
from fractions import Fraction
from pathlib import Path
from frame_quorum import (
    FCPXMLExportConfig,
    OTIOCut,
    OTIOMedia,
    write_fcpxml_bundle,
)

frame = Fraction(1001, 24000)
media = OTIOMedia(
    Path("source.mov").absolute(),
    origin=Fraction(5),
    available_start=Fraction(5),
    available_end=Fraction(5) + 20 * frame,
)
cuts = (
    OTIOCut(Fraction(5) + 2 * frame, Fraction(5) + 6 * frame),
    OTIOCut(Fraction(5) + 10 * frame, Fraction(5) + 13 * frame),
)
result = write_fcpxml_bundle(
    cuts,
    media,
    Path("new-editor-report"),
    FCPXMLExportConfig(frame_rate=Fraction(24000, 1001), width=1920, height=1080),
)
```

`render_fcpxml(cuts, media, config)` returns the same UTF-8 XML text without
writing. The pure API does not open the media path and can describe a missing
asset for later relinking. The output directory must be new. A successful write
contains `scenes.fcpxml` plus a digest-bound `audit.json`; incomplete staging
is cleaned up, and an existing target is never replaced. XML and audit together
must fit `max_output_bytes` (default and hard ceiling: 64 MiB); at most 10,000
clips are admitted. The shared writer checks short writes and retains its
publication-acknowledgment semantics.

All input times are exact integers or `Fraction`s. The editor source time is
`native_time - media.origin`; available bounds are mandatory and must contain
every selected cut. Source gaps are omitted, so the first clip starts at record
offset `0s` and later clip offsets are cumulative selected durations. The asset
uses the declared available start/duration. Every source cut coordinate and
availability bound must lie on the declared `frame_rate` lattice; off-lattice
VFR cuts fail rather than being silently rounded. The common exact rational
tick-rate limit inherited from OTIO admission is 1,000,000,000, and rendered
times also fit Apple's documented 64-bit numerator / 32-bit denominator model.
Only the first video stream is supported by the native adapter. Media rate,
size, origin, bounds and relinking behavior remain caller declarations.

## Native command and example

```bash
frame-quorum native-fcpxml source.nut --detectors luminance \
  --media-origin 0 --available-start 0 --available-end 6/25 \
  --frame-rate 25 --width 8 --height 6 --final-end 6/25 \
  --output-dir new-editor-report
python -I examples/native_fcpxml.py
```

The CLI requires complete unsampled native scene analysis. For an unknown
final scene tail, supply an explicit `--final-end`; it is not extrapolated from
nominal FPS. `--frame-rate`, dimensions and availability are explicit because
the importer-facing format cannot safely infer them from arbitrary VFR scene
times. The generated example uses a tiny lossless constant-frame-rate source
and checks decoded timestamps, source cuts and XML record offsets. It does not
perform a Final Cut Pro import.

The `.fcpxml` document references an absolute `file:` URI, percent-encoded and
XML-escaped. **It can reveal local directory names** when shared, even though
the sidecar does not repeat the path; regenerate from a nonsensitive media path
or inspect the XML before publishing. No source video is bundled. Names and
the title are escaped as XML attributes and invalid XML characters are rejected.

## Format basis and limits of evidence

Apple describes FCPXML media resources and edited timelines in its
[document guide](https://developer.apple.com/documentation/professional-video-applications/creating-fcpxml-documents),
the [FCPXML reference](https://developer.apple.com/documentation/professional-video-applications/fcpxml-reference),
[media-rep](https://developer.apple.com/documentation/professional-video-applications/media-rep),
[asset-clip](https://developer.apple.com/documentation/professional-video-applications/asset-clip),
[custom format](https://developer.apple.com/documentation/professional-video-applications/format),
and [timing attributes](https://developer.apple.com/documentation/professional-video-applications/timing-attributes).
Apple notes that off-frame timing can insert an import gap; this exporter
rejects it. Apple also says a valid file must conform to the version-specific
DTD. The public DTD article currently shows 1.10 and links older 1.9 DTD
downloads through a developer-account page. We have structurally parsed XML
and checked exact values, but have **not** validated against Apple's 1.9 DTD
or imported it into Final Cut Pro. Do not call this editor interoperability
verified until those gates are completed on macOS.

The frozen comparison project's
[FCPXML writer](https://github.com/Breakthrough/PySceneDetect/blob/24953b0bf76af17c450bc143d330eea48fc5e276/scenedetect/output/__init__.py#L341-L430)
shows the user-facing output category. This exporter is an original, narrower
implementation with explicit availability, frame-lattice and publication
contracts; it is not a byte-for-byte clone of that writer.
