# Review edited scene starts with verified stills

`native-load-overview` takes a bounded PySceneDetect-style `Start Frame` CSV
and publishes selected scene stills plus an offline HTML review page. It is
for visually checking an edited cut list without rerunning scene detection:

```bash
frame-quorum native-load-overview source.mkv --scene-csv scenes.csv \
  --images-per-scene 2 --output-dir new-review
python -I examples/native_load_overview.py
```

The destination must not exist. It receives `manifest.json`, `index.html`,
`overview.json` and verified PNG/JPEG files. The HTML has an offline content
security policy and references only the published image names. It does show
the local source path; review the page before sharing it. Ignored CSV cells
are never copied into the page or manifest.

The same strict [CSV parser](native-load-fcpxml.md) reads one-based scene
starts. `1,3,6` identifies decoded frame intervals `[0,2)`, `[2,5)` and
`[5,N)`; no timecode-column import or sorting is performed. The first pass
decodes an unsampled local, nonsymlink video stream zero from its beginning
to proved, error-free EOF, records each exact PTS and RGB digest, and checks
strictly increasing PTS. A frame or decode limit is not EOF. The second
full decode checks every frame's index, PTS, dimensions and RGB digest,
including frames not selected for stills. Selected images are encoded,
reopened and checked before the complete bundle is published without
replacing another directory. A detected source change or replay mismatch
fails the export.

The CSV's frame ordinals are independent of frame rate, so genuine VFR is
accepted for still-image review. A cut's time is the first actual frame
of the next scene. The last scene's *exclusive duration endpoint* remains
unknown even after EOF; the exporter never extrapolates one frame period.
An image position inside each half-open scene uses the existing bounded
scene-image slot/margin policy. Very short scenes may repeat a frame in
multiple slots. At most 1,000 scenes and 10,000 images can be admitted,
subject to the lower caller-configured byte/pixel/decode limits.

The imported-cut manifest binds the exact CSV bytes read by SHA-256, both
decode diagnostics, scene intervals, image pixel/encoded-file hashes and
the HTML/manifest audit. `detector_performed=false` is deliberate: a CSV
decision is not detector evidence. The current source was replay-checked
for this export, but can change afterward, so
`source_content_authenticated=false`. Failed output-stage checks do not
publish a partial directory; if publication may have succeeded without an
acknowledgment, inspect the named destination before retrying. A stdout
error after publication cannot undo the complete bundle.

Python callers may use `export_loaded_scene_overview(video, csv, output_dir,
*, video_limits=..., image_config=..., overview_config=...)`. This does not
implement general `load-scenes` chaining, `Start Timecode`, no-image HTML,
arbitrary filenames, FCPXML/VFR conversion, durable source authentication
or whole PySceneDetect parity.
