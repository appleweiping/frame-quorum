# Offline native scene overview

`export_native_scene_overview` detects scenes from one local video, verifies PNG
or JPEG stills selected from actual returned samples, and publishes a new
directory containing those stills, `manifest.json`, `index.html` and
`overview.json`. The HTML links only to the stills inside that directory. It
requires the optional PyAV video dependency; there is no network input, browser
launch, remote asset, script or new runtime dependency.

```console
frame-quorum native-scene-overview input.mkv --output-dir new-overview --images-per-scene 3 --width 640
frame-quorum native-scene-overview input.mkv -o new-jpeg-overview --image-format jpeg --image-width 320 --title "My scenes"
python examples/native_scene_overview.py
```

The `--width` and `--height` flags change encoded image pixels. The separate
`--image-width` and `--image-height` flags affect only how `<img>` elements
display in HTML; their values are bounded from 1 to 4096. `--columns` accepts
1 through 6 and chooses a fixed responsive layout. `--title` accepts 1 through
256 printable Unicode scalar characters; it is always HTML-escaped. Other
image, native video and detector flags match `native-scene-images`, including
independent per-decode and output pixel budgets. The CLI uses the same adaptive
default as `native-scenes`; the Python API's default scene config uses its
existing content detector.

```python
from frame_quorum import (
    NativeSceneImageConfig,
    NativeSceneOverviewConfig,
    export_native_scene_overview,
)

result = export_native_scene_overview(
    "input.mkv",
    "new-overview",  # Must not exist; parent must exist.
    image_config=NativeSceneImageConfig(images_per_scene=3, width=640),
    overview_config=NativeSceneOverviewConfig(title="My scenes", columns=3, image_width=320),
)
print(result.to_dict())
```

The source path is displayed as escaped plain text, never used as a link. The
report shows accepted cuts, each scene's returned-sample range, exact rational
start/last-observed times, known requested or cut endpoint, and the selected
image's native PTS and exact time. An unknown final endpoint says “Unknown
endpoint.” EOF only describes completion of the configured scan; the page never
infers whole-media duration from the last frame. Count-limited scans likewise
describe only their returned samples. The HTML is fixed UTF-8 with static styles,
semantic table headings, image alternative text and a restrictive Content
Security Policy. Browser support should be checked through a local static server
when using a platform where `file://` security rules vary.

The HTML and `manifest.json` contain the source's full local path, which may
reveal private directory names if the bundle or a screenshot is shared. Inspect
them before publishing, or regenerate the bundle from a nonsensitive path.

`manifest.json` is byte-for-byte the original image export manifest for the
same source and configuration. It records every still's encoded byte size and
SHA-256 plus observed native pixel/PTS provenance. `overview.json` binds that
image manifest's exact bytes and SHA-256 to `index.html`'s bytes and SHA-256.
The new result adds `overview_sha256` and the total byte count of **all**
published files. It does not modify the original `NativeSceneImageResult` or
its wire format.

Images, JSON and HTML share one bounded stage. The existing
`NativeSceneImageConfig.max_output_bytes` covers their aggregate output;
`max_image_bytes`, `max_manifest_bytes`, `max_html_bytes` (up to 8 MiB) and
`max_overview_bytes` (up to 64 KiB) are additional individual limits. The
complete owned file inventory, identities, byte sizes and hashes are checked
once before atomic no-replace publication. A preexisting target remains
untouched. A failed publication acknowledgement may mean the complete
directory was already moved into place; inspect the destination rather than
assuming it was rolled back. A stdout write/flush failure after publication
also leaves the completed directory in place. See the [image ownership and
input contract](native-scene-images.md) for the two native passes and cleanup
rules.

This is an original image-backed HTML profile of the frozen
[PySceneDetect `save-html` workflow](https://github.com/Breakthrough/PySceneDetect/blob/24953b0bf76af17c450bc143d330eea48fc5e276/docs/cli.rst).
Image-free HTML, reusing prior `save-images` output, writing HTML from an
independent caller-supplied scene list, filename templates, automatic browser
opening and broader image/backend interoperability remain open. The whole
reference repository's functionality, scale and release surface are still not
matched by this feature alone.
