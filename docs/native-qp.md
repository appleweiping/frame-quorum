# Native scene cuts as encoder QP I-frame requests

`native-qp` analyzes one complete local video stream and publishes an x264/x265
style QP instruction file. It writes `0 I -1` for the first decoded frame and
one `{ordinal} I -1` line for each accepted scene cut. Ordinals are **zero-based
decoded frame positions**, not PTS-derived frame estimates. The command never
runs an encoder.

```bash
frame-quorum native-qp local.mkv --detectors luminance \
  --max-frames 5000 --output-dir new-qp
python -I examples/native_qp.py
```

The Python API accepts an already-computed `NativeSceneResult`:

```python
from frame_quorum import write_native_qp_bundle

published = write_native_qp_bundle(result, "new-qp")
print(published.output_path)
```

The result must be nonempty, unsampled (`frame_step=1`), unwindowed, from one
zero-seek decode generation and terminated at actual EOF without a count limit
or cleanup error. Every decoded frame must have been returned. These conditions
make `sample_index == decode_index == encoder input ordinal`, even with VFR,
nonzero/negative timestamps or equal timestamps. A start/end window, sampling,
seek, limit-truncated run or empty source is rejected rather than shifted or
rounded. The CLI initially requires video stream zero; the pure writer records
the selected stream ordinal in its audit.

A new, never-overwritten directory contains exactly `scenes.qp` and `audit.json`.
The QP file is ASCII with LF endings. The audit records counts, stream ordinal,
byte length and SHA-256, plus `source_content_authenticated=false` and
`encoder_input_verified=false`. It does not expose a source path. A supplied
result passes structural revalidation but is not cryptographically proven to
have come from the current media. QP and audit together have a 32 MiB hard
ceiling; `--max-output-bytes` can lower it. Publication reuses the owned
no-replace directory writer, including short-write and uncertain-acknowledgment
handling. A failure after publication may leave the complete directory in
place; inspect it before retrying.

Feed an encoder **the same decoded video stream, starting at frame zero, without
dropping, duplicating, reordering or inserting frames**. Retiming, changed
decoder behavior or a different stream invalidates these ordinal requests.
The included generated CFR/VFR tests compare direct decoder ordinals and exact
QP bytes. The direct-media VFR fixture has positive nonzero, irregular PTS.
Fixture qualification with FFV1 requested ticks
`(-1000, -960, -890, -820, -730, -690, -530)` at time base `1/1000`:
Matroska decoded `(0, 40, 110, 180, 270, 310, 470)`, while NUT decoded
`(0, 2560, 7040, 11520, 17280, 19840, 30080)`; neither preserved the
negative origin. NUT rejected a repeated negative timestamp at ticks
`(-1000, -960, -960, -820, -730, -690, -530)` with an invalid-argument mux
error. The negative/equal-PTS unit cases therefore transform an otherwise
valid scene result and revalidate it; they are **not** independent real-media
oracles for those timestamp edges. No x264/x265 invocation or broad encoder
compatibility is claimed.
The frozen PySceneDetect
[`save-qp` command](https://github.com/Breakthrough/PySceneDetect/blob/24953b0bf76af17c450bc143d330eea48fc5e276/docs/cli.rst#L923-L949)
is the user-facing reference category. Its optional window-shift mode is not
part of this bounded profile; neither are Aegisub keyframes, input scene-list
CSV, arbitrary encoder QP values or multi-source composition.
