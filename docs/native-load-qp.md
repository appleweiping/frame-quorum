# Edited scene starts to encoder QP instructions

`native-load-qp` imports a bounded PySceneDetect-style `Start Frame` CSV and
publishes x264/x265-style I-frame requests without running a detector or encoder:

```bash
frame-quorum native-load-qp source.mkv --scene-csv scenes.csv --output-dir new-qp
python -I examples/native_load_qp.py
```

Python callers use `write_loaded_qp_bundle(video, csv, output_dir, *,
video_limits=..., max_input_bytes=..., max_scenes=..., max_output_bytes=...)`.
The output directory must not exist. It contains exactly `scenes.qp` and
`audit.json`; stdout reports the published result only afterward.

The existing [strict CSV loader](native-load-fcpxml.md) reads one-based starts.
`1,3,6` means scene starts at zero-based decoded ordinals `0,2,5`, so QP bytes
are exactly `0 I -1\n2 I -1\n5 I -1\n` (ASCII LF). Ignored CSV fields are
not copied or trusted. The command decodes an unsampled stream-zero local
nonsymlink video from its first frame to proved, error-free EOF, with at most
100,000 frames and lower configurable work/byte/pixel ceilings. A count limit
is not EOF. The source must be nonempty, with contiguous generation-zero
decoded ordinals and stable dimensions. A cut at or after EOF fails. It uses
decoded ordinals, never nominal FPS or PTS division; VFR is allowed if the
native decoder admits it. No final-frame duration is inferred.

The audit binds the exact read CSV bytes and published QP bytes by SHA-256,
and says `detector_performed=false`, `source_content_authenticated=false`,
`encoder_input_verified=false`. It contains no source path. The old
detector-to-QP command and its wire format are unchanged. The complete staged
bundle is byte-reconciled and published without replacing an existing target.
An unacknowledged rename may have succeeded; inspect the target before retry.
A stdout failure cannot undo a published bundle.

Feed the encoder the **same decoded video stream from frame zero**, without
inserting, dropping, reordering or duplicating frames. The CSV hash and
completed scan do not authenticate future encoder input. This is one bounded
connection of the frozen [PySceneDetect `load-scenes`](https://github.com/Breakthrough/PySceneDetect/blob/24953b0bf76af17c450bc143d330eea48fc5e276/docs/cli.rst#L627-L650)
and [`save-qp`](https://github.com/Breakthrough/PySceneDetect/blob/24953b0bf76af17c450bc143d330eea48fc5e276/docs/cli.rst#L923-L949)
workflow categories, not their general chaining, sorting, timecode column,
window shifts, encoder execution or whole-repository parity.
