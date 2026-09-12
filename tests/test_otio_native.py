from __future__ import annotations

import json
import runpy
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest
from PIL import Image

from frame_quorum import (
    DetectionConfig,
    NativeSceneConfig,
    NativeVideoConfig,
    OTIOMedia,
    detect_native_scenes,
    otio_cuts_from_native,
    render_otio,
)
from frame_quorum.cli import main
from frame_quorum.errors import ConfigurationError

av = pytest.importorskip("av", reason="install frame-quorum[video] for genuine OTIO native tests")
PTS = (5000, 5040, 5110, 5180, 5270, 5310, 5470, 5680)


def video(path):
    with av.open(str(path), "w", format="nut") as container:
        stream = container.add_stream("ffv1", rate=25)
        stream.width, stream.height, stream.pix_fmt = 8, 6, "bgr0"
        stream.time_base = stream.codec_context.time_base = Fraction(1, 1000)
        for index, pts in enumerate(PTS):
            with Image.new("RGB", (8, 6), (0, 0, 0) if index < 4 else (255, 255, 255)) as image:
                frame = av.VideoFrame.from_image(image)
            frame.pts, frame.time_base = pts, Fraction(1, 1000)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def test_real_vfr_detection_and_exact_native_to_editor_mapping(tmp_path):
    path = tmp_path / "source.nut"
    video(path)
    # This direct decode is independent of the exporter and its scene adapter.
    with av.open(str(path)) as source:
        decoded = []
        for frame in source.decode(video=0):
            with frame.to_image() as image:
                decoded.append((frame.pts * frame.time_base, image.tobytes()))
    assert [time for time, _ in decoded] == [Fraction(pts, 1000) for pts in PTS]
    assert [rgb for _, rgb in decoded] == [bytes([0 if i < 4 else 255]) * 144 for i in range(8)]
    result = detect_native_scenes(path, NativeSceneConfig(detectors=(DetectionConfig(detector="luminance"),)))
    with pytest.raises(ConfigurationError, match="unknown"):
        otio_cuts_from_native(result)
    cuts = otio_cuts_from_native(result, final_end=Fraction(57, 10))
    assert [(cut.start, cut.end) for cut in cuts] == [
        (Fraction(5), Fraction(527, 100)),
        (Fraction(527, 100), Fraction(57, 10)),
    ]
    data = json.loads(render_otio(cuts, OTIOMedia(path, Fraction(5))))
    children = data["tracks"]["children"][0]["children"]
    assert [clip["source_range"]["start_time"]["value"] for clip in children] == [0, 27]
    assert [clip["source_range"]["duration"]["value"] for clip in children] == [27, 43]
    assert data["global_start_time"]["rate"] == 100
    path.unlink()
    assert render_otio(cuts, OTIOMedia(path, Fraction(5)))


@pytest.mark.parametrize("options", [{"frame_step": 2}, {"max_frames": 4}, {"max_decoded_frames": 4}])
def test_native_adapter_does_not_promote_sampled_or_truncated_analysis(tmp_path, options):
    path = tmp_path / "source.nut"
    video(path)
    result = detect_native_scenes(path, NativeSceneConfig(video=NativeVideoConfig(**options)))
    with pytest.raises(ConfigurationError, match="complete, unsampled"):
        otio_cuts_from_native(result, final_end=Fraction(57, 10))
    path.unlink()


@pytest.mark.parametrize("final", [Fraction(5), Fraction(6)])
def test_native_explicit_final_bound_does_not_override_known_end(tmp_path, final):
    path = tmp_path / "source.nut"
    video(path)
    result = detect_native_scenes(path, NativeSceneConfig(video=NativeVideoConfig(end=Fraction(11, 2))))
    with pytest.raises(ConfigurationError):
        otio_cuts_from_native(result, final_end=final)


def test_cli_actual_native_workflow_and_existing_output_preservation(tmp_path, capsys):
    path, target = tmp_path / "source.nut", tmp_path / "report"
    video(path)
    args = [
        "native-otio",
        str(path),
        "--output-dir",
        str(target),
        "--media-origin",
        "5",
        "--final-end",
        "57/10",
        "--detectors",
        "luminance",
        "--include-audio",
    ]
    assert main(args) == 0
    outcome = json.loads(capsys.readouterr().out)
    assert outcome["clip_count"] == outcome["track_count"] == 2
    assert outcome["source_verified"] is False
    original = (target / "scenes.otio").read_bytes()
    assert json.loads(original)["metadata"]["frame_quorum"]["audio"] == "caller_declared_unverified"
    assert main(args) == 2
    assert (target / "scenes.otio").read_bytes() == original
    assert main([*args, "--frame-step", "2"]) == 2
    assert main([*args, "--available-start", "0"]) == 2
    path.unlink()


def test_cli_requires_origin_and_does_not_guess_unknown_eof(tmp_path):
    path = tmp_path / "source.nut"
    video(path)
    with pytest.raises(SystemExit):
        main(["native-otio", str(path), "-o", str(tmp_path / "missing")])
    assert main(["native-otio", str(path), "-o", str(tmp_path / "missing"), "--media-origin", "5"]) == 2
    assert not (tmp_path / "missing").exists()


def test_generated_offline_example(capsys):
    runpy.run_path(str(Path(__file__).parents[1] / "examples" / "native_otio.py"), run_name="__main__")
    outcome = json.loads(capsys.readouterr().out)
    assert outcome == {
        "cuts": [["5", "527/100"], ["527/100", "57/10"]],
        "duration": "7/10",
        "source_verified": False,
    }


def test_nondefault_video_ordinal_cannot_be_misrepresented_as_a_media_reference(tmp_path, monkeypatch):
    import frame_quorum.cli as cli

    path = tmp_path / "source.nut"
    video(path)
    result = detect_native_scenes(path)
    changed = replace(
        result, config=replace(result.config, video=replace(result.config.video, video_stream=1))
    )
    with pytest.raises(ConfigurationError, match="video_stream=0"):
        otio_cuts_from_native(changed, final_end=Fraction(57, 10))
    monkeypatch.setattr(
        cli, "detect_native_scenes", lambda *args: pytest.fail("must preflight before decoding")
    )
    assert (
        main(
            [
                "native-otio",
                str(path),
                "-o",
                str(tmp_path / "out"),
                "--media-origin",
                "5",
                "--video-stream",
                "1",
            ]
        )
        == 2
    )


def test_native_adapter_bound_and_type_before_expanding_clip_objects(tmp_path):
    with pytest.raises(ConfigurationError):
        otio_cuts_from_native(None)
    path = tmp_path / "source.nut"
    video(path)
    result = detect_native_scenes(path)
    # Fault injection proves admission before the old helper validates/expands
    # a deliberately invalid oversized source table; no new media is decoded.
    object.__setattr__(result, "scenes", result.scenes * 10001)
    with pytest.raises(ConfigurationError, match="10000 clips"):
        otio_cuts_from_native(result, final_end=Fraction(57, 10))
