from __future__ import annotations

import argparse
import json
from fractions import Fraction
from pathlib import Path

import pytest

import frame_quorum.cli as cli
import frame_quorum.native_scene_images as core
import frame_quorum.native_video as video
from frame_quorum import NativeSceneImageConfig, NativeSceneImageResult
from frame_quorum.cli import build_parser, main
from frame_quorum.errors import ConfigurationError


def test_native_scene_images_command_is_available():
    args = build_parser().parse_args(["native-scene-images", "source.mkv", "-o", "images"])
    assert args.command == "native-scene-images"


@pytest.fixture
def export_spy(monkeypatch, tmp_path):
    calls = []
    result = NativeSceneImageResult(tmp_path / "images", 2, 6, 4, 1234, "a" * 64)

    def export(source, target, scenes, images):
        calls.append((source, target, scenes, images))
        return result

    monkeypatch.setattr(cli, "export_native_scene_images", export)
    return calls, result


def test_default_cli_configs_and_complete_sorted_ascii_result(export_spy, capsys):
    calls, expected = export_spy
    assert main(["native-scene-images", "source.mkv", "-o", "images"]) == 0
    assert len(calls) == 1
    source, target, scenes, images = calls[0]
    assert source == Path("source.mkv") and target == Path("images")
    assert images == NativeSceneImageConfig()
    assert tuple(item.detector for item in scenes.detectors) == ("adaptive",)
    native = build_parser().parse_args(["native-scenes", "source.mkv"])
    assert scenes.video == cli._native_config(native)
    assert scenes.detectors == cli._detector_configs(native)
    captured = capsys.readouterr()
    assert (
        captured.out
        == json.dumps(expected.to_dict(), ensure_ascii=True, allow_nan=False, sort_keys=True) + "\n"
    )
    assert captured.err == ""


@pytest.mark.parametrize("progress", [0, -1, None, True, "all", "oversized"])
def test_short_or_invalid_stdout_progress_is_failure_after_export(export_spy, monkeypatch, capsys, progress):
    calls, _ = export_spy

    class ShortStdout:
        def write(self, text):
            return len(text) + 1 if progress == "oversized" else progress

        def flush(self):
            pytest.fail("invalid write must fail before flush")

    with monkeypatch.context() as patch:
        patch.setattr(cli.sys, "stdout", ShortStdout())
        assert main(["native-scene-images", "source.mkv", "-o", "images"]) == 2
    assert len(calls) == 1
    assert "complete native scene image result" in capsys.readouterr().err


def test_every_flag_maps_once_and_pixel_budgets_are_independent(export_spy, capsys):
    calls, _ = export_spy
    assert (
        main(
            [
                "native-scene-images",
                "source.mkv",
                "-o",
                "images",
                "--start=-1/2",
                "--end",
                "2.5",
                "--frame-step",
                "2",
                "--video-stream",
                "1",
                "--max-frames",
                "20",
                "--max-decoded-frames",
                "40",
                "--max-source-bytes",
                "5000",
                "--max-frame-pixels",
                "100",
                "--max-total-pixels",
                "1200",
                "--detectors",
                "luminance",
                "color",
                "--threshold",
                "0.6",
                "--min-scene-frames",
                "2",
                "--minimum-votes",
                "2",
                "--min-scene-samples",
                "3",
                "--window-radius",
                "3",
                "--adaptive-ratio",
                "4",
                "--min-content",
                "0.2",
                "--dark-threshold",
                "0.1",
                "--hysteresis",
                "0.03",
                "--min-dark-frames",
                "4",
                "--fade-bias",
                "-0.2",
                "--include-final-fade",
                "--images-per-scene",
                "4",
                "--sample-margin",
                "2",
                "--image-format",
                "jpeg",
                "--png-compression",
                "1",
                "--jpeg-quality",
                "72",
                "--width",
                "4",
                "--height",
                "5",
                "--interpolation",
                "lanczos",
                "--max-scenes",
                "3",
                "--max-images",
                "12",
                "--max-image-pixels",
                "40",
                "--max-image-total-pixels",
                "500",
                "--max-verification-pixels",
                "600",
                "--max-image-bytes",
                "700",
                "--max-output-bytes",
                "9000",
                "--max-manifest-bytes",
                "8000",
            ]
        )
        == 0
    )
    assert len(calls) == 1
    _, _, scenes, images = calls[0]
    assert (scenes.video.start, scenes.video.end, scenes.video.frame_step, scenes.video.video_stream) == (
        Fraction(-1, 2),
        Fraction(5, 2),
        2,
        1,
    )
    assert (scenes.video.max_frames, scenes.video.max_decoded_frames, scenes.video.max_source_bytes) == (
        20,
        40,
        5000,
    )
    assert scenes.video.max_frame_pixels == 100 and scenes.video.max_total_pixels == 1200
    assert images == NativeSceneImageConfig(
        images_per_scene=4,
        sample_margin=2,
        image_format="jpeg",
        png_compression=1,
        jpeg_quality=72,
        width=4,
        height=5,
        interpolation="lanczos",
        max_scenes=3,
        max_images=12,
        max_image_pixels=40,
        max_total_pixels=500,
        max_verification_pixels=600,
        max_image_bytes=700,
        max_output_bytes=9000,
        max_manifest_bytes=8000,
    )
    assert tuple(item.detector for item in scenes.detectors) == ("luminance", "color")
    assert (scenes.minimum_votes, scenes.min_scene_samples) == (2, 3)
    for detector in scenes.detectors:
        assert (detector.threshold, detector.min_scene_frames, detector.window_radius) == (0.6, 2, 3)
        assert (detector.adaptive_ratio, detector.min_content, detector.dark_threshold) == (4, 0.2, 0.1)
        assert (
            detector.hysteresis,
            detector.min_dark_frames,
            detector.fade_bias,
            detector.include_final_fade,
        ) == (0.03, 4, -0.2, True)
    assert json.loads(capsys.readouterr().out)["image_count"] == 6


@pytest.mark.parametrize(
    "text,expected",
    [
        ("2", Fraction(2)),
        ("0.25", Fraction(1, 4)),
        ("3/7", Fraction(3, 7)),
        ("1e-2", Fraction(1, 100)),
        ("+1.5", Fraction(3, 2)),
    ],
)
def test_scale_grammar_is_exact_not_float(text, expected, export_spy):
    calls, _ = export_spy
    assert main(["native-scene-images", "source.mkv", "-o", "images", "--scale", text]) == 0
    assert calls[0][3].scale == expected and type(calls[0][3].scale) is Fraction


@pytest.mark.parametrize(
    "text",
    [
        "0",
        "-1",
        "-1/2",
        "1/0",
        "nan",
        "inf",
        "true",
        "",
        "1e20",
        "1e-20",
        "9223372036854775808",
        "1/9223372036854775808",
    ],
)
def test_scale_invalid_domain_or_syntax_precedes_export(text, export_spy, capsys):
    calls, _ = export_spy
    with pytest.raises(SystemExit) as caught:
        main(["native-scene-images", "source.mkv", "-o", "images", "--scale=" + text])
    assert caught.value.code == 2 and calls == []
    assert "scale must" in capsys.readouterr().err


@pytest.mark.parametrize("text", ["1" * 129, "1e1000000000", "1e+1_000_000", "1e-999999999"])
def test_scale_rejects_expansion_before_fraction_construction(text, monkeypatch):
    monkeypatch.setattr(cli, "Fraction", lambda *_: pytest.fail("must reject before Fraction allocation"))
    with pytest.raises(argparse.ArgumentTypeError, match="scale"):
        cli._exact_scale(text)


@pytest.mark.parametrize(
    "options",
    [
        [],
        ["source.mkv"],
        ["-o", "images"],
        ["source.mkv", "-o", "images", "--image-format", "webp"],
        ["source.mkv", "-o", "images", "--interpolation", "area"],
        ["source.mkv", "-o", "images", "--final-end", "3"],
        ["source.mkv", "-o", "images", "--frame-margin", "1"],
        ["source.mkv", "-o", "images", "--force"],
        ["source.mkv", "-o", "images", "--images-per-scene", "1.5"],
    ],
)
def test_required_arguments_and_no_unimplemented_aliases(options, export_spy):
    calls, _ = export_spy
    with pytest.raises(SystemExit) as caught:
        main(["native-scene-images", *options])
    assert caught.value.code == 2 and calls == []


@pytest.mark.parametrize(
    "options",
    [
        ["--images-per-scene", "0"],
        ["--images-per-scene", "101"],
        ["--sample-margin", "-1"],
        ["--sample-margin", "1000001"],
        ["--png-compression", "10"],
        ["--jpeg-quality", "101"],
        ["--width", "0"],
        ["--height", "-1"],
        ["--width", "4097", "--height", "4097"],
        ["--scale", "1/2", "--width", "4"],
        ["--scale", "2", "--height", "4"],
        ["--max-scenes", "0"],
        ["--max-images", "2"],
        ["--max-image-pixels", "0"],
        ["--max-image-total-pixels", "0"],
        ["--max-verification-pixels", "0"],
        ["--max-image-bytes", "0"],
        ["--max-output-bytes", "0"],
        ["--max-manifest-bytes", "0"],
        ["--max-total-pixels", "0"],
        ["--max-frame-pixels", "0"],
        ["--frame-step", "0"],
        ["--video-stream", "-1"],
        ["--minimum-votes", "2"],
        ["--detectors", "adaptive", "adaptive"],
        ["--threshold", "nan"],
    ],
)
def test_invalid_configuration_fails_before_export(options, export_spy, capsys):
    calls, _ = export_spy
    assert main(["native-scene-images", "source.mkv", "-o", "images", *options]) == 2
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err.startswith("error: ") and calls == []


def test_help_and_existing_native_options_do_not_load_optional_decoder(monkeypatch, capsys):
    def forbidden():
        pytest.fail("help imported optional decoder")

    monkeypatch.setattr(core, "_load_av", forbidden)
    monkeypatch.setattr(video, "_load_av", forbidden)
    with pytest.raises(SystemExit) as caught:
        main(["native-scene-images", "--help"])
    assert caught.value.code == 0
    help_text = capsys.readouterr().out
    assert "--max-total-pixels" in help_text and "--max-image-total-pixels" in help_text
    assert "--max-frame-pixels" in help_text and "--max-image-pixels" in help_text
    assert "--max-verification-pixels" in help_text
    assert cli._exact_seconds("-1/2") == Fraction(-1, 2)
    args = build_parser().parse_args(["native-scenes", "source.mkv", "--max-total-pixels", "77"])
    assert cli._native_config(args).max_total_pixels == 77 and not hasattr(args, "max_image_total_pixels")


def test_existing_target_is_preserved_without_decoder_import(tmp_path, monkeypatch, capsys):
    target = tmp_path / "existing"
    target.mkdir()
    (target / "foreign").write_bytes(b"keep")
    monkeypatch.setattr(video, "_load_av", lambda: pytest.fail("must preflight target"))
    assert main(["native-scene-images", "missing.mkv", "-o", str(target)]) == 2
    assert (target / "foreign").read_bytes() == b"keep"
    assert capsys.readouterr().out == ""


def test_missing_optional_decoder_has_structured_cli_failure_without_stage(tmp_path, monkeypatch, capsys):
    source = tmp_path / "source.nut"
    source.write_bytes(b"not decoded")

    def missing():
        raise ConfigurationError("native video requires the optional frame-quorum[video] extra")

    monkeypatch.setattr(video, "_load_av", missing)
    assert main(["native-scene-images", str(source), "-o", str(tmp_path / "images")]) == 2
    captured = capsys.readouterr()
    assert captured.out == "" and "optional frame-quorum[video]" in captured.err
    assert list(tmp_path.iterdir()) == [source]
