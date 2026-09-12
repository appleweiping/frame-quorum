from __future__ import annotations

import json

import pytest
from PIL import Image

import frame_quorum.cli as cli
import frame_quorum.native_scene_images as core
from tests.test_native_scene_images_integration import (
    PTS,
    SIZE,
    TIME_BASE,
    gray,
    independently_decode,
    make_video,
    pair,
    sha256,
    textured,
)


def read_bundle(path):
    return json.loads((path / "manifest.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("image_format", ["png", "jpeg"])
def test_real_cli_once_only_provenance_and_independent_image_decode(
    tmp_path, monkeypatch, capsys, image_format
):
    source = tmp_path / "source.mkv"
    make_video(source, [textured()] * len(PTS))
    originals = independently_decode(source)
    calls = []
    export = cli.export_native_scene_images

    def observed(*args):
        calls.append(args)
        return export(*args)

    monkeypatch.setattr(cli, "export_native_scene_images", observed)
    target = tmp_path / "images"
    assert (
        cli.main(
            [
                "native-scene-images",
                str(source),
                "-o",
                str(target),
                "--image-format",
                image_format,
                "--sample-margin",
                "0",
                "--jpeg-quality",
                "70",
            ]
        )
        == 0
    )
    assert len(calls) == 1 and tuple(d.detector for d in calls[0][2].detectors) == ("adaptive",)
    captured = capsys.readouterr()
    summary = json.loads(captured.out)
    assert captured.err == "" and summary["image_count"] == 3 and summary["unique_sample_count"] == 3
    document = read_bundle(target)
    assert document["decode_passes"] == 2
    assert [row["source_sample_index"] for row in document["images"]] == [0, 3, 7]
    assert summary["manifest_sha256"] == sha256((target / "manifest.json").read_bytes())
    assert summary["total_output_bytes"] == sum(path.stat().st_size for path in target.iterdir())
    lossy = False
    for row, position in zip(document["images"], [0, 3, 7], strict=True):
        pts, tb, size, pixels = originals[position]
        assert row["source_pts"] == pts and row["source_time_base"] == pair(tb)
        assert row["source_time"] == pair(pts * tb)
        assert row["source_rgb_sha256"] == row["transformed_rgb_sha256"] == sha256(pixels)
        path = target / row["file"]
        assert row["bytes"] == path.stat().st_size and row["sha256"] == sha256(path.read_bytes())
        with Image.open(path) as image:
            assert image.format == image_format.upper() and image.size == size
            decoded = image.tobytes()
            assert row["decoded_rgb_sha256"] == sha256(decoded)
        if image_format == "png":
            assert decoded == pixels
        else:
            lossy |= decoded != pixels
    assert lossy is (image_format == "jpeg")


@pytest.mark.parametrize(
    "options,status,positions,end",
    [
        (["--frame-step", "2"], "eof", [1, 1, 2], None),
        (["--max-frames", "2"], "frame_limit", [0, 0, 1], None),
        (["--max-decoded-frames", "2"], "decode_limit", [0, 0, 1], None),
        (["--end", "5.2"], "range_end", [1, 1, 2], {"numerator": 26, "denominator": 5}),
    ],
)
def test_cli_stride_and_limited_unknown_tails(tmp_path, capsys, options, status, positions, end):
    source = tmp_path / "source.mkv"
    make_video(source, [gray(20)] * len(PTS))
    target = tmp_path / "images"
    assert cli.main(["native-scene-images", str(source), "-o", str(target), *options]) == 0
    assert json.loads(capsys.readouterr().out)["image_count"] == 3
    document = read_bundle(target)
    assert [row["source_sample_index"] for row in document["images"]] == positions
    assert document["detection_diagnostics"]["status"] == document["replay_diagnostics"]["status"] == status
    assert document["scenes"][-1]["end_time"] == end
    if options[0] == "--frame-step":
        assert [row["source_decode_index"] for row in document["images"]] == [2, 2, 4]


def test_cli_short_scenes_repeat_samples_with_scene_membership(tmp_path, capsys):
    source = tmp_path / "source.mkv"
    make_video(source, [gray(value) for value in (0, 255, 0)], PTS[:3])
    target = tmp_path / "images"
    assert (
        cli.main(
            [
                "native-scene-images",
                str(source),
                "-o",
                str(target),
                "--detectors",
                "luminance",
                "--threshold",
                "0.5",
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert (result["scene_count"], result["image_count"], result["unique_sample_count"]) == (3, 9, 3)
    assert [(row["scene_ordinal"], row["source_sample_index"]) for row in read_bundle(target)["images"]] == [
        (scene, scene) for scene in range(3) for _ in range(3)
    ]


@pytest.mark.parametrize(
    "limit",
    [
        "--max-total-pixels",
        "--max-frame-pixels",
        "--max-image-total-pixels",
        "--max-image-pixels",
        "--max-verification-pixels",
        "--max-image-bytes",
        "--max-output-bytes",
        "--max-manifest-bytes",
    ],
)
def test_cli_decode_image_verification_and_byte_limits_leave_no_bundle(tmp_path, capsys, limit):
    source = tmp_path / "source.mkv"
    make_video(source, [gray(10)] * len(PTS))
    assert cli.main(["native-scene-images", str(source), "-o", str(tmp_path / "images"), limit, "1"]) == 2
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err.startswith("error: ")
    assert list(tmp_path.iterdir()) == [source]


def test_real_cli_existing_target_and_bad_source_are_preserved(tmp_path, capsys):
    source = tmp_path / "source.mkv"
    source.write_bytes(b"not a supported container")
    target = tmp_path / "images"
    assert cli.main(["native-scene-images", str(source), "-o", str(target)]) == 2
    assert not target.exists() and capsys.readouterr().out == ""
    target.mkdir()
    foreign = target / "foreign"
    foreign.write_bytes(b"keep")
    assert cli.main(["native-scene-images", str(source), "-o", str(target)]) == 2
    assert foreign.read_bytes() == b"keep" and capsys.readouterr().out == ""


def test_cli_encoder_failure_cleans_its_staged_files(tmp_path, monkeypatch, capsys):
    source = tmp_path / "source.mkv"
    make_video(source, [gray(10)] * len(PTS))

    def fail(*args, **kwargs):
        raise OSError("controlled codec failure")

    monkeypatch.setattr(core, "_encode_image", fail)
    assert cli.main(["native-scene-images", str(source), "-o", str(tmp_path / "images")]) == 2
    assert capsys.readouterr().out == "" and list(tmp_path.iterdir()) == [source]


@pytest.mark.parametrize("failure_stage", ["write", "flush"])
def test_stdout_failure_after_real_publication_does_not_roll_back(
    tmp_path, monkeypatch, capsys, failure_stage
):
    source = tmp_path / "source.mkv"
    make_video(source, [gray(10)] * len(PTS))
    target = tmp_path / "images"
    writes = []

    class FailedStdout:
        def write(self, text):
            assert (target / "manifest.json").is_file()
            writes.append(text)
            if failure_stage == "write":
                raise BrokenPipeError("controlled stdout failure after publication")
            return len(text)

        def flush(self):
            raise BrokenPipeError("controlled stdout failure after publication")

    with monkeypatch.context() as patch:
        patch.setattr(cli.sys, "stdout", FailedStdout())
        assert cli.main(["native-scene-images", str(source), "-o", str(target)]) == 2
    assert len(writes) == 1 and json.loads(writes[0])["image_count"] == 3
    assert "controlled stdout failure" in capsys.readouterr().err
    document = read_bundle(target)
    assert len(document["images"]) == 3 and len(list(target.glob("*.png"))) == 3
    assert document["images"][0]["source_pts"] == PTS[1]
    assert document["images"][0]["source_time_base"] == pair(TIME_BASE)
    with Image.open(target / document["images"][0]["file"]) as image:
        assert image.size == SIZE
