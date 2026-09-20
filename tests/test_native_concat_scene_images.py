"""Boundary, replay-failure and safe-publication tests for composite stills."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import replace
from fractions import Fraction

import pytest
from PIL import Image

import frame_quorum.native_concat_scene_images as module
from frame_quorum import (
    DetectionConfig,
    NativeConcatClip,
    NativeConcatConfig,
    NativeConcatLimits,
    NativeConcatSceneConfig,
    NativeConcatSceneImageResult,
    NativeSceneImageConfig,
    export_native_concat_scene_images,
)
from frame_quorum.errors import ConfigurationError, OutputError, ScanError
from tests._v1_wire_snapshot import digest_summary
from tests.test_native_concat import decoder as _decoder_fixture
from tests.test_native_video import FakeFrame

decoder = _decoder_fixture


def _document(path):
    return json.loads((path / "manifest.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    "changed",
    [
        {"scene_count": 0},
        {"image_count": 3},
        {"unique_sample_count": 0},
        {"total_output_bytes": True},
        {"manifest_sha256": "A" * 64},
        {"timeline_digest": None},
    ],
)
def test_result_admission_and_exact_projection(tmp_path, changed):
    value = NativeConcatSceneImageResult(tmp_path, 2, 6, 2, 100, "a" * 64, "b" * 64)
    with pytest.raises(ConfigurationError):
        replace(value, **changed)
    assert value.to_dict() == {
        "output_dir": tmp_path.as_posix(),
        "scene_count": 2,
        "image_count": 6,
        "unique_sample_count": 2,
        "total_output_bytes": 100,
        "manifest_sha256": "a" * 64,
        "timeline_digest": "b" * 64,
    }


def _proxy_replay(monkeypatch, *, frame_change=None, diagnostic_change=None):
    real_stream = module.NativeConcatStream
    streams = []

    class Replay:
        def __init__(self, inner):
            self.inner = inner
            self.timeline = inner.timeline

        @property
        def metadata(self):
            return self.inner.metadata

        @property
        def diagnostics(self):
            value = self.inner.diagnostics
            return value if diagnostic_change is None else replace(value, **diagnostic_change)

        def __enter__(self):
            self.inner.__enter__()
            return self

        def __exit__(self, *args):
            return self.inner.__exit__(*args)

        def __iter__(self):
            for index, frame in enumerate(self.inner):
                yield frame if frame_change is None else frame_change(frame, index)

    def factory(*args):
        stream = real_stream(*args)
        streams.append(stream)
        return stream if len(streams) == 1 else Replay(stream)

    monkeypatch.setattr(module, "NativeConcatStream", factory)
    return streams


def test_signed_main_v1_wire_byte_baselines_remain_unchanged(tmp_path):
    # Recorded by tests/_v1_wire_snapshot.py against the signed main commit
    # 211e611b7535d46308e8ff000d5caa71a10e8082 on both Windows and Linux,
    # then independently reproduced on this branch. PNG compression bytes
    # differ across those platforms, so keep separate exact wire digests.
    image_hashes = {
        "win32": "238fced2a4d11409fbf9bd56e610975e5cb086cebb3d86d8011d714615484a49",
        "linux": "a01ec47bd8487e830db4065059303b4e655e2910ef325c036faf55ebc08415c5",
    }
    if sys.platform not in image_hashes:
        pytest.skip("signed-main PNG wire baseline is not recorded for this platform")
    assert digest_summary(tmp_path) == {
        "concat_scene_v1": {
            "bytes": 7588,
            "sha256": "df7d57241e491bdc2a8d4865557d4171274f731a6f2e7dce098922ae7960941c",
        },
        "single_source_image_v1": {
            "bytes": 4524,
            "sha256": image_hashes[sys.platform],
        },
    }


@pytest.mark.parametrize("changed", [[], (None,), "invalid", (), None])
def test_invalid_clips_precede_source_open(decoder, tmp_path, changed):
    with pytest.raises(ConfigurationError):
        export_native_concat_scene_images(changed, tmp_path / "out")
    assert decoder.containers == []
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("position", ["scene", "image"])
def test_invalid_config_type_precedes_source_open(decoder, tmp_path, position):
    with pytest.raises(ConfigurationError):
        export_native_concat_scene_images(
            decoder.clips,
            tmp_path / "out",
            {} if position == "scene" else None,
            {} if position == "image" else None,
        )
    assert decoder.containers == []


def test_existing_output_is_rejected_before_source_open(decoder, tmp_path):
    target = tmp_path / "existing"
    target.mkdir()
    with pytest.raises(OutputError):
        export_native_concat_scene_images(decoder.clips, target)
    assert not decoder.containers


@pytest.mark.parametrize(
    "limits",
    [
        NativeConcatLimits(max_frames=6),
        NativeConcatLimits(max_activations=3),
        NativeConcatLimits(max_source_bytes=1),
        NativeConcatLimits(max_decoded_frames=2),
        NativeConcatLimits(max_total_pixels=1),
    ],
)
def test_decoder_and_activation_budgets_never_publish_partial_output(decoder, tmp_path, limits):
    target = tmp_path / "images"
    config = NativeConcatSceneConfig(video=NativeConcatConfig(limits=limits))
    with pytest.raises((ConfigurationError, ScanError)):
        export_native_concat_scene_images(decoder.clips, target, config)
    assert not target.exists()
    assert all(container.closed for container in decoder.containers)


@pytest.mark.parametrize("field", ["max_total_pixels", "max_verification_pixels"])
def test_output_plan_limits_are_checked_before_staging(decoder, tmp_path, field):
    target = tmp_path / "images"
    config = NativeSceneImageConfig(**{field: 1})
    with pytest.raises(ConfigurationError):
        export_native_concat_scene_images(decoder.clips, target, image_config=config)
    assert not target.exists()
    assert not list(tmp_path.glob(".frame-quorum-concat-scene-images-*"))


@pytest.mark.parametrize("image_limit", [{"max_images": 3}, {"max_scenes": 1}])
def test_slot_count_after_cut_exceeds_admitted_global_limit_without_stage(decoder, tmp_path, image_limit):
    # Two visible luminance segments; 3 slots each exceed a 3-image budget.
    for frame in decoder.frames[1]:
        frame.to_image = lambda: Image.new("RGB", (2, 2), (255, 255, 255))
    config = NativeConcatSceneConfig(detectors=(DetectionConfig(detector="luminance", threshold=0.5),))
    image = NativeSceneImageConfig(**image_limit)
    with pytest.raises(ConfigurationError, match="slot count"):
        export_native_concat_scene_images(decoder.clips, tmp_path / "images", config, image)
    assert not (tmp_path / "images").exists()


def test_zero_observed_samples_rejected_without_output(decoder, tmp_path):
    target = tmp_path / "images"
    config = NativeConcatSceneConfig(video=NativeConcatConfig(start=Fraction(2, 5)))
    with pytest.raises(ScanError, match="observed"):
        export_native_concat_scene_images(decoder.clips, target, config)
    assert not target.exists()


def test_zero_duration_scene_with_one_observed_sample_keeps_repeated_slots(decoder, tmp_path):
    decoder.frames[0][:] = [FakeFrame(5000), FakeFrame(5100), FakeFrame(5100), FakeFrame(5190)]
    decoder.frames[1][:] = []
    for frame, gray in zip(decoder.frames[0], (0, 255, 0, 0), strict=True):
        frame.to_image = lambda level=gray: Image.new("RGB", (2, 2), (level, level, level))
    config = NativeConcatSceneConfig(detectors=(DetectionConfig(detector="luminance"),))
    target = tmp_path / "images"
    result = export_native_concat_scene_images(decoder.clips, target, config)
    document = _document(target)
    assert result.scene_count == 3
    middle = document["scenes"][1]
    assert middle["start_position"] == 1 and middle["end_position"] == 2
    assert middle["start_time"] == middle["end_time"] == {"numerator": 1, "denominator": 10}
    assert middle["spans"] == []
    assert [row["sample_index"] for row in document["images"]][3:6] == [1, 1, 1]
    assert all((target / row["file"]).is_file() for row in document["images"])


def test_symbolic_source_is_rejected_without_source_open(decoder, tmp_path):
    link = tmp_path / "link.mkv"
    try:
        link.symlink_to(decoder.paths[0])
    except OSError:
        pytest.skip("local symlink creation is unavailable")
    with pytest.raises(ScanError, match="symbolic-link"):
        NativeConcatClip(link, start=5, end=Fraction(26, 5))
    assert not decoder.containers


@pytest.mark.parametrize("field", ["max_image_bytes", "max_manifest_bytes", "max_output_bytes"])
def test_output_byte_limits_fail_without_partial_target(decoder, tmp_path, field):
    target = tmp_path / "images"
    with pytest.raises(OutputError):
        export_native_concat_scene_images(
            decoder.clips, target, image_config=NativeSceneImageConfig(**{field: 1})
        )
    assert not target.exists()
    assert not list(tmp_path.glob(".frame-quorum-concat-scene-images-*"))


@pytest.mark.parametrize("mutation", ["rgb", "pts", "occurrence", "dimensions"])
@pytest.mark.parametrize("sample_index", [0, 1])
def test_every_pass_two_sample_identity_or_rgb_mismatch_cleans_owned_stage(
    decoder, tmp_path, monkeypatch, mutation, sample_index
):
    # The fake corpus has one six-sample scene: three slots select 1, 2, 4.
    # Index 0 is returned but never encoded, and must still be replay-verified.
    target = tmp_path / "images"

    def change(frame, index):
        if index != sample_index:
            return frame
        if mutation == "rgb":
            native = replace(frame.native, rgb=b"\x00" * len(frame.native.rgb))
            return replace(frame, native=native)
        if mutation == "pts":
            return replace(frame, native=replace(frame.native, pts=frame.native.pts + 1))
        if mutation == "dimensions":
            native = replace(frame.native, width=1, height=4)
            return replace(frame, native=native)
        object.__setattr__(frame, "clip_index", 1)
        return frame

    streams = _proxy_replay(monkeypatch, frame_change=change)
    with pytest.raises(ScanError, match="sample differs"):
        export_native_concat_scene_images(decoder.clips, target)
    assert not target.exists()
    assert not list(tmp_path.glob(".frame-quorum-concat-scene-images-*"))
    assert all(stream.diagnostics.closed for stream in streams)


def test_pass_two_completion_diagnostics_mismatch_cleans_stage(decoder, tmp_path, monkeypatch):
    streams = _proxy_replay(monkeypatch, diagnostic_change={"opened_source_bytes": 65})
    with pytest.raises(ScanError, match="completion differs"):
        export_native_concat_scene_images(decoder.clips, tmp_path / "images")
    assert not (tmp_path / "images").exists()
    assert not list(tmp_path.glob(".frame-quorum-concat-scene-images-*"))
    assert all(stream.diagnostics.closed for stream in streams)


def test_source_stat_identity_change_after_analysis_rejects_before_stage(decoder, tmp_path, monkeypatch):
    capture = module._capture

    def altered(*args):
        result = capture(*args)
        path = decoder.paths[0]
        info = path.stat()
        os.utime(path, ns=(info.st_atime_ns, info.st_mtime_ns + 100_000_000))
        return result

    monkeypatch.setattr(module, "_capture", altered)
    with pytest.raises(ScanError, match="source changed"):
        export_native_concat_scene_images(decoder.clips, tmp_path / "images")
    assert not (tmp_path / "images").exists()


def test_same_length_owned_stage_corruption_is_caught_and_cleaned(decoder, tmp_path, monkeypatch):
    reconcile = module._reconcile

    def corrupt(stage, identity, owned, expected, budget):
        image = next(path for path in owned if path.suffix == ".png")
        raw = image.read_bytes()
        image.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
        return reconcile(stage, identity, owned, expected, budget)

    monkeypatch.setattr(module, "_reconcile", corrupt)
    with pytest.raises(OutputError, match="hash"):
        export_native_concat_scene_images(decoder.clips, tmp_path / "images")
    assert not (tmp_path / "images").exists()
    assert not list(tmp_path.glob(".frame-quorum-concat-scene-images-*"))


def test_encoder_short_write_is_not_swallowed_and_cleans_owned_stage(decoder, tmp_path, monkeypatch):
    import frame_quorum.native_scene_images as image_module

    real_writer = image_module._Writer

    class ShortWriter(real_writer):
        def write(self, data):
            if self.handle.name.endswith(".png"):
                self.failed = True
                raise OutputError("injected short output write")
            return super().write(data)

    monkeypatch.setattr(image_module, "_Writer", ShortWriter)
    with pytest.raises(OutputError, match="short output write"):
        export_native_concat_scene_images(decoder.clips, tmp_path / "images")
    assert not (tmp_path / "images").exists()
    assert not list(tmp_path.glob(".frame-quorum-concat-scene-images-*"))


def test_image_close_failure_after_analysis_cleans_owned_stage(decoder, tmp_path, monkeypatch):
    original_capture = module._capture

    def capture_then_break_close(*args):
        result = original_capture(*args)

        def broken_close(self):
            raise OSError("injected image close failure")

        monkeypatch.setattr(Image.Image, "close", broken_close)
        return result

    monkeypatch.setattr(module, "_capture", capture_then_break_close)
    with pytest.raises(OutputError, match="resource cleanup failed"):
        export_native_concat_scene_images(decoder.clips, tmp_path / "images")
    assert not (tmp_path / "images").exists()
    assert not list(tmp_path.glob(".frame-quorum-concat-scene-images-*"))


def test_foreign_racing_target_is_preserved(decoder, tmp_path, monkeypatch):
    real_publish = module._publish
    target = tmp_path / "images"

    def race(stage, output):
        output.mkdir()
        (output / "foreign.txt").write_text("owner", encoding="utf-8")
        real_publish(stage, output)

    monkeypatch.setattr(module, "_publish", race)
    with pytest.raises(OutputError, match="publication did not acknowledge"):
        export_native_concat_scene_images(decoder.clips, target)
    assert (target / "foreign.txt").read_text(encoding="utf-8") == "owner"
    assert not list(tmp_path.glob(".frame-quorum-concat-scene-images-*"))


def test_lost_rename_ack_preserves_complete_target_and_reports_uncertainty(decoder, tmp_path, monkeypatch):
    real_publish = module._publish
    target = tmp_path / "images"

    def lost(stage, output):
        real_publish(stage, output)
        raise OSError("lost acknowledgement")

    monkeypatch.setattr(module, "_publish", lost)
    with pytest.raises(OutputError, match="did not acknowledge") as caught:
        export_native_concat_scene_images(decoder.clips, target)
    assert "inspect destination" in " ".join(caught.value.__notes__)
    assert _document(target)["image_count"] == 3
    assert len(list(target.iterdir())) == 4
    assert not list(tmp_path.glob(".frame-quorum-concat-scene-images-*"))
