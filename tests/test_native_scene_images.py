from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager, suppress
from dataclasses import FrozenInstanceError, replace
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

import frame_quorum.native_scene_images as module
from frame_quorum import (
    DetectionConfig,
    NativeSceneConfig,
    NativeSceneImageConfig,
    NativeSceneImageResult,
    NativeVideoConfig,
    NativeVideoDiagnostics,
    NativeVideoFrame,
    NativeVideoMetadata,
    NativeVideoStatus,
    export_native_scene_images,
)
from frame_quorum.errors import ConfigurationError, OutputError, ScanError


def test_public_core_api_exists():
    assert NativeSceneImageConfig().images_per_scene == 3
    assert NativeSceneImageResult.__name__ == "NativeSceneImageResult"
    assert callable(export_native_scene_images)


@pytest.mark.parametrize(
    "length,count,margin,expected",
    [
        (7, 3, 1, (1, 3, 5)),
        (4, 3, 1, (1, 1, 2)),
        (2, 3, 1, (0, 0, 1)),
        (1, 3, 100, (0, 0, 0)),
        (6, 1, 0, (2,)),
    ],
)
def test_independent_selection_vectors(length, count, margin, expected):
    assert module._positions(0, length, count, margin) == expected
    assert module._positions(23, 23 + length, count, margin) == tuple(23 + x for x in expected)


def test_exhaustive_small_selection_matches_independent_rational_oracle():
    for n in range(1, 24):
        for k in range(1, 12):
            for margin in range(12):
                candidates = list(range(n))
                for _ in range(margin):
                    if len(candidates) > 2:
                        candidates = candidates[1:-1]
                a, b = candidates[0], candidates[-1]
                expected = (
                    (int(Fraction(a + b, 2)),)
                    if k == 1
                    else tuple(int(Fraction((k - 1 - j) * a + j * b, k - 1)) for j in range(k))
                )
                assert module._positions(0, n, k, margin) == expected


_INTEGER_FIELDS = [
    name for name, field in NativeSceneImageConfig.__dataclass_fields__.items() if type(field.default) is int
]


@pytest.mark.parametrize("name", _INTEGER_FIELDS)
@pytest.mark.parametrize("value", [True, False, -1, 10**100, 0.5, "1", None])
def test_config_strict_integer_admission(name, value):
    with pytest.raises(ConfigurationError):
        NativeSceneImageConfig(**{name: value})


@pytest.mark.parametrize(
    "parameters",
    [
        {"image_format": "jpg"},
        {"image_format": False},
        {"image_format": []},
        {"interpolation": "area"},
        {"interpolation": []},
        {"interpolation": None},
        {"png_compression": 10},
        {"jpeg_quality": 101},
        {"width": True},
        {"height": 0},
        {"width": 4, "height": 4, "max_image_pixels": 15},
        {"scale": 0},
        {"scale": 0.5},
        {"scale": True},
        {"scale": Fraction(1, 2**63)},
        {"scale": 1, "width": 1},
        {"scale": 1, "height": 1},
        {"images_per_scene": 4, "max_images": 3},
    ],
)
def test_config_bad_combinations_and_option_domains(parameters):
    with pytest.raises(ConfigurationError):
        NativeSceneImageConfig(**parameters)


def test_config_exact_scale_and_immutable_json_projection():
    config = NativeSceneImageConfig(scale=2, sample_margin=0, png_compression=0, jpeg_quality=0)
    assert type(config.scale) is Fraction
    assert config.to_dict()["scale"] == {"numerator": 2, "denominator": 1}
    json.dumps(config.to_dict(), allow_nan=False)
    with pytest.raises(FrozenInstanceError):
        config.width = 10


@pytest.mark.parametrize("parameters", [{"scale": Fraction(1, 100)}, {"width": 1}, {"height": 1}])
def test_resize_rejects_zero_rounding(parameters):
    with pytest.raises(ConfigurationError, match="empty"):
        module._dimensions(
            100 if "width" in parameters else 1,
            100 if "height" in parameters else 1,
            NativeSceneImageConfig(**parameters),
        )


def test_resize_pixel_ceiling_checked_before_allocation():
    with pytest.raises(ConfigurationError, match="pixel"):
        module._dimensions(2, 2, NativeSceneImageConfig(scale=2, max_image_pixels=15))
    assert module._dimensions(2, 2, NativeSceneImageConfig(scale=2, max_image_pixels=16)) == (4, 4)


@pytest.mark.parametrize(
    "field,value",
    [
        ("output_dir", "bad"),
        ("scene_count", 0),
        ("image_count", 0),
        ("unique_sample_count", 0),
        ("total_output_bytes", True),
        ("manifest_sha256", "F" * 64),
        ("manifest_sha256", None),
        ("image_count", 5),
        ("image_count", 202),
    ],
)
def test_result_strict_validation(tmp_path, field, value):
    with pytest.raises(ConfigurationError):
        replace(NativeSceneImageResult(tmp_path, 2, 6, 2, 100, "a" * 64), **{field: value})


def sample(pts=0, *, index=0, width=2, height=2, value=0, time_base=Fraction(1, 1000)):
    return NativeVideoFrame(
        pts, time_base, index, index, 0, width, height, bytes([value]) * width * height * 3
    )


@pytest.fixture
def controlled(tmp_path, monkeypatch):
    """Two owned passes with real RGB metrics and independently controlled replay."""
    source = tmp_path / "source.nut"
    source.write_bytes(b"controlled source")
    instances = []

    def install(
        frames=None,
        *,
        replay_frames=None,
        status=NativeVideoStatus.EOF,
        replay_status=None,
        metadata_change=None,
        final_metadata_change=None,
        diagnostics_change=None,
        failure=None,
    ):
        original = list(frames) if frames is not None else [sample(i, index=i) for i in range(3)]

        class Stream:
            def __init__(self, path, config):
                self.pass_index = len(instances)
                self.frames = original if self.pass_index == 0 or replay_frames is None else replay_frames
                self.closed = False
                self.count = 0
                self.metadata = NativeVideoMetadata(
                    Path(path),
                    source.stat().st_size,
                    "nut",
                    "ffv1",
                    0,
                    original[0].width if original else 2,
                    original[0].height if original else 2,
                    Fraction(1, 1000),
                    original[0].pts if original else None,
                    None,
                    None,
                    None,
                )
                if self.pass_index == 1 and metadata_change is not None:
                    self.metadata = replace(self.metadata, **metadata_change)
                instances.append(self)

            def __enter__(self):
                return self

            def __exit__(self, *args):
                self.closed = True
                if self.pass_index == 1 and final_metadata_change is not None:
                    self.metadata = replace(self.metadata, **final_metadata_change)

            def __iter__(self):
                for frame in self.frames:
                    self.count += 1
                    yield frame
                if self.pass_index == 1 and failure is not None:
                    raise failure

            @property
            def diagnostics(self):
                decoded = self.frames[-1].decode_index + 1 if self.frames else 0
                diagnostic = NativeVideoDiagnostics(
                    status if self.pass_index == 0 or replay_status is None else replay_status,
                    decoded,
                    self.count,
                    sum(f.width * f.height for f in self.frames),
                    0,
                    self.closed,
                )
                if self.pass_index == 1 and diagnostics_change is not None:
                    diagnostic = replace(diagnostic, **diagnostics_change)
                return diagnostic

        monkeypatch.setattr(module, "NativeVideoStream", Stream)
        monkeypatch.setattr(
            module, "_load_av", lambda: SimpleNamespace(__version__="controlled", library_versions={})
        )
        return source, instances

    return install


def manifest(result):
    return json.loads((result.output_dir / "manifest.json").read_text(encoding="utf-8"))


def test_exact_extreme_negative_and_equal_pts_keep_position_identity(controlled, tmp_path):
    frames = [
        sample(-(2**63) + offset, index=index, value=index * 100, time_base=Fraction(2**63 - 1, 3))
        for index, offset in enumerate([0, 0, 1])
    ]
    source, streams = controlled(frames)
    config = NativeSceneConfig(detectors=(DetectionConfig(detector="luminance", threshold=1),))
    result = export_native_scene_images(
        source, tmp_path / "images", config, NativeSceneImageConfig(sample_margin=0)
    )
    document = manifest(result)
    assert [row["source_pts"] for row in document["images"]] == [frame.pts for frame in frames]
    assert [row["source_sample_index"] for row in document["images"]] == [0, 1, 2]
    assert len({row["source_rgb_sha256"] for row in document["images"]}) == 3
    expected = frames[0].presentation_time
    assert document["images"][0]["source_time"] == {
        "numerator": expected.numerator,
        "denominator": expected.denominator,
    }
    assert document["scenes"][-1]["end_time"] is None
    assert all(stream.closed for stream in streams)
    assert result.to_dict()["output_dir"] == result.output_dir.as_posix()


@pytest.mark.parametrize(
    "change",
    [
        {"pts": 8},
        {"time_base": Fraction(2, 1000)},
        {"decode_index": 2},
        {"sample_index": 2},
        {"generation": 1},
        {"width": 1, "height": 4},
        {"rgb": bytes([1]) * 12},
    ],
)
def test_every_replay_identity_component_is_checked_even_for_unselected_samples(controlled, tmp_path, change):
    original = [sample(i, index=i) for i in range(3)]
    changed = [replace(original[0], **change), *original[1:]]
    source, streams = controlled(original, replay_frames=changed)
    with pytest.raises(ScanError, match="replay sample"):
        export_native_scene_images(source, tmp_path / "images")  # only position 1 is selected by default
    assert all(stream.closed for stream in streams)
    assert list(tmp_path.iterdir()) == [source]


@pytest.mark.parametrize(
    "options",
    [
        {"replay_frames": [sample()]},
        {"replay_frames": [sample(i, index=i) for i in range(4)]},
        {"replay_frames": [sample(i, index=i) for i in (1, 0, 2)]},
        {"replay_status": NativeVideoStatus.CLOSED},
        {"diagnostics_change": {"decoded_pixels_observed": 13}},
        {"diagnostics_change": {"closed": False}},
        {"metadata_change": {"codec_name": "other"}},
        {"final_metadata_change": {"codec_name": "other"}},
    ],
)
def test_replay_completion_and_metadata_mismatch_never_publish(controlled, tmp_path, options):
    source, _ = controlled(**options)
    with pytest.raises(ScanError, match="replay"):
        export_native_scene_images(source, tmp_path / "images")
    assert list(tmp_path.iterdir()) == [source]


def test_unknown_tail_can_have_different_native_time_bases(controlled, tmp_path):
    frames = [sample(-2), sample(-1, index=1, time_base=Fraction(1, 2000)), sample(0, index=2)]
    source, _ = controlled(frames)
    result = export_native_scene_images(
        source, tmp_path / "images", image_config=NativeSceneImageConfig(sample_margin=0)
    )
    assert [row["source_time_base"] for row in manifest(result)["images"]] == [
        {"numerator": 1, "denominator": denominator} for denominator in (1000, 2000, 1000)
    ]


@pytest.mark.parametrize(
    "field,exact,lower",
    [
        ("max_total_pixels", 12, 11),
        ("max_verification_pixels", 12, 11),
        ("max_image_pixels", 4, 3),
    ],
)
def test_pixel_boundaries_are_exact_and_preflight_before_replay(controlled, tmp_path, field, exact, lower):
    source, streams = controlled()
    export_native_scene_images(
        source, tmp_path / "good", image_config=NativeSceneImageConfig(**{field: exact})
    )
    assert len(streams) == 2
    source, streams = controlled()
    streams.clear()
    with pytest.raises(ConfigurationError, match="pixel"):
        export_native_scene_images(
            source, tmp_path / "bad", image_config=NativeSceneImageConfig(**{field: lower})
        )
    assert len(streams) == 1
    assert not (tmp_path / "bad").exists()


def test_no_samples_and_scene_count_caps_before_staging(controlled, tmp_path):
    source, streams = controlled([])
    with pytest.raises(ScanError, match="observed sample"):
        export_native_scene_images(source, tmp_path / "empty")
    streams.clear()
    source, _ = controlled([sample(i, index=i, value=value) for i, value in enumerate([0, 255, 0])])
    config = NativeSceneConfig(detectors=(DetectionConfig(detector="luminance", threshold=0.5),))
    for image_config, text in [
        (NativeSceneImageConfig(max_scenes=2), "scene count"),
        (NativeSceneImageConfig(max_images=8), "image count"),
    ]:
        streams.clear()
        with pytest.raises(ConfigurationError, match=text):
            export_native_scene_images(source, tmp_path / "limited", config, image_config)
    assert list(tmp_path.iterdir()) == [source]


@pytest.mark.parametrize("field", ["max_image_bytes", "max_manifest_bytes", "max_output_bytes"])
def test_byte_limits_fail_without_output(controlled, tmp_path, field):
    source, streams = controlled()
    with pytest.raises(OutputError, match="byte limit"):
        export_native_scene_images(
            source, tmp_path / "images", image_config=NativeSceneImageConfig(**{field: 1})
        )
    assert all(stream.closed for stream in streams)
    assert list(tmp_path.iterdir()) == [source]


def test_image_encoded_byte_boundary_exact_and_one_over(controlled, tmp_path):
    source, streams = controlled()
    result = export_native_scene_images(source, tmp_path / "baseline")
    maximum = max(row["bytes"] for row in manifest(result)["images"])
    streams.clear()
    export_native_scene_images(
        source, tmp_path / "exact", image_config=NativeSceneImageConfig(max_image_bytes=maximum)
    )
    streams.clear()
    with pytest.raises(OutputError, match="byte limit"):
        export_native_scene_images(
            source, tmp_path / "over", image_config=NativeSceneImageConfig(max_image_bytes=maximum - 1)
        )
    assert not (tmp_path / "over").exists()


@pytest.mark.parametrize("hook", ["_save_slot", "_verify_image"])
@pytest.mark.parametrize("error", [OSError("failure"), KeyboardInterrupt("stop"), SystemExit(9)])
def test_output_failures_and_control_cleanup(controlled, tmp_path, monkeypatch, hook, error):
    source, streams = controlled()

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(module, hook, fail)
    with pytest.raises(OutputError if isinstance(error, Exception) else type(error)):
        export_native_scene_images(source, tmp_path / "images")
    assert all(stream.closed for stream in streams)
    assert list(tmp_path.iterdir()) == [source]


def test_ordinary_and_control_replay_failure_cleanup(controlled, tmp_path):
    for index, error in enumerate((OSError("replay"), KeyboardInterrupt("replay"))):
        source, streams = controlled(failure=error)
        streams.clear()
        with pytest.raises(OutputError if isinstance(error, Exception) else KeyboardInterrupt):
            export_native_scene_images(source, tmp_path / f"images-{index}")
        assert all(stream.closed for stream in streams)
        assert list(tmp_path.iterdir()) == [source]


def test_source_replacement_between_passes_rejected(controlled, tmp_path, monkeypatch):
    source, streams = controlled()
    capture = module._capture

    def replace_source(*args):
        result = capture(*args)
        source.rename(tmp_path / "old-source.nut")
        source.write_bytes(b"controlled source")
        return result

    monkeypatch.setattr(module, "_capture", replace_source)
    with pytest.raises(ScanError, match="changed"):
        export_native_scene_images(source, tmp_path / "images")
    assert len(streams) == 1 and streams[0].closed
    assert not (tmp_path / "images").exists()


def test_close_failure_cleans_owned_stage(controlled, tmp_path, monkeypatch):
    source, _ = controlled()
    close_all = module._close_all

    def close_then_fail(resources, primary=None):
        close_all(resources, primary)
        raise OSError("close acknowledgment lost")

    monkeypatch.setattr(module, "_close_all", close_then_fail)
    with pytest.raises(OutputError):
        export_native_scene_images(source, tmp_path / "images")
    assert list(tmp_path.iterdir()) == [source]


def test_existing_output_not_mutated_before_decoder_import(tmp_path, monkeypatch):
    target = tmp_path / "existing"
    target.mkdir()
    keep = target / "keep.txt"
    keep.write_text("keep")
    monkeypatch.setattr(module, "_load_av", lambda: pytest.fail("no native import"))
    with pytest.raises(OutputError, match="new output"):
        export_native_scene_images("missing", target)
    assert keep.read_text() == "keep"


def test_publication_failure_preserves_foreign_target(controlled, tmp_path, monkeypatch):
    source, _ = controlled()
    publish = module._publish

    def collide(stage, target):
        target.mkdir()
        (target / "foreign").write_bytes(b"keep")
        publish(stage, target)

    monkeypatch.setattr(module, "_publish", collide)
    with pytest.raises(OutputError, match="acknowledge"):
        export_native_scene_images(source, tmp_path / "images")
    assert (tmp_path / "images" / "foreign").read_bytes() == b"keep"
    assert not list(tmp_path.glob(".frame-quorum-scene-images-*"))


def test_lost_publication_ack_keeps_published_bundle(controlled, tmp_path, monkeypatch):
    source, _ = controlled()
    publish = module._publish

    def lose_ack(stage, target):
        publish(stage, target)
        raise OSError("lost acknowledgment")

    monkeypatch.setattr(module, "_publish", lose_ack)
    with pytest.raises(OutputError, match="acknowledge") as caught:
        export_native_scene_images(source, tmp_path / "images")
    assert "never removed" in " ".join(caught.value.__cause__.__notes__)
    assert len(list((tmp_path / "images").glob("*.png"))) == 3
    assert (tmp_path / "images" / "manifest.json").is_file()


def test_cleanup_never_removes_unknown_stage_entry(controlled, tmp_path, monkeypatch):
    source, _ = controlled()

    def fail_with_foreign(frame, original, slot, stage, *args):
        (stage / "foreign").write_bytes(b"keep")
        raise OSError("failed")

    monkeypatch.setattr(module, "_save_slot", fail_with_foreign)
    with pytest.raises(OutputError, match="cleanup incomplete"):
        export_native_scene_images(source, tmp_path / "images")
    (stage,) = tmp_path.glob(".frame-quorum-scene-images-*")
    assert (stage / "foreign").read_bytes() == b"keep"


def test_manifest_counts_and_file_hashes_are_independent(controlled, tmp_path):
    source, _ = controlled()
    result = export_native_scene_images(source, tmp_path / "images")
    document = manifest(result)
    assert result.image_count == 3 and result.unique_sample_count == 1
    assert document["verified_observed_sample_count"] == 3
    assert result.total_output_bytes == sum(path.stat().st_size for path in result.output_dir.iterdir())
    assert (
        result.manifest_sha256
        == hashlib.sha256((result.output_dir / "manifest.json").read_bytes()).hexdigest()
    )
    for row in document["images"]:
        path = result.output_dir / row["file"]
        assert path.stat().st_size == row["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]
        with Image.open(path) as image:
            assert hashlib.sha256(image.tobytes()).hexdigest() == row["source_rgb_sha256"]


def test_reencoded_same_rgb_after_hash_must_not_publish_stale_provenance(controlled, tmp_path, monkeypatch):
    source, _ = controlled()
    verify = module._verify_image

    def replace_encoding(path, config, size, digest):
        with Image.open(path) as decoded:
            image = decoded.copy()
        try:
            image.save(path, format="PNG", compress_level=0)
        finally:
            image.close()
        return verify(path, config, size, digest)

    monkeypatch.setattr(module, "_verify_image", replace_encoding)
    with pytest.raises(OutputError, match="staged"):
        export_native_scene_images(source, tmp_path / "images")
    assert list(tmp_path.iterdir()) == [source]


@pytest.mark.parametrize("field", ["max_manifest_bytes", "max_output_bytes"])
def test_manifest_and_aggregate_byte_boundaries_are_exact(controlled, tmp_path, field):
    source, streams = controlled()
    baseline = export_native_scene_images(source, tmp_path / "baseline")
    document = manifest(baseline)
    image_bytes = sum(row["bytes"] for row in document["images"]) if field == "max_output_bytes" else 0
    limit = 1
    for _ in range(10):
        document["image_config"][field] = limit
        expected = image_bytes + len(
            json.dumps(document, ensure_ascii=True, allow_nan=False, sort_keys=True).encode()
        )
        if expected == limit:
            break
        limit = expected
    else:
        pytest.fail("independent manifest length did not converge")
    streams.clear()
    exact = export_native_scene_images(
        source, tmp_path / "exact", image_config=NativeSceneImageConfig(**{field: limit})
    )
    actual = (
        exact.total_output_bytes
        if field == "max_output_bytes"
        else (exact.output_dir / "manifest.json").stat().st_size
    )
    assert actual == limit
    streams.clear()
    with pytest.raises(OutputError, match="byte limit"):
        export_native_scene_images(
            source, tmp_path / "over", image_config=NativeSceneImageConfig(**{field: limit - 1})
        )
    assert not (tmp_path / "over").exists()


def test_scene_image_count_boundary_exact(controlled, tmp_path):
    source, _ = controlled([sample(i, index=i, value=value) for i, value in enumerate([0, 255, 0])])
    config = NativeSceneConfig(detectors=(DetectionConfig(detector="luminance", threshold=0.5),))
    result = export_native_scene_images(
        source, tmp_path / "exact", config, NativeSceneImageConfig(max_scenes=3, max_images=9)
    )
    assert result.scene_count == 3 and result.image_count == 9 and result.unique_sample_count == 3


def test_maximum_slot_count_for_one_scene_and_margin_clamping(controlled, tmp_path):
    source, _ = controlled([sample()])
    result = export_native_scene_images(
        source,
        tmp_path / "images",
        image_config=NativeSceneImageConfig(images_per_scene=100, sample_margin=1_000_000),
    )
    assert result.image_count == 100 and result.unique_sample_count == 1
    assert {row["source_sample_index"] for row in manifest(result)["images"]} == {0}


@pytest.mark.parametrize(
    "field,maximum",
    [
        ("images_per_scene", 100),
        ("sample_margin", 1_000_000),
        ("max_scenes", 1000),
        ("max_images", 10_000),
        ("max_image_pixels", 16_777_216),
        ("max_total_pixels", 1_000_000_000),
        ("max_verification_pixels", 1_000_000_000),
        ("max_image_bytes", 64 * 1024 * 1024),
        ("max_output_bytes", 1_000_000_000),
        ("max_manifest_bytes", 16 * 1024 * 1024),
    ],
)
def test_compiled_count_and_resource_ceilings(field, maximum):
    assert getattr(NativeSceneImageConfig(**{field: maximum}), field) == maximum
    with pytest.raises(ConfigurationError):
        NativeSceneImageConfig(**{field: maximum + 1})


@pytest.mark.parametrize(
    "scene_config,image_config", [(False, None), (None, False), (object(), None), (None, object())]
)
def test_wrong_configs_fail_without_source_or_staging(tmp_path, scene_config, image_config):
    with pytest.raises(ConfigurationError, match="configs"):
        export_native_scene_images("missing", tmp_path / "images", scene_config, image_config)
    assert list(tmp_path.iterdir()) == []


def test_mutated_nested_configuration_revalidated_before_native_import(tmp_path, monkeypatch):
    config = NativeSceneConfig(video=NativeVideoConfig())
    object.__setattr__(config.video, "frame_step", 0)
    monkeypatch.setattr(module, "_load_av", lambda: pytest.fail("native must not load"))
    with pytest.raises(ConfigurationError, match="frame_step"):
        export_native_scene_images("missing", tmp_path / "images", config)


@pytest.mark.parametrize("error", [OSError("encode"), KeyboardInterrupt("encode")])
def test_encoder_partial_write_failure_removes_owned_file(controlled, tmp_path, monkeypatch, error):
    source, streams = controlled()

    def fail(image, fp, *args, **kwargs):
        fp.write(b"partial encoded bytes")
        raise error

    monkeypatch.setattr(Image.Image, "save", fail)
    with pytest.raises(OutputError if isinstance(error, Exception) else KeyboardInterrupt):
        export_native_scene_images(source, tmp_path / "images")
    assert all(stream.closed for stream in streams)
    assert list(tmp_path.iterdir()) == [source]


def test_encoder_cannot_hide_bounded_writer_failure(controlled, tmp_path, monkeypatch):
    source, _ = controlled()

    def swallow(image, fp, *args, **kwargs):
        with suppress(OutputError):
            fp.write(b"too large")

    monkeypatch.setattr(Image.Image, "save", swallow)
    with pytest.raises(OutputError, match="swallowed"):
        export_native_scene_images(
            source, tmp_path / "images", image_config=NativeSceneImageConfig(max_image_bytes=1)
        )
    assert list(tmp_path.iterdir()) == [source]


@pytest.mark.parametrize(
    "mode,size,image_format", [("RGBA", (2, 2), "PNG"), ("RGB", (3, 2), "PNG"), ("RGB", (2, 2), "JPEG")]
)
def test_verification_rejects_wrong_shape_mode_and_codec(tmp_path, mode, size, image_format):
    path = tmp_path / "image"
    with Image.new(mode, size) as image:
        image.save(path, format=image_format)
    with pytest.raises(OutputError, match="verification"):
        module._verify_image(path, NativeSceneImageConfig(), (2, 2), "a" * 64)


def test_png_verification_rejects_valid_image_with_wrong_pixels(tmp_path):
    path = tmp_path / "image.png"
    with Image.new("RGB", (2, 2), (255, 0, 0)) as image:
        image.save(path)
    with pytest.raises(OutputError, match="PNG RGB"):
        module._verify_image(path, NativeSceneImageConfig(), (2, 2), hashlib.sha256(bytes(12)).hexdigest())


@pytest.mark.parametrize("replacement", ["file", "stage", "extra"])
def test_prepublication_reconciliation_preserves_foreign_replacements(
    controlled, tmp_path, monkeypatch, replacement
):
    source, _ = controlled()
    reconcile = module._reconcile
    foreign = None

    def alter(stage, identity, owned, expected, budget):
        nonlocal foreign
        if replacement == "stage":
            stage.rename(tmp_path / "moved-owned-stage")
            stage.mkdir()
            foreign = stage / "foreign"
            foreign.write_bytes(b"keep")
        elif replacement == "file":
            foreign = next(iter(owned))
            data = foreign.read_bytes()
            foreign.rename(tmp_path / "moved-owned-file")
            foreign.write_bytes(data)
        else:
            foreign = stage / "foreign"
            foreign.write_bytes(b"keep")
        reconcile(stage, identity, owned, expected, budget)

    monkeypatch.setattr(module, "_reconcile", alter)
    with pytest.raises(OutputError, match="cleanup incomplete"):
        export_native_scene_images(source, tmp_path / "images")
    assert foreign.is_file()
    assert not (tmp_path / "images").exists()


def test_missing_owned_file_is_rejected_by_reconciliation(controlled, tmp_path, monkeypatch):
    source, _ = controlled()
    reconcile = module._reconcile

    def remove(stage, identity, owned, expected, budget):
        next(iter(owned)).unlink()
        reconcile(stage, identity, owned, expected, budget)

    monkeypatch.setattr(module, "_reconcile", remove)
    with pytest.raises(OutputError, match="missing"):
        export_native_scene_images(source, tmp_path / "images")
    assert list(tmp_path.iterdir()) == [source]


def test_stage_identity_observation_failure_does_not_delete_unknown_stage(controlled, tmp_path, monkeypatch):
    source, _ = controlled()

    def unknown(info):
        raise OSError("identity unavailable")

    monkeypatch.setattr(module, "_identity", unknown)
    with pytest.raises(OutputError) as caught:
        export_native_scene_images(source, tmp_path / "images")
    assert "Staging identity unavailable" in " ".join(caught.value.__cause__.__notes__)
    assert len(list(tmp_path.glob(".frame-quorum-scene-images-*"))) == 1


def test_nonregular_source_refused_before_decoder_import(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "_load_av", lambda: pytest.fail("no native import"))
    with pytest.raises(ScanError, match="regular"):
        export_native_scene_images(tmp_path, tmp_path / "images")


def test_staged_file_open_identity_is_checked(tmp_path, monkeypatch):
    path = tmp_path / "owned"
    path.write_bytes(b"abc")
    identity = module._identity(path.stat())
    monkeypatch.setattr(module.os, "fstat", lambda fd: SimpleNamespace(st_dev=-1, st_ino=-1, st_size=3))
    with pytest.raises(OutputError, match="opening"):
        module._staged_digest(path, identity, 3)


@pytest.mark.parametrize("mutation", ["grow", "short"])
def test_staged_file_change_during_reconciliation(tmp_path, monkeypatch, mutation):
    path = tmp_path / "owned"
    path.write_bytes(b"abc")
    identity = module._identity(path.stat())
    managed = module._managed_file

    @contextmanager
    def altered(value):
        with managed(value) as handle:

            class Reader:
                def fileno(self):
                    return handle.fileno()

                def read(self, size):
                    return b"abcd" if mutation == "grow" else b""

            yield Reader()

    monkeypatch.setattr(module, "_managed_file", altered)
    with pytest.raises(OutputError, match=r"grew|changed"):
        module._staged_digest(path, identity, 3)


@pytest.mark.parametrize("mutation", ["inventory", "hash", "budget", "aggregate"])
def test_reconcile_rejects_unbalanced_inventory_hashes_and_budgets(tmp_path, mutation):
    stage = tmp_path / "stage"
    stage.mkdir()
    path = stage / "owned"
    path.write_bytes(b"abc")
    owned = {path: module._identity(path.stat())}
    expected = {path: (3, hashlib.sha256(b"abc").hexdigest())}
    budget = module._ByteBudget(3)
    budget.grow(3)
    if mutation == "inventory":
        expected.clear()
    elif mutation == "hash":
        expected[path] = (3, "a" * 64)
    elif mutation == "budget":
        budget.maximum = 2
    else:
        budget.total = 2
    with pytest.raises(OutputError, match="staged"):
        module._reconcile(stage, module._identity(stage.stat()), owned, expected, budget)


def test_source_change_during_final_reconciliation_refuses_publication(controlled, tmp_path, monkeypatch):
    source, _ = controlled()
    reconcile = module._reconcile

    def mutate(*args):
        reconcile(*args)
        source.write_bytes(b"changed")

    monkeypatch.setattr(module, "_reconcile", mutate)
    with pytest.raises(ScanError, match="changed"):
        export_native_scene_images(source, tmp_path / "images")
    assert list(tmp_path.iterdir()) == [source]
