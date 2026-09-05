from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

import frame_quorum.scanner as scanner_module
from frame_quorum.cli import main
from frame_quorum.contact_sheet import render_contact_sheet
from frame_quorum.errors import ConfigurationError, ScanError
from frame_quorum.models import AnimationConfig, ScanConfig, SelectionConfig
from frame_quorum.reporting import scan_manifest
from frame_quorum.scanner import scan_frames
from frame_quorum.selector import select_frames

_GIF_ONLY = ScanConfig(extensions=(".gif",))


def _animation(path: Path, count: int = 5, *, size: tuple[int, int] = (64, 48)) -> Path:
    """Write an animated container whose internal frames differ visibly."""

    internal = []
    for position in range(count):
        image = Image.new("RGB", size, ((position * 47) % 256, 40, (200 - position * 30) % 256))
        draw = ImageDraw.Draw(image)
        offset = position * 7
        draw.rectangle((offset, 6, offset + 18, 26), fill=(240, 210, 30))
        internal.append(image)
    path.parent.mkdir(parents=True, exist_ok=True)
    internal[0].save(path, save_all=True, append_images=internal[1:], duration=120, loop=0)
    return path


def test_animated_file_stays_one_frame_without_opt_in(tmp_path: Path) -> None:
    _animation(tmp_path / "clip.gif")
    frames = scan_frames(tmp_path, _GIF_ONLY)
    assert len(frames) == 1
    assert frames[0].relative_path == "clip.gif"
    assert frames[0].source_frame_index is None
    assert "source_frame_index" not in frames[0].serializable()


@pytest.mark.parametrize("name", ["clip.gif", "clip.png", "clip.webp"])
def test_expansion_traces_every_internal_frame_back_to_its_container(tmp_path: Path, name: str) -> None:
    _animation(tmp_path / name, 5)
    config = ScanConfig(extensions=(Path(name).suffix,))
    frames = scan_frames(tmp_path, config, animation=AnimationConfig())
    assert [frame.index for frame in frames] == [0, 1, 2, 3, 4]
    assert [frame.source_frame_index for frame in frames] == [0, 1, 2, 3, 4]
    assert [frame.relative_path for frame in frames] == [f"{name}#frame={index}" for index in range(5)]
    assert len({frame.path for frame in frames}) == 1
    assert {frame.path.name for frame in frames} == {name}
    assert len({frame.byte_size for frame in frames}) == 1
    assert frames[3].serializable()["source_frame_index"] == 3


def test_expanded_frames_are_measured_and_timed_independently(tmp_path: Path) -> None:
    _animation(tmp_path / "clip.gif", 4)
    frames = scan_frames(tmp_path, replace(_GIF_ONLY, frame_rate=2.0), animation=AnimationConfig())
    assert len({frame.metrics.perceptual_hash for frame in frames}) > 1
    assert [frame.timestamp for frame in frames] == [0.0, 0.5, 1.0, 1.5]


def test_expanded_and_plain_files_share_one_contiguous_index_sequence(
    image_factory: Callable[..., Path], tmp_path: Path
) -> None:
    image_factory("a_still.png")
    _animation(tmp_path / "b_clip.gif", 3)
    image_factory("c_still.png")
    config = ScanConfig(extensions=(".png", ".gif"))
    frames = scan_frames(tmp_path, config, animation=AnimationConfig())
    assert [frame.index for frame in frames] == [0, 1, 2, 3, 4]
    assert [frame.relative_path for frame in frames] == [
        "a_still.png",
        "b_clip.gif#frame=0",
        "b_clip.gif#frame=1",
        "b_clip.gif#frame=2",
        "c_still.png",
    ]
    assert [frame.source_frame_index for frame in frames] == [None, 0, 1, 2, None]


def test_still_image_keeps_its_plain_identity_when_expansion_is_enabled(
    image_factory: Callable[..., Path], tmp_path: Path
) -> None:
    image_factory("still.png")
    frames = scan_frames(tmp_path, animation=AnimationConfig())
    assert len(frames) == 1
    assert frames[0].relative_path == "still.png"
    assert frames[0].source_frame_index is None


def test_frame_cap_refuses_instead_of_truncating(tmp_path: Path) -> None:
    _animation(tmp_path / "clip.gif", 6)
    with pytest.raises(ScanError, match="holds 6 frames, above the 3-frame expansion limit"):
        scan_frames(tmp_path, _GIF_ONLY, animation=AnimationConfig(max_frames=3))


def test_decoded_byte_cap_refuses_before_any_internal_frame_is_read(tmp_path: Path) -> None:
    _animation(tmp_path / "clip.gif", 5)
    with pytest.raises(ScanError, match="would decode 46080 bytes, above the 1000-byte"):
        scan_frames(tmp_path, _GIF_ONLY, animation=AnimationConfig(max_decoded_bytes=1000))


def test_expansion_accepts_a_container_exactly_at_both_limits(tmp_path: Path) -> None:
    _animation(tmp_path / "clip.gif", 5)
    animation = AnimationConfig(max_frames=5, max_decoded_bytes=5 * 64 * 48 * 3)
    assert len(scan_frames(tmp_path, _GIF_ONLY, animation=animation)) == 5


def test_container_replaced_between_pixel_and_metadata_reads_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _animation(tmp_path / "clip.gif", 3)
    original = scanner_module.timestamp_for

    def replace_source(source: Path, index: int, config: ScanConfig) -> float | None:
        if index == 0:
            _animation(source, 3, size=(48, 32))
        return original(source, index, config)

    monkeypatch.setattr(scanner_module, "timestamp_for", replace_source)
    with pytest.raises(ScanError, match="changed while it was being scanned"):
        scan_frames(tmp_path, _GIF_ONLY, animation=AnimationConfig())


def test_container_removed_between_discovery_and_scanning_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _animation(tmp_path / "clip.gif", 3)
    missing = tmp_path / "clip.gif"

    def discover_then_remove(input_path: Path, config: ScanConfig) -> tuple[Path, ...]:
        missing.unlink()
        return (missing,)

    monkeypatch.setattr(scanner_module, "discover_images", discover_then_remove)
    with pytest.raises(ScanError, match="cannot inspect image"):
        scan_frames(tmp_path, _GIF_ONLY, animation=AnimationConfig())


def test_contact_sheet_renders_the_selected_internal_frames(tmp_path: Path) -> None:
    _animation(tmp_path / "clip.gif", 6)
    frames = scan_frames(tmp_path, _GIF_ONLY, animation=AnimationConfig())
    result = select_frames(frames, SelectionConfig(budget=3, duplicate_threshold=0.0))
    sheet = render_contact_sheet(result, tmp_path / "report" / "sheet.png")
    assert sheet.is_file()
    assert [frame.source_frame_index for frame in result.selected_frames] == [0, 3, 5]


def test_contact_sheet_rejects_an_animation_replaced_after_scanning(tmp_path: Path) -> None:
    path = _animation(tmp_path / "clip.gif", 4)
    frames = scan_frames(tmp_path, _GIF_ONLY, animation=AnimationConfig())
    result = select_frames(frames, SelectionConfig(budget=2))
    _animation(path, 4, size=(48, 32))
    with pytest.raises(ScanError, match="changed since scan"):
        render_contact_sheet(result, tmp_path / "report" / "sheet.png")


def test_scan_manifest_records_expansion_limits_only_when_requested(tmp_path: Path) -> None:
    _animation(tmp_path / "clip.gif", 3)
    animation = AnimationConfig(max_frames=8, max_decoded_bytes=2_000_000)
    frames = scan_frames(tmp_path, _GIF_ONLY, animation=animation)
    manifest = scan_manifest(frames, _GIF_ONLY, animation=animation)
    assert manifest["animation_config"] == {"max_frames": 8, "max_decoded_bytes": 2_000_000}
    assert manifest["frames"][2]["source_frame_index"] == 2
    assert "animation_config" not in scan_manifest(frames, _GIF_ONLY)


def test_manifest_rejects_a_wrong_animation_config_type(tmp_path: Path) -> None:
    _animation(tmp_path / "clip.gif", 2)
    frames = scan_frames(tmp_path, _GIF_ONLY)
    with pytest.raises(ConfigurationError, match="must be AnimationConfig or None"):
        scan_manifest(frames, _GIF_ONLY, animation=object())  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "animation",
    [
        AnimationConfig(max_frames=0),
        AnimationConfig(max_frames=True),
        AnimationConfig(max_frames=100_001),
        AnimationConfig(max_decoded_bytes=0),
        AnimationConfig(max_decoded_bytes=(1 << 40) + 1),
        AnimationConfig(max_decoded_bytes=10**400),
    ],
)
def test_invalid_animation_config(animation: AnimationConfig, tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError):
        scan_frames(tmp_path, animation=animation)


def test_scan_rejects_a_wrong_animation_config_type(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="must be AnimationConfig"):
        scan_frames(tmp_path, animation=object())  # type: ignore[arg-type]


@pytest.mark.parametrize("source_frame_index", [-1, True, 1 << 63])
def test_frame_rejects_an_invalid_source_frame_index(
    image_factory: Callable[..., Path], source_frame_index: int
) -> None:
    frame = scan_frames(image_factory("one.png"))[0]
    with pytest.raises(ConfigurationError, match="source frame index"):
        replace(frame, source_frame_index=source_frame_index).validate()


def test_cli_scan_expands_animations_and_records_the_limits(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _animation(tmp_path / "clip.gif", 4)
    assert main(["scan", str(tmp_path), "--extensions", "gif", "--expand-animations"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["summary"]["frame_count"] == 4
    assert payload["animation_config"] == {"max_frames": 64, "max_decoded_bytes": 268_435_456}
    assert payload["frames"][1]["path"] == "clip.gif#frame=1"
    assert payload["scan_config"]["extensions"] == ["gif"]


def test_cli_select_expands_animations(tmp_path: Path) -> None:
    _animation(tmp_path / "clip.gif", 6, size=(96, 64))
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    (tmp_path / "clip.gif").replace(frames_dir / "clip.gif")
    status = main(
        [
            "select",
            str(frames_dir),
            "--output-dir",
            str(tmp_path / "selection"),
            "--extensions",
            ".gif",
            "--expand-animations",
            "--budget",
            "3",
        ]
    )
    assert status == 0
    manifest = json.loads((tmp_path / "selection" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["summary"]["frame_count"] == 6
    assert manifest["animation_config"]["max_frames"] == 64
    assert manifest["selected"][0]["path"].startswith("clip.gif#frame=")


def test_cli_reports_an_exceeded_expansion_limit(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _animation(tmp_path / "clip.gif", 5)
    status = main(
        [
            "scan",
            str(tmp_path),
            "--extensions",
            ".gif",
            "--expand-animations",
            "--max-animation-frames",
            "2",
        ]
    )
    assert status == 2
    assert "expansion limit" in capsys.readouterr().err
