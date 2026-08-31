from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image

from frame_quorum.contact_sheet import render_contact_sheet
from frame_quorum.errors import ConfigurationError, ScanError
from frame_quorum.models import Frame, ScanConfig, SelectionConfig, SelectionResult
from frame_quorum.reporting import scan_manifest, selection_manifest, write_json
from frame_quorum.scanner import scan_frames
from frame_quorum.selector import select_frames


def _result(image_factory: Callable[..., Path], tmp_path: Path) -> tuple[tuple[Frame, ...], SelectionResult]:
    for index in range(5):
        image_factory(f"frame_{index}.png", pattern=index + 1)
    frames = scan_frames(tmp_path)
    return frames, select_frames(frames, SelectionConfig(budget=3, duplicate_threshold=0))


def test_scan_manifest_has_schema_and_summary(image_factory: Callable[..., Path], tmp_path: Path) -> None:
    image_factory("one.png")
    data = scan_manifest(scan_frames(tmp_path), ScanConfig())
    assert data["schema_version"] == "1.0"
    assert data["summary"]["frame_count"] == 1


def test_hash_is_serialized_as_hex(image_factory: Callable[..., Path], tmp_path: Path) -> None:
    image_factory("one.png")
    data = scan_manifest(scan_frames(tmp_path), ScanConfig())
    assert len(data["frames"][0]["metrics"]["perceptual_hash"]) == 16


def test_selection_manifest_contains_all_decisions(
    image_factory: Callable[..., Path], tmp_path: Path
) -> None:
    frames, result = _result(image_factory, tmp_path)
    data = selection_manifest(result, ScanConfig())
    assert len(data["frames"]) == len(frames)
    assert all("decision" in frame for frame in data["frames"])
    assert len(data["selected"]) == 3


def test_write_json_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "nested" / "report.json"
    rendered = write_json({"hello": "世界"}, output)
    assert json.loads(output.read_text(encoding="utf-8")) == {"hello": "世界"}
    assert rendered.endswith("\n")


def test_scan_manifest_rejects_unsafe_relative_path(
    image_factory: Callable[..., Path], tmp_path: Path
) -> None:
    frame = scan_frames(image_factory("one.png"))[0]
    with pytest.raises(ConfigurationError, match="Unicode scalar"):
        scan_manifest((replace(frame, relative_path="bad\ud800.png"),), ScanConfig())


def test_contact_sheet_is_real_png(image_factory: Callable[..., Path], tmp_path: Path) -> None:
    _, result = _result(image_factory, tmp_path)
    output = tmp_path / "sheet.png"
    render_contact_sheet(result, output, columns=2, thumbnail_width=160)
    with Image.open(output) as image:
        assert image.format == "PNG"
        assert image.width > 320
        assert image.height > 200


def test_contact_sheet_wraps_source_changed_after_scan(
    image_factory: Callable[..., Path], tmp_path: Path
) -> None:
    _, result = _result(image_factory, tmp_path)
    result.selected_frames[0].path.write_bytes(b"not an image anymore")
    with pytest.raises(ScanError, match="cannot decode selected image"):
        render_contact_sheet(result, tmp_path / "sheet.png")


def test_contact_sheet_rejects_valid_image_substitution(
    image_factory: Callable[..., Path], tmp_path: Path
) -> None:
    _, result = _result(image_factory, tmp_path)
    selected = result.selected_frames[0]
    image_factory(selected.path.name, color=(240, 10, 80), pattern=9, directory=selected.path.parent)
    with pytest.raises(ScanError, match="changed since scan"):
        render_contact_sheet(result, tmp_path / "sheet.png")


def test_contact_sheet_rejects_bad_columns(image_factory: Callable[..., Path], tmp_path: Path) -> None:
    _, result = _result(image_factory, tmp_path)
    with pytest.raises(ConfigurationError, match="columns"):
        render_contact_sheet(result, tmp_path / "bad.png", columns=0)


def test_contact_sheet_rejects_tiny_thumbnails(image_factory: Callable[..., Path], tmp_path: Path) -> None:
    _, result = _result(image_factory, tmp_path)
    with pytest.raises(ConfigurationError, match="thumbnail"):
        render_contact_sheet(result, tmp_path / "bad.png", thumbnail_width=20)


def test_contact_sheet_rejects_excessive_canvas(image_factory: Callable[..., Path], tmp_path: Path) -> None:
    _, result = _result(image_factory, tmp_path)
    with pytest.raises(ConfigurationError, match="canvas limit"):
        render_contact_sheet(result, tmp_path / "huge.png", thumbnail_width=100_000)


def test_contact_sheet_handles_unbounded_thumbnail_integer(
    image_factory: Callable[..., Path], tmp_path: Path
) -> None:
    _, result = _result(image_factory, tmp_path)
    with pytest.raises(ConfigurationError, match="canvas limit"):
        render_contact_sheet(result, tmp_path / "huge.png", thumbnail_width=10**400)


def test_empty_result_cannot_render(tmp_path: Path) -> None:
    result = select_frames([])
    with pytest.raises(ConfigurationError, match="without selected"):
        render_contact_sheet(result, tmp_path / "empty.png")


def test_checked_in_manifest_matches_current_algorithm() -> None:
    repository = Path(__file__).parents[1]
    example = repository / "examples" / "output"
    scan_config = ScanConfig(
        timestamp_mode="filename",
        timestamp_regex=r"_(?P<ts>\d+)ms$",
        timestamp_unit="milliseconds",
    )
    frames = scan_frames(example / "frames", scan_config)
    result = select_frames(frames, SelectionConfig(budget=6))
    expected = json.loads((example / "manifest.json").read_text(encoding="utf-8"))
    assert selection_manifest(result, scan_config) == expected
