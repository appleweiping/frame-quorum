from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image

from frame_quorum.contact_sheet import render_contact_sheet
from frame_quorum.errors import ConfigurationError, ScanError
from frame_quorum.models import Frame, ScanConfig, ScoreBreakdown, SelectionConfig, SelectionResult
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


def test_write_json_wraps_unbounded_integer_as_configuration_error() -> None:
    with pytest.raises(ConfigurationError, match="JSON integer"):
        write_json({"nested": {"index": 10**5000}}, None)


def test_signed_64_bit_json_boundaries_are_supported() -> None:
    minimum = -(1 << 63)
    maximum = (1 << 63) - 1
    assert json.loads(write_json({"minimum": minimum, "maximum": maximum}, None)) == {
        "minimum": minimum,
        "maximum": maximum,
    }
    with pytest.raises(ConfigurationError, match="JSON integer"):
        write_json({"below": minimum - 1}, None)


def test_maximum_frame_index_survives_selection_manifest_and_json(
    image_factory: Callable[..., Path], tmp_path: Path
) -> None:
    frame = scan_frames(image_factory("one.png"))[0]
    maximum = (1 << 63) - 1
    bounded = replace(frame, index=maximum, timestamp=0.0)
    result = select_frames((bounded,))
    rendered = write_json(selection_manifest(result, ScanConfig()), None)
    assert json.loads(rendered)["frames"][0]["index"] == maximum
    sheet = render_contact_sheet(result, tmp_path / "maximum-index.png", columns=1, thumbnail_width=96)
    with Image.open(sheet) as image:
        assert image.format == "PNG"


def test_signed_64_integer_spellings_for_continuous_values_complete_the_output_pipeline(
    image_factory: Callable[..., Path], tmp_path: Path
) -> None:
    maximum = (1 << 63) - 1
    minimum = -(1 << 63)
    frame = replace(scan_frames(image_factory("one.png"))[0], timestamp=minimum)
    scan_config = ScanConfig(frame_rate=maximum)
    selection_config = SelectionConfig(
        budget=1,
        min_gap=maximum,
        duplicate_threshold=1,
        quality_weight=maximum,
        change_weight=maximum,
        coverage_weight=maximum,
    )

    result = select_frames((frame,), selection_config)
    manifest = selection_manifest(result, scan_config)
    rendered = write_json(manifest, tmp_path / "bounded.json")

    decoded = json.loads(rendered)
    assert decoded["frames"][0]["timestamp"] == minimum
    assert decoded["scan_config"]["frame_rate"] == maximum
    assert decoded["selection_config"]["min_gap"] == maximum
    assert decoded["selection_config"]["quality_weight"] == maximum
    render_contact_sheet(result, tmp_path / "bounded.png", columns=1, thumbnail_width=96)


def test_large_finite_floats_complete_the_output_pipeline(
    image_factory: Callable[..., Path], tmp_path: Path
) -> None:
    frame = replace(scan_frames(image_factory("one.png"))[0], timestamp=-1e308)
    scan_config = ScanConfig(frame_rate=1e308)
    selection_config = SelectionConfig(
        min_gap=1e308,
        quality_weight=1e308,
        change_weight=0.0,
        coverage_weight=0.0,
    )

    result = select_frames((frame,), selection_config)
    decoded = json.loads(write_json(selection_manifest(result, scan_config), None))

    assert decoded["frames"][0]["timestamp"] == -1e308
    assert decoded["scan_config"]["frame_rate"] == 1e308
    assert decoded["selection_config"]["min_gap"] == 1e308


@pytest.mark.parametrize(
    ("record", "message"),
    [
        (ScanConfig(frame_rate=1 << 63), "frame_rate"),
        (ScanConfig(frame_rate=True), "frame_rate"),
        (SelectionConfig(min_gap=1 << 63), "min_gap"),
        (SelectionConfig(min_gap=True), "min_gap"),
        (SelectionConfig(duplicate_threshold=True), "duplicate_threshold"),
        (SelectionConfig(quality_weight=1 << 63), "weights"),
        (SelectionConfig(change_weight=1 << 63), "weights"),
        (SelectionConfig(coverage_weight=1 << 63), "weights"),
        (SelectionConfig(quality_weight=True), "weights"),
    ],
)
def test_continuous_numeric_settings_reject_unstable_integer_spellings(
    record: ScanConfig | SelectionConfig, message: str
) -> None:
    with pytest.raises(ConfigurationError, match=message):
        record.validate()


@pytest.mark.parametrize("timestamp", [1 << 63, -(1 << 63) - 1, True])
def test_frame_timestamp_rejects_unstable_integer_spellings(
    image_factory: Callable[..., Path], timestamp: int
) -> None:
    frame = replace(scan_frames(image_factory("one.png"))[0], timestamp=timestamp)
    with pytest.raises(ConfigurationError, match="timestamp"):
        frame.validate()


def test_direct_frame_serialization_enforces_integer_bounds(
    image_factory: Callable[..., Path], tmp_path: Path
) -> None:
    frame = scan_frames(image_factory("one.png"))[0]
    with pytest.raises(ConfigurationError, match="frame index"):
        replace(frame, index=10**500).serializable()
    with pytest.raises(ConfigurationError, match="frame byte size"):
        replace(frame, byte_size=True).serializable()
    with pytest.raises(ConfigurationError, match="perceptual hash"):
        replace(frame, metrics=replace(frame.metrics, perceptual_hash=1 << 64)).serializable()


def test_direct_frame_validation_rejects_invalid_record_types(
    image_factory: Callable[..., Path], tmp_path: Path
) -> None:
    frame = scan_frames(image_factory("one.png"))[0]
    with pytest.raises(ConfigurationError, match=r"pathlib\.Path"):
        replace(frame, path="one.png").validate()  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError, match="FrameMetrics"):
        replace(frame, metrics=object()).validate()  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError, match="finite number"):
        _ = replace(frame, timestamp=float("nan")).time_coordinate


def test_direct_decision_serialization_enforces_integer_bounds(
    image_factory: Callable[..., Path], tmp_path: Path
) -> None:
    _, result = _result(image_factory, tmp_path)
    decision = result.decisions[0]
    with pytest.raises(ConfigurationError, match="decision frame index"):
        replace(decision, index=10**500).serializable()
    with pytest.raises(ConfigurationError, match="decision rank"):
        replace(decision, rank=1 << 63).serializable()
    with pytest.raises(ConfigurationError, match="nearest selected frame index"):
        replace(decision, nearest_selected_index=-1).serializable()


def test_direct_decision_validation_rejects_invalid_types_and_relationships(
    image_factory: Callable[..., Path], tmp_path: Path
) -> None:
    _, result = _result(image_factory, tmp_path)
    decision = result.decisions[0]
    with pytest.raises(ConfigurationError, match="selected must be a boolean"):
        replace(decision, selected=1).validate()  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError, match="require a rank"):
        replace(decision, rank=None).validate()
    with pytest.raises(ConfigurationError, match="non-empty string"):
        replace(decision, reason="").validate()
    with pytest.raises(ConfigurationError, match="ScoreBreakdown"):
        replace(decision, scores=object()).validate()  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError, match="finite values"):
        replace(decision, scores=ScoreBreakdown(utility=float("inf"))).serializable()


def test_scan_manifest_rejects_total_byte_count_outside_signed_64_bits(
    image_factory: Callable[..., Path], tmp_path: Path
) -> None:
    frame = scan_frames(image_factory("one.png"))[0]
    maximum = (1 << 63) - 1
    frames = (replace(frame, byte_size=maximum), replace(frame, index=1, byte_size=1))
    with pytest.raises(ConfigurationError, match="total byte size"):
        scan_manifest(frames, ScanConfig())


def test_direct_selection_result_is_validated_by_report_and_contact_sheet(
    image_factory: Callable[..., Path], tmp_path: Path
) -> None:
    _, result = _result(image_factory, tmp_path)
    broken = replace(result, selected_indices=(10**500,))
    with pytest.raises(ConfigurationError, match="selected frame index"):
        selection_manifest(broken, ScanConfig())
    with pytest.raises(ConfigurationError, match="selected frame index"):
        render_contact_sheet(broken, tmp_path / "bad-result.png")


def test_direct_selection_result_cross_record_invariants(
    image_factory: Callable[..., Path], tmp_path: Path
) -> None:
    _, result = _result(image_factory, tmp_path)
    frames = result.frames
    decisions = result.decisions
    rejected = next(decision for decision in decisions if not decision.selected)
    unselected_index = rejected.index
    selected_as_rejected = replace(
        rejected,
        selected=True,
        rank=len(result.selected_indices) + 1,
    )
    state_mismatch = tuple(
        selected_as_rejected if decision.index == rejected.index else decision for decision in decisions
    )
    duplicate_rank = tuple(
        replace(decision, rank=1) if decision.selected else decision for decision in decisions
    )
    invalid_nearest = tuple(
        replace(decision, nearest_selected_index=unselected_index)
        if decision.index == rejected.index
        else decision
        for decision in decisions
    )
    invalid_results = [
        (replace(result, frames=list(frames)), "tuple of Frame"),  # type: ignore[arg-type]
        (replace(result, frames=(frames[0], frames[0], *frames[2:])), "unique"),
        (replace(result, frames=tuple(reversed(frames))), "ordered by index"),
        (replace(result, selected_indices=list(result.selected_indices)), "must be a tuple"),  # type: ignore[arg-type]
        (replace(result, selected_indices=(result.selected_indices[0],) * 2), "must be unique"),
        (replace(result, selected_indices=tuple(reversed(result.selected_indices))), "must be ordered"),
        (replace(result, selected_indices=((1 << 63) - 1,)), "must reference result frames"),
        (replace(result, decisions=list(decisions)), "tuple of FrameDecision"),  # type: ignore[arg-type]
        (replace(result, decisions=tuple(reversed(decisions))), "correspond to frames"),
        (replace(result, decisions=state_mismatch), "selected states"),
        (replace(result, decisions=duplicate_rank), "ranks must be unique"),
        (replace(result, decisions=invalid_nearest), "nearest selected indices"),
        (
            replace(result, config=replace(result.config, budget=len(result.selected_indices) - 1)),
            "selection budget",
        ),
        (
            replace(result, decisions=(replace(decisions[0], path="different.png"), *decisions[1:])),
            "decision paths",
        ),
        (replace(result, config=object()), "config must be SelectionConfig"),  # type: ignore[arg-type]
    ]
    for broken, message in invalid_results:
        with pytest.raises(ConfigurationError, match=message):
            broken.validate()


def test_decision_lookup_validates_index_and_reports_missing(
    image_factory: Callable[..., Path], tmp_path: Path
) -> None:
    _, result = _result(image_factory, tmp_path)
    with pytest.raises(ConfigurationError, match="decision lookup index"):
        result.decision_for(False)
    with pytest.raises(KeyError):
        result.decision_for((1 << 63) - 1)


def test_manifest_entry_points_reject_wrong_types_and_order(
    image_factory: Callable[..., Path], tmp_path: Path
) -> None:
    frames, result = _result(image_factory, tmp_path)
    with pytest.raises(ConfigurationError, match="result must be"):
        selection_manifest(object(), ScanConfig())  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError, match="scan_config"):
        selection_manifest(result, object())  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError, match="config must be"):
        scan_manifest(frames, object())  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError, match="tuple of Frame"):
        scan_manifest(list(frames), ScanConfig())  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError, match="unique"):
        scan_manifest((frames[0], frames[0]), ScanConfig())
    with pytest.raises(ConfigurationError, match="ordered"):
        scan_manifest(tuple(reversed(frames)), ScanConfig())


def test_json_validation_wraps_unsupported_and_circular_values() -> None:
    with pytest.raises(ConfigurationError, match="JSON-serializable"):
        write_json({"unsupported": {1, 2}}, None)
    circular: list[object] = []
    circular.append(circular)
    with pytest.raises(ConfigurationError, match="JSON-serializable"):
        write_json({"circular": circular}, None)
    assert json.loads(write_json({"tuple": (1, 2), "flag": True}, None))["tuple"] == [1, 2]


def test_contact_sheet_rejects_wrong_result_type(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="SelectionResult"):
        render_contact_sheet(object(), tmp_path / "bad.png")  # type: ignore[arg-type]


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
    with pytest.raises(ConfigurationError, match="integer between 1"):
        render_contact_sheet(result, tmp_path / "huge.png", thumbnail_width=10**5000)


@pytest.mark.parametrize("columns", [False, -1, 1 << 63, 10**500])
def test_contact_sheet_columns_have_stable_integer_bounds(
    image_factory: Callable[..., Path], tmp_path: Path, columns: int
) -> None:
    _, result = _result(image_factory, tmp_path)
    with pytest.raises(ConfigurationError, match="contact-sheet columns"):
        render_contact_sheet(result, tmp_path / "bad.png", columns=columns)


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
