from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import pytest

import frame_quorum.scanner as scanner_module
from frame_quorum.errors import ConfigurationError, ScanError
from frame_quorum.models import ScanConfig
from frame_quorum.scanner import _natural_key, _path_sort_key, scan_frames


def test_natural_filename_order(image_factory: Callable[..., Path], tmp_path: Path) -> None:
    image_factory("frame_10.png")
    image_factory("frame_2.png")
    image_factory("frame_1.png")
    frames = scan_frames(tmp_path)
    assert [frame.relative_path for frame in frames] == ["frame_1.png", "frame_2.png", "frame_10.png"]


def test_natural_key_handles_unbounded_digit_runs_without_integer_conversion() -> None:
    huge = "9" * 10_000
    assert _natural_key(f"frame_{huge}.png") > _natural_key("frame_10.png")


def test_natural_sort_has_deterministic_case_tie_break() -> None:
    assert _path_sort_key("A_2.png") != _path_sort_key("a_2.png")
    assert sorted(["a_2.png", "A_2.png"], key=_path_sort_key) == ["A_2.png", "a_2.png"]


def test_extension_matching_is_case_insensitive(image_factory: Callable[..., Path], tmp_path: Path) -> None:
    source = image_factory("source.png")
    destination = tmp_path / "FRAME.PNG"
    source.replace(destination)
    assert scan_frames(tmp_path)[0].relative_path == "FRAME.PNG"


def test_recursive_scan(image_factory: Callable[..., Path], tmp_path: Path) -> None:
    image_factory("nested.png", directory=tmp_path / "a")
    frames = scan_frames(tmp_path, ScanConfig(recursive=True))
    assert frames[0].relative_path == "a/nested.png"


def test_non_recursive_ignores_nested_images(image_factory: Callable[..., Path], tmp_path: Path) -> None:
    image_factory("nested.png", directory=tmp_path / "a")
    with pytest.raises(ScanError, match="no supported images"):
        scan_frames(tmp_path)


def test_single_file_scan(image_factory: Callable[..., Path]) -> None:
    path = image_factory("one.png", size=(37, 29))
    frame = scan_frames(path)[0]
    assert (frame.width, frame.height) == (37, 29)


def test_single_unsupported_file_is_actionable(tmp_path: Path) -> None:
    path = tmp_path / "frame.txt"
    path.write_text("not an image", encoding="utf-8")
    with pytest.raises(ScanError, match="unsupported image extension"):
        scan_frames(path)


def test_missing_input_is_actionable(tmp_path: Path) -> None:
    with pytest.raises(ScanError, match="does not exist"):
        scan_frames(tmp_path / "missing")


def test_corrupt_image_is_actionable(tmp_path: Path) -> None:
    path = tmp_path / "bad.png"
    path.write_bytes(b"not an image")
    with pytest.raises(ScanError, match="cannot decode"):
        scan_frames(tmp_path)


def test_index_timestamps_use_frame_rate(image_factory: Callable[..., Path], tmp_path: Path) -> None:
    image_factory("1.png")
    image_factory("2.png")
    frames = scan_frames(tmp_path, ScanConfig(frame_rate=4.0))
    assert [frame.timestamp for frame in frames] == [0.0, 0.25]


def test_filename_named_group_and_unit(image_factory: Callable[..., Path], tmp_path: Path) -> None:
    image_factory("capture_1500ms.png")
    config = ScanConfig(
        timestamp_mode="filename", timestamp_regex=r"_(?P<ts>\d+)ms$", timestamp_unit="milliseconds"
    )
    assert scan_frames(tmp_path, config)[0].timestamp == 1.5


def test_filename_first_group(image_factory: Callable[..., Path], tmp_path: Path) -> None:
    image_factory("at-2.75-sec.png")
    config = ScanConfig(timestamp_mode="filename", timestamp_regex=r"at-([0-9.]+)-sec")
    assert scan_frames(tmp_path, config)[0].timestamp == 2.75


def test_filename_mismatch_names_file(image_factory: Callable[..., Path], tmp_path: Path) -> None:
    image_factory("untimed.png")
    config = ScanConfig(timestamp_mode="filename", timestamp_regex=r"(\d+)ms")
    with pytest.raises(ScanError, match=r"untimed\.png"):
        scan_frames(tmp_path, config)


def test_invalid_filename_timestamp_regex_is_actionable(
    image_factory: Callable[..., Path], tmp_path: Path
) -> None:
    image_factory("frame_1.png")
    config = ScanConfig(timestamp_mode="filename", timestamp_regex="[")
    with pytest.raises(ScanError, match="invalid timestamp regex"):
        scan_frames(tmp_path, config)


def test_filename_timestamp_can_use_entire_match(image_factory: Callable[..., Path], tmp_path: Path) -> None:
    image_factory("2.5.png")
    config = ScanConfig(timestamp_mode="filename", timestamp_regex=r"[0-9.]+")
    assert scan_frames(tmp_path, config)[0].timestamp == 2.5


def test_mtime_timestamp(image_factory: Callable[..., Path]) -> None:
    path = image_factory("mtime.png")
    os.utime(path, (1234, 1234))
    assert scan_frames(path, ScanConfig(timestamp_mode="mtime"))[0].timestamp == pytest.approx(1234)


def test_none_timestamp_mode(image_factory: Callable[..., Path]) -> None:
    assert scan_frames(image_factory(), ScanConfig(timestamp_mode="none"))[0].timestamp is None


@pytest.mark.parametrize(
    "config",
    [
        ScanConfig(timestamp_mode="other"),
        ScanConfig(frame_rate=0),
        ScanConfig(frame_rate=float("nan")),
        ScanConfig(frame_rate=float("inf")),
        ScanConfig(frame_rate=True),
        ScanConfig(frame_rate=10**400),
        ScanConfig(timestamp_mode="filename"),
        ScanConfig(timestamp_regex=object()),
        ScanConfig(timestamp_regex=b"(\\d+)"),
        ScanConfig(timestamp_unit="minutes"),
        ScanConfig(extensions=()),
        ScanConfig(extensions=123),
        ScanConfig(extensions="png"),
        ScanConfig(extensions=(". png",)),
        ScanConfig(timestamp_mode=[]),
        ScanConfig(timestamp_unit=[]),
        ScanConfig(recursive=1),
    ],
)
def test_invalid_scan_config(config: ScanConfig, tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError):
        scan_frames(tmp_path, config)


def test_filename_timestamp_must_be_finite(image_factory: Callable[..., Path], tmp_path: Path) -> None:
    image_factory("capture_1e9999.png")
    config = ScanConfig(timestamp_mode="filename", timestamp_regex=r"_(.*)$")
    with pytest.raises(ScanError, match="finite"):
        scan_frames(tmp_path, config)


def test_optional_named_timestamp_group_must_match(
    image_factory: Callable[..., Path], tmp_path: Path
) -> None:
    image_factory("capture_ms.png")
    config = ScanConfig(timestamp_mode="filename", timestamp_regex=r"_(?P<ts>\d+)?ms$")
    with pytest.raises(ScanError, match="not numeric"):
        scan_frames(tmp_path, config)


def test_derived_index_timestamp_must_be_finite(image_factory: Callable[..., Path], tmp_path: Path) -> None:
    image_factory("1.png")
    image_factory("2.png")
    with pytest.raises(ScanError, match="finite"):
        scan_frames(tmp_path, ScanConfig(frame_rate=5e-324))


def test_disappearing_file_is_reported_as_scan_error(
    image_factory: Callable[..., Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = image_factory("vanishing.png")

    def remove_before_stat(source: Path, index: int, config: ScanConfig) -> float:
        source.unlink()
        return 0.0

    monkeypatch.setattr(scanner_module, "timestamp_for", remove_before_stat)
    with pytest.raises(ScanError, match="cannot inspect image"):
        scan_frames(path)
