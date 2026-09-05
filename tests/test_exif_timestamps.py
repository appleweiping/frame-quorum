from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest
from PIL import Image

from frame_quorum.cli import main
from frame_quorum.errors import ConfigurationError, ScanError
from frame_quorum.models import ScanConfig
from frame_quorum.scanner import scan_frames
from frame_quorum.timestamps import timestamp_for

_EXIF_IFD = 0x8769
_DATE_TIME = 0x0132
_DATE_TIME_ORIGINAL = 0x9003
_DATE_TIME_DIGITIZED = 0x9004
_OFFSET_TIME = 0x9010
_OFFSET_TIME_ORIGINAL = 0x9011
_OFFSET_TIME_DIGITIZED = 0x9012
_EXIF_CONFIG = ScanConfig(timestamp_mode="exif")


def _photo(
    path: Path,
    *,
    root: dict[int, object] | None = None,
    exif_ifd: dict[int, object] | None = None,
    color: tuple[int, int, int] = (30, 90, 150),
) -> Path:
    """Write a JPEG whose EXIF tags are exactly the supplied ones."""

    metadata = Image.Exif()
    for tag, value in (root or {}).items():
        metadata[tag] = value
    if exif_ifd is not None:
        sub = metadata.get_ifd(_EXIF_IFD)
        for tag, value in exif_ifd.items():
            sub[tag] = value
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (48, 32), color).save(path, exif=metadata)
    return path


def _epoch(text: str, offset: timezone = UTC) -> float:
    return datetime.strptime(text, "%Y:%m:%d %H:%M:%S").replace(tzinfo=offset).timestamp()


def test_capture_time_prefers_date_time_original(tmp_path: Path) -> None:
    _photo(
        tmp_path / "shot.jpg",
        root={_DATE_TIME: "2024:05:04 09:00:00"},
        exif_ifd={
            _DATE_TIME_ORIGINAL: "2024:05:01 12:00:00",
            _DATE_TIME_DIGITIZED: "2024:05:02 12:00:00",
        },
    )
    frames = scan_frames(tmp_path, _EXIF_CONFIG)
    assert frames[0].timestamp == _epoch("2024:05:01 12:00:00")


def test_capture_time_falls_back_to_digitized_then_file_time(tmp_path: Path) -> None:
    digitized = _photo(
        tmp_path / "digitized" / "scan.jpg",
        root={_DATE_TIME: "2024:05:04 09:00:00"},
        exif_ifd={_DATE_TIME_DIGITIZED: "2024:05:02 12:00:00"},
    )
    written = _photo(tmp_path / "written" / "edit.jpg", root={_DATE_TIME: "2024:05:04 09:00:00"})
    assert scan_frames(digitized, _EXIF_CONFIG)[0].timestamp == _epoch("2024:05:02 12:00:00")
    assert scan_frames(written, _EXIF_CONFIG)[0].timestamp == _epoch("2024:05:04 09:00:00")


def test_unset_placeholder_tag_is_skipped_rather_than_rejected(tmp_path: Path) -> None:
    _photo(
        tmp_path / "unset.jpg",
        exif_ifd={
            _DATE_TIME_ORIGINAL: "0000:00:00 00:00:00",
            _DATE_TIME_DIGITIZED: "2024:05:02 12:00:00",
        },
    )
    assert scan_frames(tmp_path, _EXIF_CONFIG)[0].timestamp == _epoch("2024:05:02 12:00:00")


def test_malformed_higher_precedence_tag_is_rejected_not_skipped(tmp_path: Path) -> None:
    _photo(
        tmp_path / "broken.jpg",
        exif_ifd={
            _DATE_TIME_ORIGINAL: "2024-05-01T12:00:00",
            _DATE_TIME_DIGITIZED: "2024:05:02 12:00:00",
        },
    )
    with pytest.raises(ScanError, match="is not a YYYY:MM:DD HH:MM:SS capture time"):
        scan_frames(tmp_path, _EXIF_CONFIG)


@pytest.mark.parametrize("value", ["2024:13:01 12:00:00", "2024:05:01 25:00:00", "", "   "])
def test_out_of_range_and_empty_capture_values_are_rejected(tmp_path: Path, value: str) -> None:
    _photo(tmp_path / "broken.jpg", exif_ifd={_DATE_TIME_ORIGINAL: value})
    with pytest.raises(ScanError, match=r"broken\.jpg"):
        scan_frames(tmp_path, _EXIF_CONFIG)


def test_non_text_capture_value_is_rejected(tmp_path: Path) -> None:
    _photo(tmp_path / "numeric.jpg", exif_ifd={_DATE_TIME_ORIGINAL: 5})
    with pytest.raises(ScanError, match=r"DateTimeOriginal in numeric\.jpg must be text, not int"):
        scan_frames(tmp_path, _EXIF_CONFIG)


def test_image_without_any_capture_tag_is_rejected(image_factory: Callable[..., Path]) -> None:
    with pytest.raises(ScanError, match="records no EXIF capture time"):
        scan_frames(image_factory("plain.png"), _EXIF_CONFIG)


@pytest.mark.parametrize(
    ("offset_text", "offset"),
    [
        ("+02:00", timezone(timedelta(hours=2))),
        ("-05:30", timezone(-timedelta(hours=5, minutes=30))),
        ("+00:00", UTC),
    ],
)
def test_recorded_utc_offset_is_applied(tmp_path: Path, offset_text: str, offset: timezone) -> None:
    _photo(
        tmp_path / "zoned.jpg",
        exif_ifd={_DATE_TIME_ORIGINAL: "2024:05:01 12:00:00", _OFFSET_TIME_ORIGINAL: offset_text},
    )
    assert scan_frames(tmp_path, _EXIF_CONFIG)[0].timestamp == _epoch("2024:05:01 12:00:00", offset)


def test_missing_and_unset_offsets_are_read_as_utc(tmp_path: Path) -> None:
    bare = _photo(tmp_path / "bare" / "a.jpg", exif_ifd={_DATE_TIME_ORIGINAL: "2024:05:01 12:00:00"})
    blank = _photo(
        tmp_path / "blank" / "b.jpg",
        exif_ifd={_DATE_TIME_ORIGINAL: "2024:05:01 12:00:00", _OFFSET_TIME_ORIGINAL: "   :  "},
    )
    expected = _epoch("2024:05:01 12:00:00")
    assert scan_frames(bare, _EXIF_CONFIG)[0].timestamp == expected
    assert scan_frames(blank, _EXIF_CONFIG)[0].timestamp == expected


@pytest.mark.parametrize("offset_text", ["+2h", "02:00", "+02:99", "+24:00"])
def test_malformed_or_out_of_range_offset_is_rejected(tmp_path: Path, offset_text: str) -> None:
    _photo(
        tmp_path / "bad-offset.jpg",
        exif_ifd={_DATE_TIME_ORIGINAL: "2024:05:01 12:00:00", _OFFSET_TIME_ORIGINAL: offset_text},
    )
    with pytest.raises(ScanError, match="UTC offset"):
        scan_frames(tmp_path, _EXIF_CONFIG)


def test_non_text_offset_is_rejected(tmp_path: Path) -> None:
    _photo(
        tmp_path / "numeric-offset.jpg",
        exif_ifd={_DATE_TIME_ORIGINAL: "2024:05:01 12:00:00", _OFFSET_TIME_ORIGINAL: 7},
    )
    with pytest.raises(ScanError, match=r"UTC offset in numeric-offset\.jpg must be text, not int"):
        scan_frames(tmp_path, _EXIF_CONFIG)


def test_each_capture_tag_uses_its_own_offset_tag(tmp_path: Path) -> None:
    digitized = _photo(
        tmp_path / "digitized" / "a.jpg",
        exif_ifd={
            _DATE_TIME_DIGITIZED: "2024:05:01 12:00:00",
            _OFFSET_TIME_DIGITIZED: "+03:00",
            _OFFSET_TIME_ORIGINAL: "-07:00",
        },
    )
    written = _photo(
        tmp_path / "written" / "b.jpg",
        root={_DATE_TIME: "2024:05:01 12:00:00", _OFFSET_TIME: "+03:00"},
    )
    expected = _epoch("2024:05:01 12:00:00", timezone(timedelta(hours=3)))
    assert scan_frames(digitized, _EXIF_CONFIG)[0].timestamp == expected
    assert scan_frames(written, _EXIF_CONFIG)[0].timestamp == expected


def test_exif_metadata_read_failure_is_actionable(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("not an image", encoding="utf-8")
    with pytest.raises(ScanError, match="cannot read EXIF metadata"):
        timestamp_for(path, 0, _EXIF_CONFIG)


def test_exif_sequence_orders_by_real_capture_time(tmp_path: Path) -> None:
    for name, moment in (
        ("a.jpg", "2024:05:01 12:00:00"),
        ("b.jpg", "2024:05:01 12:00:30"),
        ("c.jpg", "2024:05:01 12:01:00"),
    ):
        _photo(tmp_path / name, exif_ifd={_DATE_TIME_ORIGINAL: moment})
    frames = scan_frames(tmp_path, _EXIF_CONFIG)
    assert [frame.timestamp for frame in frames] == [
        _epoch("2024:05:01 12:00:00"),
        _epoch("2024:05:01 12:00:30"),
        _epoch("2024:05:01 12:01:00"),
    ]


def test_exif_mode_is_not_the_default(tmp_path: Path) -> None:
    _photo(tmp_path / "shot.jpg", exif_ifd={_DATE_TIME_ORIGINAL: "2024:05:01 12:00:00"})
    assert scan_frames(tmp_path)[0].timestamp == 0.0
    assert ScanConfig().timestamp_mode == "index"


def test_unknown_timestamp_mode_still_names_every_supported_mode(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="index, filename, mtime, exif, or none"):
        scan_frames(tmp_path, ScanConfig(timestamp_mode="capture"))


def test_cli_scan_reads_exif_capture_time(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _photo(
        tmp_path / "shot.jpg",
        exif_ifd={_DATE_TIME_ORIGINAL: "2024:05:01 12:00:00", _OFFSET_TIME_ORIGINAL: "+02:00"},
    )
    assert main(["scan", str(tmp_path), "--timestamp-mode", "exif"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["scan_config"]["timestamp_mode"] == "exif"
    assert payload["frames"][0]["timestamp"] == _epoch("2024:05:01 12:00:00", timezone(timedelta(hours=2)))


def test_cli_reports_a_missing_capture_time(
    image_factory: Callable[..., Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    image_factory("plain.png")
    assert main(["scan", str(tmp_path), "--timestamp-mode", "exif"]) == 2
    assert "records no EXIF capture time" in capsys.readouterr().err
