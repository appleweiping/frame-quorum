from __future__ import annotations

from fractions import Fraction

import pytest

from frame_quorum import FrameRate, FrameTimecode
from frame_quorum.errors import ConfigurationError


def test_rate_is_exact_and_canonical() -> None:
    assert FrameRate(60000, 2002) == FrameRate.parse("30000/1001")
    assert str(FrameRate.parse(25)) == "25"
    assert FrameRate.parse("29.97").fraction == Fraction(2997, 100)
    assert FrameRate.parse("29.97") != FrameRate(30000, 1001)
    assert FrameRate.parse(Fraction(24000, 1001)).nominal == 24
    assert FrameRate(1000).nominal == 1000
    with pytest.raises(ConfigurationError, match="SMPTE"):
        FrameTimecode(1, FrameRate.parse("29.97")).smpte()


@pytest.mark.parametrize(
    "rate", [0, -1, True, 29.97, "NaN", "1/0", "1e9", " 25", "1" * 65, "1001", 2**200, Fraction(1, 2**200)]
)
def test_invalid_rates(rate: object) -> None:
    with pytest.raises(ConfigurationError):
        FrameRate.parse(rate)  # type: ignore[arg-type]


@pytest.mark.parametrize("parts", [(1, 0), (True, 1), (25, True), (10**10, 10**10), (1001, 1)])
def test_rate_parts_are_bounded(parts: tuple[int, int]) -> None:
    with pytest.raises(ConfigurationError):
        FrameRate(*parts)


@pytest.mark.parametrize(
    "frame,label",
    [
        (0, "00:00:00;00"),
        (1799, "00:00:59;29"),
        (1800, "00:01:00;02"),
        (3598, "00:02:00;02"),
        (17981, "00:09:59;29"),
        (17982, "00:10:00;00"),
        (107892, "01:00:00;00"),
        (2589408, "24:00:00;00"),
    ],
)
def test_independently_known_drop_frame_boundaries(frame: int, label: str) -> None:
    coordinate = FrameTimecode(frame, FrameRate(30000, 1001), True)
    assert coordinate.smpte() == label
    assert FrameTimecode.parse_smpte(label, coordinate.rate) == coordinate
    assert coordinate.seconds == Fraction(frame * 1001, 30000)


@pytest.mark.parametrize(
    "rate,nominal,skip", [(FrameRate(30000, 1001), 30, 2), (FrameRate(60000, 1001), 60, 4)]
)
def test_drop_frame_against_enumerated_counter_labels(rate: FrameRate, nominal: int, skip: int) -> None:
    # Independent oracle: walk ordinary clock labels and omit forbidden counters.
    # No block arithmetic or calls into formatting helpers generate expectations.
    index = 0
    for minute in range(11):
        for second in range(60):
            for frame in range(nominal):
                if minute % 10 and second == 0 and frame < skip:
                    continue
                expected = f"00:{minute:02d}:{second:02d};{frame:02d}"
                coordinate = FrameTimecode(index, rate, True)
                assert coordinate.smpte() == expected
                assert FrameTimecode.parse_smpte(expected, rate).frame_number == index
                index += 1


@pytest.mark.parametrize(
    "label",
    [
        "00:01:00;00",
        "00:01:00;01",
        "00:00:00;30",
        "00:60:00;00",
        "00:00:60;00",
        "-00:00:00;00",
        "0:00:00;00",
        "00:00:00;00\n",
        "00:00:00,00",
        1,
    ],
)
def test_invalid_or_skipped_labels(label: object) -> None:
    with pytest.raises(ConfigurationError):
        FrameTimecode.parse_smpte(label, FrameRate(30000, 1001))  # type: ignore[arg-type]


def test_nominal_labels_are_distinct_from_elapsed_seconds() -> None:
    rate = FrameRate(24000, 1001)
    coordinate = FrameTimecode.parse_smpte("01:00:00:00", rate)
    assert coordinate.frame_number == 86400
    assert coordinate.seconds == Fraction(18018, 5)
    assert coordinate.timestamp() == "01:00:03.600"
    assert FrameTimecode.from_timestamp("01:00:03.600", rate) == coordinate
    assert FrameTimecode(2589408, FrameRate(30000, 1001), True).smpte(wrap_24_hours=True) == "00:00:00;00"
    assert FrameTimecode(3600 * 25 * 25, FrameRate(25)).smpte() == "25:00:00:00"
    assert FrameTimecode(999, FrameRate(1000)).smpte() == "00:00:00:999"


def test_rounding_is_explicit_and_ties_go_forward() -> None:
    rate = FrameRate(25)
    assert FrameTimecode.from_seconds("0.02", rate).frame_number == 1
    assert FrameTimecode.from_seconds("0.02", rate, rounding="floor").frame_number == 0
    assert FrameTimecode.from_seconds("0.001", rate, rounding="ceil").frame_number == 1
    assert FrameTimecode.from_seconds("1", rate, rounding="ceil").frame_number == 25
    assert FrameTimecode.from_seconds(1, rate).seconds == 1
    coordinate = FrameTimecode(59999, FrameRate(1000))
    assert coordinate.timestamp(precision=2) == "00:01:00.00"
    assert coordinate.timestamp(precision=0) == "00:01:00"
    assert coordinate.timestamp(precision=9) == "00:00:59.999000000"


def test_pts_shift_and_rescale_use_rational_arithmetic() -> None:
    coordinate = FrameTimecode.from_pts(99000, Fraction(1, 90000), FrameRate(24), origin_pts=9000)
    assert coordinate.frame_number == 24
    assert coordinate.shift_frames(-12).seconds == Fraction(1, 2)
    assert coordinate.rescale(FrameRate(60000, 1001)).frame_number == 60
    assert coordinate.rescale(FrameRate(60000, 1001), rounding="floor").frame_number == 59
    assert FrameTimecode.from_timestamp("00:00:01", FrameRate(25)).frame_number == 25


def test_elapsed_parser_covers_largest_frame_at_smallest_admitted_rate() -> None:
    coordinate = FrameTimecode((1 << 63) - 1, FrameRate(1, 1_000_000_000))
    assert coordinate.timestamp(precision=0) == "2562047788015215501944444:26:40"
    assert FrameTimecode.from_timestamp(coordinate.timestamp(), coordinate.rate) == coordinate


@pytest.mark.parametrize(
    "options",
    [
        (-1, FrameRate(25), False),
        (True, FrameRate(25), False),
        (2**63, FrameRate(25), False),
        (0, 25, False),
        (0, FrameRate(25), 1),
        (0, FrameRate(25), True),
        (0, FrameRate(24000, 1001), True),
    ],
)
def test_invalid_coordinates(options: tuple) -> None:
    with pytest.raises(ConfigurationError):
        FrameTimecode(*options)


def test_invalid_conversions_have_domain_errors() -> None:
    rate = FrameRate(25)
    with pytest.raises(ConfigurationError):
        FrameTimecode.from_seconds(1, 25)  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError):
        FrameTimecode.from_seconds(1, rate, rounding="bankers")  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError):
        FrameTimecode.from_seconds(-1, rate)
    with pytest.raises(ConfigurationError):
        FrameTimecode.from_seconds(10**19, rate)
    for value in ("00:00:60", "00:00:00.1234567890", "00:00:00.", "NaN", None):
        with pytest.raises(ConfigurationError):
            FrameTimecode.from_timestamp(value, rate)  # type: ignore[arg-type]
    for base in (Fraction(0), Fraction(-1), Fraction(1, 2**200), 0.01):
        with pytest.raises(ConfigurationError):
            FrameTimecode.from_pts(0, base, rate)  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError):
        FrameTimecode.from_pts(1, Fraction(1), rate, origin_pts=2)
    with pytest.raises(ConfigurationError):
        FrameTimecode(0, rate).timestamp(precision=10)
    with pytest.raises(ConfigurationError):
        FrameTimecode(0, rate).shift_frames(-1)
    with pytest.raises(ConfigurationError):
        FrameTimecode(0, rate).smpte(wrap_24_hours=1)  # type: ignore[arg-type]
