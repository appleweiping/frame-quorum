"""Exact frame/time coordinates and explicit non-drop/drop-frame labels.

Elapsed time always uses rational arithmetic. Drop-frame changes labels, never
media frames. No binary float rate is silently interpreted as an NTSC rate.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from fractions import Fraction
from typing import Literal

from .errors import ConfigurationError

Rounding = Literal["floor", "ceil", "nearest"]
_MAX_FRAME = (1 << 63) - 1
_MAX_RATE_PART = 1_000_000_000


def _integer(value: object, name: str, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ConfigurationError(f"{name} must be an integer between {minimum} and {maximum}")
    return value


def _rational(value: int | str | Fraction, name: str) -> Fraction:
    if type(value) not in (int, str, Fraction):
        raise ConfigurationError(f"{name} requires an integer, exact string, or Fraction")
    if isinstance(value, str) and (
        len(value) > 64 or re.fullmatch(r"[0-9]+(?:/[0-9]+|\.[0-9]+)?", value) is None
    ):
        raise ConfigurationError(f"{name} must be a bounded nonnegative exact number")
    if isinstance(value, int) and value.bit_length() > 128:
        raise ConfigurationError(f"{name} is too large")
    try:
        result = Fraction(value)
    except (ValueError, ZeroDivisionError) as exc:
        raise ConfigurationError(f"invalid {name}") from exc
    if result < 0 or result.numerator.bit_length() > 128 or result.denominator.bit_length() > 128:
        raise ConfigurationError(f"{name} is outside the supported rational range")
    return result


def _round(value: Fraction, rounding: Rounding) -> int:
    if rounding not in ("floor", "ceil", "nearest"):
        raise ConfigurationError("rounding must be floor, ceil, or nearest")
    whole, remainder = divmod(value.numerator, value.denominator)
    if rounding == "ceil":
        return whole + bool(remainder)
    if rounding == "nearest":
        return whole + (remainder * 2 >= value.denominator)
    return whole


@dataclass(frozen=True, slots=True)
class FrameRate:
    """Reduced exact frame rate in (0, 1000], with bounded integer parts."""

    numerator: int
    denominator: int = 1

    def __post_init__(self) -> None:
        _integer(self.numerator, "rate numerator", 1, _MAX_RATE_PART)
        _integer(self.denominator, "rate denominator", 1, _MAX_RATE_PART)
        fraction = Fraction(self.numerator, self.denominator)
        if fraction > 1000:
            raise ConfigurationError("frame rate cannot exceed 1000")
        object.__setattr__(self, "numerator", fraction.numerator)
        object.__setattr__(self, "denominator", fraction.denominator)

    @classmethod
    def parse(cls, value: int | str | Fraction) -> FrameRate:
        fraction = _rational(value, "frame rate")
        return cls(fraction.numerator, fraction.denominator)

    @property
    def fraction(self) -> Fraction:
        return Fraction(self.numerator, self.denominator)

    @property
    def nominal(self) -> int:
        """Integer counter frequency for supported SMPTE-style text labels."""
        if self.denominator == 1:
            return self.numerator
        if self.denominator == 1001 and self.numerator in (24000, 30000, 60000, 120000):
            return self.numerator // 1000
        raise ConfigurationError("SMPTE labels require an integer or supported exact NTSC rate")

    @property
    def drop_count(self) -> int:
        if self.denominator == 1001 and self.numerator in (30000, 60000):
            return self.numerator // 15000
        raise ConfigurationError("drop-frame requires exactly 30000/1001 or 60000/1001")

    def __str__(self) -> str:
        return str(self.numerator) if self.denominator == 1 else f"{self.numerator}/{self.denominator}"


@dataclass(frozen=True, slots=True)
class FrameTimecode:
    """Nonnegative frame coordinate; hours do not wrap at midnight implicitly."""

    frame_number: int
    rate: FrameRate
    drop_frame: bool = False

    def __post_init__(self) -> None:
        _integer(self.frame_number, "frame_number", 0, _MAX_FRAME)
        if not isinstance(self.rate, FrameRate):
            raise ConfigurationError("rate must be FrameRate")
        if type(self.drop_frame) is not bool:
            raise ConfigurationError("drop_frame must be boolean")
        if self.drop_frame:
            _ = self.rate.drop_count

    @property
    def seconds(self) -> Fraction:
        return self.frame_number / self.rate.fraction

    @classmethod
    def from_seconds(
        cls,
        seconds: int | str | Fraction,
        rate: FrameRate,
        *,
        rounding: Rounding = "nearest",
        drop_frame: bool = False,
    ) -> FrameTimecode:
        if not isinstance(rate, FrameRate):
            raise ConfigurationError("rate must be FrameRate")
        return cls(_round(_rational(seconds, "seconds") * rate.fraction, rounding), rate, drop_frame)

    @classmethod
    def from_timestamp(
        cls,
        timestamp: str,
        rate: FrameRate,
        *,
        rounding: Rounding = "nearest",
        drop_frame: bool = False,
    ) -> FrameTimecode:
        """Parse elapsed HH:MM:SS[.fraction], never a frame-counter label."""
        match = (
            re.fullmatch(r"([0-9]{2,25}):([0-5][0-9]):([0-5][0-9])(?:\.([0-9]{1,9}))?", timestamp)
            if type(timestamp) is str
            else None
        )
        if match is None:
            raise ConfigurationError("timestamp must be HH:MM:SS with up to nine fractional digits")
        hours, minutes, seconds, decimal = match.groups()
        elapsed = Fraction(int(hours) * 3600 + int(minutes) * 60 + int(seconds))
        if decimal:
            elapsed += Fraction(int(decimal), 10 ** len(decimal))
        return cls.from_seconds(elapsed, rate, rounding=rounding, drop_frame=drop_frame)

    @classmethod
    def from_pts(
        cls,
        pts: int,
        time_base: Fraction,
        rate: FrameRate,
        *,
        origin_pts: int = 0,
        rounding: Rounding = "nearest",
        drop_frame: bool = False,
    ) -> FrameTimecode:
        """Convert an explicit decoder PTS/timebase to a CFR coordinate.

        This is quantization, not a VFR decoder or a claim of exact VFR indexing.
        The caller retains raw PTS if sub-frame timing must be preserved.
        """
        _integer(pts, "pts", -_MAX_FRAME, _MAX_FRAME)
        _integer(origin_pts, "origin_pts", -_MAX_FRAME, _MAX_FRAME)
        if type(time_base) is not Fraction or time_base <= 0:
            raise ConfigurationError("time_base must be a positive Fraction")
        _rational(time_base, "time_base")
        return cls.from_seconds(
            (pts - origin_pts) * time_base, rate, rounding=rounding, drop_frame=drop_frame
        )

    @classmethod
    def parse_smpte(cls, label: str, rate: FrameRate) -> FrameTimecode:
        match = (
            re.fullmatch(r"([0-9]{2,16}):([0-5][0-9]):([0-5][0-9])([:;])([0-9]{2,4})", label)
            if type(label) is str
            else None
        )
        if match is None or not isinstance(rate, FrameRate):
            raise ConfigurationError("SMPTE label must be HH:MM:SS:FF or HH:MM:SS;FF with a FrameRate")
        hours, minutes, seconds, separator, frames = match.groups()
        frame = int(frames)
        nominal = rate.nominal
        if frame >= nominal:
            raise ConfigurationError("frame field must be below the nominal frame rate")
        minute = int(hours) * 60 + int(minutes)
        number = (minute * 60 + int(seconds)) * nominal + frame
        drop = separator == ";"
        if drop:
            skipped = rate.drop_count
            if minute % 10 and int(seconds) == 0 and frame < skipped:
                raise ConfigurationError("label names a skipped drop-frame counter")
            number -= skipped * (minute - minute // 10)
        return cls(number, rate, drop)

    def smpte(self, *, wrap_24_hours: bool = False) -> str:
        if type(wrap_24_hours) is not bool:
            raise ConfigurationError("wrap_24_hours must be boolean")
        nominal = self.rate.nominal
        number = self.frame_number
        if self.drop_frame:
            skipped = self.rate.drop_count
            # Count complete ten-minute blocks, then one full and nine short minutes.
            ordinary = nominal * 60
            shortened = ordinary - skipped
            blocks, remaining = divmod(number, ordinary + 9 * shortened)
            if remaining < ordinary:
                minute, within = 0, remaining
            else:
                minute, within = divmod(remaining - ordinary, shortened)
                minute += 1
                within += skipped
            total_minutes = blocks * 10 + minute
            seconds, frame = divmod(within, nominal)
        else:
            total_seconds, frame = divmod(number, nominal)
            total_minutes, seconds = divmod(total_seconds, 60)
        hours, minutes = divmod(total_minutes, 60)
        if wrap_24_hours:
            hours %= 24
        separator = ";" if self.drop_frame else ":"
        width = max(2, len(str(nominal - 1)))
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}{separator}{frame:0{width}d}"

    def timestamp(self, *, precision: int = 3) -> str:
        _integer(precision, "precision", 0, 9)
        scale = 10**precision
        ticks = _round(self.seconds * scale, "nearest")
        whole, fraction = divmod(ticks, scale)
        minutes, seconds = divmod(whole, 60)
        hours, minutes = divmod(minutes, 60)
        suffix = f".{fraction:0{precision}d}" if precision else ""
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}{suffix}"

    def shift_frames(self, count: int) -> FrameTimecode:
        _integer(count, "frame shift", -_MAX_FRAME, _MAX_FRAME)
        return FrameTimecode(self.frame_number + count, self.rate, self.drop_frame)

    def rescale(
        self, rate: FrameRate, *, rounding: Rounding = "nearest", drop_frame: bool = False
    ) -> FrameTimecode:
        return FrameTimecode.from_seconds(self.seconds, rate, rounding=rounding, drop_frame=drop_frame)


__all__ = ["FrameRate", "FrameTimecode", "Rounding"]
