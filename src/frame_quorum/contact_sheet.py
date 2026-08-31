"""Render selected frames and decision facts into a compact image."""

from __future__ import annotations

import warnings
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps, UnidentifiedImageError

from .errors import ConfigurationError, ScanError
from .metrics import measure_image
from .models import SelectionResult

_BACKGROUND = "#101722"
_CARD = "#1b2736"
_TEXT = "#f5f7fa"
_MUTED = "#a8b7c7"
_ACCENT = "#58d6a9"
_MAX_CANVAS_PIXELS = 50_000_000


def render_contact_sheet(
    result: SelectionResult,
    destination: str | Path,
    *,
    columns: int = 3,
    thumbnail_width: int = 320,
) -> Path:
    """Create a PNG contact sheet using only selected source images."""

    if isinstance(columns, bool) or not isinstance(columns, int) or columns < 1:
        raise ConfigurationError("contact-sheet columns must be at least one")
    if isinstance(thumbnail_width, bool) or not isinstance(thumbnail_width, int) or thumbnail_width < 96:
        raise ConfigurationError("thumbnail width must be at least 96 pixels")
    selected = result.selected_frames
    if not selected:
        raise ConfigurationError("cannot render a contact sheet without selected frames")

    columns = min(columns, len(selected))
    margin, gap, header_height, label_height = 28, 18, 84, 74
    thumb_height = (thumbnail_width * 5 + 4) // 8
    card_width = thumbnail_width
    card_height = thumb_height + label_height
    rows = (len(selected) + columns - 1) // columns
    width = 2 * margin + columns * card_width + (columns - 1) * gap
    height = header_height + margin + rows * card_height + (rows - 1) * gap + margin
    if width * height > _MAX_CANVAS_PIXELS:
        raise ConfigurationError(f"contact sheet would exceed the {_MAX_CANVAS_PIXELS:,}-pixel canvas limit")
    sheet = Image.new("RGB", (width, height), _BACKGROUND)
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    decisions = {decision.index: decision for decision in result.decisions}
    draw.text((margin, 24), "FRAME QUORUM", fill=_ACCENT, font=font)
    draw.text(
        (margin, 47),
        f"{len(selected)} of {len(result.frames)} frames selected  |  budget {result.config.budget}",
        fill=_MUTED,
        font=font,
    )

    for position, frame in enumerate(selected):
        row, column = divmod(position, columns)
        x = margin + column * (card_width + gap)
        y = header_height + row * (card_height + gap)
        draw.rounded_rectangle((x, y, x + card_width, y + card_height), radius=10, fill=_CARD)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(frame.path) as opened:
                    rendered = ImageOps.exif_transpose(opened).convert("RGB")
                    dimensions = rendered.size
                    metrics = measure_image(rendered)
                    byte_size = frame.path.stat().st_size
                    if (
                        dimensions != (frame.width, frame.height)
                        or metrics != frame.metrics
                        or byte_size != frame.byte_size
                    ):
                        raise ScanError(f"selected image changed since scan: {frame.path}")
                    thumb = ImageOps.fit(
                        rendered,
                        (card_width, thumb_height),
                    )
        except (
            Image.DecompressionBombError,
            Image.DecompressionBombWarning,
            UnidentifiedImageError,
            OSError,
            ValueError,
        ) as error:
            raise ScanError(f"cannot decode selected image {frame.path}: {error}") from error
        sheet.paste(thumb, (x, y))
        decision = decisions[frame.index]
        title = f"#{frame.index:03d}  {frame.relative_path}"
        timestamp = "untimed" if frame.timestamp is None else f"t={frame.timestamp:.3f}s"
        score = f"rank {decision.rank}  |  utility {decision.scores.utility:.3f}  |  {timestamp}"
        draw.text((x + 12, y + thumb_height + 13), _truncate(title, card_width, font), fill=_TEXT, font=font)
        draw.text((x + 12, y + thumb_height + 39), score, fill=_MUTED, font=font)

    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, format="PNG", optimize=True)
    return output


def _truncate(text: str, width: int, font: ImageFont.ImageFont | ImageFont.FreeTypeFont) -> str:
    maximum = max(8, width // 7)
    return text if len(text) <= maximum else f"{text[: maximum - 3]}..."
