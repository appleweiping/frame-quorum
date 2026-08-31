from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from PIL import Image, ImageDraw


@pytest.fixture
def image_factory(tmp_path: Path) -> Callable[..., Path]:
    counter = 0

    def create(
        name: str | None = None,
        *,
        color: tuple[int, int, int] = (20, 40, 80),
        pattern: int = 0,
        size: tuple[int, int] = (160, 100),
        directory: Path | None = None,
    ) -> Path:
        nonlocal counter
        counter += 1
        destination = directory or tmp_path
        destination.mkdir(parents=True, exist_ok=True)
        path = destination / (name or f"frame_{counter:03d}.png")
        image = Image.new("RGB", size, color)
        draw = ImageDraw.Draw(image)
        if pattern:
            for step in range(pattern):
                x = (step * 29 + pattern * 7) % max(1, size[0] - 18)
                y = (step * 17) % max(1, size[1] - 18)
                draw.rectangle((x, y, x + 16, y + 16), fill=(230, 180 - step % 80, 40 + step % 150))
        image.save(path)
        return path

    return create
