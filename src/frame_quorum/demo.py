"""Generate a self-contained synthetic sequence for demonstrations and tests."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from .errors import ConfigurationError


def create_demo_sequence(directory: str | Path, *, overwrite: bool = False) -> Path:
    """Create 18 deterministic frames with repeats, transitions, and detail."""

    root = Path(directory)
    expected_names = tuple(f"frame_{index * 500:05d}ms.png" for index in range(18))
    if root.is_symlink():
        raise ConfigurationError(f"refusing to use a symbolic link as the demo frame directory: {root}")
    if root.exists() and not root.is_dir():
        raise ConfigurationError(f"demo frame path is not a directory: {root}")
    if root.exists() and any(root.iterdir()):
        if not overwrite:
            raise ConfigurationError(f"demo frame directory is not empty: {root}")
        expected = set(expected_names)
        unexpected = [
            item
            for item in root.iterdir()
            if item.name not in expected or not item.is_file() or item.is_symlink()
        ]
        if unexpected:
            raise ConfigurationError(f"refusing to overwrite a demo directory with unrelated content: {root}")
        for name in expected_names:
            generated_frame = root / name
            if generated_frame.exists():
                generated_frame.unlink()
    root.mkdir(parents=True, exist_ok=True)
    for index in range(18):
        image = _scene(index)
        image.save(root / f"frame_{index * 500:05d}ms.png", optimize=True)
    return root


def _scene(index: int) -> Image.Image:
    width, height = 640, 400
    palette = ["#20355a", "#155d5c", "#7b3f35", "#4a376d"]
    phase = min(3, index // 5)
    image = Image.new("RGB", (width, height), palette[phase])
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 285, width, height), fill="#111923")
    draw.line((0, 334, width, 334), fill="#f2cf57", width=5)
    draw.line((0, 360, width, 360), fill="#f2cf57", width=5)
    offset = (index % 5) * 92
    draw.rounded_rectangle((55 + offset, 245, 170 + offset, 315), radius=18, fill="#e8eef6")
    draw.rectangle((80 + offset, 225, 142 + offset, 270), fill="#58d6a9")
    if phase >= 1:
        for step in range(8):
            x = 30 + step * 82
            draw.ellipse((x, 48 + (step % 2) * 30, x + 34, 82 + (step % 2) * 30), fill="#f9ab55")
    if phase >= 2:
        draw.polygon(((440, 75), (555, 185), (325, 185)), fill="#f45b69")
        draw.rectangle((389, 185, 492, 277), fill="#c7d4e5")
    if phase >= 3:
        for line in range(10):
            draw.line((20, 18 + line * 18, 270, 18 + line * 18), fill="#9fb8d8", width=2)
    draw.text((22, 370), f"synthetic scene {phase + 1} / frame {index:02d}", fill="#ffffff")
    return image
