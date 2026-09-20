"""Fixed, offline HTML presentation of verified native scene image bundles.

The exporter in ``native_scene_images`` owns all media, files and publication.
This module only validates the overview contract and renders bounded chunks from
the already verified scene and image records; it never opens a video or path.
"""

from __future__ import annotations

import base64
import hashlib
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from fractions import Fraction
from html import escape
from pathlib import Path
from typing import Any, Protocol

from .errors import ConfigurationError, OutputError
from .native_scenes import NativeScene
from .native_video import NativeVideoDiagnostics, NativeVideoMetadata, _integer

_STYLE = """
:root { color-scheme: light; font-family: system-ui, sans-serif; line-height: 1.5; }
body { max-width: 90rem; margin: 1.5rem auto; padding: 0 1rem; color: #17212b; background: #fff; }
header, section { margin-bottom: 1.5rem; }
code, time { font-family: ui-monospace, monospace; overflow-wrap: anywhere; }
.notice { border-left: .3rem solid #9a6700; padding: .5rem 1rem; background: #fff8df; }
.table-scroll { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; }
caption { text-align: left; font-weight: bold; margin-bottom: .5rem; }
th, td { border: 1px solid #a5adb5; text-align: left; vertical-align: top; padding: .6rem; }
thead { background: #eef2f5; }
.gallery { display: grid; gap: .65rem; min-width: 14rem; }
.columns-1 { grid-template-columns: repeat(1, minmax(0, 1fr)); }
.columns-2 { grid-template-columns: repeat(2, minmax(0, 1fr)); }
.columns-3 { grid-template-columns: repeat(3, minmax(0, 1fr)); }
.columns-4 { grid-template-columns: repeat(4, minmax(0, 1fr)); }
.columns-5 { grid-template-columns: repeat(5, minmax(0, 1fr)); }
.columns-6 { grid-template-columns: repeat(6, minmax(0, 1fr)); }
figure { margin: 0; min-width: 0; }
img { display: block; max-width: 100%; object-fit: contain; }
figcaption { overflow-wrap: anywhere; font-size: .875rem; }
@media (max-width: 48rem) { .gallery { grid-template-columns: minmax(0, 1fr); } }
""".strip()
_STYLE_HASH = base64.b64encode(hashlib.sha256(_STYLE.encode("utf-8")).digest()).decode("ascii")
_CSP = (
    "default-src 'none'; img-src 'self'; style-src 'sha256-"
    + _STYLE_HASH
    + "'; object-src 'none'; base-uri 'none'; form-action 'none'"
)


class _SceneView(Protocol):
    @property
    def metadata(self) -> NativeVideoMetadata: ...

    @property
    def diagnostics(self) -> NativeVideoDiagnostics: ...

    @property
    def scenes(self) -> tuple[NativeScene, ...]: ...

    @property
    def cut_times(self) -> tuple[Fraction, ...]: ...


@dataclass(frozen=True, slots=True)
class NativeSceneOverviewConfig:
    title: str = "Scene overview"
    columns: int = 3
    image_width: int | None = None
    image_height: int | None = None
    max_html_bytes: int = 8 * 1024 * 1024
    max_overview_bytes: int = 64 * 1024

    def __post_init__(self) -> None:
        if (
            type(self.title) is not str
            or not 1 <= len(self.title) <= 256
            or any(
                not character.isprintable() or 0xD800 <= ord(character) <= 0xDFFF for character in self.title
            )
        ):
            raise ConfigurationError("overview title must be 1-256 printable Unicode scalar characters")
        _integer(self.columns, "columns", 1, 6)
        for name in ("image_width", "image_height"):
            if getattr(self, name) is not None:
                _integer(getattr(self, name), name, 1, 4096)
        _integer(self.max_html_bytes, "max_html_bytes", 1, 8 * 1024 * 1024)
        _integer(self.max_overview_bytes, "max_overview_bytes", 1, 64 * 1024)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class NativeSceneOverviewResult:
    output_dir: Path
    scene_count: int
    image_count: int
    unique_sample_count: int
    total_output_bytes: int
    image_manifest_sha256: str
    html_sha256: str
    overview_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.output_dir, Path):
            raise ConfigurationError("output_dir must be a pathlib.Path")
        _integer(self.scene_count, "scene_count", 1, 1000)
        _integer(self.image_count, "image_count", self.scene_count, 10_000)
        _integer(self.unique_sample_count, "unique_sample_count", self.scene_count, self.image_count)
        _integer(self.total_output_bytes, "total_output_bytes", 1, 1_000_000_000)
        if self.image_count % self.scene_count:
            raise ConfigurationError("each scene must have the same number of image slots")
        if self.image_count // self.scene_count > 100:
            raise ConfigurationError("image count exceeds per-scene ceiling")
        for name in ("image_manifest_sha256", "html_sha256", "overview_sha256"):
            value = getattr(self, name)
            if (
                type(value) is not str
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ConfigurationError(f"{name} must be a SHA-256 hex digest")

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "output_dir": self.output_dir.as_posix()}


def _rational(value: Fraction) -> str:
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def _pair_text(pair: dict[str, int]) -> str:
    if (
        type(pair) is not dict
        or pair.keys() != {"numerator", "denominator"}
        or type(pair["numerator"]) is not int
        or type(pair["denominator"]) is not int
        or pair["denominator"] < 1
    ):
        raise OutputError("overview image time is not an exact rational pair")
    numerator, denominator = pair["numerator"], pair["denominator"]
    return str(numerator) if denominator == 1 else f"{numerator}/{denominator}"


def _chunks(
    scenes: _SceneView,
    images: Sequence[dict[str, Any]],
    image_format: str,
    images_per_scene: int,
    config: NativeSceneOverviewConfig,
    *,
    imported: bool = False,
) -> Iterator[bytes]:
    """Yield escaped UTF-8 HTML without caller-controlled markup or resource URLs."""
    if len(images) != len(scenes.scenes) * images_per_scene:
        raise OutputError("overview image count differs from verified scene slots")
    extension = "png" if image_format == "png" else "jpg"
    names: set[str] = set()
    for scene in scenes.scenes:
        for image_index in range(images_per_scene):
            row = images[scene.ordinal * images_per_scene + image_index]
            name = f"scene-{scene.ordinal + 1:06d}-image-{image_index + 1:06d}.{extension}"
            if (
                row["scene_ordinal"] != scene.ordinal
                or row["image_index"] != image_index
                or row["file"] != name
            ):
                raise OutputError("overview image filename or slot differs from verified image record")
            if (
                type(row["scene_ordinal"]) is not int
                or type(row["image_index"]) is not int
                or any(
                    type(row[key]) is not int
                    for key in ("source_sample_index", "source_pts", "width", "height")
                )
                or row["source_sample_index"] < 0
                or row["width"] < 1
                or row["height"] < 1
            ):
                raise OutputError("overview image record contains invalid numeric provenance")
            _pair_text(row["source_time"])
            if name in names:
                raise OutputError("overview image filename is duplicated")
            names.add(name)

    def emit(markup: str) -> bytes:
        return markup.encode("utf-8")

    title = escape(config.title, quote=True)
    source = escape(scenes.metadata.path.as_posix(), quote=True)
    status = escape(scenes.diagnostics.status.value, quote=True)
    yield emit('<!doctype html>\n<html lang="en"><head><meta charset="utf-8">\n')
    yield emit(f'<meta http-equiv="Content-Security-Policy" content="{escape(_CSP, quote=True)}">\n')
    yield emit('<meta name="viewport" content="width=device-width, initial-scale=1">\n')
    yield emit(f"<title>{title}</title><style>{_STYLE}</style></head><body>\n")
    yield emit(f"<header><h1>{title}</h1><p>Source: <code>{source}</code></p>")
    if imported:
        yield emit(
            f"<p>{len(scenes.scenes)} imported scenes; {len(images)} verified image slots; "
            f"{scenes.diagnostics.returned_frames} decoded frames. Capture status: <code>{status}</code>.</p>"
        )
        yield emit('<p class="notice">Scene boundaries come from the supplied Start Frame CSV. ')
    else:
        yield emit(
            f"<p>{len(scenes.scenes)} observed scenes; {len(images)} verified image slots; "
            f"{scenes.diagnostics.returned_frames} returned samples. "
            f"Capture status: <code>{status}</code>.</p>"
        )
        yield emit('<p class="notice">This report covers returned samples only. ')
    if scenes.scenes[-1].end_time is None:
        yield emit("The final scene endpoint is unknown; no media duration is inferred.")
    else:
        yield emit("The final endpoint follows the requested analysis range, not inferred media duration.")
    yield emit("</p></header>\n")
    yield emit(
        '<section aria-labelledby="cuts"><h2 id="cuts">Imported cuts</h2>'
        if imported
        else '<section aria-labelledby="cuts"><h2 id="cuts">Observed cuts</h2>'
    )
    if scenes.cut_times:
        yield emit("<ol>")
        for cut in scenes.cut_times:
            yield emit(f"<li><time>{_rational(cut)} s</time></li>")
        yield emit("</ol>")
    else:
        yield emit(
            "<p>No imported cut in the decoded frames.</p>"
            if imported
            else "<p>No accepted cut in the observed samples.</p>"
        )
    yield emit("</section>\n")
    yield emit('<section aria-labelledby="scenes"><h2 id="scenes">Scenes and stills</h2>')
    yield emit('<div class="table-scroll"><table>')
    yield emit(
        "<caption>Imported scene intervals and verified still images</caption>"
        if imported
        else "<caption>Observed scene intervals and verified still images</caption>"
    )
    yield emit(
        '<thead><tr><th scope="col">Scene</th><th scope="col">Decoded frame range</th>'
        if imported
        else '<thead><tr><th scope="col">Scene</th><th scope="col">Returned sample range</th>'
    )
    yield emit('<th scope="col">Start</th><th scope="col">Last observed</th>')
    yield emit('<th scope="col">End</th><th scope="col">Verified stills</th></tr></thead><tbody>\n')
    for scene in scenes.scenes:
        yield emit(
            f'<tr><th scope="row">{scene.ordinal + 1}</th>'
            f"<td>{scene.start_position}-{scene.end_position - 1} "
            f"({scene.sample_count} {'frames' if imported else 'samples'})</td>"
        )
        yield emit(f"<td><time>{_rational(scene.start_time)} s</time></td>")
        yield emit(f"<td><time>{_rational(scene.last_sample_time)} s</time></td>")
        if scene.end_time is None:
            yield emit("<td><strong>Unknown endpoint</strong></td>")
        else:
            end = _rational(scene.end_time)
            reason = escape(scene.end_reason, quote=True)
            yield emit(f"<td><time>{end} s</time> ({reason})</td>")
        yield emit(f'<td><div class="gallery columns-{config.columns}">')
        for image_index in range(images_per_scene):
            row = images[scene.ordinal * images_per_scene + image_index]
            name = f"scene-{scene.ordinal + 1:06d}-image-{image_index + 1:06d}.{extension}"
            sample_time = _pair_text(row["source_time"])
            alt = escape(
                f"Scene {scene.ordinal + 1}, still {image_index + 1}, "
                f"{'decoded frame' if imported else 'returned sample'} "
                f"{row['source_sample_index']} at {sample_time} seconds",
                quote=True,
            )
            if config.image_width is None and config.image_height is None:
                dimensions = f' width="{row["width"]}" height="{row["height"]}"'
            else:
                dimensions = (f' width="{config.image_width}"' if config.image_width is not None else "") + (
                    f' height="{config.image_height}"' if config.image_height is not None else ""
                )
            yield emit(f'<figure><img src="{name}" alt="{alt}"{dimensions} loading="lazy">')
            yield emit(
                f"<figcaption>Still {image_index + 1}: "
                f"{'frame' if imported else 'sample'} {row['source_sample_index']}; "
                f"PTS {row['source_pts']}; <time>{sample_time} s</time></figcaption></figure>"
            )
        yield emit("</div></td></tr>\n")
    yield emit("</tbody></table></div></section></body></html>\n")


__all__ = ["NativeSceneOverviewConfig", "NativeSceneOverviewResult"]
