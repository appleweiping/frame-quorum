"""Generate a VFR source and independently inspect an offline HTML scene bundle.

Run with or without ``python -O`` after installing ``frame-quorum[video]``.
Only this example's temporary directory is removed when it exits.
"""

from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from html.parser import HTMLParser
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

from frame_quorum import (
    DetectionConfig,
    NativeSceneConfig,
    NativeSceneImageConfig,
    NativeSceneOverviewConfig,
    export_native_scene_overview,
)


class _Images(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.sources: list[str] = []
        self.tags: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.tags.append(tag)
        if tag == "img":
            source = dict(attrs).get("src")
            if source is None:
                raise RuntimeError("overview image has no source")
            self.sources.append(source)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    import av

    times = (5000, 5040, 5110, 5180, 5270, 5310, 5470, 5680)
    size = (16, 12)
    originals = tuple(
        bytes(
            (180 if index >= 4 else 0) + index + (x + y) % 16
            for y in range(size[1])
            for x in range(size[0])
            for _channel in range(3)
        )
        for index in range(len(times))
    )
    with TemporaryDirectory(prefix="frame-quorum-overview-") as directory:
        source = Path(directory) / "generated.nut"
        with av.open(str(source), "w", format="nut") as container:
            stream = container.add_stream("ffv1", rate=25)
            stream.width, stream.height, stream.pix_fmt = *size, "bgr0"
            stream.time_base = stream.codec_context.time_base = Fraction(1, 1000)
            for pts, pixels in zip(times, originals, strict=True):
                with Image.frombytes("RGB", size, pixels) as image:
                    frame = av.VideoFrame.from_image(image)
                frame.pts, frame.time_base = pts, Fraction(1, 1000)
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)

        result = export_native_scene_overview(
            source,
            Path(directory) / "overview",
            NativeSceneConfig(detectors=(DetectionConfig(detector="luminance", threshold=0.5),)),
            NativeSceneImageConfig(images_per_scene=2, sample_margin=0, width=8, interpolation="nearest"),
            NativeSceneOverviewConfig(title="Generated VFR scenes", columns=2, image_width=120),
        )
        target = result.output_dir
        manifest_bytes = (target / "manifest.json").read_bytes()
        html_bytes = (target / "index.html").read_bytes()
        overview_bytes = (target / "overview.json").read_bytes()
        manifest = json.loads(manifest_bytes)
        overview = json.loads(overview_bytes)
        parser = _Images()
        parser.feed(html_bytes.decode("utf-8"))
        parser.close()
        positions = (0, 3, 4, 7)
        rows = manifest["images"]
        _require((result.scene_count, result.image_count) == (2, 4), "unexpected scene/image count")
        _require(
            [row["source_sample_index"] for row in rows] == list(positions), "incorrect sample selection"
        )
        _require(
            parser.sources == [row["file"] for row in rows], "HTML image links differ from verified slots"
        )
        _require("script" not in parser.tags and "iframe" not in parser.tags, "unsafe HTML element")
        _require(manifest["scenes"][-1]["end_time"] is None, "unknown final endpoint was invented")
        _require(overview["capture_status"] == "eof", "unexpected decode completion")
        _require(
            overview["image_manifest"]["sha256"] == hashlib.sha256(manifest_bytes).hexdigest(),
            "image binding",
        )
        _require(overview["html"]["sha256"] == hashlib.sha256(html_bytes).hexdigest(), "HTML binding")
        _require(result.overview_sha256 == hashlib.sha256(overview_bytes).hexdigest(), "overview binding")
        _require(
            result.total_output_bytes == sum(path.stat().st_size for path in target.iterdir()), "byte count"
        )
        for row, position in zip(rows, positions, strict=True):
            _require(
                row["source_rgb_sha256"] == hashlib.sha256(originals[position]).hexdigest(), "source pixels"
            )
            _require(Fraction(**row["source_time"]) == Fraction(times[position], 1000), "source time")
            path = target / row["file"]
            _require(row["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest(), "encoded image")
            expected = bytes(
                originals[position][((2 * y + 1) * size[0] + (2 * x + 1)) * 3 + channel]
                for y in range(6)
                for x in range(8)
                for channel in range(3)
            )
            with Image.open(path) as image:
                _require(image.format == "PNG" and image.size == (8, 6), "image type or size")
                _require(image.tobytes() == expected, "independent pixel-center oracle")
        print(
            json.dumps(
                {
                    "scenes": 2,
                    "images": 4,
                    "html_links_verified": True,
                    "native_times_verified": True,
                    "exact_pixels_verified": True,
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
