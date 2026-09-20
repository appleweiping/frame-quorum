"""Import edited scene starts into VFR-aware, replay-verified stills and HTML."""

from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

from frame_quorum import NativeSceneImageConfig, export_loaded_scene_overview


def main() -> None:
    import av

    with TemporaryDirectory(prefix="frame-quorum-load-overview-example-") as directory:
        root = Path(directory)
        source, csv_path = root / "colors.mkv", root / "scenes.csv"
        timestamps = (5000, 5040, 5110, 5180, 5270, 5310, 5470, 5680)
        colors = (0, 8, 16, 24, 220, 228, 236, 244)
        with av.open(str(source), "w", format="matroska") as container:
            stream = container.add_stream("ffv1", rate=25)
            stream.width, stream.height, stream.pix_fmt = 8, 6, "bgr0"
            stream.time_base = stream.codec_context.time_base = Fraction(1, 1000)
            for color, timestamp in zip(colors, timestamps, strict=True):
                with Image.new("RGB", (8, 6), (color, color, color)) as image:
                    frame = av.VideoFrame.from_image(image)
                frame.pts, frame.time_base = timestamp, Fraction(1, 1000)
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        with av.open(str(source)) as container:
            decoded = tuple(container.decode(video=0))
            observed = tuple(Fraction(frame.pts) * frame.time_base for frame in decoded)
            observed_colors = []
            for frame in decoded:
                with frame.to_image() as image:
                    observed_colors.append(image.getpixel((0, 0)))
        if observed != tuple(Fraction(value, 1000) for value in timestamps):
            raise RuntimeError("independent decoder did not preserve VFR timestamps")
        if tuple(observed_colors) != tuple((value, value, value) for value in colors):
            raise RuntimeError("independent decoder did not preserve generated pixels")
        csv_path.write_text("Scene Number,Start Frame\n1,1\n2,3\n3,6\n", encoding="utf-8")
        output = export_loaded_scene_overview(
            source,
            csv_path,
            root / "review",
            image_config=NativeSceneImageConfig(images_per_scene=2, sample_margin=0),
        )
        manifest = json.loads((output.output_dir / "manifest.json").read_bytes())
        if [(row["start_position"], row["end_position"]) for row in manifest["scenes"]] != [
            (0, 2),
            (2, 5),
            (5, 8),
        ]:
            raise RuntimeError("imported CSV did not yield expected decoded frame partitions")
        if [row["source_sample_index"] for row in manifest["images"]] != [0, 1, 2, 4, 5, 7]:
            raise RuntimeError("still positions differ from independent scene slot expectations")
        if manifest["scenes"][-1]["end_time"] is not None:
            raise RuntimeError("the last observed frame does not prove the exclusive scene endpoint")
        if output.csv_sha256 != hashlib.sha256(csv_path.read_bytes()).hexdigest():
            raise RuntimeError("published overview did not bind the imported CSV bytes")
        print(json.dumps({"scenes": output.scene_count, "images": output.image_count, "vfr": True}))


if __name__ == "__main__":
    main()
