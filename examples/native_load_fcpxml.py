"""Generate CFR media, import scene-list CSV, and verify an editor timeline."""

from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory
from xml.etree import ElementTree

from PIL import Image

from frame_quorum import write_loaded_fcpxml_bundle

_CSV = (
    b"Scene Number,Start Frame,Start Timecode,Start Time (seconds),End Frame,"
    b"End Timecode,End Time (seconds),Length (frames),Length (timecode),Length (seconds)\n"
    b"1,1,00:00:00.000,0.000,2,00:00:00.080,0.080,2,00:00:00.080,0.080\n"
    b"2,3,00:00:00.080,0.080,5,00:00:00.200,0.200,3,00:00:00.120,0.120\n"
    b"3,6,00:00:00.200,0.200,7,00:00:00.280,0.280,2,00:00:00.080,0.080\n"
)


def main() -> None:
    import av

    with TemporaryDirectory(prefix="frame-quorum-load-fcpxml-example-") as directory:
        root = Path(directory)
        source = root / "colors.nut"
        colors = (
            (0, 0, 0),
            (20, 20, 20),
            (80, 80, 80),
            (100, 100, 100),
            (120, 120, 120),
            (200, 200, 200),
            (220, 220, 220),
        )
        with av.open(str(source), "w", format="nut") as container:
            stream = container.add_stream("ffv1", rate=25)
            stream.width, stream.height, stream.pix_fmt = 8, 6, "bgr0"
            for ordinal, color in enumerate(colors):
                with Image.new("RGB", (8, 6), color) as image:
                    frame = av.VideoFrame.from_image(image)
                frame.pts, frame.time_base = ordinal, Fraction(1, 25)
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
                    observed_colors.append(image.convert("RGB").getpixel((0, 0)))
        if observed != tuple(Fraction(index, 25) for index in range(7)):
            raise RuntimeError("independent decoder did not preserve generated CFR timestamps")
        if tuple(observed_colors) != colors:
            raise RuntimeError("independent decoder did not preserve generated frame colors")
        csv_path = root / "scenes.csv"
        csv_path.write_bytes(_CSV)
        output = write_loaded_fcpxml_bundle(
            source,
            csv_path,
            root / "editor",
            frame_rate=Fraction(25),
            final_end=Fraction(7, 25),
        )
        xml = ElementTree.parse(output.output_dir / "scenes.fcpxml").getroot()
        clips = xml.findall("./library/event/project/sequence/spine/asset-clip")
        intervals = [(clip.attrib["start"], clip.attrib["offset"], clip.attrib["duration"]) for clip in clips]
        if intervals != [
            ("0s", "0s", "2/25s"),
            ("2/25s", "2/25s", "3/25s"),
            ("1/5s", "1/5s", "2/25s"),
        ]:
            raise RuntimeError("loaded CSV cuts did not produce exact FCPXML intervals")
        audit = json.loads((output.output_dir / "audit.json").read_bytes())
        if audit["csv_sha256"] != hashlib.sha256(_CSV).hexdigest() or audit["editor_import_verified"]:
            raise RuntimeError("CSV audit does not bind the generated input or overclaims editor import")
        print(
            json.dumps(
                {
                    "scene_count": output.scene_count,
                    "frame_count": output.frame_count,
                    "editor_import_verified": False,
                }
            )
        )


if __name__ == "__main__":
    main()
