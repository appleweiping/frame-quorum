"""Generate a tiny CFR source and inspect an offline FCPXML cuts-only export."""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory
from xml.etree import ElementTree

from PIL import Image

from frame_quorum import (
    DetectionConfig,
    FCPXMLExportConfig,
    NativeSceneConfig,
    OTIOMedia,
    detect_native_scenes,
    otio_cuts_from_native,
    write_fcpxml_bundle,
)


def main() -> None:
    import av

    with TemporaryDirectory(prefix="frame-quorum-fcpxml-example-") as directory:
        source = Path(directory) / "neutral.nut"
        with av.open(str(source), "w", format="nut") as container:
            stream = container.add_stream("ffv1", rate=25)
            stream.width, stream.height, stream.pix_fmt = 8, 6, "bgr0"
            for index in range(6):
                with Image.new("RGB", (8, 6), (0, 0, 0) if index < 3 else (255, 255, 255)) as image:
                    frame = av.VideoFrame.from_image(image)
                frame.pts, frame.time_base = index, Fraction(1, 25)
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        with av.open(str(source)) as container:
            observed = [frame.pts * frame.time_base for frame in container.decode(video=0)]
        if observed != [Fraction(index, 25) for index in range(6)]:
            raise RuntimeError("decoded CFR timestamps differ from the generated source")
        analysis = detect_native_scenes(
            source, NativeSceneConfig(detectors=(DetectionConfig(detector="luminance"),))
        )
        cuts = otio_cuts_from_native(analysis, final_end=Fraction(6, 25))
        if [(cut.start, cut.end) for cut in cuts] != [
            (Fraction(0), Fraction(3, 25)),
            (Fraction(3, 25), Fraction(6, 25)),
        ]:
            raise RuntimeError("native scene cuts differ from the independent expected intervals")
        result = write_fcpxml_bundle(
            cuts,
            OTIOMedia(source, Fraction(0), Fraction(0), Fraction(6, 25)),
            Path(directory) / "editor",
            FCPXMLExportConfig(frame_rate=Fraction(25), width=8, height=6),
        )
        root = ElementTree.parse(result.output_dir / "scenes.fcpxml").getroot()
        clips = root.findall("./library/event/project/sequence/spine/asset-clip")
        if [(clip.attrib["start"], clip.attrib["offset"], clip.attrib["duration"]) for clip in clips] != [
            ("0s", "0s", "3/25s"),
            ("3/25s", "3/25s", "3/25s"),
        ]:
            raise RuntimeError("FCPXML did not preserve exact source and record intervals")
        print(json.dumps({"clip_count": result.clip_count, "editor_import_verified": False}))


if __name__ == "__main__":
    main()
