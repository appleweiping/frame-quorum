"""Native encoder QP interchange; RED first against the pre-feature baseline."""

from __future__ import annotations

import hashlib
import json
import runpy
from contextlib import contextmanager
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest
from PIL import Image

import frame_quorum
from frame_quorum import DetectionConfig, NativeSceneConfig, NativeVideoConfig, detect_native_scenes
from frame_quorum.cli import build_parser, main
from frame_quorum.errors import ConfigurationError, OutputError
from frame_quorum.native_qp import write_native_qp_bundle


def test_public_native_qp_writer_exists() -> None:
    assert callable(frame_quorum.write_native_qp_bundle)


def test_native_qp_cli_command_exists() -> None:
    arguments = build_parser().parse_args(["native-qp", "source.nut", "--output-dir", "new"])
    assert arguments.command == "native-qp"


VALUES = (0, 0, 255, 255, 255, 0, 0)


def _video(
    path: Path,
    pts: tuple[int, ...],
    *,
    values: tuple[int, ...] = VALUES,
    time_base: Fraction = Fraction(1, 1000),
) -> None:
    av = pytest.importorskip("av")
    with av.open(str(path), "w", format="nut" if path.suffix == ".nut" else "matroska") as container:
        stream = container.add_stream("ffv1", rate=25)
        stream.width, stream.height, stream.pix_fmt = 8, 6, "bgr0"
        stream.time_base = stream.codec_context.time_base = time_base
        for value, timestamp in zip(values, pts, strict=True):
            with Image.new("RGB", (8, 6), (value, value, value)) as image:
                frame = av.VideoFrame.from_image(image)
            frame.pts, frame.time_base = timestamp, time_base
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def _direct(path: Path) -> list[tuple[int, Fraction, int]]:
    av = pytest.importorskip("av")
    with av.open(str(path)) as container:
        rows = []
        for frame in container.decode(video=0):
            with frame.to_rgb().to_image() as image:
                rows.append((frame.pts, Fraction(frame.time_base), image.getpixel((0, 0))[0]))
        return rows


def _result(path: Path, video: NativeVideoConfig | None = None):
    return detect_native_scenes(
        path,
        NativeSceneConfig(
            video=NativeVideoConfig() if video is None else video,
            detectors=(DetectionConfig(detector="luminance", threshold=0.5),),
        ),
    )


@pytest.mark.parametrize(
    "name,pts",
    [
        ("cfr.mkv", (0, 40, 80, 120, 160, 200, 240)),
        ("vfr.mkv", (5000, 5040, 5110, 5180, 5270, 5310, 5470)),
    ],
)
def test_cfr_vfr_full_source_qp_uses_direct_decoder_ordinals(tmp_path: Path, name, pts) -> None:
    source = tmp_path / name
    _video(source, pts)
    assert _direct(source) == [(pts[i], Fraction(1, 1000), VALUES[i]) for i in range(len(pts))]
    scenes = _result(source)
    assert scenes.cut_positions == (2, 5)
    assert [row.sample.decode_index for row in scenes.statistics] == list(range(len(VALUES)))
    output = write_native_qp_bundle(scenes, tmp_path / "qp")
    data = output.output_path.read_bytes()
    assert data == b"0 I -1\n2 I -1\n5 I -1\n"
    audit = json.loads((output.output_path.parent / "audit.json").read_bytes())
    assert audit == {
        "kind": "frame-quorum-native-qp",
        "schema_version": 1,
        "qp_format": "x264-x265-qpfile-v1",
        "video_stream": 0,
        "frame_count": 7,
        "cut_count": 2,
        "qp_bytes": len(data),
        "qp_sha256": hashlib.sha256(data).hexdigest(),
        "source_content_authenticated": False,
        "encoder_input_verified": False,
    }
    assert output.qp_bytes == len(data)
    assert output.total_output_bytes == len(data) + len(
        (output.output_path.parent / "audit.json").read_bytes()
    )
    assert output.qp_sha256 == audit["qp_sha256"]
    with pytest.raises(OutputError):
        write_native_qp_bundle(scenes, output.output_path.parent)
    assert output.output_path.read_bytes() == data


def test_qp_refuses_sample_window_limit_and_wrong_input_before_staging(tmp_path: Path) -> None:
    source = tmp_path / "source.mkv"
    _video(source, (0, 40, 80, 120, 160, 200, 240))
    cases = (
        _result(source, NativeVideoConfig(frame_step=2)),
        _result(source, NativeVideoConfig(start=Fraction(1, 25))),
        _result(source, NativeVideoConfig(end=Fraction(1, 5))),
        _result(source, NativeVideoConfig(max_frames=3)),
        _result(source, NativeVideoConfig(max_decoded_frames=3)),
    )
    for index, case in enumerate(cases):
        target = tmp_path / f"invalid-{index}"
        with pytest.raises(ConfigurationError):
            write_native_qp_bundle(case, target)
        assert not target.exists()
    with pytest.raises(ConfigurationError):
        write_native_qp_bundle(object(), tmp_path / "wrong")
    assert not (tmp_path / "wrong").exists()


def test_qp_total_byte_budget_rejects_before_staging(tmp_path: Path) -> None:
    source = tmp_path / "source.mkv"
    _video(source, (0, 40, 80, 120, 160, 200, 240))
    scenes = _result(source)
    for limit in (0, 1, 2):
        with pytest.raises((ConfigurationError, OutputError)):
            write_native_qp_bundle(scenes, tmp_path / f"low-{limit}", max_output_bytes=limit)
        assert not (tmp_path / f"low-{limit}").exists()
    output = write_native_qp_bundle(scenes, tmp_path / "exact")
    assert write_native_qp_bundle(scenes, tmp_path / "boundary", max_output_bytes=output.total_output_bytes)
    with pytest.raises(OutputError):
        write_native_qp_bundle(scenes, tmp_path / "one-under", max_output_bytes=output.total_output_bytes - 1)
    assert not (tmp_path / "one-under").exists()


def test_qp_result_revalidation_rejects_forged_decoded_index(tmp_path: Path) -> None:
    source = tmp_path / "source.mkv"
    _video(source, (0, 40, 80, 120, 160, 200, 240))
    scenes = _result(source)
    first = scenes.statistics[0]
    forged = replace(first.sample, decode_index=1)
    object.__setattr__(first, "sample", forged)
    with pytest.raises(ConfigurationError):
        write_native_qp_bundle(scenes, tmp_path / "forged")
    assert not (tmp_path / "forged").exists()


def test_qp_rejects_empty_and_nonzero_generation_result(tmp_path: Path) -> None:
    source = tmp_path / "source.mkv"
    _video(source, (0, 40, 80, 120, 160, 200, 240))
    ordinary = _result(source)
    empty = replace(
        ordinary,
        diagnostics=replace(ordinary.diagnostics, decoded_frames=0, returned_frames=0),
        statistics=(),
        scenes=(),
    )
    with pytest.raises(ConfigurationError, match="every decoded source frame"):
        write_native_qp_bundle(empty, tmp_path / "empty")
    generation = replace(ordinary.diagnostics, generation=1)
    object.__setattr__(ordinary, "diagnostics", generation)
    with pytest.raises(ConfigurationError):
        write_native_qp_bundle(ordinary, tmp_path / "generation")
    assert not (tmp_path / "empty").exists() and not (tmp_path / "generation").exists()


def test_one_frame_emits_only_initial_i_frame(tmp_path: Path) -> None:
    source = tmp_path / "one.mkv"
    _video(source, (0,), values=(0,))
    scenes = _result(source)
    assert scenes.cut_positions == ()
    output = write_native_qp_bundle(scenes, tmp_path / "one-qp")
    assert output.output_path.read_bytes() == b"0 I -1\n"
    assert output.frame_count == 1 and output.cut_count == 0


def test_negative_origin_result_does_not_quantize_pts_into_frame_numbers(tmp_path: Path) -> None:
    # Synthetic result semantics only: FFV1 Matroska/NUT probes normalize a
    # requested negative start to zero, so they are not valid media oracles.
    source = tmp_path / "source.mkv"
    _video(source, (5000, 5040, 5110, 5180, 5270, 5310, 5470))
    positive = _result(source)
    shift_ticks = -6000
    shift_time = Fraction(shift_ticks, 1000)
    shifted = replace(
        positive,
        statistics=tuple(
            replace(row, sample=replace(row.sample, pts=row.sample.pts + shift_ticks))
            for row in positive.statistics
        ),
        scenes=tuple(
            replace(
                scene,
                start_time=scene.start_time + shift_time,
                last_sample_time=scene.last_sample_time + shift_time,
                end_time=None if scene.end_time is None else scene.end_time + shift_time,
            )
            for scene in positive.scenes
        ),
    )
    assert shifted.statistics[0].sample.pts < 0
    assert shifted.cut_positions == (2, 5)
    output = write_native_qp_bundle(shifted, tmp_path / "negative-qp")
    assert output.output_path.read_bytes() == b"0 I -1\n2 I -1\n5 I -1\n"


def test_equal_pts_across_cut_still_uses_decoded_ordinal(tmp_path: Path) -> None:
    # Synthetic result semantics only: FFV1 NUT rejects our repeated-PTS mux.
    source = tmp_path / "source.mkv"
    _video(source, (5000, 5040, 5110, 5180, 5270, 5310, 5470))
    ordinary = _result(source)
    rows = list(ordinary.statistics)
    rows[2] = replace(rows[2], sample=replace(rows[2].sample, pts=rows[1].sample.pts))
    scenes = list(ordinary.scenes)
    scenes[0] = replace(scenes[0], end_time=rows[2].sample.presentation_time)
    scenes[1] = replace(scenes[1], start_time=rows[2].sample.presentation_time)
    repeated = replace(ordinary, statistics=tuple(rows), scenes=tuple(scenes))
    assert repeated.statistics[1].sample.presentation_time == repeated.statistics[2].sample.presentation_time
    output = write_native_qp_bundle(repeated, tmp_path / "equal-qp")
    assert output.output_path.read_bytes() == b"0 I -1\n2 I -1\n5 I -1\n"


def test_cli_publishes_complete_qp_and_rejects_window_before_decoder(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    import frame_quorum.cli as cli

    source = tmp_path / "source.mkv"
    _video(source, (0, 40, 80, 120, 160, 200, 240))
    target = tmp_path / "cli-qp"
    assert main(["native-qp", str(source), "--detectors", "luminance", "-o", str(target)]) == 0
    reported = json.loads(capsys.readouterr().out)
    assert reported["frame_count"] == 7 and reported["cut_count"] == 2
    assert reported["encoder_input_verified"] is False
    assert (target / "scenes.qp").read_bytes() == b"0 I -1\n2 I -1\n5 I -1\n"

    def forbidden(*_args, **_kwargs):
        raise AssertionError("invalid QP config must reject before native decoding")

    monkeypatch.setattr(cli, "detect_native_scenes", forbidden)
    for option in (["--start", "0"], ["--end", "1"], ["--frame-step", "2"], ["--video-stream", "1"]):
        assert main(["native-qp", str(source), "-o", str(tmp_path / "invalid-cli"), *option]) == 2
        assert "QP" in capsys.readouterr().err
    assert not (tmp_path / "invalid-cli").exists()


def test_qp_publication_short_write_and_racing_foreign_target(tmp_path: Path, monkeypatch) -> None:
    import frame_quorum.native_measurements as publication

    source = tmp_path / "source.mkv"
    _video(source, (0, 40, 80, 120, 160, 200, 240))
    scenes = _result(source)
    original_managed = publication._managed_file

    @contextmanager
    def short(path, owned):
        with original_managed(path, owned) as handle:

            class Partial:
                def write(self, chunk):
                    return handle.write(chunk[:-1])

            yield Partial()

    monkeypatch.setattr(publication, "_managed_file", short)
    with pytest.raises(OutputError, match="short write"):
        write_native_qp_bundle(scenes, tmp_path / "short")
    assert not (tmp_path / "short").exists()
    assert not list(tmp_path.glob(".frame-quorum-measurements-*"))
    monkeypatch.setattr(publication, "_managed_file", original_managed)

    original_publish = publication._publish

    def race(stage, target):
        target.mkdir()
        (target / "foreign.txt").write_text("do not replace", encoding="utf-8")
        original_publish(stage, target)

    monkeypatch.setattr(publication, "_publish", race)
    with pytest.raises(OutputError):
        write_native_qp_bundle(scenes, tmp_path / "raced")
    assert (tmp_path / "raced" / "foreign.txt").read_text(encoding="utf-8") == "do not replace"
    assert not list(tmp_path.glob(".frame-quorum-measurements-*"))


@pytest.mark.parametrize("corrupted_name", ["scenes.qp", "audit.json"])
def test_qp_prepublication_reconcile_rejects_same_length_corruption(
    tmp_path: Path, monkeypatch, corrupted_name: str
) -> None:
    import frame_quorum.native_measurements as publication

    source = tmp_path / "source.mkv"
    _video(source, (0, 40, 80, 120, 160, 200, 240))
    scenes = _result(source)
    original = publication._managed_file

    @contextmanager
    def corrupt(path, owned):
        with original(path, owned) as handle:

            class Corrupt:
                def write(self, chunk):
                    if path.name == corrupted_name:
                        chunk = bytes((chunk[0] ^ 1,)) + chunk[1:]
                    return handle.write(chunk)

            yield Corrupt()

    monkeypatch.setattr(publication, "_managed_file", corrupt)
    with pytest.raises(OutputError, match="digest"):
        write_native_qp_bundle(scenes, tmp_path / "corrupt")
    assert not (tmp_path / "corrupt").exists()
    assert not list(tmp_path.glob(".frame-quorum-measurements-*"))


def test_qp_prepublication_reconcile_rejects_silent_short_write(tmp_path: Path, monkeypatch) -> None:
    import frame_quorum.native_measurements as publication

    source = tmp_path / "source.mkv"
    _video(source, (0, 40, 80, 120, 160, 200, 240))
    scenes = _result(source)
    original = publication._managed_file

    @contextmanager
    def silently_shorten(path, owned):
        with original(path, owned) as handle:

            class Shorten:
                def write(self, chunk):
                    if path.name == "scenes.qp":
                        handle.write(chunk[:-1])
                        return len(chunk)
                    return handle.write(chunk)

            yield Shorten()

    monkeypatch.setattr(publication, "_managed_file", silently_shorten)
    with pytest.raises(OutputError, match="size"):
        write_native_qp_bundle(scenes, tmp_path / "shortened")
    assert not (tmp_path / "shortened").exists()
    assert not list(tmp_path.glob(".frame-quorum-measurements-*"))


def test_qp_published_bundle_survives_lost_return(tmp_path: Path, monkeypatch) -> None:
    import frame_quorum.native_measurements as publication

    source = tmp_path / "source.mkv"
    _video(source, (0, 40, 80, 120, 160, 200, 240))
    scenes = _result(source)
    original = publication._publish

    def lose_return(stage, target):
        original(stage, target)
        raise OSError("simulated lost publication return")

    monkeypatch.setattr(publication, "_publish", lose_return)
    with pytest.raises(OutputError) as caught:
        write_native_qp_bundle(scenes, tmp_path / "published")
    assert (tmp_path / "published" / "scenes.qp").read_bytes() == b"0 I -1\n2 I -1\n5 I -1\n"
    assert "inspect" in " ".join(caught.value.__cause__.__notes__)


def test_qp_cli_uncertain_publication_reports_inspect_target(tmp_path: Path, monkeypatch, capsys) -> None:
    import frame_quorum.native_measurements as publication

    source = tmp_path / "source.mkv"
    _video(source, (0, 40, 80, 120, 160, 200, 240))
    target = tmp_path / "qp"
    original = publication._publish

    def lose_return(stage, destination):
        original(stage, destination)
        raise OSError("simulated lost publication return")

    monkeypatch.setattr(publication, "_publish", lose_return)
    assert main(["native-qp", str(source), "--detectors", "luminance", "-o", str(target)]) == 2
    stderr = capsys.readouterr().err
    assert "inspect" in stderr and str(target) in stderr
    assert (target / "scenes.qp").read_bytes() == b"0 I -1\n2 I -1\n5 I -1\n"
    assert not list(tmp_path.glob(".frame-quorum-measurements-*"))


def test_qp_close_failure_never_publishes_partial_bundle(tmp_path: Path, monkeypatch) -> None:
    import frame_quorum.native_measurements as publication

    source = tmp_path / "source.mkv"
    _video(source, (0, 40, 80, 120, 160, 200, 240))
    scenes = _result(source)
    original = publication._managed_file

    @contextmanager
    def bad_close(path, owned):
        with original(path, owned) as handle:
            yield handle
        raise OSError("simulated close failure")

    monkeypatch.setattr(publication, "_managed_file", bad_close)
    with pytest.raises(OutputError, match="publication failed"):
        write_native_qp_bundle(scenes, tmp_path / "not-published")
    assert not (tmp_path / "not-published").exists()
    assert not list(tmp_path.glob(".frame-quorum-measurements-*"))


def test_generated_offline_example_checks_direct_ordinal_oracle(capsys) -> None:
    runpy.run_path(str(Path(__file__).parents[1] / "examples" / "native_qp.py"), run_name="__main__")
    assert json.loads(capsys.readouterr().out) == {
        "frame_count": 7,
        "cut_count": 2,
        "encoder_input_verified": False,
    }


@pytest.mark.parametrize("failure", ["short", "flush"])
def test_qp_cli_stdout_failure_does_not_delete_published_bundle(tmp_path: Path, monkeypatch, failure) -> None:
    import frame_quorum.cli as cli

    source = tmp_path / "source.mkv"
    _video(source, (0, 40, 80, 120, 160, 200, 240))
    target = tmp_path / "qp"
    args = build_parser().parse_args(
        ["native-qp", str(source), "-o", str(target), "--detectors", "luminance"]
    )

    class BrokenOutput:
        def write(self, payload):
            return len(payload) - 1 if failure == "short" else len(payload)

        def flush(self):
            if failure == "flush":
                raise OSError("simulated stdout flush failure")

    monkeypatch.setattr(cli.sys, "stdout", BrokenOutput())
    with pytest.raises(OutputError if failure == "short" else OSError):
        cli._handle_native_qp(args)
    assert (target / "scenes.qp").read_bytes() == b"0 I -1\n2 I -1\n5 I -1\n"
