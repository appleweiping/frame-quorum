from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from fractions import Fraction
from html.parser import HTMLParser
from pathlib import Path

import pytest

from frame_quorum import (
    NativeSceneImageConfig,
    NativeSceneOverviewConfig,
    NativeVideoConfig,
    export_loaded_scene_overview,
)
from frame_quorum.errors import ConfigurationError, OutputError, ScanError


class _Images(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.sources: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "img":
            self.sources.extend(value for name, value in attrs if name == "src" and value is not None)


def _csv(path: Path, starts: tuple[int, ...]) -> None:
    rows = ["Scene Number,Start Frame,Ignored"]
    rows.extend(f'{number},{frame},"<script>secret</script>"' for number, frame in enumerate(starts, 1))
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _video(path: Path) -> None:
    from tests.test_native_scene_images_integration import gray, make_video

    make_video(path, [gray(value) for value in (0, 8, 16, 24, 220, 228, 236, 244)])


def test_loaded_overview_public_api_and_command_exist() -> None:
    from frame_quorum.cli import build_parser

    assert callable(export_loaded_scene_overview)
    args = build_parser().parse_args(
        ["native-load-overview", "video.nut", "--scene-csv", "scenes.csv", "--output-dir", "new"]
    )
    assert args.command == "native-load-overview"


def test_real_vfr_imported_stills_and_html_use_decoded_ordinals_not_detector(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.test_native_scene_images_integration import (
        PTS,
        TIME_BASE,
        assert_image_row,
        gray,
        independently_decode,
        make_video,
    )

    source, csv_path, target = tmp_path / "source.mkv", tmp_path / "scenes.csv", tmp_path / "overview"
    originals = [gray(value) for value in (0, 8, 16, 24, 220, 228, 236, 244)]
    make_video(source, originals)
    _csv(csv_path, (1, 3, 6))
    independent = independently_decode(source)
    assert [row[0] * row[1] for row in independent] == [pts * TIME_BASE for pts in PTS]

    import frame_quorum.native_scene_images as module

    def no_detection(*args: object, **kwargs: object) -> None:
        raise AssertionError("imported scenes must not run a detector")

    monkeypatch.setattr(module, "_capture", no_detection)
    result = export_loaded_scene_overview(
        source, csv_path, target, image_config=NativeSceneImageConfig(images_per_scene=2, sample_margin=0)
    )
    document = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
    overview = json.loads((target / "overview.json").read_text(encoding="utf-8"))
    assert result.scene_count == 3 and result.image_count == 6
    assert result.csv_sha256 == hashlib.sha256(csv_path.read_bytes()).hexdigest()
    assert document["kind"] == "frame-quorum-loaded-scene-images"
    assert document["selection"] == "imported-start-frame-ordinals-v1"
    assert document["detector_performed"] is False
    assert document["source_content_authenticated"] is False
    assert document["capture_diagnostics"]["status"] == "eof"
    assert document["replay_diagnostics"]["status"] == "eof"
    assert [(scene["start_position"], scene["end_position"]) for scene in document["scenes"]] == [
        (0, 2),
        (2, 5),
        (5, 8),
    ]
    assert document["scenes"][-1]["end_time"] is None
    assert [row["source_sample_index"] for row in document["images"]] == [0, 1, 2, 4, 5, 7]
    for row, ordinal in zip(document["images"], (0, 1, 2, 4, 5, 7), strict=True):
        assert_image_row(target, row, independent[ordinal], sample_index=ordinal, decode_index=ordinal)
    html = (target / "index.html").read_text(encoding="utf-8")
    parsed = _Images()
    parsed.feed(html)
    assert parsed.sources == [row["file"] for row in document["images"]]
    assert "Imported cuts" in html and "Unknown endpoint" in html
    assert "<script>secret</script>" not in html
    assert overview["csv_sha256"] == result.csv_sha256
    assert overview["final_scene_endpoint_known"] is False


def test_invalid_csv_fails_before_decoder_or_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import frame_quorum.native_scene_images as module

    source, csv_path, target = tmp_path / "unused.mkv", tmp_path / "scenes.csv", tmp_path / "out"
    csv_path.write_text("Scene Number,Start Frame\n1,1\n2,1\n", encoding="utf-8")

    def unopened(*args: object, **kwargs: object) -> None:
        raise AssertionError("decoder opened before CSV validation")

    monkeypatch.setattr(module, "NativeVideoStream", unopened)
    with pytest.raises(ConfigurationError, match="strictly increasing"):
        export_loaded_scene_overview(source, csv_path, target)
    assert not target.exists()


@pytest.mark.parametrize("starts", [(1, 9), (1, 3, 10)])
def test_imported_cut_outside_eof_never_publishes(tmp_path: Path, starts: tuple[int, ...]) -> None:
    source, csv_path, target = tmp_path / "source.mkv", tmp_path / "scenes.csv", tmp_path / "out"
    _video(source)
    _csv(csv_path, starts)
    with pytest.raises(ConfigurationError, match="outside the decoded video"):
        export_loaded_scene_overview(source, csv_path, target)
    assert not target.exists()


def test_limit_hit_is_not_accepted_as_eof(tmp_path: Path) -> None:
    source, csv_path, target = tmp_path / "source.mkv", tmp_path / "scenes.csv", tmp_path / "out"
    _video(source)
    _csv(csv_path, (1, 3))
    with pytest.raises(ConfigurationError, match="frame limit"):
        export_loaded_scene_overview(
            source,
            csv_path,
            target,
            video_limits=NativeVideoConfig(max_frames=7, max_decoded_frames=9),
        )
    assert not target.exists()


def test_short_scenes_repeat_slots_and_keep_unknown_final_endpoint(tmp_path: Path) -> None:
    source, csv_path, target = tmp_path / "source.mkv", tmp_path / "scenes.csv", tmp_path / "out"
    _video(source)
    _csv(csv_path, (1, 2, 8))
    result = export_loaded_scene_overview(
        source, csv_path, target, image_config=NativeSceneImageConfig(images_per_scene=3, sample_margin=0)
    )
    document = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
    assert result.image_count == 9
    assert [row["source_sample_index"] for row in document["images"]] == [0, 0, 0, 1, 3, 6, 7, 7, 7]
    assert document["scenes"][-1]["end_time"] is None


def test_single_scene_has_no_cut_and_escapes_title(tmp_path: Path) -> None:
    source, csv_path, target = tmp_path / "source.mkv", tmp_path / "scenes.csv", tmp_path / "out"
    _video(source)
    _csv(csv_path, (1,))
    result = export_loaded_scene_overview(
        source,
        csv_path,
        target,
        image_config=NativeSceneImageConfig(images_per_scene=1),
        overview_config=NativeSceneOverviewConfig(title="<unsafe> & review"),
    )
    assert result.scene_count == 1
    html = (target / "index.html").read_text(encoding="utf-8")
    assert "No imported cut" in html
    assert "&lt;unsafe&gt; &amp; review" in html
    assert "<unsafe>" not in html


def test_replay_rgb_mutation_in_unselected_frame_cannot_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import frame_quorum.native_scene_images as module

    source, csv_path, target = tmp_path / "source.mkv", tmp_path / "scenes.csv", tmp_path / "out"
    _video(source)
    _csv(csv_path, (1, 3, 6))
    original = module.NativeVideoStream
    opened = 0

    class ChangedReplay:
        def __init__(self, path: Path, config: NativeVideoConfig) -> None:
            nonlocal opened
            self.pass_index = opened
            opened += 1
            self.inner = original(path, config)

        def __enter__(self) -> ChangedReplay:
            self.inner.__enter__()
            return self

        def __exit__(self, *args: object) -> object:
            return self.inner.__exit__(*args)

        @property
        def metadata(self):
            return self.inner.metadata

        @property
        def diagnostics(self):
            return self.inner.diagnostics

        def __iter__(self):
            for frame in self.inner:
                if self.pass_index == 1 and frame.decode_index == 3:
                    yield replace(frame, rgb=bytes(len(frame.rgb)))
                else:
                    yield frame

    monkeypatch.setattr(module, "NativeVideoStream", ChangedReplay)
    with pytest.raises(ScanError, match="replay sample"):
        export_loaded_scene_overview(
            source, csv_path, target, image_config=NativeSceneImageConfig(images_per_scene=1)
        )
    assert opened == 2 and not target.exists()
    assert not list(tmp_path.glob(".frame-quorum-scene-images-*"))


def test_nonincreasing_pts_and_invalid_window_fail_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import frame_quorum.native_scene_images as module

    source, csv_path, target = tmp_path / "source.mkv", tmp_path / "scenes.csv", tmp_path / "out"
    _video(source)
    _csv(csv_path, (1, 3))
    original = module.NativeVideoStream

    class RepeatedTime:
        def __init__(self, path: Path, config: NativeVideoConfig) -> None:
            self.inner = original(path, config)

        def __enter__(self) -> RepeatedTime:
            self.inner.__enter__()
            return self

        def __exit__(self, *args: object) -> object:
            return self.inner.__exit__(*args)

        @property
        def metadata(self):
            return self.inner.metadata

        @property
        def diagnostics(self):
            return self.inner.diagnostics

        def __iter__(self):
            previous = None
            for frame in self.inner:
                if frame.decode_index == 3:
                    assert previous is not None
                    yield replace(frame, pts=previous.pts, time_base=previous.time_base)
                else:
                    yield frame
                previous = frame

    with pytest.raises(ConfigurationError, match="complete stream-zero"):
        export_loaded_scene_overview(
            source, csv_path, target, video_limits=NativeVideoConfig(start=Fraction(0))
        )
    monkeypatch.setattr(module, "NativeVideoStream", RepeatedTime)
    with pytest.raises(ScanError, match="strictly increase"):
        export_loaded_scene_overview(source, csv_path, target)
    assert not target.exists()


def test_same_length_staged_image_corruption_is_reconciled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import frame_quorum.native_scene_images as module

    source, csv_path, target = tmp_path / "source.mkv", tmp_path / "scenes.csv", tmp_path / "out"
    _video(source)
    _csv(csv_path, (1, 3))
    original = module._reconcile

    def corrupt(stage, identity, owned, expected, budget):
        image = next(stage.glob("scene-*.png"))
        data = bytearray(image.read_bytes())
        data[-1] ^= 1
        image.write_bytes(data)
        return original(stage, identity, owned, expected, budget)

    monkeypatch.setattr(module, "_reconcile", corrupt)
    with pytest.raises(OutputError, match=r"digest|hash|staged"):
        export_loaded_scene_overview(source, csv_path, target)
    assert not target.exists()


def test_existing_target_and_uncertain_publication_are_not_removed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import frame_quorum.native_scene_images as module

    source, csv_path, target = tmp_path / "source.mkv", tmp_path / "scenes.csv", tmp_path / "out"
    _video(source)
    _csv(csv_path, (1, 3))
    target.mkdir()
    keep = target / "keep.txt"
    keep.write_text("foreign", encoding="ascii")
    with pytest.raises(OutputError):
        export_loaded_scene_overview(source, csv_path, target)
    assert keep.read_text(encoding="ascii") == "foreign"
    keep.unlink()
    target.rmdir()
    publish = module._publish

    def lose_ack(stage: Path, output: Path) -> None:
        publish(stage, output)
        raise OSError("lost rename acknowledgment")

    monkeypatch.setattr(module, "_publish", lose_ack)
    with pytest.raises(OutputError, match="inspect"):
        export_loaded_scene_overview(source, csv_path, target)
    assert (target / "index.html").is_file()
    assert (target / "overview.json").is_file()


def test_cli_real_export_and_stdout_failure_does_not_undo_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from frame_quorum import cli

    source, csv_path, target = tmp_path / "source.mkv", tmp_path / "scenes.csv", tmp_path / "out"
    _video(source)
    _csv(csv_path, (1, 3))
    args = [
        "native-load-overview",
        str(source),
        "--scene-csv",
        str(csv_path),
        "--output-dir",
        str(target),
        "--images-per-scene",
        "1",
    ]
    assert cli.main(args) == 0
    reported = json.loads(capsys.readouterr().out)
    assert reported["scene_count"] == 2
    assert reported["detector_performed"] is False
    assert (target / "index.html").is_file()

    another = tmp_path / "stdout-failed"
    args[args.index(str(target))] = str(another)

    class ShortOutput:
        encoding = "utf-8"

        def write(self, text: str) -> int:
            return len(text) - 1

        def flush(self) -> None:
            pytest.fail("short stdout write must fail before flush")

    monkeypatch.setattr(cli.sys, "stdout", ShortOutput())
    assert cli.main(args) == 2
    assert (another / "index.html").is_file()
    assert (another / "overview.json").is_file()
