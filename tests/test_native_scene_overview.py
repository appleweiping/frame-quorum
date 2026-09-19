"""Public HTML scene-overview acceptance, including the missing API/CLI RED."""

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from fractions import Fraction
from html.parser import HTMLParser
from pathlib import Path

import pytest
from PIL import Image

import frame_quorum as package
import frame_quorum.native_scene_images as core
from frame_quorum.cli import build_parser
from frame_quorum.errors import ConfigurationError, OutputError
from tests.test_native_scene_images_integration import (
    PTS,
    gray,
    independently_decode,
    make_video,
    scene_config,
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class OverviewParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tags: list[str] = []
        self.images: list[dict[str, str | None]] = []
        self.meta: list[dict[str, str | None]] = []
        self.text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.tags.append(tag)
        if tag == "img":
            self.images.append(dict(attrs))
        if tag == "meta":
            self.meta.append(dict(attrs))

    def handle_data(self, data: str) -> None:
        self.text.append(data)


def test_native_scene_overview_api_is_available():
    assert callable(package.export_native_scene_overview)


def test_native_scene_overview_cli_is_available():
    args = build_parser().parse_args(["native-scene-overview", "source.mkv", "-o", "overview"])
    assert args.command == "native-scene-overview"


@pytest.mark.parametrize(
    "invalid",
    [
        {"title": ""},
        {"title": "x" * 257},
        {"title": "invalid\ud800"},
        {"title": "line\nbreak"},
        {"columns": 0},
        {"columns": 7},
        {"columns": True},
        {"image_width": 0},
        {"image_height": 4097},
        {"max_html_bytes": 0},
        {"max_overview_bytes": 65537},
    ],
)
def test_overview_config_rejects_invalid_values_before_media(invalid):
    with pytest.raises(ConfigurationError):
        package.NativeSceneOverviewConfig(**invalid)


@pytest.mark.parametrize(
    "scene_count,image_count,unique_count,message",
    [
        (2, 3, 2, "same number of image slots"),
        (1, 101, 1, "per-scene ceiling"),
    ],
)
def test_overview_result_preserves_original_per_scene_invariants(
    tmp_path, scene_count, image_count, unique_count, message
):
    with pytest.raises(ConfigurationError, match=message):
        package.NativeSceneOverviewResult(
            tmp_path, scene_count, image_count, unique_count, 123, "a" * 64, "b" * 64, "c" * 64
        )


def test_overview_result_accepts_exact_hundred_images_per_scene(tmp_path):
    result = package.NativeSceneOverviewResult(tmp_path, 2, 200, 2, 123, "a" * 64, "b" * 64, "c" * 64)
    assert result.image_count == 200 and result.scene_count == 2


@pytest.mark.parametrize("image_format", ["png", "jpeg"])
def test_real_vfr_html_binds_original_images_and_escapes_input(tmp_path, image_format):
    source = tmp_path / "source&clip.mkv"
    originals = [gray(value) for value in (0, 8, 16, 24, 220, 228, 236, 244)]
    make_video(source, originals)
    independently_observed = independently_decode(source)
    images = package.NativeSceneImageConfig(
        images_per_scene=2, sample_margin=0, width=8, image_format=image_format, jpeg_quality=89
    )
    scenes = scene_config()
    old = package.export_native_scene_images(source, tmp_path / "old-images", scenes, images)
    title = 'Review <script src="https://invalid.example/x"> & Ω'
    result = package.export_native_scene_overview(
        source,
        tmp_path / "overview",
        scenes,
        images,
        package.NativeSceneOverviewConfig(title=title, columns=2, image_width=60),
    )
    target = result.output_dir
    old_manifest = (old.output_dir / "manifest.json").read_bytes()
    manifest = (target / "manifest.json").read_bytes()
    overview_bytes = (target / "overview.json").read_bytes()
    html_bytes = (target / "index.html").read_bytes()
    document, overview_document = json.loads(manifest), json.loads(overview_bytes)
    assert manifest == old_manifest
    assert result.image_manifest_sha256 == old.manifest_sha256 == sha256(manifest)
    assert result.html_sha256 == sha256(html_bytes)
    assert result.overview_sha256 == sha256(overview_bytes)
    assert result.total_output_bytes == sum(path.stat().st_size for path in target.iterdir())
    assert result.scene_count == 2 and result.image_count == 4 and result.unique_sample_count == 4
    assert overview_document["kind"] == "frame-quorum-native-scene-overview"
    assert overview_document["image_manifest"] == {
        "file": "manifest.json",
        "bytes": len(manifest),
        "sha256": sha256(manifest),
    }
    assert overview_document["html"] == {
        "file": "index.html",
        "bytes": len(html_bytes),
        "sha256": sha256(html_bytes),
    }
    assert {path.name for path in target.iterdir()} == {
        "manifest.json",
        "index.html",
        "overview.json",
        *(row["file"] for row in document["images"]),
    }
    parser = OverviewParser()
    parser.feed(html_bytes.decode("utf-8"))
    parser.close()
    assert not {"script", "iframe", "form", "a"}.intersection(parser.tags)
    assert title in "".join(parser.text)
    assert source.as_posix() in "".join(parser.text)
    assert b"&lt;script" in html_bytes and b"source&amp;clip" in html_bytes
    policies = [row["content"] for row in parser.meta if row.get("http-equiv") == "Content-Security-Policy"]
    assert len(policies) == 1 and "default-src 'none'" in policies[0]
    assert "img-src 'self'" in policies[0] and "style-src 'sha256-" in policies[0]
    assert [row["src"] for row in parser.images] == [row["file"] for row in document["images"]]
    assert all(row["width"] == "60" and "height" not in row and row["alt"] for row in parser.images)
    assert [row["source_sample_index"] for row in document["images"]] == [0, 3, 4, 7]
    assert "Unknown endpoint" in "".join(parser.text)
    assert "5 s" in "".join(parser.text) and "142/25 s" in "".join(parser.text)
    for row in document["images"]:
        position = row["source_sample_index"]
        assert row["source_pts"] == PTS[position]
        assert row["source_rgb_sha256"] == sha256(independently_observed[position][3])
        new_image = target / row["file"]
        old_image = old.output_dir / row["file"]
        assert new_image.read_bytes() == old_image.read_bytes()
        with Image.open(new_image) as decoded:
            decoded.load()
            assert decoded.format == image_format.upper()
            assert decoded.size == (8, 6)
            assert row["decoded_rgb_sha256"] == sha256(decoded.tobytes())


@pytest.mark.parametrize(
    "video,status,tail",
    [
        (package.NativeVideoConfig(max_frames=2), "frame_limit", "Unknown endpoint"),
        (package.NativeVideoConfig(end=Fraction(26, 5)), "range_end", "26/5 s"),
    ],
)
def test_real_limited_or_known_tail_is_explicit(tmp_path, video, status, tail):
    source = tmp_path / "source.mkv"
    make_video(source, [gray(value) for value in range(8)])
    result = package.export_native_scene_overview(
        source, tmp_path / "overview", scene_config(video), package.NativeSceneImageConfig(images_per_scene=1)
    )
    assert json.loads((result.output_dir / "overview.json").read_bytes())["capture_status"] == status
    html = (result.output_dir / "index.html").read_text(encoding="utf-8")
    assert tail in html and status in html


def test_html_exact_byte_limit_and_one_below_cleanup(tmp_path):
    source = tmp_path / "source.mkv"
    make_video(source, [gray(value) for value in range(8)])
    baseline = package.export_native_scene_overview(source, tmp_path / "baseline")
    exact = (baseline.output_dir / "index.html").stat().st_size
    accepted = package.export_native_scene_overview(
        source, tmp_path / "exact", overview_config=package.NativeSceneOverviewConfig(max_html_bytes=exact)
    )
    assert (accepted.output_dir / "index.html").stat().st_size == exact
    with pytest.raises(OutputError, match="byte limit"):
        package.export_native_scene_overview(
            source,
            tmp_path / "one-below",
            overview_config=package.NativeSceneOverviewConfig(max_html_bytes=exact - 1),
        )
    assert not (tmp_path / "one-below").exists()
    assert not list(tmp_path.glob(".frame-quorum-scene-images-*"))


def test_aggregate_budget_covers_images_manifest_html_and_overview(tmp_path):
    source = tmp_path / "source.mkv"
    make_video(source, [gray(value) for value in range(8)])
    baseline = package.export_native_scene_overview(source, tmp_path / "baseline")
    target = baseline.output_dir
    manifest = json.loads((target / "manifest.json").read_bytes())
    overview = json.loads((target / "overview.json").read_bytes())
    image_bytes = sum(row["bytes"] for row in manifest["images"])
    html_bytes = (target / "index.html").stat().st_size
    limit = 1
    for _ in range(10):
        manifest["image_config"]["max_output_bytes"] = limit
        encoded_manifest = json.dumps(manifest, ensure_ascii=True, allow_nan=False, sort_keys=True).encode()
        overview["image_manifest"]["bytes"] = len(encoded_manifest)
        overview["image_manifest"]["sha256"] = sha256(encoded_manifest)
        encoded_overview = json.dumps(overview, ensure_ascii=True, allow_nan=False, sort_keys=True).encode()
        expected = image_bytes + len(encoded_manifest) + html_bytes + len(encoded_overview)
        if expected == limit:
            break
        limit = expected
    else:
        pytest.fail("independent aggregate JSON-length calculation did not converge")
    exact = package.export_native_scene_overview(
        source, tmp_path / "exact", image_config=package.NativeSceneImageConfig(max_output_bytes=limit)
    )
    assert exact.total_output_bytes == limit
    with pytest.raises(OutputError, match="byte limit"):
        package.export_native_scene_overview(
            source,
            tmp_path / "one-below",
            image_config=package.NativeSceneImageConfig(max_output_bytes=limit - 1),
        )
    assert not (tmp_path / "one-below").exists()


@pytest.mark.parametrize("error", [OSError("render failed"), KeyboardInterrupt("stop")])
def test_render_failure_keeps_original_control_and_removes_owned_stage(tmp_path, monkeypatch, error):
    source = tmp_path / "source.mkv"
    make_video(source, [gray(value) for value in range(8)])

    def fail(*_args):
        yield b"partially generated"
        raise error

    monkeypatch.setattr(core, "_chunks", fail)
    with pytest.raises(OutputError if isinstance(error, Exception) else type(error)):
        package.export_native_scene_overview(source, tmp_path / "overview")
    assert list(tmp_path.iterdir()) == [source]


def test_html_short_write_is_rejected_before_publication(tmp_path, monkeypatch):
    source = tmp_path / "source.mkv"
    make_video(source, [gray(value) for value in range(8)])
    managed = core._managed_file

    class ShortHandle:
        def __init__(self, handle):
            self.handle = handle

        def __getattr__(self, name):
            return getattr(self.handle, name)

        def write(self, data):
            return self.handle.write(data[:-1])

    @contextmanager
    def short_html(path, owned=None):
        with managed(path, owned) as handle:
            yield ShortHandle(handle) if path.name == "index.html" and owned is not None else handle

    monkeypatch.setattr(core, "_managed_file", short_html)
    with pytest.raises(OutputError, match="short output write"):
        package.export_native_scene_overview(source, tmp_path / "overview")
    assert list(tmp_path.iterdir()) == [source]


@pytest.mark.parametrize(
    "field,value",
    [
        ("file", "../outside.png"),
        ("width", '1" onerror="alert(1)'),
        ("source_time", {"numerator": "<script>", "denominator": 1}),
    ],
)
def test_untrusted_image_record_cannot_inject_html_or_external_image(tmp_path, monkeypatch, field, value):
    source = tmp_path / "source.mkv"
    make_video(source, [gray(value) for value in range(8)])
    save = core._save_slot

    def poison(*args):
        row = save(*args)
        row[field] = value
        return row

    monkeypatch.setattr(core, "_save_slot", poison)
    with pytest.raises(OutputError, match=r"filename or slot|numeric provenance|exact rational"):
        package.export_native_scene_overview(source, tmp_path / "overview")
    assert list(tmp_path.iterdir()) == [source]


@pytest.mark.parametrize("name", ["index.html", "overview.json"])
def test_reconciliation_rejects_substituted_overview_file(tmp_path, monkeypatch, name):
    source = tmp_path / "source.mkv"
    make_video(source, [gray(value) for value in range(8)])
    reconcile = core._reconcile

    def substitute(stage, identity, owned, expected, budget):
        path = stage / name
        path.rename(tmp_path / f"moved-{name}")
        path.write_bytes((tmp_path / f"moved-{name}").read_bytes())
        reconcile(stage, identity, owned, expected, budget)

    monkeypatch.setattr(core, "_reconcile", substitute)
    with pytest.raises(OutputError, match="cleanup incomplete"):
        package.export_native_scene_overview(source, tmp_path / "overview")
    assert not (tmp_path / "overview").exists()
    assert (tmp_path / f"moved-{name}").exists()


def test_preexisting_target_and_lost_publication_ack_do_not_delete_data(tmp_path, monkeypatch):
    source = tmp_path / "source.mkv"
    make_video(source, [gray(value) for value in range(8)])
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "foreign").write_bytes(b"keep")
    with pytest.raises(OutputError, match="new output"):
        package.export_native_scene_overview(source, occupied)
    assert (occupied / "foreign").read_bytes() == b"keep"
    publish = core._publish

    def lose_ack(stage, target):
        publish(stage, target)
        raise OSError("lost acknowledgment")

    monkeypatch.setattr(core, "_publish", lose_ack)
    target = tmp_path / "overview"
    with pytest.raises(OutputError, match="acknowledge") as caught:
        package.export_native_scene_overview(source, target)
    assert "never removed" in " ".join(caught.value.__cause__.__notes__)
    assert {"manifest.json", "index.html", "overview.json"}.issubset(path.name for path in target.iterdir())


def test_cli_maps_distinct_display_and_encoded_dimensions_once(tmp_path, monkeypatch, capsys):
    import frame_quorum.cli as cli

    calls = []
    expected = package.NativeSceneOverviewResult(
        tmp_path / "overview", 2, 4, 4, 100, "a" * 64, "b" * 64, "c" * 64
    )

    def export(*args):
        calls.append(args)
        return expected

    monkeypatch.setattr(cli, "export_native_scene_overview", export)
    assert (
        cli.main(
            [
                "native-scene-overview",
                "source.mkv",
                "-o",
                "overview",
                "--width",
                "8",
                "--image-width",
                "60",
                "--image-height",
                "45",
                "--title",
                "A < B",
                "--columns",
                "2",
            ]
        )
        == 0
    )
    assert len(calls) == 1
    source, output, scenes, images, overview = calls[0]
    assert source == Path("source.mkv") and output == Path("overview")
    assert tuple(item.detector for item in scenes.detectors) == ("adaptive",)
    assert images.width == 8 and images.height is None
    assert (overview.image_width, overview.image_height, overview.columns, overview.title) == (
        60,
        45,
        2,
        "A < B",
    )
    assert (
        capsys.readouterr().out
        == json.dumps(expected.to_dict(), ensure_ascii=True, allow_nan=False, sort_keys=True) + "\n"
    )


def test_cli_real_export_and_output_error_keeps_published_bundle(tmp_path, monkeypatch, capsys):
    import frame_quorum.cli as cli

    source = tmp_path / "source.mkv"
    make_video(source, [gray(value) for value in range(8)])
    target = tmp_path / "overview"
    assert cli.main(["native-scene-overview", str(source), "-o", str(target), "--images-per-scene", "1"]) == 0
    reported = json.loads(capsys.readouterr().out)
    assert reported["html_sha256"] == sha256((target / "index.html").read_bytes())
    real = cli.export_native_scene_overview

    class ShortStdout:
        def write(self, text):
            return len(text) - 1

        def flush(self):
            pytest.fail("short write must fail before flush")

    monkeypatch.setattr(cli, "export_native_scene_overview", real)
    with monkeypatch.context() as patch:
        patch.setattr(cli.sys, "stdout", ShortStdout())
        assert cli.main(["native-scene-overview", str(source), "-o", str(tmp_path / "another")]) == 2
    assert (tmp_path / "another" / "index.html").is_file()
