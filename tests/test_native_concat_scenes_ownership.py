import copy
from dataclasses import replace
from fractions import Fraction

import pytest

import frame_quorum.native_concat_scenes as module
import frame_quorum.native_video as native
from frame_quorum import NativeConcatConfig, NativeConcatSceneConfig, detect_native_concat_scenes
from frame_quorum.errors import ConfigurationError, ScanError
from tests.test_native_concat import decoder as _decoder_fixture

decoder = _decoder_fixture


@pytest.mark.parametrize("stage", ["enter", "callback"])
@pytest.mark.parametrize("resource", ["_frames", "_file"])
@pytest.mark.parametrize("primary_type", [ValueError, KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("cleanup_type", [ValueError, KeyboardInterrupt, SystemExit])
def test_explicit_owner_retains_failed_native_resource_and_selected_control(
    decoder, monkeypatch, stage, resource, primary_type, cleanup_type
):
    primary, cleanup = primary_type("primary"), cleanup_type("cleanup")
    owners, calls, allow = [], [], [False]
    original_stream, original_open = module.NativeConcatStream, native.NativeVideoStream._open

    def capture(*args, **kwargs):
        owner = original_stream(*args, **kwargs)
        owners.append(owner)
        return owner

    class Fault:
        def __init__(self, target):
            self.target = target

        def __getattr__(self, name):
            return getattr(self.target, name)

        def __iter__(self):
            return self

        def __next__(self):
            return next(self.target)

        def close(self):
            calls.append(resource)
            if not allow[0]:
                raise cleanup
            self.target.close()

    def install(child):
        setattr(child, resource, Fault(getattr(child, resource)))

    def opening(child):
        original_open(child)
        if stage == "enter":
            install(child)
            raise primary

    def callback(sample):
        install(owners[0]._child)
        raise primary

    monkeypatch.setattr(module, "NativeConcatStream", capture)
    monkeypatch.setattr(native.NativeVideoStream, "_open", opening)
    with pytest.raises(BaseException) as caught:
        detect_native_concat_scenes(decoder.clips, on_sample=callback)
    selected = caught.value
    expected = cleanup if isinstance(primary, Exception) and not isinstance(cleanup, Exception) else primary
    if not isinstance(expected, Exception):
        assert selected is expected
    elif stage == "callback":
        assert selected is primary
    else:
        assert isinstance(selected, ScanError)
        assert selected.__cause__ is primary
    owner = selected.native_concat_cleanup
    assert owner is owners.pop() and not owners
    for error in (selected, primary, cleanup):
        error.__traceback__ = error.__context__ = error.__cause__ = None
    assert not owner.diagnostics.closed
    assert ("frames.close" if resource == "_frames" else "file.close") in owner.diagnostics.cleanup_errors
    assert all(item.closed for item in decoder.containers)
    if resource == "_frames":
        assert all(reader.handle.closed for reader in decoder.readers)
    acquired = len(decoder.containers)
    allow[0] = True
    owner.close()
    assert owner.diagnostics.closed
    assert len(decoder.containers) == acquired
    assert all(reader.handle.closed for reader in decoder.readers)


@pytest.mark.parametrize("mode", ["probe-only", "empty-playback", "range-end", "intervals-end"])
def test_actual_terminal_child_status_modes(decoder, mode):
    if mode == "probe-only":
        video = NativeConcatConfig(start=Fraction(2, 5))
        expected = ("closed", (1, 1), 0)
    elif mode == "empty-playback":
        decoder.frames = [[], []]
        video = NativeConcatConfig()
        expected = ("eof", (2, 2), 0)
    elif mode == "range-end":
        video = NativeConcatConfig(end=Fraction(1, 10))
        expected = ("range_end", (2, 1), 2)
    else:
        video = NativeConcatConfig()
        expected = ("range_end", (2, 2), 6)
    result = detect_native_concat_scenes(decoder.clips, NativeConcatSceneConfig(video=video))
    diagnostic = result.diagnostics
    assert (
        diagnostic.last_child_status,
        diagnostic.source_activations,
        diagnostic.returned_frames,
    ) == expected


class Hostile:
    def __getattribute__(self, name):
        raise AssertionError("untrusted attribute")

    def __eq__(self, other):
        raise AssertionError("untrusted comparison")

    def __len__(self):
        raise AssertionError("untrusted length")

    def __hash__(self):
        raise AssertionError("untrusted hash")


@pytest.mark.parametrize("slot", ["clips", "offsets", "digest", "path", "start", "end", "video_stream"])
@pytest.mark.parametrize("standalone", [False, True])
def test_alternate_shared_manifest_revalidates_before_user_protocols(decoder, slot, standalone):
    result = copy.deepcopy(detect_native_concat_scenes(decoder.clips))
    timeline = replace(result.timeline)
    scene = result.scenes[0]
    scene = replace(scene, spans=tuple(replace(span, timeline=timeline) for span in scene.spans))
    target = timeline if slot in {"clips", "offsets", "digest"} else timeline.clips[0]
    object.__setattr__(target, slot, Hostile())
    with pytest.raises(ConfigurationError):
        if standalone:
            scene.to_dict()
        else:
            replace(result, scenes=(scene,)).to_dict()


@pytest.mark.parametrize("alternate", [False, True])
def test_one_or_two_timeline_validations_not_per_sample_or_span(decoder, monkeypatch, alternate):
    result = detect_native_concat_scenes(decoder.clips)
    if alternate:
        timeline = replace(result.timeline)
        scene = result.scenes[0]
        scene = replace(scene, spans=tuple(replace(span, timeline=timeline) for span in scene.spans))
        object.__setattr__(result, "scenes", (scene,))
    checked = []
    original = module._timeline

    def counted(value):
        checked.append(value)
        return original(value)

    monkeypatch.setattr(module, "_timeline", counted)
    result.to_dict()
    assert len(checked) == (2 if alternate else 1)
