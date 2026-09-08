"""Verified local video/PCM16 clips on a shared exact source-audio epoch."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from collections.abc import Iterable
from dataclasses import asdict, dataclass, replace
from fractions import Fraction
from pathlib import Path
from tempfile import mkdtemp
from typing import Any

from .errors import ConfigurationError, OutputError, ScanError
from .native_av import NativeAVSplitConfig, _audio_epoch, _AudioBlock, _AudioReader
from .native_splitting import (
    NativeClip,
    NativeSplitResult,
    _ByteBudget,
    _cleanup,
    _Encoder,
    _hash_file,
    _identity,
    _managed_file,
    _publish,
    _target_path,
    _verify,
    _Writer,
)
from .native_video import (
    _INT64,
    NativeVideoConfig,
    NativeVideoFrame,
    NativeVideoStatus,
    NativeVideoStream,
    _fraction,
    _integer,
    _load_av,
    _pair,
)


@dataclass(frozen=True, slots=True)
class NativeAVSplitResult(NativeSplitResult):
    audio_sample_count: int

    def __post_init__(self) -> None:
        NativeSplitResult.__post_init__(self)
        _integer(self.audio_sample_count, "audio_sample_count", self.clip_count, 1 << 36)

    def to_dict(self) -> dict[str, Any]:
        return {**NativeSplitResult.to_dict(self), "audio_sample_count": self.audio_sample_count}


class _AVEncoder(_Encoder):
    def __init__(
        self,
        path: Path,
        first: NativeVideoFrame,
        sound: _AudioBlock,
        epoch: Fraction,
        config: NativeAVSplitConfig,
        budget: _ByteBudget,
        av: Any,
        owned: dict[Path, tuple[int, int]],
    ) -> None:
        denominator = math.lcm(first.time_base.denominator, epoch.denominator)
        _integer(denominator, "AV output tick denominator", 1, 1_000_000_000)
        self.config = config
        self.audio_stream: Any = None
        self.audio_first: Fraction | None = None
        self.audio_samples = 0
        self.audio_hash = hashlib.sha256()
        self.rate, self.channels = sound.rate, sound.channels
        self._packet_time: Fraction | None = None
        self._last_packet_time: Fraction | None = None
        self.peak_packet_bytes = 0
        super().__init__(
            path,
            first,
            budget,
            av,
            owned,
            origin=epoch,
            time_base=Fraction(1, denominator),
            mux_options={"max_interleave_delta": "1", "flush_packets": "1"},
        )
        try:
            self.audio_stream = self.container.add_stream("pcm_s16le", rate=self.rate)
            self.audio_stream.layout = "mono" if self.channels == 1 else "stereo"
            self.audio_stream.time_base = self.audio_stream.codec_context.time_base = Fraction(1, self.rate)
            self.audio_stream.codec_context.thread_count = 1
            self.container.start_encoding()
            if self.audio_stream.time_base != Fraction(1, self.rate):
                raise ScanError("NUT audio clock does not match the source sample lattice")
        except BaseException as error:
            self.close(error)
            raise

    def write(self, frame: NativeVideoFrame) -> None:
        self._packet_time = frame.presentation_time - self.origin
        super().write(frame)

    def write_audio(self, block: _AudioBlock) -> None:
        if (block.rate, block.channels) != (self.rate, self.channels):
            raise ScanError("clip audio format changed")
        if self.audio_first is not None and block.start != self.audio_first + Fraction(
            self.audio_samples, self.rate
        ):
            raise ScanError("selected clip audio is discontinuous")
        time = block.start - self.origin
        ticks = time * self.rate
        if ticks.denominator != 1:
            raise ScanError("clip audio timestamp is not representable on the output sample grid")
        _integer(ticks.numerator, "rebased audio pts", 0, _INT64)
        frame = self.av.AudioFrame(
            format="s16", layout="mono" if self.channels == 1 else "stereo", samples=block.samples
        )
        frame.sample_rate, frame.pts, frame.time_base = self.rate, ticks.numerator, Fraction(1, self.rate)
        frame.planes[0].update(block.pcm)
        self._packet_time = time
        self._mux_packets(self.audio_stream.encode(frame))
        if self.audio_first is None:
            self.audio_first = block.start
        self.audio_hash.update(block.pcm)
        self.audio_samples += block.samples

    def _mux_packets(self, packets: Iterable[Any]) -> None:
        count = 0
        for packet in packets:
            count += 1
            if count != 1 or self._packet_time is None:
                raise ScanError("fixed AV codecs produced delayed or multiple packets")
            size = _integer(packet.size, "encoded AV packet bytes", 1, self.config.max_pending_packet_bytes)
            time_base = _fraction(packet.time_base, "encoded packet time base", positive=True)
            pts = _integer(packet.pts, "encoded packet pts", 0, _INT64)
            dts = _integer(packet.dts, "encoded packet dts", 0, _INT64)
            time = pts * time_base
            if (
                dts != pts
                or time != self._packet_time
                or (self._last_packet_time is not None and time < self._last_packet_time)
            ):
                raise ScanError("fixed AV codec timestamps changed or reordered")
            self.peak_packet_bytes = max(self.peak_packet_bytes, size)
            self.container.mux(packet)
            if self.writer.failed:
                raise OutputError("native AV writer failed during mux")
            self._last_packet_time = time
        if self._packet_time is not None and count != 1:
            raise ScanError("fixed AV codec did not return its immediate packet")

    def finish(self) -> None:
        primary: BaseException | None = None
        try:
            self._packet_time = None
            self._mux_packets(self.stream.encode())
            self._mux_packets(self.audio_stream.encode())
        except BaseException as error:
            primary = error
            raise
        finally:
            self.close(primary)
        if self.writer.failed:
            raise OutputError("native AV writer failed while finishing")


def _verify_audio(encoder: _AVEncoder, remaining: int, remaining_blocks: int) -> dict[str, Any]:
    if remaining < encoder.audio_samples:
        raise OutputError("native AV verification audio sample budget exhausted")
    if remaining_blocks < 1:
        raise OutputError("native AV verification audio block budget exhausted")
    options = replace(
        encoder.config,
        audio_stream=0,
        max_source_bytes=encoder.config.max_output_bytes,
        max_source_audio_samples=remaining,
        max_audio_blocks=remaining_blocks,
    )
    digest = hashlib.sha256()
    samples = 0
    first: Fraction | None = None
    with _AudioReader(encoder.path, options, encoder.av) as reader:
        if (
            len(reader.container.streams) != 2
            or len(reader.container.streams.audio) != 1
            or len(reader.container.streams.video) != 1
        ):
            raise OutputError("native AV output track inventory differs")
        if reader.container.streams.audio[0].codec_context.name != "pcm_s16le":
            raise OutputError("native AV output audio codec differs")
        while (block := reader.next()) is not None:
            if first is None:
                first = block.start
            if samples + block.samples > encoder.audio_samples:
                raise OutputError("native AV output has extra audio samples")
            if block.rate != encoder.rate or block.channels != encoder.channels:
                raise OutputError("native AV output audio format differs")
            digest.update(block.pcm)
            samples += block.samples
    if (
        encoder.audio_first is None
        or first != encoder.audio_first - encoder.origin
        or samples != encoder.audio_samples
        or digest.digest() != encoder.audio_hash.digest()
    ):
        raise OutputError("native AV output PCM bytes, timing or count differs")
    return {
        "sample_rate": encoder.rate,
        "channels": encoder.channels,
        "samples": samples,
        "decoded_blocks": reader.blocks,
        "source_first": _pair(encoder.audio_first),
        "output_first": _pair(first),
        "pcm_s16le_sha256": digest.hexdigest(),
    }


def _clips(value: tuple[NativeClip, ...], maximum: int) -> None:
    if type(value) is not tuple or not 1 <= len(value) <= maximum:
        raise ConfigurationError("clips must be a nonempty bounded tuple")
    for index, clip in enumerate(value):
        if type(clip) is not NativeClip:
            raise ConfigurationError("clips must contain NativeClip values")
        clip.__post_init__()
        if index and clip.start < value[index - 1].end:
            raise ConfigurationError("clips must be sorted and non-overlapping")


def split_native_av(
    source: str | Path,
    output_dir: str | Path,
    clips: tuple[NativeClip, ...],
    config: NativeAVSplitConfig | None = None,
) -> NativeAVSplitResult:
    options = NativeAVSplitConfig() if config is None else config
    if type(options) is not NativeAVSplitConfig:
        raise ConfigurationError("config must be NativeAVSplitConfig")
    options.__post_init__()
    _clips(clips, options.max_clips)
    if sys.byteorder != "little":
        raise ConfigurationError("native AV PCM16 splitting requires a little-endian host")
    target = _target_path(output_dir)
    budget = _ByteBudget(options.max_output_bytes)
    stage: Path | None = None
    stage_identity: tuple[int, int] | None = None
    owned: dict[Path, tuple[int, int]] = {}
    active: _AVEncoder | None = None
    encoders: list[_AVEncoder] = []
    selected_video = selected_audio = 0
    publication_attempted = False
    try:
        av = _load_av()
        av.codec.Codec("ffv1", "w")
        av.codec.Codec("pcm_s16le", "w")
        with (
            _AudioReader(source, options, av) as audio,
            NativeVideoStream(
                source,
                NativeVideoConfig(
                    video_stream=options.video_stream,
                    end=clips[-1].end,
                    max_frames=options.max_decoded_frames,
                    max_decoded_frames=options.max_decoded_frames,
                    max_source_bytes=options.max_source_bytes,
                    max_frame_pixels=options.max_frame_pixels,
                    max_total_pixels=options.max_source_pixels,
                ),
            ) as video,
        ):
            if audio.fingerprint != video._fingerprint:
                raise ScanError("AV readers observed different source identities")
            sound = audio.next()
            frame = next(video, None)
            if sound is None or frame is None or audio.anchor is None:
                raise ScanError("native AV source requires nonempty video and audio")
            stage = Path(mkdtemp(prefix=".frame-quorum-av-split-", dir=target.parent))
            stage_identity = _identity(stage.lstat())
            for ordinal, clip in enumerate(clips):
                while frame is not None and frame.presentation_time < clip.start:
                    frame = next(video, None)
                while sound is not None and sound.end <= clip.start:
                    sound = audio.next()
                part = (
                    None
                    if sound is None
                    else sound.trim(clip.start, clip.end, maximum=options.max_audio_samples - selected_audio)
                )
                if frame is None or frame.presentation_time >= clip.end or part is None:
                    raise ScanError("requested AV clip contains no video frames or audio samples")
                active = _AVEncoder(
                    stage / f"clip-{ordinal:06d}.nut",
                    frame,
                    part,
                    _audio_epoch(clip.start, audio.anchor, audio.rate),
                    options,
                    budget,
                    av,
                    owned,
                )
                encoders.append(active)
                while (frame is not None and frame.presentation_time < clip.end) or part is not None:
                    if part is not None and (
                        frame is None
                        or frame.presentation_time >= clip.end
                        or part.start <= frame.presentation_time
                    ):
                        if selected_audio + part.samples > options.max_audio_samples:
                            raise ScanError("native AV selected audio sample limit exceeded")
                        active.write_audio(part)
                        selected_audio += part.samples
                        if sound is not None and sound.end < clip.end:
                            sound = audio.next()
                            part = (
                                None
                                if sound is None
                                else sound.trim(
                                    clip.start, clip.end, maximum=options.max_audio_samples - selected_audio
                                )
                            )
                        else:
                            part = None
                    else:
                        if frame is None:
                            raise ScanError("native AV frame cursor is unavailable")
                        if selected_video >= options.max_frames:
                            raise ScanError("native AV retained video frame limit exceeded")
                        active.write(frame)
                        selected_video += 1
                        frame = next(video, None)
                active.finish()
                active = None
            if video.diagnostics.status not in (NativeVideoStatus.EOF, NativeVideoStatus.RANGE_END):
                raise ScanError("native AV refuses truncated source video")
            audio_status = "range_end" if sound is not None and sound.end >= clips[-1].end else audio.status
            if audio_status not in ("eof", "range_end"):
                raise ScanError("native AV refuses truncated source audio")
            source_metadata = video.metadata.to_dict()
            audio_source = {
                "stream_index": audio.stream_index,
                "anchor": _pair(audio.anchor),
                "sample_rate": audio.rate,
                "channels": audio.channels,
                "decoded_blocks": audio.blocks,
                "decoded_samples": audio.samples,
                "status": audio_status,
            }
        rows = []
        verification_pixels = verification_samples = verification_blocks = 0
        for ordinal, (encoder, clip) in enumerate(zip(encoders, clips, strict=True)):
            frames, pixels = _verify(encoder, options, options.max_verification_pixels - verification_pixels)
            verification_pixels += pixels
            sound_row = _verify_audio(
                encoder,
                options.max_verification_audio_samples - verification_samples,
                options.max_verification_audio_blocks - verification_blocks,
            )
            verification_samples += encoder.audio_samples
            verification_blocks += sound_row["decoded_blocks"]
            rows.append(
                {
                    "ordinal": ordinal,
                    "file": encoder.path.name,
                    "start": _pair(clip.start),
                    "end": _pair(clip.end),
                    "epoch": _pair(encoder.origin),
                    "video_frames": frames,
                    "audio": sound_row,
                    "byte_size": encoder.writer.size,
                    "sha256": _hash_file(encoder.path, options.max_output_bytes),
                    "peak_packet_bytes": encoder.peak_packet_bytes,
                }
            )
        document = {
            "kind": "frame-quorum-native-av-split",
            "schema_version": 1,
            "source_video": source_metadata,
            "source_audio": audio_source,
            "source_video_diagnostics": video.diagnostics.to_dict(),
            "container": "nut",
            "video_codec": "ffv1",
            "pixel_format": "bgr0",
            "audio_codec": "pcm_s16le",
            "rebasing": "source-audio-grid-floor",
            "clips": rows,
            "frame_count": selected_video,
            "audio_sample_count": selected_audio,
            "verification_pixels": verification_pixels,
            "verification_audio_samples": verification_samples,
            "verification_audio_blocks": verification_blocks,
            "limits": asdict(options),
            "pyav_version": av.__version__,
            "native_libraries": {key: list(value) for key, value in av.library_versions.items()},
        }
        with _managed_file(stage / "manifest.json", owned) as handle:
            writer = _Writer(handle, budget, options.max_manifest_bytes)
            for text in json.JSONEncoder(ensure_ascii=True, allow_nan=False, sort_keys=True).iterencode(
                document
            ):
                writer.write(text.encode("ascii"))
        result = NativeAVSplitResult(
            target,
            len(clips),
            selected_video,
            budget.total,
            _hash_file(stage / "manifest.json", options.max_manifest_bytes),
            selected_audio,
        )
        publication_attempted = True
        _publish(stage, target)
        stage = None
        return result
    except BaseException as primary:
        if publication_attempted:
            primary.add_note(
                f"Publication did not acknowledge success; inspect destination {target}. "
                "A moved destination is never removed by failure cleanup."
            )
        cleanup_primary = primary
        try:
            if active is not None:
                active.close(primary)
        except BaseException as error:
            cleanup_primary = error
        if stage is not None:
            if stage_identity is None:
                cleanup_primary.add_note(f"Staging identity unavailable; inspect {stage}; not removed.")
            else:
                _cleanup(stage, cleanup_primary, owned, stage_identity)
        if cleanup_primary is not primary:
            raise cleanup_primary from primary
        if isinstance(primary, (ConfigurationError, ScanError, OutputError)) or not isinstance(
            primary, Exception
        ):
            raise
        raise OutputError(
            f"native AV publication did not acknowledge success; inspect {target}"
            if publication_attempted
            else "native AV splitting failed before publication"
        ) from primary


__all__ = ["NativeAVSplitResult", "split_native_av"]
