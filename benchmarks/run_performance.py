"""Run a local, machine-readable selector throughput benchmark.

This script uses precomputed synthetic records, so it measures selection and
report construction rather than image decoder performance. Results are
machine-specific and are intentionally not checked into the repository.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

from frame_quorum import Frame, SelectionConfig, select_frames
from frame_quorum.models import FrameMetrics


def _sequence(frame_count: int) -> tuple[Frame, ...]:
    frames = []
    for index in range(frame_count):
        fraction = (index % 101) / 100
        metrics = FrameMetrics(
            perceptual_hash=(index * 0x9E3779B97F4A7C15) & ((1 << 64) - 1),
            luminance=fraction,
            entropy=(index % 97) / 96,
            sharpness=(index % 89) / 88,
            colorfulness=(index % 83) / 82,
            mean_red=fraction,
            mean_green=(index % 79) / 78,
            mean_blue=(index % 73) / 72,
        )
        frames.append(
            Frame(
                index=index,
                path=Path(f"frame_{index:09d}.png"),
                relative_path=f"frame_{index:09d}.png",
                timestamp=float(index) / 30,
                width=1920,
                height=1080,
                byte_size=1_000_000,
                metrics=metrics,
            )
        )
    return tuple(frames)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=10_000)
    parser.add_argument("--budget", type=int, default=8)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--assert-max-seconds",
        type=float,
        help="return status 1 when the median exceeds this machine-specific ceiling",
    )
    args = parser.parse_args(argv)
    if args.frames < 1 or args.frames > 1_000_000:
        parser.error("--frames must be between 1 and 1000000")
    if args.budget < 1 or args.budget > args.frames:
        parser.error("--budget must be between 1 and --frames")
    if args.warmups < 0 or args.warmups > 100:
        parser.error("--warmups must be between 0 and 100")
    if args.repeats < 1 or args.repeats > 100:
        parser.error("--repeats must be between 1 and 100")
    if args.assert_max_seconds is not None and args.assert_max_seconds <= 0:
        parser.error("--assert-max-seconds must be greater than zero")

    frames = _sequence(args.frames)
    config = SelectionConfig(budget=args.budget, duplicate_threshold=0.0)
    for _ in range(args.warmups):
        select_frames(frames, config)
    samples = []
    selected_count = 0
    for _ in range(args.repeats):
        started = time.perf_counter()
        selected_count = len(select_frames(frames, config).selected_indices)
        samples.append(time.perf_counter() - started)
    median = statistics.median(samples)
    report = {
        "schema_version": "1.0",
        "kind": "frame-quorum-performance",
        "environment": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
        },
        "protocol": {
            "frame_count": args.frames,
            "budget": args.budget,
            "warmups": args.warmups,
            "repeats": args.repeats,
            "scope": "selection over precomputed synthetic Frame records",
        },
        "result": {
            "selected_count": selected_count,
            "seconds": [round(sample, 9) for sample in samples],
            "median_seconds": round(median, 9),
            "minimum_seconds": round(min(samples), 9),
            "maximum_seconds": round(max(samples), 9),
            "median_frames_per_second": round(args.frames / median, 3),
        },
    }
    rendered = json.dumps(report, indent=2, ensure_ascii=True, allow_nan=False) + "\n"
    if args.output is None:
        sys.stdout.write(rendered)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8", newline="\n")
    return int(args.assert_max_seconds is not None and median > args.assert_max_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
