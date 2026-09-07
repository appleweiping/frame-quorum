"""Regenerate algorithmic benchmark artifacts with explicit runtime provenance."""

from pathlib import Path

from frame_quorum import BenchmarkConfig, ScanConfig, SelectionConfig, scan_frames, source_content_digest
from frame_quorum.benchmark import benchmark_manifest, render_benchmark_svg, run_benchmark
from frame_quorum.reporting import write_json

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "benchmark"

scan_config = ScanConfig()
frames = scan_frames(HERE / "output" / "frames", scan_config)
result = run_benchmark(
    frames,
    SelectionConfig(budget=6),
    BenchmarkConfig(random_seed=1729, random_trials=8),
    scan_config=scan_config,
    source_digest=source_content_digest(frames),
)
write_json(benchmark_manifest(result), OUTPUT / "benchmark.json")
render_benchmark_svg(result, OUTPUT / "benchmark.svg")
