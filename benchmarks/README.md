# Performance protocol

`run_performance.py` measures selector throughput over already measured `Frame` records. It deliberately excludes
filesystem discovery and image decoding, whose cost depends on codecs and source dimensions.

```bash
python benchmarks/run_performance.py --frames 10000 --budget 8 --repeats 5
```

The JSON report records the Python implementation, platform, samples, median, and exact workload. A ceiling is
available for controlled runners:

```bash
python benchmarks/run_performance.py --assert-max-seconds 3.0 --output performance.json
```

Do not compare results across unlike hardware, Python builds, thermal states, or background loads. Establish a
baseline on the target runner and use the ceiling only to catch large regressions. The unit suite separately guards
the same workload's correctness and a deliberately generous small-workload latency ceiling.
