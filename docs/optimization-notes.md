# Optimization Notes

This document is the performance engineering log. Each change should begin with
a hypothesis and end with measured results plus a correctness check.

## Baseline hypotheses

### H1: Fusion reduces long-sequence memory pressure

The materialized path writes and rereads a quadratic score matrix. The fused
path should use less memory and eventually win as sequence length increases.
The current fused kernel may lose at small sizes because its serial key loop and
block-wide barriers limit occupancy and instruction throughput.

Measure sequence lengths 128, 256, 512, 1024, and 2048 at dimensions 64 and 128.
Record both latency and workspace bytes. Use profiler memory counters to verify
the traffic explanation rather than inferring it from time alone.

### H2: The fused kernel is synchronization bound

It currently performs a full block reduction and several barriers per key.
Collect wave occupancy, VALU utilization, and synchronization-related counters
available on the installed `gfx1201` profiler. Compare wave-sized reductions
against the block-wide baseline.

### H3: Q reuse should move to LDS or registers

Every key iteration rereads Q. Cache the query row once, then measure the effect
on cache misses, memory traffic, occupancy, and latency. LDS use can reduce
occupancy, so the counter evidence matters.

Status: implemented as the separate `fused-cached` variant; target-GPU
measurement is pending.

## Optimization roadmap

| Stage | Change | Status | Evidence required |
| --- | --- | --- | --- |
| 1 | Cache Q in LDS | Implemented, unmeasured | Latency, occupancy, cache counters |
| 2 | Tile K/V through LDS | Planned | Latency, occupancy, cache counters |
| 3 | Wave-level reductions | Planned | Correctness on wave32/wave64, barrier count |
| 4 | Vectorized aligned loads | Planned | Dimension sweep, sanitizer/static checks |
| 5 | FP16/BF16 with FP32 accumulation | Planned | Error distributions, throughput |
| 6 | RDNA WMMA score tiles | Planned | ISA inspection, utilization, speedup |
| 7 | Double-buffered tiles | Planned | Occupancy and stall analysis |
| 8 | Shape-aware autotuning | Planned | Held-out shape results |
| 9 | PyTorch custom operator | Planned | Framework parity and end-to-end latency |

## Experiment rules

- Lock the GPU performance profile when possible and record driver/ROCm versions.
- Run warmups before sampling and use at least 100 timed iterations for small shapes.
- Repeat complete trials, report the median, and retain raw CSV files.
- Verify each optimized variant against the CPU oracle before timing large shapes.
- Change one meaningful variable per experiment.
- Do not claim physical bandwidth from the CLI's logical-byte estimate.

## Result log template

```text
Date:
Commit:
GPU / target:
Driver / ROCm:
Shape and mode:
Hypothesis:
Code change:
Correctness (max absolute / relative error):
Latency before / after:
Profiler evidence:
Conclusion and next experiment:
```
