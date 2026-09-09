# Results

| Report | Card | Precision | Kernel | Commit |
| --- | --- | --- | --- | --- |
| [rx9070xt-fp32.md](rx9070xt-fp32.md) | RX 9070 XT (`gfx1201`) | FP32 | `7.2.3-arch1-3` | `20d8093` |

The headline finding is that the fused path is 1.84x to 6.37x *slower* than the
materialized baseline at every measured shape, and that caching Q in LDS changes
almost nothing. Both are explained by profiler data in the report.

The publication process below is what those numbers went through.

```mermaid
flowchart LR
    ENV[Install supported ROCm environment] --> BUILD[Compile for gfx1201]
    BUILD --> CHECK[Verify every small shape]
    CHECK --> SWEEP[Run three complete sweeps]
    SWEEP --> PROFILE[Capture trace and profiler counters]
    PROFILE --> REVIEW[Review raw data and explanations]
    REVIEW --> PUBLISH[Commit CSV, report, and charts]
```

A result is publishable only when it includes:

- Raw CSV data from all custom variants and PyTorch SDPA
- At least three complete trials for the reported matrix
- Maximum absolute and relative correctness errors
- Commit SHA, driver, ROCm runtime, architecture, and test configuration
- Median and p95 latency rather than a selected best iteration
- A profiler-backed explanation of the main bottleneck
- Explicit disclosure of clock, power, and thermal settings

The first report compares `materialized`, `fused`, and `fused-cached` at sequence
lengths 128, 256, 512, and 1024 with dimensions 64 and 128. Larger shapes follow
after correctness and memory behavior are understood.

