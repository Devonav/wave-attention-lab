# Dimension-sized LDS experiment

The dynamic-LDS candidate is **rejected for promotion**: correctness passed,
but latency changes were mixed and several shapes regressed. The default
kernel retains fixed LDS. Next: test compile-time dimension specialization.

## Reproduction and environment

Measured 2026-09-09 on RX 9070 XT (gfx1201), Arch kernel 7.2.3-arch1-3,
HIP runtime/driver 70253211, AMD clang 22.0.0. Both builds used CMake Release
HIP flags `-O3 -DNDEBUG`. Baseline source is a `git archive` of `74a2c38`;
its CSV revision is `unknown` because the archive has no `.git` directory.
Candidate source is the same kernel plus [candidate.patch](candidate.patch);
its CSV revision is `74a2c38-dirty`. These labels are retained unmodified.

FP32, B=1, H=8, N=128/256/512/1024, D=64/128, both mask modes,
10 warmups and 100 timed iterations. Three trials per shape and version,
48 rows per version. Baseline/candidate ordering alternated between trials.
Every run used `--verify`; maximum absolute error across both versions was
1.34110451e-7. The candidate also passed all 144 GPU boundary test cases.

Stock automatic performance settings; no clock, fan, or power changes were
made. A post-run sample showed edge/junction/memory temperatures 37/45/64 C,
31 W, sclk 30 MHz and mclk 772 MHz (idle). Continuous thermal/clock telemetry
was not captured; timings should not be interpreted as isolated occupancy
measurements or statistically established improvements.

To repeat, build baseline and patched candidate in separate source directories:

```bash
python3 results/lds-sizing/run_experiment.py \
  --baseline /path/to/baseline/wave_bench \
  --candidate /path/to/candidate/wave_bench \
  --output out/lds-repeat
```

The output directory must not already exist. Raw data: [baseline.csv](baseline.csv)
and [candidate.csv](candidate.csv). Each table cell is the median across three
trial medians (or trial p95s); speedup is baseline median / candidate median.

## Results

| Causal | N | D | Baseline ms | Candidate ms | Speedup | Baseline p95 ms | Candidate p95 ms |
| :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| False | 128 | 64 | 0.1520 | 0.2265 | 0.671x | 0.1572 | 0.2318 |
| False | 128 | 128 | 0.2707 | 0.2756 | 0.982x | 0.2797 | 0.2869 |
| False | 256 | 64 | 0.4438 | 0.4029 | 1.102x | 0.4900 | 0.4245 |
| False | 256 | 128 | 0.9746 | 0.9990 | 0.976x | 1.1637 | 1.2825 |
| False | 512 | 64 | 1.9035 | 1.4041 | 1.356x | 2.3829 | 1.5398 |
| False | 512 | 128 | 3.3163 | 3.3714 | 0.984x | 3.8908 | 3.9304 |
| False | 1024 | 64 | 3.6878 | 4.7885 | 0.770x | 10.8271 | 5.8067 |
| False | 1024 | 128 | 10.9918 | 11.2469 | 0.977x | 13.0791 | 13.2256 |
| True | 128 | 64 | 0.1517 | 0.2247 | 0.675x | 0.1531 | 0.2281 |
| True | 128 | 128 | 0.2664 | 0.2725 | 0.978x | 0.2739 | 0.2788 |
| True | 256 | 64 | 0.3696 | 0.3851 | 0.960x | 0.3994 | 0.4068 |
| True | 256 | 128 | 0.5930 | 0.6073 | 0.976x | 0.6360 | 0.6586 |
| True | 512 | 64 | 1.0496 | 0.8539 | 1.229x | 1.1670 | 0.9414 |
| True | 512 | 128 | 1.5144 | 1.5728 | 0.963x | 1.7501 | 1.7969 |
| True | 1024 | 64 | 2.7983 | 2.2211 | 1.260x | 3.6463 | 2.7010 |
| True | 1024 | 128 | 3.7737 | 3.8657 | 0.976x | 5.1803 | 5.3593 |

## Profiler evidence

Non-causal N=1024, 2 warmups and 10 timed iterations; mean duration includes
all 12 dispatches. These captures are separate from the benchmark trials.

| Version | D | Reported LDS bytes | VGPRs | Scratch bytes | Mean kernel ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| [baseline](baseline-d64/redvsblue/32762_kernel_trace.csv) | 64 | 32768 | 176 | 0 | 12.2083 |
| [baseline](baseline-d128/redvsblue/32772_kernel_trace.csv) | 128 | 32768 | 176 | 0 | 14.8569 |
| [candidate](candidate-d64/redvsblue/32783_kernel_trace.csv) | 64 | 0 | 40 | 0 | 6.0889 |
| [candidate](candidate-d128/redvsblue/32794_kernel_trace.csv) | 128 | 0 | 40 | 0 | 15.1329 |

The candidate launch requests 16 KiB at D=64 and 32 KiB at D=128, but these
traces report zero LDS for its dynamic allocation. That field cannot verify
the candidate's actual total allocation. VGPR count changes from 176 to 40,
so code generation also changed substantially. No occupancy counters were
collected; neither reduced register usage nor requested LDS proves higher
occupancy or explains the mixed timing result by itself.
