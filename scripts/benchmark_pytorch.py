#!/usr/bin/env python3
"""Benchmark PyTorch SDPA on ROCm using the Wave Attention CSV schema."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path


FIELDS = (
    "timestamp_utc",
    "git_sha",
    "device",
    "architecture",
    "runtime_version",
    "driver_version",
    "backend",
    "algorithm",
    "dtype",
    "batch",
    "heads",
    "sequence",
    "dimension",
    "causal",
    "seed",
    "warmup",
    "iterations",
    "average_ms",
    "median_ms",
    "p95_ms",
    "min_ms",
    "max_ms",
    "tflops",
    "logical_bandwidth_gbs",
    "workspace_bytes",
    "max_abs_error",
    "max_rel_error",
)


def parse_int_list(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("expected a comma-separated list of positive integers")
    return values


def percentile_95(samples: list[float]) -> float:
    ordered = sorted(samples)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def git_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def benchmark_shape(torch, functional, args, sequence: int, dimension: int) -> dict[str, object]:
    dtype = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[args.dtype]
    torch.manual_seed(args.seed)
    tensors = [
        torch.rand(
            (args.batch, args.heads, sequence, dimension),
            device="cuda",
            dtype=dtype,
        ) - 0.5
        for _ in range(3)
    ]

    for _ in range(args.warmup):
        output = functional.scaled_dot_product_attention(
            *tensors,
            dropout_p=0.0,
            is_causal=args.causal,
        )
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(args.iterations)]
    stops = [torch.cuda.Event(enable_timing=True) for _ in range(args.iterations)]
    for start, stop in zip(starts, stops):
        start.record()
        output = functional.scaled_dot_product_attention(
            *tensors,
            dropout_p=0.0,
            is_causal=args.causal,
        )
        stop.record()
    torch.cuda.synchronize()
    samples = [start.elapsed_time(stop) for start, stop in zip(starts, stops)]
    median_ms = statistics.median(samples)
    # Causal attention evaluates only the lower triangle, N(N+1)/2 query-key
    # pairs rather than N^2, so counting the full square overstates throughput by
    # close to 2x at large N.
    pairs = sequence * (sequence + 1) / 2 if args.causal else sequence * sequence
    operations = 4.0 * args.batch * args.heads * pairs * dimension
    logical_bytes = 4.0 * args.batch * args.heads * sequence * dimension * output.element_size()
    properties = torch.cuda.get_device_properties(0)

    return {
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_sha": git_sha(),
        "device": torch.cuda.get_device_name(0),
        "architecture": getattr(properties, "gcnArchName", "unknown"),
        "runtime_version": torch.version.hip or "not-rocm",
        "driver_version": "reported-by-runtime",
        "backend": "pytorch",
        "algorithm": "sdpa-auto",
        "dtype": args.dtype,
        "batch": args.batch,
        "heads": args.heads,
        "sequence": sequence,
        "dimension": dimension,
        "causal": int(args.causal),
        "seed": args.seed,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "average_ms": statistics.fmean(samples),
        "median_ms": median_ms,
        "p95_ms": percentile_95(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "tflops": operations / (median_ms * 1.0e9),
        "logical_bandwidth_gbs": logical_bytes / (median_ms * 1.0e6),
        "workspace_bytes": -1,
        "max_abs_error": "",
        "max_rel_error": "",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("out/benchmarks.csv"))
    parser.add_argument("--sequences", type=parse_int_list, default=parse_int_list("128,256,512,1024"))
    parser.add_argument("--dimensions", type=parse_int_list, default=parse_int_list("64,128"))
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dtype", choices=("fp32", "fp16", "bf16"), default="fp32")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--seed", type=int, default=87218)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--append", action="store_true")
    args = parser.parse_args()

    if args.output.exists() and not args.append:
        parser.error(f"output already exists: {args.output} (use --append to add rows)")
    if min(args.batch, args.heads, args.iterations) <= 0 or args.warmup < 0:
        parser.error("batch, heads, and iterations must be positive; warmup must be non-negative")

    try:
        import torch
        import torch.nn.functional as functional
    except ImportError as error:
        parser.error(f"PyTorch is required: {error}")
    if not torch.cuda.is_available() or torch.version.hip is None:
        parser.error("a ROCm-enabled PyTorch installation and visible AMD GPU are required")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not args.output.exists() or args.output.stat().st_size == 0
    with args.output.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        if needs_header:
            writer.writeheader()
        for sequence in args.sequences:
            for dimension in args.dimensions:
                row = benchmark_shape(torch, functional, args, sequence, dimension)
                writer.writerow(row)
                stream.flush()
                print(
                    f"sdpa-auto {args.dtype} B={args.batch} H={args.heads} "
                    f"N={sequence} D={dimension}: {row['median_ms']:.4f} ms",
                    flush=True,
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

