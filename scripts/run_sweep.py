#!/usr/bin/env python3
"""Run the standard HIP attention experiment matrix."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def parse_int_list(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("expected a comma-separated list of positive integers")
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("out/benchmarks.csv"))
    parser.add_argument("--sequences", type=parse_int_list, default=parse_int_list("128,256,512,1024"))
    parser.add_argument("--dimensions", type=parse_int_list, default=parse_int_list("64,128"))
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--verify-through", type=int, default=256)
    parser.add_argument("--append", action="store_true")
    args = parser.parse_args()

    executable = args.executable.resolve()
    if not executable.is_file():
        parser.error(f"benchmark executable does not exist: {executable}")
    if args.heads <= 0 or args.iterations <= 0 or args.warmup < 0:
        parser.error("heads and iterations must be positive; warmup must be non-negative")
    if args.output.exists() and not args.append:
        parser.error(f"output already exists: {args.output} (use --append to add rows)")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    for sequence in args.sequences:
        for dimension in args.dimensions:
            for algorithm in ("materialized", "fused", "fused-cached"):
                command = [
                    str(executable),
                    "--backend", "hip",
                    "--algorithm", algorithm,
                    "--heads", str(args.heads),
                    "--sequence", str(sequence),
                    "--dimension", str(dimension),
                    "--warmup", str(args.warmup),
                    "--iterations", str(args.iterations),
                    "--csv", str(args.output),
                ]
                if args.causal:
                    command.append("--causal")
                if sequence <= args.verify_through:
                    command.append("--verify")
                print("+", " ".join(command), flush=True)
                subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
