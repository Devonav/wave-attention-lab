#!/usr/bin/env python3
"""Turn wave_bench CSV rows into a compact Markdown comparison."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


KEY_FIELDS = ("device", "dtype", "batch", "heads", "sequence", "dimension", "causal")


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    required = set(KEY_FIELDS) | {
        "algorithm",
        "average_ms",
        "workspace_bytes",
        "max_abs_error",
    }
    if not rows:
        raise ValueError("benchmark CSV is empty")
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"benchmark CSV is missing columns: {', '.join(sorted(missing))}")
    return rows


def summarize(rows: list[dict[str, str]]) -> str:
    groups: dict[tuple[str, ...], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        key = tuple(row[field] for field in KEY_FIELDS)
        groups[key][row["algorithm"]] = row

    lines = [
        "# Wave Attention Benchmark Results",
        "",
        "| Device | Type | B | H | N | D | Causal | Materialized | Fused | Cached | PyTorch SDPA | Cached vs baseline | Cached vs SDPA | Cached p95 | Workspace saved | Max abs error |",
        "| --- | --- | ---: | ---: | ---: | ---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    comparisons = 0
    for key in sorted(groups):
        variants = groups[key]
        required_variants = {"materialized", "fused", "fused-cached"}
        if not required_variants.issubset(variants):
            continue
        materialized = variants["materialized"]
        fused = variants["fused"]
        cached = variants["fused-cached"]
        materialized_ms = float(materialized.get("median_ms") or materialized["average_ms"])
        fused_ms = float(fused.get("median_ms") or fused["average_ms"])
        cached_ms = float(cached.get("median_ms") or cached["average_ms"])
        cached_p95 = float(cached.get("p95_ms") or cached["average_ms"])
        speedup = materialized_ms / cached_ms if cached_ms else 0.0
        sdpa = variants.get("sdpa-auto")
        sdpa_ms = float(sdpa.get("median_ms") or sdpa["average_ms"]) if sdpa else None
        sdpa_speedup = sdpa_ms / cached_ms if sdpa_ms is not None and cached_ms else None
        workspace_saved = int(materialized["workspace_bytes"]) - int(cached["workspace_bytes"])
        error = max(
            float(materialized["max_abs_error"]),
            float(fused["max_abs_error"]),
            float(cached["max_abs_error"]),
        )
        device, dtype, batch, heads, sequence, dimension, causal = key
        lines.append(
            f"| {device} | {dtype} | {batch} | {heads} | {sequence} | {dimension} | "
            f"{'yes' if causal == '1' else 'no'} | {materialized_ms:.4f} ms | "
            f"{fused_ms:.4f} ms | {cached_ms:.4f} ms | "
            f"{f'{sdpa_ms:.4f} ms' if sdpa_ms is not None else '-'} | {speedup:.2f}x | "
            f"{f'{sdpa_speedup:.2f}x' if sdpa_speedup is not None else '-'} | "
            f"{cached_p95:.4f} ms | {workspace_saved:,} B | {error:.3g} |"
        )
        comparisons += 1
    if comparisons == 0:
        raise ValueError("no shapes contain materialized, fused, and fused-cached results")
    lines.extend([
        "",
        "Generated from raw `wave_bench` CSV rows. Timing covers GPU kernels only.",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = summarize(load_rows(args.csv))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
        print(args.output)
    else:
        print(report, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
