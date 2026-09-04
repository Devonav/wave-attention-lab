#!/usr/bin/env python3
"""Turn wave_bench CSV rows into a compact Markdown comparison."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


KEY_FIELDS = ("device", "batch", "heads", "sequence", "dimension", "causal")


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    required = set(KEY_FIELDS) | {"algorithm", "average_ms", "workspace_bytes", "max_abs_error"}
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
        "| Device | B | H | N | D | Causal | Materialized ms | Fused ms | Speedup | Workspace saved | Max abs error |",
        "| --- | ---: | ---: | ---: | ---: | :---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    comparisons = 0
    for key in sorted(groups):
        variants = groups[key]
        if "materialized" not in variants or "fused" not in variants:
            continue
        materialized = variants["materialized"]
        fused = variants["fused"]
        materialized_ms = float(materialized["average_ms"])
        fused_ms = float(fused["average_ms"])
        speedup = materialized_ms / fused_ms if fused_ms else 0.0
        workspace_saved = int(materialized["workspace_bytes"]) - int(fused["workspace_bytes"])
        error = max(float(materialized["max_abs_error"]), float(fused["max_abs_error"]))
        device, batch, heads, sequence, dimension, causal = key
        lines.append(
            f"| {device} | {batch} | {heads} | {sequence} | {dimension} | "
            f"{'yes' if causal == '1' else 'no'} | {materialized_ms:.4f} | {fused_ms:.4f} | "
            f"{speedup:.2f}x | {workspace_saved:,} B | {error:.3g} |"
        )
        comparisons += 1
    if comparisons == 0:
        raise ValueError("no shapes contain both materialized and fused results")
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

