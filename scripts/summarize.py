#!/usr/bin/env python3
"""Turn wave_bench CSV rows into a compact Markdown comparison."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import median


# Grouping uses architecture rather than device. The HIP backend reports
# hipDeviceProp_t::name ("AMD Radeon RX 9070 XT") while PyTorch reports
# torch.cuda.get_device_name ("AMD Radeon Graphics"), so keying on the marketing
# name would never join SDPA rows to the custom-kernel rows. Both backends derive
# architecture from gcnArchName, which agrees exactly.
KEY_FIELDS = ("architecture", "dtype", "batch", "heads", "sequence", "dimension", "causal")


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


def trial_median(trials: list[dict[str, str]], field: str) -> float:
    """Median across trials of a per-trial statistic.

    Each CSV row is one complete trial that already aggregated its own
    iterations, so combining trials means taking the median of their medians.
    Reporting a single trial would defeat the repeated-measurement requirement in
    results/README.md.
    """
    return median(float(trial.get(field) or trial["average_ms"]) for trial in trials)


def summarize(rows: list[dict[str, str]]) -> str:
    groups: dict[tuple[str, ...], dict[str, list[dict[str, str]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        key = tuple(row[field] for field in KEY_FIELDS)
        groups[key][row["algorithm"]].append(row)

    lines = [
        "# Wave Attention Benchmark Results",
        "",
        "| Device | Type | B | H | N | D | Causal | Materialized | Fused | Cached | PyTorch SDPA | Cached vs baseline | Cached vs SDPA | Cached p95 | Workspace saved | Max abs error |",
        "| --- | --- | ---: | ---: | ---: | ---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    comparisons = 0
    trial_counts: set[int] = set()
    for key in sorted(groups):
        variants = groups[key]
        required_variants = {"materialized", "fused", "fused-cached"}
        if not required_variants.issubset(variants):
            continue
        materialized = variants["materialized"]
        fused = variants["fused"]
        cached = variants["fused-cached"]
        materialized_ms = trial_median(materialized, "median_ms")
        fused_ms = trial_median(fused, "median_ms")
        cached_ms = trial_median(cached, "median_ms")
        cached_p95 = trial_median(cached, "p95_ms")
        speedup = materialized_ms / cached_ms if cached_ms else 0.0
        sdpa = variants.get("sdpa-auto")
        sdpa_ms = trial_median(sdpa, "median_ms") if sdpa else None
        sdpa_speedup = sdpa_ms / cached_ms if sdpa_ms is not None and cached_ms else None
        workspace_saved = int(materialized[0]["workspace_bytes"]) - int(
            cached[0]["workspace_bytes"]
        )
        error = max(
            float(trial["max_abs_error"])
            for trials in (materialized, fused, cached)
            for trial in trials
        )
        trial_counts.update(len(trials) for trials in variants.values())
        architecture, dtype, batch, heads, sequence, dimension, causal = key
        device = materialized[0]["device"]
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
    trials = (
        str(trial_counts.pop())
        if len(trial_counts) == 1
        else f"{min(trial_counts)}-{max(trial_counts)}"
    )
    lines.extend([
        "",
        f"Generated from raw `wave_bench` CSV rows over {trials} trial(s) per cell; "
        "each figure is the median across trials of that trial's own median. "
        "Timing covers GPU kernels only.",
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
