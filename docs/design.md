# Design

## Problem definition

For each batch item and attention head, compute:

```text
softmax(Q K^T / sqrt(d)) V
```

The tensors use `[batch, heads, sequence, head_dimension]` layout with the head
dimension contiguous. Causal mode masks keys whose position is greater than the
query position.

## Reference path

The CPU path computes one score row at a time. It subtracts the maximum score
before exponentiation and therefore acts as a stable semantic oracle. Seeded
FP32 inputs in `[-0.5, 0.5]` make failures reproducible.

## Materialized GPU path

The baseline has three dispatches:

1. Compute every entry of `Q K^T` and write the full score matrix.
2. Normalize each row with stable softmax.
3. Multiply the probability matrix by `V`.

This path is easy to inspect, but it writes and rereads `B * H * N^2` FP32
values. That quadratic intermediate becomes increasingly expensive as sequence
length grows.

## Fused GPU path

One block owns one query row. It streams over keys while maintaining a running
maximum `m`, normalization term `l`, and output accumulator `o`. For a new score
`x`:

```text
m_new = max(m, x)
alpha = exp(m - m_new)
beta  = exp(x - m_new)
l_new = alpha * l + beta
o_new = alpha * o + beta * V
```

The final output is `o / l`. Rescaling on every step preserves numerical
stability without storing the score matrix. The current implementation uses a
simple block reduction and is readable by design; it is a baseline for the
optimizations in `optimization-notes.md`.

## Measurement contract

GPU timing uses HIP events around kernel dispatches only. Allocation and host to
device copies occur before the timed region. Warmup iterations run before the
events. The output is copied back after timing.

Reported TFLOP/s uses `4 * B * H * N^2 * d` operations, counting a multiply-add
as two operations for both `QK^T` and `PV`. Logical bandwidth counts one read of
Q, K, and V plus one write of O; it is a comparison metric, not measured DRAM
traffic. Profiler counters are required for claims about physical bandwidth.

