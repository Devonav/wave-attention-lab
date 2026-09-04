# Architecture

This page shows how the repository grows from a kernel experiment into an
end-to-end AMD attention engineering project. Solid boxes are implemented;
dashed boxes describe the next integration boundary.

## System layers

```mermaid
flowchart TB
    subgraph Workload[Model workload]
        L[Llama-style attention]
        P[PyTorch SDPA baseline]
    end

    subgraph Integration[Integration layer]
        E[Future PyTorch custom operator]
        C[Benchmark CLI]
    end

    subgraph Kernels[HIP kernel layer]
        M[Materialized baseline]
        F[Fused online softmax]
        FC[Fused plus cached Q]
    end

    subgraph Evidence[Evidence layer]
        O[CPU correctness oracle]
        T[HIP event timing]
        R[ROCm profiler counters]
        A[CSV and Markdown reports]
    end

    L -. future dispatch .-> E
    P --> C
    E -. future device tensors .-> Kernels
    C --> Kernels
    Kernels --> O
    Kernels --> T
    Kernels --> R
    O --> A
    T --> A
    R --> A
```

## Kernel data flow

```mermaid
flowchart LR
    subgraph Baseline[Materialized path: three dispatches]
        BQ[Q and K] --> S[Score kernel]
        S --> GMEM[Write B x H x N x N scores]
        GMEM --> SM[Softmax kernel]
        SM --> GMEM2[Write probabilities]
        GMEM2 --> PV[Value kernel]
        BV[V] --> PV
        PV --> BO[Output]
    end

    subgraph Fused[Fused-cached path: one dispatch]
        FQ[Q row] --> LDS[Cache Q in LDS]
        FK[Stream K rows] --> DOT[Block dot product]
        LDS --> DOT
        DOT --> ONLINE[Online max and normalizer]
        FV[Stream V rows] --> ACC[Rescaled output accumulator]
        ONLINE --> ACC
        ACC --> FO[Output]
    end
```

The materialized path is intentionally simple, but its intermediate grows with
`N^2`. The fused paths keep only the running softmax state and output
accumulator. `fused-cached` additionally removes repeated global reads of Q.

## Online softmax state

```mermaid
stateDiagram-v2
    [*] --> Empty: m = -inf, l = 0, o = 0
    Empty --> ReadScore: load K tile and compute x
    ReadScore --> Rescale: m_new = max(m, x)
    Rescale --> Normalize: alpha = exp(m - m_new)
    Normalize --> Accumulate: beta = exp(x - m_new)
    Accumulate --> ReadScore: l = alpha*l + beta; o = alpha*o + beta*V
    Accumulate --> Finalize: all keys consumed
    Finalize --> [*]: output = o / l
```

Subtracting the running maximum keeps exponentials bounded. When a new maximum
appears, the previous denominator and output accumulator are rescaled before
the new value is added.

## Benchmark lifecycle

```mermaid
sequenceDiagram
    participant CLI as wave_bench
    participant Host as Host memory
    participant GPU as HIP runtime
    participant Kernel as Attention kernel
    participant Report as CSV report

    CLI->>Host: Generate deterministic Q, K, V
    Host->>GPU: Allocate and copy inputs
    loop Warmup
        GPU->>Kernel: Dispatch without recording
    end
    GPU->>GPU: Synchronize
    loop Timed iterations
        GPU->>Kernel: Record start, dispatch, record stop
        GPU->>GPU: Resolve HIP event time
    end
    GPU->>Host: Copy final output
    CLI->>Host: Compare with CPU oracle
    CLI->>Report: Write median, p95, errors, and environment
```

Allocation and host-device transfer are deliberately outside the timing window.
Each timed iteration produces a sample, allowing median and tail latency to be
reported rather than hiding variation inside one aggregate measurement.

## Optimization loop

```mermaid
flowchart LR
    H[Write one falsifiable hypothesis] --> B[Capture baseline]
    B --> P[Collect profiler evidence]
    P --> I[Implement one change]
    I --> C{Correct versus CPU oracle?}
    C -- No --> D[Diagnose and fix]
    D --> C
    C -- Yes --> M{Median and p95 improve?}
    M -- No --> K[Keep evidence, revert the idea]
    M -- Yes --> R[Record result with commit SHA]
    K --> H
    R --> H
```

This loop is the central artifact of the project. A fast number without its
source revision, environment, correctness result, and profiler explanation is
not accepted as an optimization result.

## Target PyTorch and LLM path

```mermaid
flowchart TB
    QKV[PyTorch Q, K, V device tensors] --> OP[wave_attention custom operator]
    OP --> PICK{Shape and dtype}
    PICK --> FP32[FP32 cached-fused kernel]
    PICK -.-> MP[Future FP16 or BF16 WMMA kernel]
    PICK -.-> KV[Future decode and paged KV-cache kernel]
    FP32 --> OUT[PyTorch output tensor]
    MP -.-> OUT
    KV -.-> OUT
    OUT --> PREFILL[Prefill benchmark]
    OUT --> DECODE[Decode benchmark]
    OUT --> SFT[LoRA or SFT training experiment]
```

The custom operator comes after device-level evidence. That order keeps
framework overhead from obscuring kernel behavior while the algorithm is still
changing.

