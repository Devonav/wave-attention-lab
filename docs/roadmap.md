# Roadmap

```mermaid
flowchart LR
    M1[1. Credible measurements] --> M2[2. RDNA4 kernel optimization]
    M2 --> M3[3. PyTorch custom operator]
    M3 --> M4[4. Llama prefill and decode]
    M4 --> M5[5. Training backward path]
    M5 --> M6[6. Guarded autotuning agent]
    M6 --> M7[7. Upstream ROCm contribution]
```

## Milestone 1: Credible measurements

- [x] CPU reference and causal semantics
- [x] Materialized and online-softmax HIP kernels
- [x] Query-cached fused experiment
- [x] Median, p95, min, max, device, runtime, and revision metadata
- [x] PyTorch SDPA performance harness
- [ ] Compile all HIP variants for `gfx1201`
- [ ] Publish RX 9070 XT raw measurements
- [ ] Publish runtime trace and profiler comparison

Exit criterion: all small shapes pass correctness and a checked-in report
explains the baseline using measurements rather than estimates.

## Milestone 2: RDNA4 optimization

- [ ] Cache and tile K/V through LDS
- [ ] Replace block reductions with measured wave-level reductions
- [ ] Add aligned vector loads and tail handling
- [ ] Add FP16 and BF16 inputs with FP32 accumulation
- [ ] Evaluate WMMA score tiles on RDNA4
- [ ] Add shape-aware deterministic autotuning

Exit criterion: each accepted optimization has before/after profiler evidence
and no correctness regressions across the test matrix.

## Milestone 3: PyTorch custom operator

- [ ] Accept contiguous PyTorch device tensors without host copies
- [ ] Register forward and meta implementations
- [ ] Support causal MHA and GQA shapes
- [ ] Compare outputs and gradients against PyTorch SDPA
- [ ] Package reproducibly for the supported ROCm/PyTorch pair

## Milestone 4: LLM inference

- [ ] Integrate a small Llama-style attention module
- [ ] Separate prefill and single-token decode benchmarks
- [ ] Add append-KV and paged KV-cache paths
- [ ] Report tokens per second, tail latency, and peak VRAM

## Milestone 5: Training and post-training

- [ ] Implement and validate attention backward
- [ ] Run a small LoRA/SFT workload
- [ ] Profile forward, backward, optimizer, and data movement separately
- [ ] Add a rollout-generation study representative of GRPO workloads

## Milestone 6: Agentic autotuning

- [ ] Define a constrained tile and launch-parameter search space
- [ ] Reject any candidate that fails correctness
- [ ] Require repeated statistically meaningful measurements
- [ ] Store every proposal, compiler result, profile, and decision
- [ ] Allow promotion only through a reviewed configuration file

The tuning agent will not initially rewrite arbitrary kernel code. Constraining
its actions makes experiments reproducible and keeps correctness gating explicit.

