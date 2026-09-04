#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace wave {

struct AttentionShape {
    int batch = 1;
    int heads = 8;
    int sequence = 128;
    int head_dimension = 64;

    void validate() const;
    std::size_t qkv_elements() const;
    std::size_t rows() const;
};

struct AttentionInputs {
    std::vector<float> query;
    std::vector<float> key;
    std::vector<float> value;
};

enum class HipAlgorithm {
    materialized,
    fused,
};

struct RunResult {
    std::vector<float> output;
    double average_ms = 0.0;
    double tflops = 0.0;
    double logical_bandwidth_gbs = 0.0;
    std::size_t workspace_bytes = 0;
    std::string device;
};

AttentionInputs make_inputs(const AttentionShape& shape, std::uint64_t seed);

std::vector<float> attention_cpu(
    const AttentionShape& shape,
    const std::vector<float>& query,
    const std::vector<float>& key,
    const std::vector<float>& value,
    bool causal);

RunResult benchmark_cpu(
    const AttentionShape& shape,
    const AttentionInputs& inputs,
    bool causal,
    int warmup,
    int iterations);

bool hip_backend_available();

RunResult benchmark_hip(
    const AttentionShape& shape,
    const AttentionInputs& inputs,
    HipAlgorithm algorithm,
    bool causal,
    int warmup,
    int iterations);

double max_absolute_error(const std::vector<float>& expected, const std::vector<float>& actual);
double max_relative_error(const std::vector<float>& expected, const std::vector<float>& actual);

}  // namespace wave

