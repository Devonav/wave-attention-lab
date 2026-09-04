#include "wave/attention.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

namespace wave {
namespace {

void validate_inputs(
    const AttentionShape& shape,
    const std::vector<float>& query,
    const std::vector<float>& key,
    const std::vector<float>& value) {
    shape.validate();
    const auto expected = shape.qkv_elements();
    if (query.size() != expected || key.size() != expected || value.size() != expected) {
        throw std::invalid_argument("Q, K, and V sizes must match the attention shape");
    }
}

double estimated_tflops(const AttentionShape& shape, double milliseconds) {
    if (milliseconds <= 0.0) {
        return 0.0;
    }
    const double operations = 4.0 * shape.batch * shape.heads * shape.sequence *
                              shape.sequence * shape.head_dimension;
    return operations / (milliseconds * 1.0e9);
}

double logical_bandwidth(const AttentionShape& shape, double milliseconds) {
    if (milliseconds <= 0.0) {
        return 0.0;
    }
    const double bytes = 4.0 * static_cast<double>(shape.qkv_elements()) * sizeof(float);
    return bytes / (milliseconds * 1.0e6);
}

void populate_timing_statistics(RunResult& result, std::vector<double> samples) {
    if (samples.empty()) {
        throw std::invalid_argument("timing samples must not be empty");
    }
    result.average_ms = 0.0;
    for (const double sample : samples) {
        result.average_ms += sample;
    }
    result.average_ms /= static_cast<double>(samples.size());

    std::sort(samples.begin(), samples.end());
    const auto count = samples.size();
    result.median_ms = count % 2 == 0
        ? (samples[count / 2 - 1] + samples[count / 2]) / 2.0
        : samples[count / 2];
    const auto p95_index = static_cast<std::size_t>(
        std::ceil(0.95 * static_cast<double>(count)) - 1.0);
    result.p95_ms = samples[std::min(p95_index, count - 1)];
    result.minimum_ms = samples.front();
    result.maximum_ms = samples.back();
}

}  // namespace

void AttentionShape::validate() const {
    if (batch <= 0 || heads <= 0 || sequence <= 0 || head_dimension <= 0) {
        throw std::invalid_argument("all shape dimensions must be positive");
    }
}

std::size_t AttentionShape::qkv_elements() const {
    validate();
    return static_cast<std::size_t>(batch) * heads * sequence * head_dimension;
}

std::size_t AttentionShape::rows() const {
    validate();
    return static_cast<std::size_t>(batch) * heads * sequence;
}

AttentionInputs make_inputs(const AttentionShape& shape, std::uint64_t seed) {
    const auto count = shape.qkv_elements();
    AttentionInputs inputs;
    inputs.query.resize(count);
    inputs.key.resize(count);
    inputs.value.resize(count);

    std::mt19937_64 generator(seed);
    std::uniform_real_distribution<float> distribution(-0.5f, 0.5f);
    for (auto* tensor : {&inputs.query, &inputs.key, &inputs.value}) {
        std::generate(tensor->begin(), tensor->end(), [&] { return distribution(generator); });
    }
    return inputs;
}

std::vector<float> attention_cpu(
    const AttentionShape& shape,
    const std::vector<float>& query,
    const std::vector<float>& key,
    const std::vector<float>& value,
    bool causal) {
    validate_inputs(shape, query, key, value);

    const int n = shape.sequence;
    const int d = shape.head_dimension;
    const float scale = 1.0f / std::sqrt(static_cast<float>(d));
    std::vector<float> output(shape.qkv_elements(), 0.0f);
    std::vector<float> scores(static_cast<std::size_t>(n));

    for (std::size_t row = 0; row < shape.rows(); ++row) {
        const int query_index = static_cast<int>(row % n);
        const std::size_t head_base = (row / n) * static_cast<std::size_t>(n) * d;
        const std::size_t query_base = row * d;
        const int key_limit = causal ? query_index + 1 : n;

        float row_max = -std::numeric_limits<float>::infinity();
        for (int key_index = 0; key_index < key_limit; ++key_index) {
            float dot = 0.0f;
            const std::size_t key_base = head_base + static_cast<std::size_t>(key_index) * d;
            for (int column = 0; column < d; ++column) {
                dot += query[query_base + column] * key[key_base + column];
            }
            scores[key_index] = dot * scale;
            row_max = std::max(row_max, scores[key_index]);
        }

        float denominator = 0.0f;
        for (int key_index = 0; key_index < key_limit; ++key_index) {
            scores[key_index] = std::exp(scores[key_index] - row_max);
            denominator += scores[key_index];
        }

        for (int key_index = 0; key_index < key_limit; ++key_index) {
            const float probability = scores[key_index] / denominator;
            const std::size_t value_base = head_base + static_cast<std::size_t>(key_index) * d;
            for (int column = 0; column < d; ++column) {
                output[query_base + column] += probability * value[value_base + column];
            }
        }
    }
    return output;
}

RunResult benchmark_cpu(
    const AttentionShape& shape,
    const AttentionInputs& inputs,
    bool causal,
    int warmup,
    int iterations) {
    if (warmup < 0 || iterations <= 0) {
        throw std::invalid_argument("warmup must be non-negative and iterations must be positive");
    }

    std::vector<float> output;
    for (int iteration = 0; iteration < warmup; ++iteration) {
        output = attention_cpu(shape, inputs.query, inputs.key, inputs.value, causal);
    }

    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(iterations));
    for (int iteration = 0; iteration < iterations; ++iteration) {
        const auto start = std::chrono::steady_clock::now();
        output = attention_cpu(shape, inputs.query, inputs.key, inputs.value, causal);
        const auto stop = std::chrono::steady_clock::now();
        samples.push_back(std::chrono::duration<double, std::milli>(stop - start).count());
    }

    RunResult result;
    result.output = std::move(output);
    populate_timing_statistics(result, std::move(samples));
    result.tflops = estimated_tflops(shape, result.median_ms);
    result.logical_bandwidth_gbs = logical_bandwidth(shape, result.median_ms);
    result.workspace_bytes = static_cast<std::size_t>(shape.sequence) * sizeof(float);
    result.device = "CPU reference";
    result.architecture = "host";
    result.runtime_version = "C++17";
    result.driver_version = "n/a";
    return result;
}

double max_absolute_error(const std::vector<float>& expected, const std::vector<float>& actual) {
    if (expected.size() != actual.size()) {
        throw std::invalid_argument("cannot compare outputs with different sizes");
    }
    double maximum = 0.0;
    for (std::size_t index = 0; index < expected.size(); ++index) {
        maximum = std::max(maximum, std::abs(static_cast<double>(expected[index]) - actual[index]));
    }
    return maximum;
}

double max_relative_error(const std::vector<float>& expected, const std::vector<float>& actual) {
    if (expected.size() != actual.size()) {
        throw std::invalid_argument("cannot compare outputs with different sizes");
    }
    double maximum = 0.0;
    for (std::size_t index = 0; index < expected.size(); ++index) {
        const double denominator = std::max(1.0e-6, std::abs(static_cast<double>(expected[index])));
        maximum = std::max(
            maximum,
            std::abs(static_cast<double>(expected[index]) - actual[index]) / denominator);
    }
    return maximum;
}

#ifndef WAVE_HAS_HIP
bool hip_backend_available() {
    return false;
}

RunResult benchmark_hip(
    const AttentionShape&,
    const AttentionInputs&,
    HipAlgorithm,
    bool,
    int,
    int) {
    throw std::runtime_error("HIP backend was not built; configure with -DWAVE_ENABLE_HIP=ON");
}
#endif

}  // namespace wave
