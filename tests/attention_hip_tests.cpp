#include "wave/attention.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>

int main() {
    if (!wave::hip_backend_available()) {
        std::cout << "HIP device unavailable; skipping GPU correctness tests\n";
        return 77;
    }
    try {
        // Exercise partial query blocks, key tiles, and wave fragments, plus
        // multiple batches/heads to catch shared-memory indexing mistakes.
        for (const int sequence : {1, 7, 8, 9, 31, 32, 33, 65}) {
            for (const int dimension : {1, 31, 32, 33, 63, 64, 65, 127, 128}) {
                for (const bool causal : {false, true}) {
                    const wave::AttentionShape shape{2, 2, sequence, dimension};
                    const auto inputs = wave::make_inputs(shape, 42);
                    const auto expected = wave::attention_cpu(
                        shape, inputs.query, inputs.key, inputs.value, causal);
                    const auto actual = wave::benchmark_hip(
                        shape, inputs, wave::HipAlgorithm::tiled, causal, 0, 1).output;
                    if (actual.size() != expected.size()) {
                        throw std::runtime_error("output size mismatch");
                    }
                    for (std::size_t i = 0; i < expected.size(); ++i) {
                        if (!std::isfinite(actual[i]) ||
                            std::abs(actual[i] - expected[i]) >
                                1.0e-4f + 1.0e-4f * std::abs(expected[i])) {
                            std::cerr << "N=" << sequence << " D=" << dimension
                                      << " causal=" << causal << " element=" << i << '\n';
                            throw std::runtime_error("tiled output differs from CPU oracle");
                        }
                    }
                }
            }
        }
        std::cout << "all 144 tiled HIP correctness cases passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "test failure: " << error.what() << '\n';
        return 1;
    }
}
