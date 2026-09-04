#include "wave/attention.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

void expect_near(float actual, float expected, float tolerance, const char* label) {
    if (std::abs(actual - expected) > tolerance) {
        throw std::runtime_error(std::string(label) + " mismatch");
    }
}

void test_uniform_attention() {
    const wave::AttentionShape shape{1, 1, 3, 2};
    const std::vector<float> query(6, 0.0f);
    const std::vector<float> key(6, 0.0f);
    const std::vector<float> value{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    const auto output = wave::attention_cpu(shape, query, key, value, false);
    for (int row = 0; row < 3; ++row) {
        expect_near(output[row * 2], 3.0f, 1.0e-6f, "uniform column zero");
        expect_near(output[row * 2 + 1], 4.0f, 1.0e-6f, "uniform column one");
    }
}

void test_causal_attention() {
    const wave::AttentionShape shape{1, 1, 3, 1};
    const std::vector<float> query(3, 0.0f);
    const std::vector<float> key(3, 0.0f);
    const std::vector<float> value{3.0f, 6.0f, 12.0f};
    const auto output = wave::attention_cpu(shape, query, key, value, true);
    expect_near(output[0], 3.0f, 1.0e-6f, "causal row zero");
    expect_near(output[1], 4.5f, 1.0e-6f, "causal row one");
    expect_near(output[2], 7.0f, 1.0e-6f, "causal row two");
}

void test_deterministic_inputs() {
    const wave::AttentionShape shape{1, 2, 4, 8};
    const auto first = wave::make_inputs(shape, 42);
    const auto second = wave::make_inputs(shape, 42);
    if (first.query != second.query || first.key != second.key || first.value != second.value) {
        throw std::runtime_error("seeded inputs are not deterministic");
    }
}

void test_heads_do_not_mix() {
    const wave::AttentionShape shape{1, 2, 2, 1};
    const std::vector<float> query(4, 0.0f);
    const std::vector<float> key(4, 0.0f);
    const std::vector<float> value{1.0f, 3.0f, 10.0f, 14.0f};
    const auto output = wave::attention_cpu(shape, query, key, value, false);
    expect_near(output[0], 2.0f, 1.0e-6f, "head zero row zero");
    expect_near(output[1], 2.0f, 1.0e-6f, "head zero row one");
    expect_near(output[2], 12.0f, 1.0e-6f, "head one row zero");
    expect_near(output[3], 12.0f, 1.0e-6f, "head one row one");
}

void test_large_scores_remain_finite() {
    const wave::AttentionShape shape{1, 1, 2, 1};
    const std::vector<float> query{1000.0f, 1000.0f};
    const std::vector<float> key{1000.0f, 999.0f};
    const std::vector<float> value{7.0f, -3.0f};
    const auto output = wave::attention_cpu(shape, query, key, value, false);
    for (const float element : output) {
        if (!std::isfinite(element)) {
            throw std::runtime_error("large finite scores produced a non-finite result");
        }
        expect_near(element, 7.0f, 1.0e-5f, "stable softmax output");
    }
}

void test_invalid_shape() {
    try {
        wave::AttentionShape{1, 1, 0, 64}.validate();
    } catch (const std::invalid_argument&) {
        return;
    }
    throw std::runtime_error("invalid shape was accepted");
}

}  // namespace

int main() {
    try {
        test_uniform_attention();
        test_causal_attention();
        test_deterministic_inputs();
        test_heads_do_not_mix();
        test_large_scores_remain_finite();
        test_invalid_shape();
        std::cout << "all CPU correctness tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "test failure: " << error.what() << '\n';
        return 1;
    }
}
