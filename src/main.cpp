#include "wave/attention.hpp"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

#ifndef WAVE_GIT_SHA
#define WAVE_GIT_SHA "unknown"
#endif

struct Options {
    wave::AttentionShape shape;
    std::string backend = "cpu";
    wave::HipAlgorithm algorithm = wave::HipAlgorithm::fused;
    bool causal = false;
    bool verify = false;
    int warmup = 3;
    int iterations = 20;
    std::uint64_t seed = 87218;
    std::string csv_path;
};

void print_help() {
    std::cout
        << "wave_bench - scaled dot-product attention kernel lab\n\n"
        << "Options:\n"
        << "  --backend cpu|hip        Execution backend (default: cpu)\n"
        << "  --algorithm fused|fused-cached|materialized\n"
        << "  --batch N                Batch size (default: 1)\n"
        << "  --heads N                Attention heads (default: 8)\n"
        << "  --sequence N             Sequence length (default: 128)\n"
        << "  --dimension N            Head dimension (default: 64)\n"
        << "  --warmup N               Untimed iterations (default: 3)\n"
        << "  --iterations N           Timed iterations (default: 20)\n"
        << "  --seed N                 Input seed (default: 87218)\n"
        << "  --causal                 Apply a causal mask\n"
        << "  --verify                 Compare HIP output with the CPU oracle\n"
        << "  --csv PATH               Append one benchmark row to PATH\n"
        << "  --help                    Show this message\n";
}

std::string require_value(int& index, int argc, char** argv) {
    if (index + 1 >= argc) {
        throw std::invalid_argument(std::string("missing value for ") + argv[index]);
    }
    return argv[++index];
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--help") {
            print_help();
            std::exit(0);
        } else if (argument == "--backend") {
            options.backend = require_value(index, argc, argv);
            if (options.backend != "cpu" && options.backend != "hip") {
                throw std::invalid_argument("backend must be cpu or hip");
            }
        } else if (argument == "--algorithm") {
            const auto value = require_value(index, argc, argv);
            if (value == "fused") {
                options.algorithm = wave::HipAlgorithm::fused;
            } else if (value == "fused-cached") {
                options.algorithm = wave::HipAlgorithm::fused_cached;
            } else if (value == "materialized") {
                options.algorithm = wave::HipAlgorithm::materialized;
            } else {
                throw std::invalid_argument(
                    "algorithm must be fused, fused-cached, or materialized");
            }
        } else if (argument == "--batch") {
            options.shape.batch = std::stoi(require_value(index, argc, argv));
        } else if (argument == "--heads") {
            options.shape.heads = std::stoi(require_value(index, argc, argv));
        } else if (argument == "--sequence") {
            options.shape.sequence = std::stoi(require_value(index, argc, argv));
        } else if (argument == "--dimension") {
            options.shape.head_dimension = std::stoi(require_value(index, argc, argv));
        } else if (argument == "--warmup") {
            options.warmup = std::stoi(require_value(index, argc, argv));
        } else if (argument == "--iterations") {
            options.iterations = std::stoi(require_value(index, argc, argv));
        } else if (argument == "--seed") {
            options.seed = std::stoull(require_value(index, argc, argv));
        } else if (argument == "--csv") {
            options.csv_path = require_value(index, argc, argv);
        } else if (argument == "--causal") {
            options.causal = true;
        } else if (argument == "--verify") {
            options.verify = true;
        } else {
            throw std::invalid_argument("unknown option: " + argument);
        }
    }
    options.shape.validate();
    if (options.warmup < 0 || options.iterations <= 0) {
        throw std::invalid_argument("warmup must be non-negative and iterations must be positive");
    }
    return options;
}

const char* algorithm_name(wave::HipAlgorithm algorithm) {
    switch (algorithm) {
        case wave::HipAlgorithm::materialized:
            return "materialized";
        case wave::HipAlgorithm::fused:
            return "fused";
        case wave::HipAlgorithm::fused_cached:
            return "fused-cached";
    }
    throw std::invalid_argument("unknown HIP algorithm");
}

std::string utc_timestamp() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t time = std::chrono::system_clock::to_time_t(now);
    std::tm utc{};
#ifdef _WIN32
    gmtime_s(&utc, &time);
#else
    gmtime_r(&time, &utc);
#endif
    std::ostringstream stream;
    stream << std::put_time(&utc, "%Y-%m-%dT%H:%M:%SZ");
    return stream.str();
}

void append_csv(const Options& options, const wave::RunResult& result, double absolute_error, double relative_error) {
    const bool needs_header = !std::filesystem::exists(options.csv_path) ||
                              std::filesystem::file_size(options.csv_path) == 0;
    std::ofstream stream(options.csv_path, std::ios::app);
    if (!stream) {
        throw std::runtime_error("could not open CSV output: " + options.csv_path);
    }
    if (needs_header) {
        stream << "timestamp_utc,git_sha,device,architecture,runtime_version,driver_version,"
                  "backend,algorithm,dtype,batch,heads,sequence,dimension,causal,seed,warmup,iterations,"
                  "average_ms,median_ms,p95_ms,min_ms,max_ms,tflops,logical_bandwidth_gbs,"
                  "workspace_bytes,max_abs_error,max_rel_error\n";
    }
    stream << utc_timestamp() << ',' << WAVE_GIT_SHA << ',' << std::quoted(result.device) << ','
           << result.architecture << ',' << result.runtime_version << ',' << result.driver_version << ','
           << options.backend << ','
           << (options.backend == "hip" ? algorithm_name(options.algorithm) : "reference") << ",fp32,"
           << options.shape.batch << ',' << options.shape.heads << ',' << options.shape.sequence << ','
           << options.shape.head_dimension << ',' << (options.causal ? 1 : 0) << ',' << options.seed << ','
           << options.warmup << ',' << options.iterations << ',' << std::setprecision(9)
           << result.average_ms << ',' << result.median_ms << ',' << result.p95_ms << ','
           << result.minimum_ms << ',' << result.maximum_ms << ',' << result.tflops << ','
           << result.logical_bandwidth_gbs << ',' << result.workspace_bytes << ','
           << absolute_error << ',' << relative_error << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        const auto inputs = wave::make_inputs(options.shape, options.seed);

        wave::RunResult result;
        if (options.backend == "hip") {
            result = wave::benchmark_hip(
                options.shape,
                inputs,
                options.algorithm,
                options.causal,
                options.warmup,
                options.iterations);
        } else {
            result = wave::benchmark_cpu(
                options.shape,
                inputs,
                options.causal,
                options.warmup,
                options.iterations);
        }

        double absolute_error = 0.0;
        double relative_error = 0.0;
        if (options.verify && options.backend == "hip") {
            const auto reference = wave::attention_cpu(
                options.shape,
                inputs.query,
                inputs.key,
                inputs.value,
                options.causal);
            absolute_error = wave::max_absolute_error(reference, result.output);
            relative_error = wave::max_relative_error(reference, result.output);
        }

        const double checksum = std::accumulate(result.output.begin(), result.output.end(), 0.0);
        std::cout << std::fixed << std::setprecision(4)
                  << "device:               " << result.device << '\n'
                  << "backend:              " << options.backend << '\n'
                  << "algorithm:            "
                  << (options.backend == "hip" ? algorithm_name(options.algorithm) : "reference") << '\n'
                  << "shape:                [" << options.shape.batch << ", " << options.shape.heads
                  << ", " << options.shape.sequence << ", " << options.shape.head_dimension << "]\n"
                  << "causal:               " << (options.causal ? "yes" : "no") << '\n'
                  << "average_ms:           " << result.average_ms << '\n'
                  << "median_ms:            " << result.median_ms << '\n'
                  << "p95_ms:               " << result.p95_ms << '\n'
                  << "min_ms:               " << result.minimum_ms << '\n'
                  << "max_ms:               " << result.maximum_ms << '\n'
                  << "estimated_tflops:     " << result.tflops << '\n'
                  << "logical_bandwidth:    " << result.logical_bandwidth_gbs << " GB/s\n"
                  << "workspace_bytes:      " << result.workspace_bytes << '\n'
                  << "architecture:         " << result.architecture << '\n'
                  << "runtime_version:      " << result.runtime_version << '\n'
                  << "driver_version:       " << result.driver_version << '\n'
                  << "git_sha:              " << WAVE_GIT_SHA << '\n'
                  << "output_checksum:      " << checksum << '\n';
        if (options.verify && options.backend == "hip") {
            std::cout << "max_absolute_error:   " << absolute_error << '\n'
                      << "max_relative_error:   " << relative_error << '\n';
            if (absolute_error > 2.0e-3) {
                std::cerr << "verification failed: absolute error exceeds 2e-3\n";
                return 2;
            }
        }
        if (!options.csv_path.empty()) {
            append_csv(options, result, absolute_error, relative_error);
            std::cout << "csv:                  " << options.csv_path << '\n';
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
