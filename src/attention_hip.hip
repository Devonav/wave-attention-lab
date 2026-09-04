#include "wave/attention.hpp"

#include <hip/hip_runtime.h>

#include <cmath>
#include <sstream>
#include <stdexcept>
#include <string>

namespace wave {
namespace {

constexpr int kThreads = 256;

void check_hip(hipError_t result, const char* expression) {
    if (result != hipSuccess) {
        std::ostringstream message;
        message << expression << " failed: " << hipGetErrorString(result);
        throw std::runtime_error(message.str());
    }
}

#define WAVE_HIP_CHECK(expression) check_hip((expression), #expression)

__global__ void score_kernel(
    const float* query,
    const float* key,
    float* scores,
    int rows,
    int sequence,
    int dimension,
    float scale,
    bool causal) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = rows * sequence;
    if (index >= total) {
        return;
    }

    const int row = index / sequence;
    const int key_index = index % sequence;
    const int query_index = row % sequence;
    if (causal && key_index > query_index) {
        scores[index] = -INFINITY;
        return;
    }

    const int query_base = row * dimension;
    const int head_base = (row / sequence) * sequence * dimension;
    const int key_base = head_base + key_index * dimension;
    float dot = 0.0f;
    for (int column = 0; column < dimension; ++column) {
        dot += query[query_base + column] * key[key_base + column];
    }
    scores[index] = dot * scale;
}

__global__ void softmax_kernel(float* scores, int rows, int sequence, bool causal) {
    __shared__ float reduction[kThreads];
    const int row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    const int query_index = row % sequence;
    const int key_limit = causal ? query_index + 1 : sequence;

    float local_max = -INFINITY;
    for (int key_index = threadIdx.x; key_index < key_limit; key_index += blockDim.x) {
        local_max = fmaxf(local_max, scores[row * sequence + key_index]);
    }
    reduction[threadIdx.x] = local_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            reduction[threadIdx.x] = fmaxf(reduction[threadIdx.x], reduction[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    const float row_max = reduction[0];

    float local_sum = 0.0f;
    for (int key_index = threadIdx.x; key_index < key_limit; key_index += blockDim.x) {
        const float probability = expf(scores[row * sequence + key_index] - row_max);
        scores[row * sequence + key_index] = probability;
        local_sum += probability;
    }
    reduction[threadIdx.x] = local_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            reduction[threadIdx.x] += reduction[threadIdx.x + stride];
        }
        __syncthreads();
    }
    const float denominator = reduction[0];

    for (int key_index = threadIdx.x; key_index < sequence; key_index += blockDim.x) {
        scores[row * sequence + key_index] = key_index < key_limit
            ? scores[row * sequence + key_index] / denominator
            : 0.0f;
    }
}

__global__ void apply_value_kernel(
    const float* probabilities,
    const float* value,
    float* output,
    int rows,
    int sequence,
    int dimension) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = rows * dimension;
    if (index >= total) {
        return;
    }
    const int row = index / dimension;
    const int column = index % dimension;
    const int head_base = (row / sequence) * sequence * dimension;
    float result = 0.0f;
    for (int key_index = 0; key_index < sequence; ++key_index) {
        result += probabilities[row * sequence + key_index] *
                  value[head_base + key_index * dimension + column];
    }
    output[index] = result;
}

__global__ void fused_attention_kernel(
    const float* query,
    const float* key,
    const float* value,
    float* output,
    int rows,
    int sequence,
    int dimension,
    float scale,
    bool causal) {
    __shared__ float reduction[kThreads];
    __shared__ float running_max;
    __shared__ float running_sum;
    __shared__ float alpha;
    __shared__ float beta;

    const int row = blockIdx.x;
    const int lane = threadIdx.x;
    if (row >= rows) {
        return;
    }

    const int query_index = row % sequence;
    const int head_base = (row / sequence) * sequence * dimension;
    const int query_base = row * dimension;
    const int key_limit = causal ? query_index + 1 : sequence;
    float accumulator = 0.0f;

    if (lane == 0) {
        running_max = -INFINITY;
        running_sum = 0.0f;
    }
    __syncthreads();

    for (int key_index = 0; key_index < key_limit; ++key_index) {
        float partial = 0.0f;
        for (int column = lane; column < dimension; column += blockDim.x) {
            partial += query[query_base + column] *
                       key[head_base + key_index * dimension + column];
        }
        reduction[lane] = partial;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
            if (lane < stride) {
                reduction[lane] += reduction[lane + stride];
            }
            __syncthreads();
        }

        if (lane == 0) {
            const float score = reduction[0] * scale;
            const float next_max = fmaxf(running_max, score);
            alpha = running_sum == 0.0f ? 0.0f : expf(running_max - next_max);
            beta = expf(score - next_max);
            running_sum = running_sum * alpha + beta;
            running_max = next_max;
        }
        __syncthreads();

        if (lane < dimension) {
            accumulator = accumulator * alpha +
                          beta * value[head_base + key_index * dimension + lane];
        }
        __syncthreads();
    }

    if (lane < dimension) {
        output[query_base + lane] = accumulator / running_sum;
    }
}

class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t bytes) : bytes_(bytes) {
        WAVE_HIP_CHECK(hipMalloc(&pointer_, bytes));
    }

    ~DeviceBuffer() {
        if (pointer_ != nullptr) {
            hipFree(pointer_);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    float* get() { return static_cast<float*>(pointer_); }
    const float* get() const { return static_cast<const float*>(pointer_); }

private:
    void* pointer_ = nullptr;
    std::size_t bytes_ = 0;
};

void launch_attention(
    const AttentionShape& shape,
    HipAlgorithm algorithm,
    bool causal,
    const float* query,
    const float* key,
    const float* value,
    float* output,
    float* scores) {
    const int rows = static_cast<int>(shape.rows());
    const int sequence = shape.sequence;
    const int dimension = shape.head_dimension;
    const float scale = 1.0f / std::sqrt(static_cast<float>(dimension));

    if (algorithm == HipAlgorithm::materialized) {
        const int score_count = rows * sequence;
        hipLaunchKernelGGL(
            score_kernel,
            dim3((score_count + kThreads - 1) / kThreads),
            dim3(kThreads),
            0,
            0,
            query,
            key,
            scores,
            rows,
            sequence,
            dimension,
            scale,
            causal);
        hipLaunchKernelGGL(
            softmax_kernel,
            dim3(rows),
            dim3(kThreads),
            0,
            0,
            scores,
            rows,
            sequence,
            causal);
        const int output_count = rows * dimension;
        hipLaunchKernelGGL(
            apply_value_kernel,
            dim3((output_count + kThreads - 1) / kThreads),
            dim3(kThreads),
            0,
            0,
            scores,
            value,
            output,
            rows,
            sequence,
            dimension);
    } else {
        if (dimension > kThreads) {
            throw std::invalid_argument("fused HIP kernel supports head dimensions up to 256");
        }
        hipLaunchKernelGGL(
            fused_attention_kernel,
            dim3(rows),
            dim3(kThreads),
            0,
            0,
            query,
            key,
            value,
            output,
            rows,
            sequence,
            dimension,
            scale,
            causal);
    }
    WAVE_HIP_CHECK(hipGetLastError());
}

double tflops(const AttentionShape& shape, double milliseconds) {
    const double operations = 4.0 * shape.batch * shape.heads * shape.sequence *
                              shape.sequence * shape.head_dimension;
    return operations / (milliseconds * 1.0e9);
}

double logical_bandwidth(const AttentionShape& shape, double milliseconds) {
    const double bytes = 4.0 * static_cast<double>(shape.qkv_elements()) * sizeof(float);
    return bytes / (milliseconds * 1.0e6);
}

}  // namespace

bool hip_backend_available() {
    int device_count = 0;
    return hipGetDeviceCount(&device_count) == hipSuccess && device_count > 0;
}

RunResult benchmark_hip(
    const AttentionShape& shape,
    const AttentionInputs& inputs,
    HipAlgorithm algorithm,
    bool causal,
    int warmup,
    int iterations) {
    shape.validate();
    if (inputs.query.size() != shape.qkv_elements() ||
        inputs.key.size() != shape.qkv_elements() ||
        inputs.value.size() != shape.qkv_elements()) {
        throw std::invalid_argument("Q, K, and V sizes must match the attention shape");
    }
    if (warmup < 0 || iterations <= 0) {
        throw std::invalid_argument("warmup must be non-negative and iterations must be positive");
    }
    if (!hip_backend_available()) {
        throw std::runtime_error("no HIP-capable GPU is visible");
    }

    const std::size_t tensor_bytes = shape.qkv_elements() * sizeof(float);
    const std::size_t score_bytes = shape.rows() * shape.sequence * sizeof(float);
    DeviceBuffer device_query(tensor_bytes);
    DeviceBuffer device_key(tensor_bytes);
    DeviceBuffer device_value(tensor_bytes);
    DeviceBuffer device_output(tensor_bytes);
    DeviceBuffer device_scores(algorithm == HipAlgorithm::materialized ? score_bytes : sizeof(float));

    WAVE_HIP_CHECK(hipMemcpy(device_query.get(), inputs.query.data(), tensor_bytes, hipMemcpyHostToDevice));
    WAVE_HIP_CHECK(hipMemcpy(device_key.get(), inputs.key.data(), tensor_bytes, hipMemcpyHostToDevice));
    WAVE_HIP_CHECK(hipMemcpy(device_value.get(), inputs.value.data(), tensor_bytes, hipMemcpyHostToDevice));

    for (int iteration = 0; iteration < warmup; ++iteration) {
        launch_attention(
            shape,
            algorithm,
            causal,
            device_query.get(),
            device_key.get(),
            device_value.get(),
            device_output.get(),
            device_scores.get());
    }
    WAVE_HIP_CHECK(hipDeviceSynchronize());

    hipEvent_t start = nullptr;
    hipEvent_t stop = nullptr;
    WAVE_HIP_CHECK(hipEventCreate(&start));
    WAVE_HIP_CHECK(hipEventCreate(&stop));
    WAVE_HIP_CHECK(hipEventRecord(start));
    for (int iteration = 0; iteration < iterations; ++iteration) {
        launch_attention(
            shape,
            algorithm,
            causal,
            device_query.get(),
            device_key.get(),
            device_value.get(),
            device_output.get(),
            device_scores.get());
    }
    WAVE_HIP_CHECK(hipEventRecord(stop));
    WAVE_HIP_CHECK(hipEventSynchronize(stop));

    float total_ms = 0.0f;
    WAVE_HIP_CHECK(hipEventElapsedTime(&total_ms, start, stop));
    hipEventDestroy(start);
    hipEventDestroy(stop);

    RunResult result;
    result.output.resize(shape.qkv_elements());
    WAVE_HIP_CHECK(hipMemcpy(result.output.data(), device_output.get(), tensor_bytes, hipMemcpyDeviceToHost));
    result.average_ms = total_ms / iterations;
    result.tflops = tflops(shape, result.average_ms);
    result.logical_bandwidth_gbs = logical_bandwidth(shape, result.average_ms);
    result.workspace_bytes = algorithm == HipAlgorithm::materialized ? score_bytes : 0;

    hipDeviceProp_t properties{};
    WAVE_HIP_CHECK(hipGetDeviceProperties(&properties, 0));
    result.device = properties.name;
    return result;
}

}  // namespace wave

