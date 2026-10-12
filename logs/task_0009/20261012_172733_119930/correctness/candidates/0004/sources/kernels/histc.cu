#include <cuda_runtime.h>
#include <torch/types.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Kernel for histogram
// Each thread processes multiple elements using grid-stride loop
__global__ void histc_kernel(
    const float* __restrict__ input,
    long long num_elements,
    int bins,
    float min_val,
    float max_val,
    float* __restrict__ output) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // 计算 bin 宽度
    float bin_width = (max_val - min_val) / (float)bins;

    for (long long i = idx; i < num_elements; i += stride) {
        float val = input[i];
        // 忽略 NaN 和范围外值
        if (isnan(val) || val < min_val || val > max_val) {
            continue;
        }
        // 计算 bin 索引
        int bin = (int)((val - min_val) / bin_width);
        // 处理 val == max_val 的情况
        if (bin >= bins) bin = bins - 1;
        // 原子加
        atomicAdd(&output[bin], 1.0f);
    }
}

// Host launcher
void histc_forward_launcher(
    const float* input,
    long long num_elements,
    int bins,
    float min_val,
    float max_val,
    float* output,
    cudaStream_t stream) {

    int block_size = 256;
    // 根据元素数量决定 grid 大小，但限制最大 grid 以避免过多 block
    long long desired_blocks = (num_elements + block_size - 1) / block_size;
    int max_blocks = 65535; // 可以更大，但这里限制
    int blocks = (int)min(desired_blocks, (long long)max_blocks);

    histc_kernel<<<blocks, block_size, 0, stream>>>(
        input, num_elements, bins, min_val, max_val, output);
}

namespace py = pybind11;

// Binding wrapper
torch::Tensor histc_forward(torch::Tensor input, int64_t bins, double min_val, double max_val) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(bins > 0, "bins must be positive");

    auto output = torch::zeros({bins}, input.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    histc_forward_launcher(
        input.data_ptr<float>(),
        input.numel(),
        (int)bins,
        (float)min_val,
        (float)max_val,
        output.data_ptr<float>(),
        stream
    );

    return output;
}

void register_histc(pybind11::module& m) {
    m.def("histc_forward", &histc_forward,
          "Histogram forward",
          py::arg("input"), py::arg("bins"), py::arg("min"), py::arg("max"));
}

REGISTER_BINDING(histc, register_histc);
