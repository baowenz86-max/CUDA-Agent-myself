#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// 声明 launcher 函数
extern "C" void compute_histc_launcher(
    const float* input,
    int64_t total_elements,
    float* output,
    int64_t bins,
    float min_val,
    float max_val,
    cudaStream_t stream
);

extern "C" void cat_histc_launcher(
    const float* input,
    float* output,
    int64_t total_elements,
    cudaStream_t stream
);

// compute_histc 包装函数
torch::Tensor compute_histc(torch::Tensor input, int64_t bins, double min, double max) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(bins > 0, "bins must be positive");

    auto output = torch::empty({bins}, input.options());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    compute_histc_launcher(
        input.data_ptr<float>(),
        input.numel(),
        output.data_ptr<float>(),
        bins,
        static_cast<float>(min),
        static_cast<float>(max),
        stream
    );

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "compute_histc_launcher failed");
    return output;
}

// cat_histc 包装函数
torch::Tensor cat_histc(torch::Tensor histc_list, int64_t num_images, int64_t bins) {
    TORCH_CHECK(histc_list.is_cuda(), "histc_list must be a CUDA tensor");
    TORCH_CHECK(histc_list.dtype() == torch::kFloat32, "histc_list must be float32");
    TORCH_CHECK(histc_list.is_contiguous(), "histc_list must be contiguous");
    TORCH_CHECK(num_images > 0, "num_images must be positive");
    TORCH_CHECK(bins > 0, "bins must be positive");
    TORCH_CHECK(histc_list.numel() == num_images * bins, "histc_list numel mismatch");

    auto output = torch::empty({num_images * bins}, histc_list.options());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    cat_histc_launcher(
        histc_list.data_ptr<float>(),
        output.data_ptr<float>(),
        histc_list.numel(),
        stream
    );

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "cat_histc_launcher failed");
    return output;
}

// 注册函数
void register_custom_ops(pybind11::module& m) {
    m.def("compute_histc", &compute_histc,
          "Compute histogram of a CUDA tensor",
          py::arg("input"),
          py::arg("bins"),
          py::arg("min"),
          py::arg("max"));
    m.def("cat_histc", &cat_histc,
          "Concatenate histograms along dim 0",
          py::arg("histc_list"),
          py::arg("num_images"),
          py::arg("bins"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
