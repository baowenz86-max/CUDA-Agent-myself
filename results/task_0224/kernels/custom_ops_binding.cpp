// Binding for hardtanh and avgpool2d custom CUDA kernels
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>

#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void hardtanh_forward_launcher(
    float* output,
    const float* input,
    long long total_elements,
    float min_val,
    float max_val,
    cudaStream_t stream);

extern "C" void avgpool2d_forward_launcher(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    int kernel_size, int stride, int padding,
    cudaStream_t stream);

// Utility to convert scalar->int pair (tuple of two ints) or int to [int,int]
static inline std::pair<int,int> parse_pair(const pybind11::object& v) {
    if (pybind11::isinstance<pybind11::tuple>(v)) {
        pybind11::tuple t = v.cast<pybind11::tuple>();
        TORCH_CHECK(t.size() == 2, "Expected tuple of size 2");
        return {t[0].cast<int>(), t[1].cast<int>()};
    } else {
        int val = v.cast<int>();
        return {val, val};
    }
}

// hardtanh_forward binding
torch::Tensor hardtanh_forward(torch::Tensor input, double min_val = -1.0, double max_val = 1.0) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(input.numel() > 0, "Input cannot be empty");

    // Ensure input is contiguous; if not, make contiguous (hardtanh is elementwise, safe)
    auto input_contig = input.contiguous();
    auto output = torch::empty_like(input_contig);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    hardtanh_forward_launcher(
        output.data_ptr<float>(),
        input_contig.data_ptr<float>(),
        static_cast<long long>(input_contig.numel()),
        static_cast<float>(min_val),
        static_cast<float>(max_val),
        stream);

    // If input was non-contiguous, output is contiguous but Python expects same shape/stride as input; 
    // however reference output is contiguous from a contiguous input. Since input in this task is contiguous,
    // this branch is not taken. To be safe, return contiguous output with original shape (same as input).
    // But PyTorch AvgPool2d returns contiguous; hardtanh preserves layout. We'll return output as is.
    // If input was non-contiguous, output shape matches but stride may differ; but contract allows strides support,
    // so we can return output with contiguous stride. For exact match, we could reshape to original strides, but
    // that would require as_strided; not necessary for this task as input is contiguous. We'll keep output.
    return output;
}

// avgpool2d_forward binding
torch::Tensor avgpool2d_forward(torch::Tensor input, pybind11::object kernel_size, pybind11::object stride, pybind11::object padding) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(input.dim() == 4, "Input must be 4D (N,C,H,W)");
    TORCH_CHECK(input.size(0) > 0 && input.size(1) > 0 && input.size(2) > 0 && input.size(3) > 0, "Input dimensions must be positive");

    auto kp = parse_pair(kernel_size);
    auto sp = parse_pair(stride);
    auto pp = parse_pair(padding);

    // The CUDA kernel implementation only supports exactly kernel=(2,2), stride=(1,1), padding=(0,0)
    // as per the contract. Enforce this to avoid silent incorrect results.
    TORCH_CHECK(kp.first == 2 && kp.second == 2, "Only kernel_size=(2,2) is supported");
    TORCH_CHECK(sp.first == 1 && sp.second == 1, "Only stride=(1,1) is supported");
    TORCH_CHECK(pp.first == 0 && pp.second == 0, "Only padding=(0,0) is supported");

    int N = static_cast<int>(input.size(0));
    int C = static_cast<int>(input.size(1));
    int H = static_cast<int>(input.size(2));
    int W = static_cast<int>(input.size(3));

    // Compute output dimensions
    int H_out = (H + 2 * pp.first - kp.first) / sp.first + 1;
    int W_out = (W + 2 * pp.second - kp.second) / sp.second + 1;

    auto output = torch::empty({N, C, H_out, W_out}, input.options());

    // The avgpool kernel expects input contiguous
    auto input_contig = input.contiguous();

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    avgpool2d_forward_launcher(
        input_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        2, 1, 0,  // hardcoded for enforced case
        stream);

    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("hardtanh_forward", &hardtanh_forward,
          "hardtanh forward (clamp)",
          pybind11::arg("input"),
          pybind11::arg("min_val") = -1.0,
          pybind11::arg("max_val") = 1.0);

    m.def("avgpool2d_forward", &avgpool2d_forward,
          "avgpool2d forward",
          pybind11::arg("input"),
          pybind11::arg("kernel_size"),
          pybind11::arg("stride"),
          pybind11::arg("padding"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
