#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void constant_pad3d_launcher(
    const float* input, float* output,
    int64_t N, int64_t C, int64_t D_in, int64_t H_in, int64_t W_in,
    int64_t D_out, int64_t H_out, int64_t W_out,
    int64_t pad_left, int64_t pad_right,
    int64_t pad_top, int64_t pad_bottom,
    int64_t pad_front, int64_t pad_back,
    cudaStream_t stream);

extern "C" void bias_add_launcher(
    const float* input, const float* bias, float* output,
    int64_t total_elements, int64_t C, int64_t spatial_size,
    cudaStream_t stream);

// PyTorch wrapper for constant_pad3d
// signature must match contract: input tensor, int64_t pad
// We ignore the pad value and use the fixed padding (1,2,1,2,1,2) as per task.
torch::Tensor constant_pad3d(torch::Tensor input, int64_t pad) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 5, "input must be 5D");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "input must be float32");

    auto N = input.size(0);
    auto C = input.size(1);
    auto D_in = input.size(2);
    auto H_in = input.size(3);
    auto W_in = input.size(4);

    // Hardcoded padding (left, right, top, bottom, front, back) = (1,2,1,2,1,2)
    int64_t pad_left = 1;
    int64_t pad_right = 2;
    int64_t pad_top = 1;
    int64_t pad_bottom = 2;
    int64_t pad_front = 1;
    int64_t pad_back = 2;

    int64_t D_out = D_in + pad_front + pad_back;
    int64_t H_out = H_in + pad_top + pad_bottom;
    int64_t W_out = W_in + pad_left + pad_right;

    auto output = torch::empty({N, C, D_out, H_out, W_out}, input.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    constant_pad3d_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, D_in, H_in, W_in,
        D_out, H_out, W_out,
        pad_left, pad_right,
        pad_top, pad_bottom,
        pad_front, pad_back,
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "constant_pad3d kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

// PyTorch wrapper for bias_add
torch::Tensor bias_add(torch::Tensor input, torch::Tensor bias) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 5, "input must be 5D");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(bias.scalar_type() == torch::kFloat32, "bias must be float32");

    auto C = input.size(1);
    TORCH_CHECK(bias.numel() == C, "bias must have C elements");

    auto bias_contig = bias.contiguous();

    auto N = input.size(0);
    auto D = input.size(2);
    auto H = input.size(3);
    auto W = input.size(4);
    int64_t spatial_size = D * H * W;
    int64_t total_elements = input.numel();

    auto output = torch::empty_like(input);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    bias_add_launcher(
        input.data_ptr<float>(),
        bias_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        total_elements,
        C,
        spatial_size,
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "bias_add kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

// Registration
void register_custom_ops(pybind11::module& m) {
    m.def("constant_pad3d", &constant_pad3d, "ConstantPad3d forward",
          py::arg("input"), py::arg("pad"));
    m.def("bias_add", &bias_add, "Bias add forward",
          py::arg("input"), py::arg("bias"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
