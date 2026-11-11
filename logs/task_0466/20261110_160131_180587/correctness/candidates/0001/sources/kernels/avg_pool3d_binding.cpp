#include <torch/types.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declaration of CUDA launcher from custom_ops.cu
extern "C" void avg_pool3d_forward_launcher(
    const float* input,
    float* output,
    int N, int C, int D, int H, int W,
    int in_stride_n, int in_stride_c, int in_stride_d, int in_stride_h, int in_stride_w,
    int out_D, int out_H, int out_W,
    cudaStream_t stream);

// PyTorch wrapper for 3D average pooling (2x2x2, stride 2x2x2)
torch::Tensor avg_pool3d_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 5, "Input must be a 5D tensor (N, C, D, H, W)");

    int N = x.size(0);
    int C = x.size(1);
    int D = x.size(2);
    int H = x.size(3);
    int W = x.size(4);

    TORCH_CHECK(D > 0 && H > 0 && W > 0, "Spatial dimensions must be positive");

    int out_D = D / 2;
    int out_H = H / 2;
    int out_W = W / 2;

    auto output = torch::empty({N, C, out_D, out_H, out_W}, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    avg_pool3d_forward_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, D, H, W,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
        out_D, out_H, out_W,
        stream);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

// Registration function
void register_avg_pool3d_forward(pybind11::module& m) {
    m.def("avg_pool3d_forward", &avg_pool3d_forward,
          "3D average pooling with kernel_size=2 and stride=2",
          py::arg("x"));
}

// Auto-register
REGISTER_BINDING(avg_pool3d_forward, register_avg_pool3d_forward);