#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

extern "C" void upsample_bilinear_2d_launcher(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    float scale_factor,
    cudaStream_t stream);

torch::Tensor upsample_bilinear_2d(torch::Tensor x, double scale_factor) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 4, "Input must have 4 dimensions (N,C,H,W)");
    TORCH_CHECK(scale_factor > 0.0, "scale_factor must be positive");
    
    // Ensure input is contiguous (already done in model_new.py, but keep for safety)
    auto x_contig = x.contiguous();
    int N = x_contig.size(0);
    int C = x_contig.size(1);
    int H = x_contig.size(2);
    int W = x_contig.size(3);
    
    int outH = static_cast<int>(H * scale_factor);
    int outW = static_cast<int>(W * scale_factor);
    TORCH_CHECK(outH > 0 && outW > 0, "Invalid output dimensions");
    
    // Allocate output with same options (dtype, device)
    auto output = torch::empty({N, C, outH, outW}, x_contig.options());
    
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    
    upsample_bilinear_2d_launcher(
        x_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        static_cast<float>(scale_factor),
        stream);
    
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));
    
    return output;
}

void register_upsample_bilinear_2d(pybind11::module& m) {
    m.def("upsample_bilinear_2d", &upsample_bilinear_2d,
          "Bilinear 2D upsampling (align_corners=False)",
          py::arg("x"), py::arg("scale_factor"));
}

REGISTER_BINDING(upsample_bilinear_2d, register_upsample_bilinear_2d);
