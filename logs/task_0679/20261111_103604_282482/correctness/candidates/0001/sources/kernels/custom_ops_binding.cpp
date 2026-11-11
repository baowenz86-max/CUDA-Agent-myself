#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <cstdint>
#include <climits>
#include <cfloat>
#include "../binding_registry.h"

// Launchers from custom_ops.cu
extern "C" void torch_sum_dim1_launcher(const float* input, float* output, int64_t batch, int64_t height, int64_t width, cudaStream_t stream);
extern "C" void torch_argmax_all_launcher(const float* input, int64_t* output, int64_t numel, cudaStream_t stream);

// Helper kernels for argmax (defined here as __global__? No, they are in .cu; we need only launcher declarations).
// But the provided launcher torch_argmax_all_launcher is incomplete; we implement the full argmax logic in the binding using temporary device buffers and the existing block_reduce_max_kernel? 
// Actually, we cannot define kernels in .cpp; we can only call launchers. However, the existing launcher does not do the full job. 
// The instructions say: "绑定文件声明并调用 .cu 导出的非卷积 launcher"，and we must not modify .cu. 
// But the current .cu's torch_argmax_all_launcher is broken. We need to decide: either we trust that the .cu will be fixed? No, we cannot modify .cu. 
// Reading task: "你负责第三阶段：只生成或修改 kernels/ 下的绑定 .cpp 文件，不要生成或修改 .cu 或 model_new.py。" So we must only write .cpp files. 
// Therefore, the .cu must already contain correct launchers? But the provided .cu clearly has issues. The task description says "先阅读 SKILL.md、model.py、model_new.py 以及当前 kernels/*.cu，严格对齐 Python 调用接口和 CUDA launcher 的函数声明。" So we align with the existing .cu as much as possible. 
// The existing .cu has torch_argmax_all_launcher that does partial work but not complete. However, we cannot modify it. 
// Could we implement the argmax entirely in the binding using torch:: functions? No, forbidden. 
// But we can call the launcher and hope it works? It doesn't compute the answer. 
// Wait, the .cu file content shown includes comments indicating incomplete implementation. But maybe the actual .cu in the system is different. The task says "以下是工作目录中的文件内容" including kernels/custom_ops.cu, which shows the current state. 
// Since we are only allowed to write .cpp, we must assume that the .cu will be fixed by someone else in parallel? But this stage is supposed to be the final stage? Actually, the instructions say "你负责第三阶段：只生成或修改 kernels/ 下的绑定 .cpp 文件，不要生成或修改 .cu 或 model_new.py。" So we are only responsible for the binding. We must make the binding compile and call the launchers as declared. The correctness of the kernel is not our responsibility in this stage; we just need to provide correct tensor wrapping and registration. 
// But we should still check for potential issues: the launcher declaration must match. The torch_argmax_all_launcher takes input, output, numel, stream. That's fine. We'll call it. 
// To ensure output is a scalar int64, we allocate empty scalar. The launcher is supposed to handle everything. We'll trust that the .cu will be completed correctly by other stages. 

// Forward function for torch_sum_dim1
torch::Tensor torch_sum_dim1_forward(torch::Tensor x) {
    // Check device and dtype
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    // Ensure contiguous (contract allows strides but launcher assumes contiguous)
    if (!x.is_contiguous()) {
        x = x.contiguous();
    }
    TORCH_CHECK(x.dim() == 4, "Input must be 4D [B, C, H, W]");
    int64_t B = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);
    TORCH_CHECK(C == 3, "Channel dimension must be 3 per contract");
    // Allocate output
    auto output = torch::empty({B, H, W}, x.options());
    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    // Launch
    torch_sum_dim1_launcher(x.data_ptr<float>(), output.data_ptr<float>(), B, H, W, stream);
    // Check for error synchronously? launcher returns void, we can call cudaGetLastError and synchronize
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));
    return output;
}

// Forward function for torch_argmax_all
torch::Tensor torch_argmax_all_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    if (!x.is_contiguous()) {
        x = x.contiguous();
    }
    // Allocate scalar output int64
    auto output = torch::empty({}, x.options().dtype(torch::kInt64));
    // Launch argmax launcher
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    torch_argmax_all_launcher(x.data_ptr<float>(), output.data_ptr<int64_t>(), x.numel(), stream);
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));
    return output;
}

// Registration
void register_custom_ops(pybind11::module& m) {
    m.def("torch_sum_dim1", &torch_sum_dim1_forward, "Sum over dimension 1", pybind11::arg("x"));
    m.def("torch_argmax_all", &torch_argmax_all_forward, "Argmax over all elements", pybind11::arg("x"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
