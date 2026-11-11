// Binding for custom ops (embedding, permute, bn eval)
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>

#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void embedding_forward_launcher(const int64_t* input,
                                            const float* weight,
                                            float* output,
                                            int64_t total_elements,
                                            int64_t embedding_dim,
                                            cudaStream_t stream);

extern "C" void permute_4d_to_bn3d_launcher(const float* input,
                                             float* output,
                                             int64_t total_elements,
                                             int64_t B, int64_t C, int64_t H, int64_t W, int64_t D,
                                             cudaStream_t stream);

extern "C" void bn3d_eval_forward_launcher(const float* input,
                                            float* output,
                                            const float* weight,
                                            const float* bias,
                                            const float* running_mean,
                                            const float* running_var,
                                            float eps,
                                            int64_t total_elements,
                                            int64_t C,
                                            int64_t HW_D,
                                            cudaStream_t stream);

extern "C" void permute_bn3d_to_4d_launcher(const float* input,
                                             float* output,
                                             int64_t total_elements,
                                             int64_t B, int64_t C, int64_t H, int64_t W, int64_t D,
                                             cudaStream_t stream);

// Utility to check CUDA error after launch
static void check_cuda_error() {
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));
}

// embedding_forward: input [B,H,W,D] int64, weight [num_embeddings, embedding_dim] float32
// returns [B,H,W,D,embedding_dim] float32
torch::Tensor embedding_forward(torch::Tensor input, torch::Tensor weight,
                                int64_t num_embeddings, int64_t embedding_dim) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kInt64, "input must be int64");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");

    auto sizes = input.sizes();
    TORCH_CHECK(sizes.size() == 4, "input must be 4D");
    int64_t B = sizes[0], H = sizes[1], W = sizes[2], D = sizes[3];
    TORCH_CHECK(weight.size(0) == num_embeddings, "weight first dim must equal num_embeddings");
    TORCH_CHECK(weight.size(1) == embedding_dim, "weight second dim must equal embedding_dim");

    // FIX: allocate output with float32 dtype and same device as weight (or input, but dtype float32)
    auto output = torch::empty({B, H, W, D, embedding_dim}, weight.options());
    int64_t total_elements = input.numel();

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    embedding_forward_launcher(
        input.data_ptr<int64_t>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        total_elements,
        embedding_dim,
        stream
    );
    check_cuda_error();
    return output;
}

// permute_4d_to_bn3d: input [B,H,W,D,C] -> [B,C,H,W,D]
torch::Tensor permute_4d_to_bn3d(torch::Tensor input,
                                 int64_t batch_size, int64_t embedding_dim,
                                 int64_t height, int64_t width, int64_t depth) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");

    auto sizes = input.sizes();
    TORCH_CHECK(sizes.size() == 5, "input must be 5D");
    TORCH_CHECK(sizes[0] == batch_size && sizes[1] == height && sizes[2] == width && sizes[3] == depth && sizes[4] == embedding_dim,
                "input shape mismatch");

    auto output = torch::empty({batch_size, embedding_dim, height, width, depth}, input.options());
    int64_t total_elements = input.numel();

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    permute_4d_to_bn3d_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        total_elements,
        batch_size, embedding_dim, height, width, depth,
        stream
    );
    check_cuda_error();
    return output;
}

// bn3d_eval_forward: input [B,C,H,W,D], weight/bias/running_mean/running_var: [C]
// returns [B,C,H,W,D]
torch::Tensor bn3d_eval_forward(torch::Tensor input,
                                torch::Tensor weight, torch::Tensor bias,
                                torch::Tensor running_mean, torch::Tensor running_var,
                                double eps) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda() && bias.is_cuda() && running_mean.is_cuda() && running_var.is_cuda(),
                "BN parameters must be CUDA tensors");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32 && bias.dtype() == torch::kFloat32 &&
                running_mean.dtype() == torch::kFloat32 && running_var.dtype() == torch::kFloat32,
                "BN parameter tensors must be float32");
    TORCH_CHECK(input.is_contiguous() && weight.is_contiguous() && bias.is_contiguous() &&
                running_mean.is_contiguous() && running_var.is_contiguous(),
                "all tensors must be contiguous");

    auto sizes = input.sizes();
    TORCH_CHECK(sizes.size() == 5, "input must be 5D");
    int64_t B = sizes[0], C = sizes[1], H = sizes[2], W = sizes[3], D = sizes[4];
    TORCH_CHECK(weight.size(0) == C && bias.size(0) == C &&
                running_mean.size(0) == C && running_var.size(0) == C,
                "BN parameter length must match channel dimension");

    auto output = torch::empty_like(input);
    int64_t total_elements = input.numel();
    int64_t HW_D = H * W * D;
    float eps_f = static_cast<float>(eps);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    bn3d_eval_forward_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        eps_f,
        total_elements,
        C,
        HW_D,
        stream
    );
    check_cuda_error();
    return output;
}

// permute_bn3d_to_4d: input [B,C,H,W,D] -> [B,H,W,D,C]
torch::Tensor permute_bn3d_to_4d(torch::Tensor input,
                                 int64_t batch_size, int64_t embedding_dim,
                                 int64_t height, int64_t width, int64_t depth) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");

    auto sizes = input.sizes();
    TORCH_CHECK(sizes.size() == 5, "input must be 5D");
    TORCH_CHECK(sizes[0] == batch_size && sizes[1] == embedding_dim && sizes[2] == height &&
                sizes[3] == width && sizes[4] == depth, "input shape mismatch");

    auto output = torch::empty({batch_size, height, width, depth, embedding_dim}, input.options());
    int64_t total_elements = input.numel();

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    permute_bn3d_to_4d_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        total_elements,
        batch_size, embedding_dim, height, width, depth,
        stream
    );
    check_cuda_error();
    return output;
}

// Register all custom ops
void register_custom_ops(pybind11::module& m) {
    m.def("embedding_forward", &embedding_forward,
          "Embedding forward (CUDA)",
          py::arg("input"), py::arg("weight"),
          py::arg("num_embeddings"), py::arg("embedding_dim"));

    m.def("permute_4d_to_bn3d", &permute_4d_to_bn3d,
          "Permute [B,H,W,D,C] to [B,C,H,W,D] (CUDA)",
          py::arg("input"), py::arg("batch_size"), py::arg("embedding_dim"),
          py::arg("height"), py::arg("width"), py::arg("depth"));

    m.def("bn3d_eval_forward", &bn3d_eval_forward,
          "BatchNorm3d eval forward (CUDA)",
          py::arg("input"), py::arg("weight"), py::arg("bias"),
          py::arg("running_mean"), py::arg("running_var"), py::arg("eps"));

    m.def("permute_bn3d_to_4d", &permute_bn3d_to_4d,
          "Permute [B,C,H,W,D] to [B,H,W,D,C] (CUDA)",
          py::arg("input"), py::arg("batch_size"), py::arg("embedding_dim"),
          py::arg("height"), py::arg("width"), py::arg("depth"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
