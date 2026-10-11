#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <limits>
#include "../binding_registry.h"
#include "attention_launchers.h"

at::Tensor fused_attention_forward(
    at::Tensor x, at::Tensor y, at::Tensor in_weight, at::Tensor in_bias,
    at::Tensor out_weight, at::Tensor out_bias,
    at::Tensor query_projection, at::Tensor key_projection, at::Tensor value_projection,
    at::Tensor query, at::Tensor key, at::Tensor value, at::Tensor probabilities,
    at::Tensor context, at::Tensor merged, at::Tensor projected, at::Tensor output,
    int64_t heads, bool batch_first, int64_t repeats, double scale) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    for (const auto& tensor : {x, y, in_weight, in_bias, out_weight, out_bias,
                              query_projection, key_projection, value_projection,
                              query, key, value, probabilities, context, merged, projected, output}) {
        TORCH_CHECK(tensor.is_cuda() && tensor.device() == x.device(), "All tensors must share a CUDA device");
        TORCH_CHECK(tensor.scalar_type() == at::kFloat, "Only float32 is supported");
        TORCH_CHECK(tensor.is_contiguous(), "All tensors must be contiguous");
    }
    TORCH_CHECK(x.dim() == 3 && y.dim() == 3, "Inputs must be 3D");
    int64_t batch = x.size(batch_first ? 0 : 1);
    int64_t queries = x.size(batch_first ? 1 : 0);
    int64_t keys = y.size(batch_first ? 1 : 0);
    int64_t embed = x.size(2);
    TORCH_CHECK(embed > 0 && heads > 0 && embed % heads == 0, "Invalid embedding or head count");
    TORCH_CHECK(repeats >= 0, "repeats must be nonnegative");
    TORCH_CHECK(y.size(batch_first ? 0 : 1) == batch && y.size(2) == embed, "Input dimensions mismatch");
    const int64_t limit = std::numeric_limits<int>::max();
    TORCH_CHECK(batch <= limit && queries <= limit && keys <= limit && embed <= limit && heads <= limit && repeats <= limit,
                "Dimensions exceed supported limits");
    TORCH_CHECK(batch * queries <= limit && batch * keys <= limit && batch * heads <= limit,
                "GEMM dimensions exceed supported limits");
    TORCH_CHECK(in_weight.dim() == 2 && in_weight.size(0) == 3 * embed && in_weight.size(1) == embed,
                "in_proj_weight must have shape [3E, E]");
    TORCH_CHECK(out_weight.dim() == 2 && out_weight.size(0) == embed && out_weight.size(1) == embed,
                "out_proj_weight must have shape [E, E]");
    TORCH_CHECK(in_bias.dim() == 1 && in_bias.numel() == 3 * embed && out_bias.dim() == 1 && out_bias.numel() == embed,
                "Invalid bias dimensions");
    TORCH_CHECK(output.dim() == 3 && output.size(2) == embed &&
                output.size(0) == (batch_first ? batch : queries) &&
                output.size(1) == (batch_first ? queries : batch) * repeats, "Invalid output shape");
    if (batch == 0 || queries == 0 || repeats == 0)
        return output;
    TORCH_CHECK(query_projection.numel() == batch * queries * embed && query.numel() == batch * queries * embed &&
                context.numel() == batch * queries * embed && merged.numel() == batch * queries * embed &&
                projected.numel() == batch * queries * embed, "Invalid query workspace size");
    TORCH_CHECK(key_projection.numel() == batch * keys * embed && value_projection.numel() == batch * keys * embed &&
                key.numel() == batch * keys * embed && value.numel() == batch * keys * embed, "Invalid key workspace size");
    TORCH_CHECK(probabilities.numel() == batch * heads * queries * keys, "Invalid score workspace size");
    c10::cuda::CUDAGuard guard(x.device());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(x.get_device()).stream();
    if (keys == 0) {
        C10_CUDA_CHECK(cudaMemsetAsync(projected.data_ptr<float>(), 0, projected.numel() * sizeof(float), stream));
    } else {
        int batch_count = static_cast<int>(batch);
        int query_count = static_cast<int>(queries);
        int key_count = static_cast<int>(keys);
        int features = static_cast<int>(embed);
        int head_count = static_cast<int>(heads);
        const float* weights = in_weight.data_ptr<float>();
        const float* biases = in_bias.data_ptr<float>();
        attention_gemm(x.data_ptr<float>(), weights, query_projection.data_ptr<float>(), batch * queries, features, features, stream);
        attention_gemm(y.data_ptr<float>(), weights + embed * embed, key_projection.data_ptr<float>(), batch * keys, features, features, stream);
        attention_gemm(y.data_ptr<float>(), weights + 2 * embed * embed, value_projection.data_ptr<float>(), batch * keys, features, features, stream);
        attention_pack(query_projection.data_ptr<float>(), biases, query.data_ptr<float>(), batch_count, query_count, features, head_count, batch_first, stream);
        attention_pack(key_projection.data_ptr<float>(), biases + embed, key.data_ptr<float>(), batch_count, key_count, features, head_count, batch_first, stream);
        attention_pack(value_projection.data_ptr<float>(), biases + 2 * embed, value.data_ptr<float>(), batch_count, key_count, features, head_count, batch_first, stream);
        attention_scores(query.data_ptr<float>(), key.data_ptr<float>(), probabilities.data_ptr<float>(), batch * heads, query_count, key_count, embed / heads, stream);
        attention_softmax(probabilities.data_ptr<float>(), batch * heads * queries, key_count, stream);
        attention_values(probabilities.data_ptr<float>(), value.data_ptr<float>(), context.data_ptr<float>(), batch * heads, query_count, key_count, embed / heads, stream);
        attention_merge(context.data_ptr<float>(), merged.data_ptr<float>(), batch_count, query_count, features, head_count, batch_first, stream);
        attention_gemm(merged.data_ptr<float>(), out_weight.data_ptr<float>(), projected.data_ptr<float>(), batch * queries, features, features, stream);
    }
    attention_epilogue(projected.data_ptr<float>(), out_bias.data_ptr<float>(), output.data_ptr<float>(),
                       batch, queries, embed, repeats, static_cast<float>(scale), batch_first, stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

void register_fused_attention(pybind11::module& module) {
    module.def("fused_attention_forward", &fused_attention_forward);
}

REGISTER_BINDING(fused_attention, register_fused_attention);
