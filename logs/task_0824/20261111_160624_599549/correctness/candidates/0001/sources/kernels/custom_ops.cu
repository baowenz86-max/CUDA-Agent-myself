// CUDA kernels for custom ops in model_new.py
// All kernels use float32 only, as contract specifies.
#include <cuda_runtime.h>
#include <cstdint>

// -----------------------------------------------------------------------------
// Utility: index to multi-dim, multi-dim to index
// 4D tensor with strides
struct Stride4D { int64_t s0, s1, s2, s3; };

__device__ __forceinline__ int64_t get_linear_index(int64_t i0, int64_t i1, int64_t i2, int64_t i3,
                                                    const Stride4D& strides) {
    return i0 * strides.s0 + i1 * strides.s1 + i2 * strides.s2 + i3 * strides.s3;
}

// -----------------------------------------------------------------------------
// Kernel 1: BatchNorm2d inference (training=false only for now)
// input: [B, C, H, W] contiguous
// weight, bias, running_mean, running_var: length C
// output: same shape as input
__global__ void batch_norm_2d_inference_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    float eps,
    int64_t num_elements,
    int64_t C,
    int64_t HW)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t linear = idx; linear < num_elements; linear += stride) {
        int64_t c = (linear / HW) % C;
        float mean = running_mean[c];
        float var = running_var[c];
        float inv_std = rsqrtf(var + eps);
        float gamma = (weight != nullptr) ? weight[c] : 1.0f;
        float beta = (bias != nullptr) ? bias[c] : 0.0f;
        output[linear] = (input[linear] - mean) * inv_std * gamma + beta;
    }
}

// Launcher for batch_norm_2d_forward
// Note: training is ignored; this op is only called with training=False in model_new.py
// If training=true is required later, this launcher would need updating to update running stats.
extern "C" void batch_norm_2d_forward_launcher(
    const float* input,
    float* output,
    const float* weight,
    const float* bias,
    const float* running_mean,
    const float* running_var,
    float eps,
    bool training,
    int64_t num_elements,
    int64_t C,
    int64_t HW,
    cudaStream_t stream)
{
    if (training) {
        // Not implemented: contract says training=false for this model.
        // In binding, training should be checked; here we just copy if training is true for safety.
        // But to avoid silent errors, we simply return (binding should not call with true).
        return;
    }
    const int threads = 256;
    int blocks = (int)((num_elements + threads - 1) / threads);
    // Cap grid to avoid too many blocks; grid-stride loop handles rest
    if (blocks > 65535) blocks = 65535;
    batch_norm_2d_inference_kernel<<<blocks, threads, 0, stream>>>(
        input, output, weight, bias, running_mean, running_var, eps,
        num_elements, C, HW);
}

// -----------------------------------------------------------------------------
// Kernel 2: Permute contiguous for 4D tensors
// Input contiguous with shape [d0, d1, d2, d3]
// Output contiguous with permuted shape according to perm[4]
// We assume output pointer is different and is contiguous with the new layout.
__global__ void permute_contiguous_4d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t in_shape[4],
    int64_t in_strides[4],
    int64_t out_shape[4],
    int64_t out_strides[4],
    int64_t num_elements)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t linear_out = idx; linear_out < num_elements; linear_out += stride) {
        // compute output multi-index
        int64_t rem = linear_out;
        int64_t i0 = rem / out_strides[0]; rem %= out_strides[0];
        int64_t i1 = rem / out_strides[1]; rem %= out_strides[1];
        int64_t i2 = rem / out_strides[2]; rem %= out_strides[2];
        int64_t i3 = rem;

        // map to input indices using perm
        // perm gives destination dim for each input dim: output_dim_i = perm[input_dim_i]
        // So we need inverse: input_dim_j = inverse_perm[output_dim_j]
        // We'll store inverse_perm in passed arrays.
        int64_t in_multi[4] = {0,0,0,0};
        in_multi[inverse_perm[0]] = i0;
        in_multi[inverse_perm[1]] = i1;
        in_multi[inverse_perm[2]] = i2;
        in_multi[inverse_perm[3]] = i3;

        int64_t linear_in = in_multi[0] * in_strides[0] + in_multi[1] * in_strides[1]
                          + in_multi[2] * in_strides[2] + in_multi[3] * in_strides[3];
        output[linear_out] = input[linear_in];
    }
}

// We need inverse_perm accessible in kernel. We'll pass as constant memory or computed on host.
// Easier: use a struct with perm and inverse_perm arrays; allocate in constant memory.
// For simplicity, we pass a small array via kernel parameter (max 4).
__global__ void permute_contiguous_4d_kernel_v2(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t in_strides[4],
    int64_t out_strides[4],
    int64_t out_shape[4],
    int64_t num_elements)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t linear_out = idx; linear_out < num_elements; linear_out += stride) {
        // compute output indices i0,i1,i2,i3 from linear_out using out_strides
        int64_t rem = linear_out;
        int64_t i0 = rem / out_strides[0]; rem %= out_strides[0];
        int64_t i1 = rem / out_strides[1]; rem %= out_strides[1];
        int64_t i2 = rem / out_strides[2]; rem %= out_strides[2];
        int64_t i3 = rem;

        // compute input linear index: input dims are (dim0_in_dim, dim1_in_dim, dim2_in_dim, dim3_in_dim)
        // but we have the relation: output[i0,i1,i2,i3] corresponds to input with dims permuted.
        // Since perm is provided by caller, we could pass perm and compute on the fly. But simpler: on host we compute the inverse permutation and pass it as four ints.
        // However kernel needs to know which input dimension corresponds to each output dimension.
        // We'll pass perm[0..3] (output dim for each input dim) and compute inverse manually.
        // But we don't want to compute inverse in kernel. We'll pass inverse_perm[0..3] as parameters.
        // The host launcher will compute and pass inverse_perm.
        // For now, we assume that kernel receives inverse_perm as constant values embedded in the function through template? Instead, we pass through kernel arguments.
        // We'll define kernel with extra params: int64_t ip0, ip1, ip2, ip3 (inverse permutation).
        // But that makes signature messy. We'll use a struct passed by value.
    }
}

// We'll create a templated kernel with permutation as template parameters
// to avoid dynamic indexing? Since perm is fixed per call, but launcher can instantiate all 24 permutations.
template<int P0, int P1, int P2, int P3>
__global__ void permute_contiguous_4d_kernel_templated(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t in_strides[4],
    int64_t out_strides[4],
    int64_t out_shape[4],
    int64_t num_elements)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t linear_out = idx; linear_out < num_elements; linear_out += stride) {
        // compute output multi-index using out_strides
        int64_t rem = linear_out;
        int64_t i0 = rem / out_strides[0]; rem %= out_strides[0];
        int64_t i1 = rem / out_strides[1]; rem %= out_strides[1];
        int64_t i2 = rem / out_strides[2]; rem %= out_strides[2];
        int64_t i3 = rem;

        // Now map output dims to input dims: input_dim[Pj] = output_dim_j
        // That means input multi-index coordinates:
        int64_t input_indices[4];
        input_indices[P0] = i0;
        input_indices[P1] = i1;
        input_indices[P2] = i2;
        input_indices[P3] = i3;

        int64_t linear_in = input_indices[0] * in_strides[0] + input_indices[1] * in_strides[1]
                          + input_indices[2] * in_strides[2] + input_indices[3] * in_strides[3];
        output[linear_out] = input[linear_in];
    }
}

// Launcher for permute_contiguous_4d
// Input is contiguous with shape [d0,d1,d2,d3]. Output is contiguous with shape permuted.
// The binding will allocate output with proper sizes and pass pointers.
extern "C" void permute_contiguous_4d_launcher(
    const float* input,
    float* output,
    int64_t in_shape[4],
    int64_t perm[4],
    int64_t num_elements,
    cudaStream_t stream)
{
    // compute input strides (contiguous row-major)
    int64_t in_strides[4];
    in_strides[3] = 1;
    in_strides[2] = in_shape[3];
    in_strides[1] = in_strides[2] * in_shape[2];
    in_strides[0] = in_strides[1] * in_shape[1];

    // output shape = [in_shape[perm[0]], in_shape[perm[1]], in_shape[perm[2]], in_shape[perm[3]]]
    int64_t out_shape[4];
    for (int i = 0; i < 4; ++i) out_shape[i] = in_shape[perm[i]];
    int64_t out_strides[4];
    out_strides[3] = 1;
    out_strides[2] = out_shape[3];
    out_strides[1] = out_strides[2] * out_shape[2];
    out_strides[0] = out_strides[1] * out_shape[1];

    const int threads = 256;
    int blocks = (int)((num_elements + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;

    // Dispatch based on permutation (only 24 possibilities, but we handle common ones and fallback to generic?)
    // We'll implement a generic kernel that uses arrays, to avoid 24 instantiations.
    // But we already have a version that uses dynamic indexing; we'll use a generic kernel with inverse permutation.
    // We'll compute inverse_perm on host and pass it via constant memory or as args.
    // To keep it simple, we'll use a kernel that takes inverse_perm as four int64_t arguments.
    // We'll define that kernel here.
    // But we already have the templated version. We'll instead implement a generic kernel using local arrays.
    // We'll define a kernel below.
    // For now, this launcher will call a generic kernel defined outside.
    // However, to avoid code duplication, we'll define a generic permute kernel that uses arrays.
    // Since the launcher is external C, we can call that kernel.
    // We'll implement it as global function with name permute_contiguous_4d_generic_kernel.
    // We'll include its definition before launcher.
    // We'll call it with appropriate block/grid.
    // Note: need to pass in_strides, out_strides, out_shape.
    // We'll allocate kernel parameters on stack and pass to kernel.
    // The kernel signature: (input, output, in_strides, out_strides, out_shape, num_elements)
    // But it also needs to know permutation to compute input index from output index.
    // In that kernel, we can compute input index by using out_strides and out_shape to get output multi-index,
    // then use perm array to map to input multi-index. But perm array is on host; need to pass to device.
    // We can pass perm as an array on constant memory or use a struct. We'll pass perm as four int64_t fields.
    // So kernel signature includes p0,p1,p2,p3 and in_strides, out_strides, out_shape.
    // We'll adopt this approach.
    // We'll define kernel with 4 permutation ints and arrays.
    // We'll call it from here.
    // We'll do this below.
}

// Generic permute kernel with inverse permutation arguments
__global__ void permute_contiguous_4d_generic_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t in_strides[4],
    int64_t out_strides[4],
    int64_t out_shape[4],
    int64_t num_elements)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t linear_out = idx; linear_out < num_elements; linear_out += stride) {
        // output multi-index
        int64_t rem = linear_out;
        int64_t i0 = rem / out_strides[0]; rem %= out_strides[0];
        int64_t i1 = rem / out_strides[1]; rem %= out_strides[1];
        int64_t i2 = rem / out_strides[2]; rem %= out_strides[2];
        int64_t i3 = rem;

        // Map to input indices using permutation stored as constant memory? We'll use
        // a static __constant__ array set by launcher via cudaMemcpyToSymbol.
        // Since the launcher is C, it can call cudaMemcpyToSymbol.
        // But to keep pure kernel, we can pass perm as four int64_t parameters.
        // Let's modify to have p0..p3.
    }
}

// Instead of complicating, we'll implement a simpler approach: since both input and output
// are contiguous, we can define a kernel that uses a precomputed mapping function based on
// the permutation. We'll writelauncher code that selects a templated instantiation.
// We'll implement 24 explicit instantiations? That's too much.
// Alternative: use inverse permutation computed on host, then pass as four int64_t, and kernel computes input linear index as:
// input_linear = i0*in_strides[inv_perm[0]] + i1*in_strides[inv_perm[1]] + i2*in_strides[inv_perm[2]] + i3*in_strides[inv_perm[3]]
// That way we need inv_perm[4] and in_strides[4].
// We'll define kernel with two arrays of 4 int64_t: in_strides, inv_perm.
// And output multi-index computed from out_strides (which are determined by output shape but we can compute on the fly using out_shape).
// Actually, we can compute output multi-index from out_shape and linear_out, similar to before.
// We'll pass in_strides, inv_perm, out_shape, num_elements.
// The kernel will compute i0..i3 from linear_out and out_shape, then compute in_linear as sum of i_j * in_strides[inv_perm[j]] (since output dim j corresponds to input dim inv_perm[j]).
// This is efficient and general.
// Let's implement that.

__global__ void permute_contiguous_4d_kernel_final(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t in_strides[4],
    int64_t inv_perm[4],
    int64_t out_shape[4],
    int64_t num_elements)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t linear_out = idx; linear_out < num_elements; linear_out += stride) {
        // compute output multi-index using out_shape (contiguous row-major)
        int64_t rem = linear_out;
        int64_t i3 = rem % out_shape[3]; rem /= out_shape[3];
        int64_t i2 = rem % out_shape[2]; rem /= out_shape[2];
        int64_t i1 = rem % out_shape[1]; rem /= out_shape[1];
        int64_t i0 = rem;

        // input linear index
        int64_t linear_in = i0 * in_strides[inv_perm[0]]
                          + i1 * in_strides[inv_perm[1]]
                          + i2 * in_strides[inv_perm[2]]
                          + i3 * in_strides[inv_perm[3]];
        output[linear_out] = input[linear_in];
    }
}

extern "C" void permute_contiguous_4d_launcher(
    const float* input,
    float* output,
    int64_t in_shape[4],
    int64_t perm[4],
    int64_t num_elements,
    cudaStream_t stream)
{
    int64_t in_strides[4];
    in_strides[3] = 1;
    in_strides[2] = in_shape[3];
    in_strides[1] = in_strides[2] * in_shape[2];
    in_strides[0] = in_strides[1] * in_shape[1];

    int64_t out_shape[4];
    for (int i = 0; i < 4; ++i) out_shape[i] = in_shape[perm[i]];

    // compute inverse permutation: inv_perm[output_dim] = input_dim
    int64_t inv_perm[4];
    for (int i = 0; i < 4; ++i) inv_perm[perm[i]] = i;

    const int threads = 256;
    int blocks = (int)((num_elements + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;

    permute_contiguous_4d_kernel_final<<<blocks, threads, 0, stream>>>(
        input, output, in_strides, inv_perm, out_shape, num_elements);
}

// -----------------------------------------------------------------------------
// Kernel 3: view operations (reshape contiguous tensor to different dims)
// Since view does not change data, we can just return a tensor with same storage.
// However, binding might allocate a new tensor with same storage or copy.
// To be safe, we provide a copy kernel that copies num_elements from input to output if pointers differ.
// If pointers are same, no copy needed.
__global__ void copy_kernel(const float* __restrict__ input, float* __restrict__ output, int64_t num_elements)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < num_elements; i += stride) {
        output[i] = input[i];
    }
}

extern "C" void view_contiguous_copy_launcher(
    const float* input,
    float* output,
    int64_t num_elements,
    cudaStream_t stream)
{
    if (input == output) return;
    const int threads = 256;
    int blocks = (int)((num_elements + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;
    copy_kernel<<<blocks, threads, 0, stream>>>(input, output, num_elements);
}

// -----------------------------------------------------------------------------
// Kernel 4: MaxPool1d forward
// Input: [B, H, L] contiguous
// Output: [B, H, L_out] contiguous
// kernel_size, stride, padding, dilation, ceil_mode given
__global__ void max_pool1d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t B,
    int64_t H,
    int64_t L,
    int64_t L_out,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    bool ceil_mode)
{
    int64_t total_out = B * H * L_out;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t step = blockDim.x * gridDim.x;
    for (int64_t linear = idx; linear < total_out; linear += step) {
        // decompose: linear = ((b*H + h)*L_out + l_out)
        int64_t l_out = linear % L_out;
        int64_t bh = linear / L_out;
        int64_t h = bh % H;
        int64_t b = bh / H;

        // compute start index in input for this output position
        int64_t start = l_out * stride - padding;
        float max_val = -INFINITY;
        bool found = false;
        for (int k = 0; k < kernel_size; ++k) {
            int64_t in_idx = start + k * dilation;
            if (in_idx >= 0 && in_idx < L) {
                float val = input[((b * H + h) * L) + in_idx];
                if (!found || val > max_val) {
                    max_val = val;
                    found = true;
                }
            }
        }
        // PyTorch: if no valid indices (should not happen with padding=1,kernel=3,dilation=1,stride=2, L_out=192 for L=384)
        // but we still output something. If not found, output -inf or 0? PyTorch pads with -inf before max.
        // For safety, if !found, output 0.
        output[linear] = found ? max_val : 0.0f;
    }
}

extern "C" void max_pool1d_forward_launcher(
    const float* input,
    float* output,
    int64_t B,
    int64_t H,
    int64_t L,
    int64_t L_out,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    bool ceil_mode,
    cudaStream_t stream)
{
    int64_t total_out = B * H * L_out;
    if (total_out == 0) return;
    const int threads = 256;
    int blocks = (int)((total_out + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;
    max_pool1d_kernel<<<blocks, threads, 0, stream>>>(
        input, output, B, H, L, L_out, kernel_size, stride, padding, dilation, ceil_mode);
}

// -----------------------------------------------------------------------------
// End of custom_ops.cu
