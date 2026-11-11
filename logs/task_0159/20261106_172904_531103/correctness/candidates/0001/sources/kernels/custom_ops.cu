#include <cuda_runtime.h>
#include <cstdint>

// Kernel: view_as_real
// Input: complex64 tensor of shape [B, H, W] with arbitrary strides (in elements)
// Output: float32 tensor of shape [B, H, W, 2], contiguous, with last dim size 2
__global__ void view_as_real_kernel(
    const float2* __restrict__ input,
    float* __restrict__ output,
    int64_t total_complex,
    int64_t stride_b, int64_t stride_h, int64_t stride_w)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (int64_t i = idx; i < total_complex; i += stride) {
        // Compute 3D coordinates from flat index assuming contiguous input layout
        int64_t w = i % stride_w;
        int64_t tmp = i / stride_w;
        int64_t h = tmp % stride_h;
        int64_t b = tmp / stride_h;

        // Given that input is usually contiguous, but we must handle arbitrary strides.
        // The flat index i is based on contiguous logical indexing, but input may not be contiguous.
        // To access the actual element, use provided strides:
        int64_t input_offset = b * stride_b + h * stride_h + w * stride_w;
        float2 val = input[input_offset];
        int64_t out_offset = i * 2;
        output[out_offset] = val.x;
        output[out_offset + 1] = val.y;
    }
}

// Launcher for view_as_real
// input_data: pointer to complex64 elements
// output_data: pointer to float32 output, size total_complex*2
// dims: array of 3 int64_t [B, H, W]
// strides: array of 3 int64_t [stride_b, stride_h, stride_w] in elements
// stream: CUDA stream
extern "C" void view_as_real_launcher(
    const void* input_data,
    float* output_data,
    const int64_t* dims,
    const int64_t* strides,
    cudaStream_t stream)
{
    int64_t total_complex = dims[0] * dims[1] * dims[2];
    int threads = 256;
    int blocks = (int)((total_complex + threads - 1) / threads);
    // Grid cap not needed since total_complex ~ 64M, but use grid-stride loop anyway.
    view_as_real_kernel<<<blocks, threads, 0, stream>>>(
        (const float2*)input_data,
        output_data,
        total_complex,
        strides[0], strides[1], strides[2]
    );
}

// Kernel: constant_pad
// Input: float32 tensor of shape [B, H, W, 2] with arbitrary strides (in elements)
// Output: float32 tensor of shape [B, H, W+2, 4], contiguous, last dim size 4
// Pad: pad_left=1, pad_right=1, pad_top=1, pad_bottom=1 (applied to last two dims: W and H)
// Value: pad_value (float)
// Contract: pad parameter is tuple (1,1,1,1) -> (left, right, top, bottom)
__global__ void constant_pad_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t B, int64_t H_in, int64_t W_in,
    int64_t stride_b, int64_t stride_h, int64_t stride_w,
    int64_t out_H, int64_t out_W,
    float pad_value)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total_out = B * out_H * out_W * 2; // output elements count (including last dim 2? Wait output last dim is 4)
    // Actually output shape [B, H+2, W+2, 4], so last dim is 4.
    // But input last dim was 2. The padding is applied to last two dims: W and H? Let's verify from contract.
    // Contract says: pad=(1,1,1,1) -> (left, right, top, bottom) applied to last two dims.
    // However reference output shape is [64,1024,1026,4]. Input shape after view_as_real is [64,1024,1024,2].
    // If we pad last two dims each by 1, we would get [64,1025,1025,3]? But output is [64,1024,1026,4]. So something else.
    // Let's examine F.pad semantics: pad is (left, right, top, bottom) for last two dims of input.
    // Input after view_as_real is [64,1024,1024,2]. Last two dims are (1024,2). Padding left=1, right=1, top=1, bottom=1 would give (1026,4) for those dims, resulting in [64,1024,1026,4]? Wait the second-to-last dim is 1024, padding top/bottom on that dim would change it to 1026, but output second-to-last dim is 1024, not 1026. So maybe pad_width is (1,1,1,1) but applied to original 3D complex tensor before view_as_real? But task_contract says constant_pad input is [64,1024,1024,2].
    // Let's check reference steps: step_001 view_as_real yields [64,1024,1024,2]. step_002 F.pad with pad_width=(1,1,1,1) on this 4D tensor.
    // PyTorch F.pad for 4D tensor with 4-element pad: pads last dim by (1,1), second-to-last dim by (1,1). So:
    // dim -1 (size 2) -> size 4 (pad 1 each side)
    // dim -2 (size 1024) -> size 1026 (pad 1 each side)
    // dim -3 (size 1024) -> unchanged (since pad list only 4 elements)
    // dim -4 (size 64) -> unchanged
    // Result: [64, 1024, 1026, 4]. Yes, that matches output shape.
    // So constant_pad operates on 4D input, padding last two dims (dim -1 and dim -2).
    // Our input shape: B=64, H=1024 (dim -3), W=1024 (dim -2), C=2 (dim -1).
    // Output: B=64, H_out=1024, W_out=1026, C_out=4.
    // We need to map each output element to input element or pad_value.
    // Use grid-stride loop over all output elements (total = B*H_out*W_out*C_out).
    int64_t total_out = B * H_out * W_out * 4;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (int64_t i = idx; i < total_out; i += stride) {
        // Compute output coordinates
        int64_t c = i % 4; // last dim size 4
        int64_t tmp = i / 4;
        int64_t w_out = tmp % W_out;
        tmp /= W_out;
        int64_t h_out = tmp % H_out;
        int64_t b = tmp / H_out;

        // Input coordinates: w_in = w_out - 1, h_in = h_out, c_in = c - 1
        int64_t w_in = w_out - 1;
        int64_t h_in = h_out;
        int64_t c_in = c - 1;

        bool is_pad = (w_in < 0 || w_in >= W_in || c_in < 0 || c_in >= 2);
        // Note: h_in is always valid because H_out = H_in.

        if (is_pad) {
            output[i] = pad_value;
        } else {
            // Compute input offset using input strides (in elements)
            // Input shape [B, H_in, W_in, 2], strides [stride_b, stride_h, stride_w, 1] (since last dim contiguous? Actually arbitrary stride for last dim too)
            // But we are given stride_b, stride_h, stride_w, but not last dim stride. Assume last dim stride is 1 (contiguous) as typical, but maybe not.
            // The contract for view_as_real output stride is [2097152, 2048, 2, 1], meaning last dim contiguous.
            // So we can assume last dim stride = 1 for constant_pad input.
            // However, to be safe, we could pass all strides, but we only have 3 stride params. Since view_as_real outputs contiguous with last dim size 2, stride last = 1.
            // We'll compute offset as b*stride_b + h_in*stride_h + w_in*stride_w + c_in*1.
            int64_t offset = b * stride_b + h_in * stride_h + w_in * stride_w + c_in;
            output[i] = input[offset];
        }
    }
}

// Launcher for constant_pad
// input_data: pointer to float32 input [B, H_in, W_in, 2]
// output_data: pointer to float32 output [B, H_out, W_out, 4]
// dims: array of 3 int64_t? Actually input shape is 4D but B,H,W are first three dims, last dim always 2.
// We pass B, H_in, W_in, strides for first three dims.
// pad value as float
// output dims computed from input and pad.
extern "C" void constant_pad_launcher(
    const float* input_data,
    float* output_data,
    int64_t B, int64_t H_in, int64_t W_in,
    int64_t stride_b, int64_t stride_h, int64_t stride_w,
    float pad_value,
    cudaStream_t stream)
{
    int64_t H_out = H_in; // no padding on H dimension (since pad only last two dims? Actually pad on last two dims: W and C, not H)
    int64_t W_out = W_in + 2;
    int64_t total_out = B * H_out * W_out * 4;
    int threads = 256;
    int blocks = (int)((total_out + threads - 1) / threads);
    constant_pad_kernel<<<blocks, threads, 0, stream>>>(
        input_data, output_data, B, H_in, W_in,
        stride_b, stride_h, stride_w,
        H_out, W_out, pad_value);
}
