#include <cuda_runtime.h>
#include <cstdint>

// Helper for ceil division with positive integers
__host__ __device__ inline int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

// MaxPool2d kernel: supports arbitrary strides, dilation, padding, ceil_mode
// Input: x (N,C,H,W), output: y (N,C,Hout,Wout)
// Parameters: kernel_size, stride, padding, dilation, ceil_mode (passed as ints)
// For simplicity, only square kernel, stride, padding, dilation? 
// Reference uses non-square: kernel_size=(3,5), stride=(2,2), padding=0, dilation=1.
// We will support separate H and W dimensions as separate ints? No, contract passes single int for kernel_size, stride, padding, dilation but actual call passes 6 scalars: kernel_size_h, kernel_size_w, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w, ceil_mode?
// model_new.py passes self.maxpool.kernel_size and self.maxpool.stride which are tuples, but binding layer will likely pass tuple elements as separate ints.
// The contract lists parameters: x, kernel_size, stride, padding, dilation, ceil_mode, each int. But actual MaxPool2d has two values for each. However, the contract says "The six scalar parameters correspond to the fields of the nn.MaxPool2d module; they are always expanded from tuples and passed as individual ints/bool."
// That suggests the binding layer will pass separate ints for height and width dimensions. We need to define launcher with 6 ints after x: kh, kw, sh, sw, ph, pw, and bool ceil_mode, dilation is 1? Actually dilation also tuple but as per contract likely passed as two ints dh, dw? The contract says six scalar parameters but lists only kernel_size, stride, padding, dilation, ceil_mode. Possibly each is expanded into two. But given the call in model_new.py: cuda_extension.maxpool2d_forward(x, self.maxpool.kernel_size, self.maxpool.stride, self.maxpool.padding, self.maxpool.dilation, self.maxpool.ceil_mode). If self.maxpool.kernel_size is a tuple of 2 ints, Python will pass that tuple as a single argument to pybind, but the launcher signature must accept a tuple? However, the binding layer will likely unpack tuples into separate ints. The contract indicates that binding should pass them as individual ints. We can design launcher to accept 9 ints: kh, kw, sh, sw, ph, pw, dh, dw, ceil_mode? But the contract says parameters are 6 items; maybe they compress each tuple into its first component? Not clear.
// Safer: define launcher with explicit 9 scalar ints for kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w, ceil_mode. The binding layer can split tuples accordingly. We'll document that.

extern "C" void maxpool2d_launcher(
    float* output,
    const float* input,
    int N, int C, int H, int W,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w,
    bool ceil_mode,
    cudaStream_t stream)
{
    // Output dimensions
    int Hout, Wout;
    if (ceil_mode) {
        Hout = ceil_div(H + 2 * padding_h - dilation_h * (kernel_h - 1) - 1, stride_h) + 1;
        Wout = ceil_div(W + 2 * padding_w - dilation_w * (kernel_w - 1) - 1, stride_w) + 1;
    } else {
        Hout = (H + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
        Wout = (W + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;
    }

    int total_outputs = N * C * Hout * Wout;
    int threads = 256;
    int blocks = (total_outputs + threads - 1) / threads;
    // Cap blocks to avoid exceeding grid dimension, but use grid-stride loop inside kernel
    int max_blocks = 65535;
    if (blocks > max_blocks) blocks = max_blocks;

    maxpool2d_kernel<<<blocks, threads, 0, stream>>>(
        output, input,
        N, C, H, W,
        Hout, Wout,
        kernel_h, kernel_w,
        stride_h, stride_w,
        padding_h, padding_w,
        dilation_h, dilation_w,
        total_outputs
    );
}

__global__ void maxpool2d_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    int N, int C, int H, int W,
    int Hout, int Wout,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w,
    int total_outputs)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int out_idx = idx; out_idx < total_outputs; out_idx += stride) {
        // Decompose output index
        int ow = out_idx % Wout;
        int oh = (out_idx / Wout) % Hout;
        int oc = (out_idx / (Wout * Hout)) % C;
        int on = out_idx / (Wout * Hout * C);

        // Compute pooling window start (input coordinates)
        int h_start = oh * stride_h - padding_h;
        int w_start = ow * stride_w - padding_w;

        float max_val = -1e30f; // very negative for max

        for (int kh = 0; kh < kernel_h; ++kh) {
            int h_in = h_start + kh * dilation_h;
            if (h_in < 0 || h_in >= H) continue;
            for (int kw = 0; kw < kernel_w; ++kw) {
                int w_in = w_start + kw * dilation_w;
                if (w_in < 0 || w_in >= W) continue;
                // Input index with strides: assume contiguous? We cannot assume contiguity per contract.
                // We'll compute using flat index with strides? But launcher currently doesn't get strides.
                // The contract says input may have arbitrary strides, but our launcher only gets raw pointer and dimensions. Need to pass strides as well.
                // We'll add stride parameters: sN, sC, sH, sW.
                // Alternatively, we can assume contiguous because model_new.py passes tensor from torch.randn which is contiguous, and maxpool output is contiguous? Actually maxpool output may not be contiguous depending on stride, but torch maxpool returns contiguous tensor usually. But to be safe, we should pass strides.
                // We'll add int strideN, strideC, strideH, strideW to launcher and kernel.
                // However, we already have stride variable name conflict with pooling stride; rename.
                // We'll need to modify launcher signature to include input strides, but the binding layer will handle that.
                // For now, we'll assume contiguous for simplicity? But contract explicitly says "supports strides" meaning input may be non-contiguous.
                // Let's include strides in launcher. The binding layer can pass them from tensor.
                // Let's redefine launcher with additional parameters: input_strides[4]? We'll just pass four ints: sN, sC, sH, sW.
                // For output, we can assume output is contiguous because we allocate it as contiguous (torch::empty_like? Actually we allocate via torch::empty_like(input) which may not be contiguous if input is not. But we need output contiguous for our indexing? We can also create output with torch::empty({N,C,Hout,Wout}, options) which is contiguous. The binding layer should allocate contiguous output.
                // So output indexing can be linear: ((n*C + c)*Hout + oh)*Wout + ow.
                // We'll compute output pointer as output[out_idx] directly.
                // For input, compute using strides.
                // We'll need to store input strides in kernel.
                // Here we'll include those in launcher.
            }
        }
        output[out_idx] = max_val;
    }
}

// We'll correct the kernel below with strides handled.
// Actually, let's write a complete correct version after considering all aspects.

// We'll rewrite custom_ops.cu fully.
