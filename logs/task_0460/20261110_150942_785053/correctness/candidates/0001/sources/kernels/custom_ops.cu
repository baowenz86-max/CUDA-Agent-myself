#include <cuda_runtime.h>
#include <stdint.h>

// This file contains CUDA kernels and host launchers for custom ops used by the model.
// It must not depend on torch, ATen, or pybind. All tensor handling is done in the binding layer.

// Kernel: copies input contiguous buffer to output contiguous buffer.
// Used by identity_new_tensor: copy [512,128,128] contiguous tensor.
// grid-stride loop for generality.
__global__ void copy_kernel_float(const float* __restrict__ input,
                                  float* __restrict__ output,
                                  int64_t numel) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < numel; i += stride) {
        output[i] = input[i];
    }
}

// Kernel: stacks a list of contiguous tensors along a new dimension 1.
// Input: each tensor has shape [B, H, W] -> flattened per tensor: B*H*W
// Output: shape [B, C, H, W], where C is number of tensors.
// We assume all input tensors are contiguous and have same shape (B,H,W).
// Output is contiguous.
__global__ void stack_kernel_float(const float* __restrict__ input,
                                   float* __restrict__ output,
                                   int64_t B,
                                   int64_t C,
                                   int64_t H,
                                   int64_t W) {
    int64_t total = B * C * H * W;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    int64_t per_tensor = B * H * W;
    for (int64_t i = idx; i < total; i += stride) {
        // decompose output index: i = (((b * C + c) * H + h) * W + w)
        int64_t tmp = i;
        int64_t w = tmp % W;
        tmp /= W;
        int64_t h = tmp % H;
        tmp /= H;
        int64_t c = tmp % C;
        int64_t b = tmp / C;
        int64_t src_idx = (b * H + h) * W + w;
        output[i] = input[c * per_tensor + src_idx];
    }
}

// Launcher for unbind_dim1. Since unbind returns views, no CUDA work is needed.
// The binding layer will create Python tuple of tensor views using torch::unbind-like slicing.
// We provide a no-op launcher for completeness; it takes raw pointer and does nothing.
extern "C" void unbind_dim1_launcher(const float* x,
                                     int64_t B,
                                     int64_t C,
                                     int64_t H,
                                     int64_t W,
                                     cudaStream_t stream) {
    // No kernel launch; views are handled in C++.
    (void)x; (void)B; (void)C; (void)H; (void)W; (void)stream;
}

// Launcher for identity_new_tensor: copies data to a new contiguous tensor.
extern "C" void identity_copy_launcher(const float* input,
                                        float* output,
                                        int64_t numel,
                                        cudaStream_t stream) {
    if (numel <= 0) return;
    int threads = 256;
    int blocks = (int)((numel + threads - 1) / threads);
    // Limit blocks to a reasonable max to avoid excessive grid dimensions.
    if (blocks > 65535) blocks = 65535;
    copy_kernel_float<<<blocks, threads, 0, stream>>>(input, output, numel);
}

// Launcher for stack_dim1: stacks C contiguous tensors into output [B,C,H,W].
extern "C" void stack_dim1_launcher(const float* const* inputs,
                                    int64_t B,
                                    int64_t C,
                                    int64_t H,
                                    int64_t W,
                                    float* output,
                                    cudaStream_t stream) {
    if (B <= 0 || C <= 0 || H <= 0 || W <= 0) return;
    int64_t total = B * C * H * W;
    // For simplicity, we assume the inputs are stored as an array of pointers in device memory?
    // Better approach: the binding layer can pass a single pointer to a contiguous array of all input data
    // if all inputs are concatenated, but the contract says they are separate tensors.
    // Since we cannot assume device pointer array availability easily without extra CUDA copies,
    // we can instead copy each input tensor one by one into the correct slice of output using a loop.
    // However, the launcher signature should be convenient for binding: we can accept a pointer to an array of pointers on host,
    // but that requires host-device copy. Alternatively, the binding layer can pass each raw data pointer separately.
    // Due to limitations, we instead provide a launcher that takes a single input pointer to a contiguous block
    // containing all C tensors concatenated (which the binding layer can create by copying or by using torch.cat).
    // But that would add an extra copy, not ideal. Since the model_new.py calls identity_new_tensor on each unbind result,
    // after identity_new_tensor each tensor is contiguous and separate. The binding layer will have access to each tensor's data_ptr.
    // To keep the .cu independent, we can define a launcher that accepts a pointer to an array of device pointers allocated by binding.
    // However, we cannot use torch in .cu. We can implement stack by launching a kernel for each input tensor,
    // with an offset marking the slice. We'll provide a launcher that loops over C and launches one kernel per tensor,
    // using the provided pointers. But we need pointer to pointer. We can use std::vector or allocate device memory for pointer array
    // inside the launcher, but that would require cudaMalloc and cudaMemcpy, which is acceptable (raw CUDA).
    // We'll implement a launcher that takes a host pointer to an array of device pointers (float** hostPtrs)
    // and handles the device-side copies.
    // To simplify, the binding layer can pass a single pointer to a contiguous array of all data if it concatenates,
    // but that would change semantics. Let's define the launcher to accept a pointer to an array of raw device pointers.
    // Since we cannot allocate device memory for pointer array in the .cu without knowing C at runtime, we can use
    // cudaMalloc and cudaMemcpy internally, then launch a single kernel that reads from device pointer array.
    // However, we are only allowed to use CUDA runtime, so this is fine.
    // We will implement stack_kernel that takes a float** deviceInputPtrs, and we provide a launcher that
    // allocates device memory for the pointer array, copies host pointers to device, launches kernel, and frees.
    // But we must ensure the host pointer array remains valid during launch.
    // The binding layer can pass a std::vector<float*> converted to raw array.
    // We'll define stack_dim1_launcher with signature:
    //   void stack_dim1_launcher(float** device_input_ptrs, int64_t B, int64_t C, int64_t H, int64_t W, float* output, cudaStream_t stream)
    // where device_input_ptrs is already a device pointer to an array of device pointers; the binding layer can create it using cudaMalloc and cudaMemcpy.
    // To avoid that complexity, we could instead pass a single pointer to a contiguous array of inputs (if the binding layer concatenates them).
    // But model_new.py expects stack_dim1 to be called with list of tensors; the binding layer can easily create a contiguous concatenation? That would be an extra copy.
    // Given performance is not critical for this simple model, we can accept a host pointer to array of device pointers and perform copies sequentially.
    // We'll implement a launcher that loops over C and uses cudaMemcpyAsync from each input to the corresponding output slice, but that may be slower.
    // Better: launch one kernel per input with appropriate offsets, using pointer arithmetic on a single base pointer? No, each tensor has its own pointer.
    // For simplicity and correctness, we'll provide a launcher that takes an array of device pointers allocated by the binding. But we cannot force the binding to do that.
    // Let's check the contract: binding fragments show stack_dim1 parameter is torch::Tensor tensors, not a list? The contract says "kind: tensor" but shape "list of 32 tensors", so binding likely receives a list of tensors.
    // The binding layer will handle conversion to a vector of data_ptr<float>().
    // We'll design the launcher to accept a host pointer to an array of float* (pointers to device memory), C, B, H, W, output, stream.
    // We'll then inside the launcher create a device array of pointers via cudaMalloc and cudaMemcpyAsync, launch stack_kernel_with_ptrs, and free.
    // This keeps the .cu free of torch and works correctly.
    
    // Actually, we can simplify: since we are not required to optimize heavily, we can just launch a copy kernel for each input tensor into the proper output slice,
    // using a simple loop in the launcher. The kernel copies one tensor's data to output + offset.
    // This uses multiple kernel launches but is simple and correct.
    // We'll do that: for each c in [0,C), launch copy_kernel_float on input[c] to output + c*per_tensor, with numel = B*H*W.
    // The launcher receives input as an array of pointers in host memory: const float** input_ptrs.
    // This avoids device pointer array allocation and kernel complexity.
    // We'll use a helper kernel that copies numel elements.
    // We already have copy_kernel_float, so we can reuse it.
    // We'll add a launcher that loops over c and launches copy_kernel_float with offset.
    // This is straightforward and does not require extra device memory.
    // The binding layer must pass a std::vector<float*> or a raw pointer to an array of float* on host.
    // We'll document the signature.
    // Due to these considerations, we adjust the launcher signature for stack_dim1 to:
    //   void stack_dim1_launcher(const float* const* input_ptrs, int64_t B, int64_t C, int64_t H, int64_t W, float* output, cudaStream_t stream)
    // where input_ptrs is a host pointer to an array of C device pointers.
}

// We will now write the actual stack_dim1_launcher implementation using per-tensor copy launches.
extern "C" void stack_dim1_launcher(const float* const* input_ptrs,
                                    int64_t B,
                                    int64_t C,
                                    int64_t H,
                                    int64_t W,
                                    float* output,
                                    cudaStream_t stream) {
    if (B <= 0 || C <= 0 || H <= 0 || W <= 0) return;
    int64_t per_tensor = B * H * W;
    int threads = 256;
    int blocks = (int)((per_tensor + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;
    for (int64_t c = 0; c < C; ++c) {
        copy_kernel_float<<<blocks, threads, 0, stream>>>(
            input_ptrs[c], output + c * per_tensor, per_tensor);
    }
}

// Note: The unbind_dim1_launcher is a no-op because unbind returns views, handled in binding.
// The binding layer will create a tuple of tensors by slicing the original tensor using as_strided or narrow.
// No CUDA kernel is required for unbind.
