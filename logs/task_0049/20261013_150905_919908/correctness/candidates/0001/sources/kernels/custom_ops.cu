#include <cuda_runtime.h>

// This file intentionally contains no non-convolution CUDA kernels.
// The custom operator 'convt1d_bias_round' includes a ConvTranspose1d,
// which is implemented using cuDNN in the C++ binding layer (Stage 3).
// The subsequent bias additions and rounding can be fused in the binding
// or provided as separate kernels in the same .cu file if needed.
// Per the task instructions, this file contains only the required header.
