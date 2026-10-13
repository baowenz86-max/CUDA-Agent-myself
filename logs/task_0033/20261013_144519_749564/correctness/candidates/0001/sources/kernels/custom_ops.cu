#include <cuda_runtime.h>

// This file intentionally contains no CUDA kernels.
// All required operations are implemented in C++ binding files using cuDNN for ConvTranspose
// and straightforward elementwise/normalization logic for LayerNorm and addcmul.
// Since no non-convolution CUDA computation is required, this file only includes the CUDA runtime header.