// This file intentionally contains no custom CUDA kernels.
// The model operations are:
//   1. ConvTranspose1d: must be implemented with cuDNN (by the C++ binding layer),
//      not as a custom CUDA kernel.
//   2. Flatten: view operation that does not require any data movement,
//      so it will be handled in the binding wrapper.
// Therefore there are no non-convolution compute kernels to implement here.
#include <cuda_runtime.h>
