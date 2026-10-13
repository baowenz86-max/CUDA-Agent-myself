// This file intentionally contains no CUDA kernels.
// The only operator, conv_transpose1d_forward, uses cuDNN and will be implemented
// in a separate .cpp binding file (not in this .cu) because cuDNN API calls are
// not allowed in .cu files per the task restrictions.
#include <cuda_runtime.h>
