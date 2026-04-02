#ifndef KERNELS_CUH
#define KERNELS_CUH

#include <cuda_runtime.h>
namespace Kernels {
inline __global__ void dgemm_naive(int M, int N, int K, const float *A,
                                   const float *B, float *C) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < M && col < N) {
    float tmp = 0.0f;

    for (int i = 0; i < K; ++i) {
      tmp += A[row * K + i] * B[i * N + col];
    }
    C[row * N + col] = tmp;
  }
}
} // namespace Kernels
#endif
