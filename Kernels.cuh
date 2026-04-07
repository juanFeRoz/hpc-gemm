#ifndef KERNELS_CUH
#define KERNELS_CUH

#include <cuda_runtime.h>
namespace Kernels {
template <int BLOCKSIZE>
__global__ void sgemm_naive(int M, int N, int K, const float *A, const float *B,
                            float *C) {
  int row = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
  int col = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

  if (row < M && col < N) {
    float tmp = 0.0;
    for (int i = 0; i < K; ++i) {
      tmp += A[row * K + i] * B[i * N + col];
    }
    C[row * N + col] = tmp;
  }
}
} // namespace Kernels
#endif
