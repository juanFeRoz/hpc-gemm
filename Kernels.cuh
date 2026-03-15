#ifndef KERNELS_CUH
#define KERNELS_CUH

#include <cuda_runtime.h>
namespace Kernels {
template <int BK, int BN, int BM>
__global__ void dgemm(int M, int N, int K, const double *A, const double *B,
                      double *C) {
  __shared__ double sA[BM][BK];
  __shared__ double sB[BK][BN];

  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  double tmp = 0.0;
  for (int i = 0; i < K; i += BK) {
    sA[threadIdx.y][threadIdx.x] = A[row * K + (i + threadIdx.x)];
    sB[threadIdx.y][threadIdx.x] = B[(i + threadIdx.y) * N + col];

    __syncthreads();
    for (int j = 0; j < BK; ++j) {
      tmp += sA[threadIdx.y][j] * sB[j][threadIdx.x];
    }
  }
  if (row < M && col < N) {
    C[row * N + col] = tmp;
  }
}
} // namespace Kernels
#endif
