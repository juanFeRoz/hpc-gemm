#ifndef KERNELS_CUH
#define KERNELS_CUH

#include <cuda_runtime.h>

namespace Kernels {

template <const int BM, const int BN, const int BK, const int TM>
__global__ void sgemm1DTiling(int M, int N, int K, const float *A,
                              const float *B, float *C) {
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  const int threadCol = threadIdx.x % BN;
  const int threadRow = threadIdx.x / BN;

  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  A += cRow * BM * K;
  B += cCol * BN;
  C += (cRow * BM + threadRow * TM) * N + cCol * BN;

  const uint innerColA = threadIdx.x % BK;
  const uint innerRowA = threadIdx.x / BK;
  const uint innerColB = threadIdx.x % BN;
  const uint innerRowB = threadIdx.x / BN;

  float threadResults[TM] = {0.0f};

  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    As[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
    Bs[innerRowB * BN + innerColB] = B[innerRowB * N + innerColB];

    __syncthreads();

    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
      float tmpB = Bs[dotIdx * BN + threadCol];
      for (uint resIdx = 0; resIdx < TM; ++resIdx) {
        threadResults[resIdx] +=
            As[(threadRow * TM + resIdx) * BK + dotIdx] * tmpB;
      }
    }

    __syncthreads();

    A += BK;
    B += BK * N;
  }

  for (uint resIdx = 0; resIdx < TM; ++resIdx) {
    if ((cRow * BM + threadRow * TM + resIdx) < M &&
        (cCol * BN + threadCol) < N) {
      C[resIdx * N + threadCol] = threadResults[resIdx];
    }
  }
}
} // namespace Kernels
#endif
