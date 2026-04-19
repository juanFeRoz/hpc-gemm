#include <algorithm>
#include <cuda_runtime.h>
#include <random>
#include <vector>

template <typename Container> void fillWithRandom(Container &c) {
  static std::random_device rd;
  static std::mt19937 gen(rd());
  using T = typename Container::value_type;
  std::uniform_real_distribution<T> dis(0.0, 1.0);
  std::generate(std::begin(c), std::end(c), [&]() { return dis(gen); });
}

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

int main(int argc, char *argv[]) {
  size_t N = 512;
  int blockSize = 16;

  std::vector<float> A(N * N);
  std::vector<float> B(N * N);
  fillWithRandom(A);
  fillWithRandom(B);

  float *dA, *dB, *dC;
  size_t size = N * N * sizeof(float);

  cudaMalloc(&dA, size);
  cudaMalloc(&dB, size);
  cudaMalloc(&dC, size);

  cudaMemcpy(dA, A.data(), size, cudaMemcpyHostToDevice);
  cudaMemcpy(dB, B.data(), size, cudaMemcpyHostToDevice);

  dim3 dimBlock((32 * 32) / 8);
  dim3 dimGrid((N + 32 - 1) / 32, (N + 32 - 1) / 32);
  sgemm1DTiling<32, 32, 8, 8><<<dimGrid, dimBlock>>>(N, N, N, dA, dB, dC);

  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);

  return 0;
}
