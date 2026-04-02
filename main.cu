#include "Kernels.cuh"
#include "Matrix.hpp"
#include <cuda_runtime.h>

int main() {
  const size_t N = 2880;
  Matrix<float> A(N, N);
  Matrix<float> B(N, N);
  Matrix<float> C(N, N);

  A.fillRandom();
  B.fillRandom();

  float *dA;
  float *dB;
  float *dC;

  size_t size = N * N * sizeof(float);

  cudaMalloc(&dA, size);
  cudaMalloc(&dB, size);
  cudaMalloc(&dC, size);

  cudaMemcpy(dA, A.data(), size, cudaMemcpyHostToDevice);
  cudaMemcpy(dB, B.data(), size, cudaMemcpyHostToDevice);

  const int tile = 64;
  dim3 dimBlock(tile, tile);
  dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x,
               (N + dimBlock.y - 1) / dimBlock.y);

  Kernels::dgemm_naive<<<dimGrid, dimBlock>>>(N, N, N, dA, dB, dC);

  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);

  return 0;
}
