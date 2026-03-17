#include "Kernels.cuh"
#include "Matrix.hpp"
#include <cuda_runtime.h>

int main() {
  const size_t N = 2880;
  Matrix<double> A(N, N);
  Matrix<double> B(N, N);
  Matrix<double> C(N, N);

  A.fillRandom();
  B.fillRandom();

  double *dA;
  double *dB;
  double *dC;

  size_t size = N * N * sizeof(double);

  cudaMalloc(&dA, size);
  cudaMalloc(&dB, size);
  cudaMalloc(&dC, size);

  cudaMemcpy(dA, A.data(), size, cudaMemcpyHostToDevice);
  cudaMemcpy(dB, B.data(), size, cudaMemcpyHostToDevice);

  dim3 dimBlock(32, 32);
  dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x,
               (N + dimBlock.y - 1) / dimBlock.y);

  Kernels::dgemm<32, 32, 32><<<dimGrid, dimBlock>>>(N, N, N, dA, dB, dC);

  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);

  return 0;
}
