#include "Kernels.cuh"
#include "Matrix.hpp"
#include <cuda_runtime.h>
#include <iostream>

int main(int argc, char *argv[]) {
  size_t N = 512;
  int tile = 8;
  if (argc >= 2)
    N = std::atoi(argv[1]);
  if (argc >= 3)
    tile = std::atoi(argv[2]);

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

  dim3 dimBlock(tile, tile);
  dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x,
               (N + dimBlock.y - 1) / dimBlock.y);

  Kernels::dgemm_naive<<<dimGrid, dimBlock>>>(N, N, N, dA, dB, dC);
  cudaDeviceSynchronize();

  const int iterations = 10;
  for (int i = 0; i < iterations; ++i) {
    Kernels::dgemm_naive<<<dimGrid, dimBlock>>>(N, N, N, dA, dB, dC);
  }
  cudaDeviceSynchronize();

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
  }
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);

  return 0;
}
