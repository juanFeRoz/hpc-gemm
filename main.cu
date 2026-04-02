#include "Kernels.cuh"
#include "Matrix.hpp"
#include <cuda_runtime.h>
#include <iostream>

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

  cudaEvent_t start;
  cudaEvent_t end;

  cudaEventCreate(&start);
  cudaEventCreate(&end);

  cudaEventRecord(start);
  Kernels::dgemm_naive<<<dimGrid, dimBlock>>>(N, N, N, dA, dB, dC);
  cudaEventRecord(end);

  cudaEventSynchronize(end);

  double milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, end);
  double seconds = milliseconds / 1000.0;
  double gflops = (2.0 * N * N * N) / (seconds * 1e9);

  std::cout << "Tiempo: " << milliseconds << " ms" << std::endl;
  std::cout << "Rendimiento: " << gflops << " GFLOPS" << std::endl;

  cudaMemcpy(C.data(), dC, size, cudaMemcpyDeviceToHost);

  cudaEventDestroy(start);
  cudaEventDestroy(end);
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);

  return 0;
}
