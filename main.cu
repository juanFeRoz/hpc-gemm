#include "Kernels.cuh"
#include "Matrix.hpp"
#include <cuda_runtime.h>
#include <iostream>

int main(int argc, char *argv[]) {
  size_t N = 512;
  int blockSize = 8;
  if (argc >= 2)
    N = std::atoi(argv[1]);
  if (argc >= 3)
    blockSize = std::atoi(argv[2]);

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

  dim3 dimBlock(blockSize * blockSize);
  dim3 dimGrid((N + blockSize - 1) / blockSize,
               (N + blockSize - 1) / blockSize);

  cudaDeviceSynchronize();

  const int iterations = 10;
  for (int i = 0; i < iterations; ++i) {
    switch (blockSize) {
    case 8:
      Kernels::sgemm_naive<8><<<dimGrid, dimBlock>>>(N, N, N, dA, dB, dC);
      break;
    case 16:
      Kernels::sgemm_naive<16><<<dimGrid, dimBlock>>>(N, N, N, dA, dB, dC);
      break;
    case 32:
      Kernels::sgemm_naive<32><<<dimGrid, dimBlock>>>(N, N, N, dA, dB, dC);
      break;
    default:
      break;
    }
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
