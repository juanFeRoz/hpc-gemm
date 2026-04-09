#include "Kernels.cuh"
#include "Matrix.hpp"
#include <cuda_runtime.h>
#include <iostream>

int main(int argc, char *argv[]) {
  size_t N = 512;
  int blockSize = 16;
  if (argc >= 2)
    N = std::atoi(argv[1]);
  if (argc >= 3)
    blockSize = std::atoi(argv[2]);

  Matrix<float> A(N, N);
  Matrix<float> B(N, N);
  Matrix<float> C(N, N);

  A.fillRandom();
  B.fillRandom();

  float *dA, *dB, *dC;
  size_t size = N * N * sizeof(float);

  cudaMalloc(&dA, size);
  cudaMalloc(&dB, size);
  cudaMalloc(&dC, size);

  cudaMemcpy(dA, A.data(), size, cudaMemcpyHostToDevice);
  cudaMemcpy(dB, B.data(), size, cudaMemcpyHostToDevice);

  const int iterations = 10;
  for (int i = 0; i < iterations; ++i) {
    if (blockSize == 8) {
      dim3 dimBlock((8 * 8) / 8);
      dim3 dimGrid((N + 8 - 1) / 8, (N + 8 - 1) / 8);
      Kernels::sgemm1DTiling<8, 8, 8, 8>
          <<<dimGrid, dimBlock>>>(N, N, N, dA, dB, dC);
    } else if (blockSize == 16) {
      dim3 dimBlock((16 * 16) / 8);
      dim3 dimGrid((N + 16 - 1) / 16, (N + 16 - 1) / 16);
      Kernels::sgemm1DTiling<16, 16, 8, 8>
          <<<dimGrid, dimBlock>>>(N, N, N, dA, dB, dC);
    } else if (blockSize == 32) {
      dim3 dimBlock((32 * 32) / 8);
      dim3 dimGrid((N + 32 - 1) / 32, (N + 32 - 1) / 32);
      Kernels::sgemm1DTiling<32, 32, 8, 8>
          <<<dimGrid, dimBlock>>>(N, N, N, dA, dB, dC);
    }
  }

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
  }

  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);

  return 0;
}
