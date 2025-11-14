#include <iostream>
#include <hip/hip_runtime.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;
typedef float T;

__global__ void simpleGEMM(
    const T* A, const T* B, T* C, 
    int M, int N, int K) 
{
    int n = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int m = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;

    if (m < M && n < N) {
        T sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[m * K + k] * B[k * N + n];
        }
        C[m * N + n] = sum;
    }
}

// Host function callable by Python
float run_gemm_trial(int block_x, int block_y) {
    const int M = 64, N = 64, K = 64; 
    size_t size_A = M * K * sizeof(T);
    size_t size_B = K * N * sizeof(T);
    size_t size_C = M * N * sizeof(T);
    
    // 1. Host Memory Setup (for initialization/cleanup)
    T *h_A, *h_B, *h_C;
    h_A = (T*)malloc(size_A);
    h_B = (T*)malloc(size_B);
    h_C = (T*)malloc(size_C);
    for (int i = 0; i < M * K; ++i) h_A[i] = (T)i + 1.0f;
    for (int i = 0; i < K * N; ++i) h_B[i] = (T)1.0f;

    // 2. Device Memory Setup and Copy
    T *d_A, *d_B, *d_C;
    hipMalloc((void**)&d_A, size_A);
    hipMalloc((void**)&d_B, size_B);
    hipMalloc((void**)&d_C, size_C);
    hipMemcpy(d_A, h_A, size_A, hipMemcpyHostToDevice);
    hipMemcpy(d_B, h_B, size_B, hipMemcpyHostToDevice);

    // 3. Launch Configuration
    dim3 block(block_x, block_y);
    dim3 grid(
        (N + block.x - 1) / block.x, // Grid size for N dimension
        (M + block.y - 1) / block.y  // Grid size for M dimension
    );

    // 4. Time and Launch the Kernel
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

    hipEventRecord(start);
    hipLaunchKernelGGL(simpleGEMM, grid, block, 0, 0, d_A, d_B, d_C, M, N, K);
    hipEventRecord(stop);
    hipEventSynchronize(stop);

    float milliseconds = 0;
    hipEventElapsedTime(&milliseconds, start, stop);

    // 5. Cleanup
    hipFree(d_A); hipFree(d_B); hipFree(d_C);
    free(h_A); free(h_B); free(h_C);
    
    return milliseconds;
}

// PYBIND11 BINDING
PYBIND11_MODULE(hip_gemm_module, m) {
    m.doc() = "pybind11 wrapper for HIP GEMM block size optimization.";
    m.def("run_gemm_trial", &run_gemm_trial, 
          "Runs the GEMM kernel, accepts block_x and block_y, returns time (ms)");
}