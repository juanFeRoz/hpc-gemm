#include <algorithm>
#include <cuda_runtime.h>
#include <random>
#include <vector>

std::mt19937 make_rng(unsigned int seed = std::random_device{}()) {
  return std::mt19937(seed);
}

template <typename Container>
void fillWithRandom(Container &c, std::mt19937 &gen) {
  using T = typename Container::value_type;
  std::uniform_real_distribution<T> dis(0.0f, 1.0f);
  std::generate(std::begin(c), std::end(c), [&]() { return dis(gen); });
}

template <const int BM, const int BN, const int BK, const int TM>
__global__ void sgemm_blocktiling_1d_kernel(int num_rows_a, int num_cols_b,
                                            int num_cols_a, float alpha,
                                            const float *matrix_a,
                                            const float *matrix_b, float beta,
                                            float *matrix_c) {
  const uint block_row = blockIdx.x;
  const uint block_col = blockIdx.y;

  __shared__ float tile_a[BM * BK];
  __shared__ float tile_b[BK * BN];

  const uint thread_row = threadIdx.x / BN;
  const uint thread_col = threadIdx.x % BN;

  const int global_row = block_row * BM + thread_row * TM;
  const int global_col = block_col * BN + thread_col;

  matrix_a += block_row * BM * num_cols_a;
  matrix_b += block_col * BN;
  matrix_c += block_row * BM * num_cols_b + block_col * BN;

  float thread_results[TM] = {0.0f};

  for (int tile_idx = 0; tile_idx < num_cols_a; tile_idx += BK) {
    const uint a_row = threadIdx.x / BK;
    const uint a_col = threadIdx.x % BK;
    if ((block_row * BM + a_row) < num_rows_a &&
        (tile_idx + a_col) < num_cols_a)
      tile_a[a_row * BK + a_col] = matrix_a[a_row * num_cols_a + a_col];
    else
      tile_a[a_row * BK + a_col] = 0.0f;

    const uint b_row = threadIdx.x / BN;
    const uint b_col = threadIdx.x % BN;
    if ((tile_idx + b_row) < num_cols_a &&
        (block_col * BN + b_col) < num_cols_b)
      tile_b[b_row * BN + b_col] = matrix_b[b_row * num_cols_b + b_col];
    else
      tile_b[b_row * BN + b_col] = 0.0f;

    __syncthreads();

    matrix_a += BK;
    matrix_b += BK * num_cols_b;

    for (uint dot_idx = 0; dot_idx < BK; ++dot_idx) {
      float b_tmp = tile_b[dot_idx * BN + thread_col];
      for (uint res_idx = 0; res_idx < TM; ++res_idx) {
        thread_results[res_idx] +=
            tile_a[(thread_row * TM + res_idx) * BK + dot_idx] * b_tmp;
      }
    }
    __syncthreads();
  }

  for (uint res_idx = 0; res_idx < TM; ++res_idx) {
    int row = global_row + res_idx;
    if (row < num_rows_a && global_col < num_cols_b) {
      matrix_c[(thread_row * TM + res_idx) * num_cols_b + thread_col] =
          alpha * thread_results[res_idx] +
          beta *
              matrix_c[(thread_row * TM + res_idx) * num_cols_b + thread_col];
    }
  }
}

extern "C" float run_kernel(int M, int N, int K, int BM, int BN, int BK, int TM,
                            unsigned int seed) {
  auto gen = make_rng(seed);
  std::vector<float> A(M * K), B(K * N);
  fillWithRandom(A, gen);
  fillWithRandom(B, gen);

  float *dA, *dB, *dC;
  cudaMalloc(&dA, M * K * sizeof(float));
  cudaMalloc(&dB, K * N * sizeof(float));
  cudaMalloc(&dC, M * N * sizeof(float));
  cudaMemset(dC, 0, M * N * sizeof(float));
  cudaMemcpy(dA, A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dB, B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice);

  dim3 dimBlock((BM / TM) * BN);
  dim3 dimGrid((M + BM - 1) / BM, (N + BN - 1) / BN);
  const float alpha = 1.0f, beta = 0.0f;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  if (BM == 8 && BN == 8 && BK == 8 && TM == 1)
    sgemm_blocktiling_1d_kernel<8, 8, 8, 1>
        <<<dimGrid, dimBlock>>>(M, N, K, alpha, dA, dB, beta, dC);
  else if (BM == 16 && BN == 16 && BK == 8 && TM == 2)
    sgemm_blocktiling_1d_kernel<16, 16, 8, 2>
        <<<dimGrid, dimBlock>>>(M, N, K, alpha, dA, dB, beta, dC);
  else if (BM == 32 && BN == 32 && BK == 8 && TM == 4)
    sgemm_blocktiling_1d_kernel<32, 32, 8, 4>
        <<<dimGrid, dimBlock>>>(M, N, K, alpha, dA, dB, beta, dC);
  else if (BM == 16 && BN == 16 && BK == 16 && TM == 1)
    sgemm_blocktiling_1d_kernel<16, 16, 16, 1>
        <<<dimGrid, dimBlock>>>(M, N, K, alpha, dA, dB, beta, dC);
  else if (BM == 32 && BN == 32 && BK == 16 && TM == 2)
    sgemm_blocktiling_1d_kernel<32, 32, 16, 2>
        <<<dimGrid, dimBlock>>>(M, N, K, alpha, dA, dB, beta, dC);
  else if (BM == 32 && BN == 32 && BK == 32 && TM == 1)
    sgemm_blocktiling_1d_kernel<32, 32, 32, 1>
        <<<dimGrid, dimBlock>>>(M, N, K, alpha, dA, dB, beta, dC);
  else {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    return -1.0f;
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float elapsed_ms = 0.0f;
  cudaEventElapsedTime(&elapsed_ms, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);

  return elapsed_ms;
}
