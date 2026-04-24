#include <algorithm>
#include <hip/hip_runtime.h>
#include <random>
#include <vector>

std::mt19937 make_rng(unsigned int seed = std::random_device{}())
{
  return std::mt19937(seed);
}

template <typename Container>
void fillWithRandom(Container &c, std::mt19937 &gen)
{
  using T = typename Container::value_type;
  std::uniform_real_distribution<T> dis(0.0f, 1.0f);
  std::generate(std::begin(c), std::end(c), [&]()
                { return dis(gen); });
}

template <const int BM, const int BN, const int BK, const int TM>
__global__ void sgemm_blocktiling_1d_kernel(int num_rows_a, int num_cols_b,
                                            int num_cols_a, float alpha,
                                            const float *matrix_a,
                                            const float *matrix_b, float beta,
                                            float *matrix_c)
{
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

  for (int tile_idx = 0; tile_idx < num_cols_a; tile_idx += BK)
  {
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

    for (uint dot_idx = 0; dot_idx < BK; ++dot_idx)
    {
      float b_tmp = tile_b[dot_idx * BN + thread_col];
      for (uint res_idx = 0; res_idx < TM; ++res_idx)
      {
        thread_results[res_idx] +=
            tile_a[(thread_row * TM + res_idx) * BK + dot_idx] * b_tmp;
      }
    }
    __syncthreads();
  }

  for (uint res_idx = 0; res_idx < TM; ++res_idx)
  {
    int row = global_row + res_idx;
    if (row < num_rows_a && global_col < num_cols_b)
    {
      matrix_c[(thread_row * TM + res_idx) * num_cols_b + thread_col] =
          alpha * thread_results[res_idx] +
          beta *
              matrix_c[(thread_row * TM + res_idx) * num_cols_b + thread_col];
    }
  }
}

// Template parameters must be defined at compile time via -D_BM, -D_BN, -D_BK, -D_TM flags
#ifndef _BM
#define _BM 32
#endif
#ifndef _BN
#define _BN 32
#endif
#ifndef _BK
#define _BK 32
#endif
#ifndef _TM
#define _TM 1
#endif

extern "C" float run_kernel(int M, int N, int K, int BM_arg, int BN_arg, int BK_arg, int TM_arg,
                            unsigned int seed)
{
  // Verify that runtime parameters match compile-time template parameters
  if (BM_arg != _BM || BN_arg != _BN || BK_arg != _BK || TM_arg != _TM)
  {
    return -1.0f; // Parameter mismatch
  }

  auto gen = make_rng(seed);
  std::vector<float> A(M * K), B(K * N);
  fillWithRandom(A, gen);
  fillWithRandom(B, gen);

  float *dA, *dB, *dC;
  hipMalloc(&dA, M * K * sizeof(float));
  hipMalloc(&dB, K * N * sizeof(float));
  hipMalloc(&dC, M * N * sizeof(float));
  hipMemset(dC, 0, M * N * sizeof(float));
  hipMemcpy(dA, A.data(), M * K * sizeof(float), hipMemcpyHostToDevice);
  hipMemcpy(dB, B.data(), K * N * sizeof(float), hipMemcpyHostToDevice);

  dim3 dimBlock((_BM / _TM) * _BN);
  dim3 dimGrid((M + _BM - 1) / _BM, (N + _BN - 1) / _BN);
  const float alpha = 1.0f, beta = 0.0f;

  hipEvent_t start, stop;
  hipEventCreate(&start);
  hipEventCreate(&stop);

  hipEventRecord(start);
  sgemm_blocktiling_1d_kernel<_BM, _BN, _BK, _TM>
      <<<dimGrid, dimBlock>>>(M, N, K, alpha, dA, dB, beta, dC);
  hipEventRecord(stop);
  hipEventSynchronize(stop);

  float elapsed_ms = 0.0f;
  hipEventElapsedTime(&elapsed_ms, start, stop);
  hipEventDestroy(start);
  hipEventDestroy(stop);

  hipFree(dA);
  hipFree(dB);
  hipFree(dC);

  return elapsed_ms;
}
