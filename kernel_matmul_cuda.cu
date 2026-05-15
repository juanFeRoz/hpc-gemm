#include <algorithm>
#include <cuda_runtime.h>
#include <random>
#include <vector>

using uint = unsigned int;

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

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void sgemm_blocktiling_2d_kernel(int num_rows_a, int num_cols_b, int num_cols_a,
                                            float alpha, const float *matrix_a,
                                            const float *matrix_b, float beta,
                                            float *matrix_c)
{
    const uint block_row = blockIdx.x;
    const uint block_col = blockIdx.y;

    __shared__ float tile_a[BM * BK];
    __shared__ float tile_b[BK * BN];

    const uint thread_row = threadIdx.y;
    const uint thread_col = threadIdx.x;
    const uint num_threads = (BM / TM) * (BN / TN);
    const uint thread_id = thread_row * (BN / TN) + thread_col;

    matrix_a += block_row * BM * num_cols_a;
    matrix_b += block_col * BN;
    matrix_c += block_row * BM * num_cols_b + block_col * BN;

    float thread_results[TM * TN] = {0.0f};
    float register_m[TM] = {0.0f};
    float register_n[TN] = {0.0f};

    for (uint block_k_idx = 0; block_k_idx < num_cols_a; block_k_idx += BK)
    {
        for (uint load_offset = 0; load_offset < BM * BK; load_offset += num_threads)
        {
            uint load_idx = thread_id + load_offset;
            if (load_idx < BM * BK)
            {
                uint a_row = load_idx / BK;
                uint a_col = load_idx % BK;
                tile_a[load_idx] = matrix_a[a_row * num_cols_a + a_col];
            }
        }

        for (uint load_offset = 0; load_offset < BK * BN; load_offset += num_threads)
        {
            uint load_idx = thread_id + load_offset;
            if (load_idx < BK * BN)
            {
                uint b_row = load_idx / BN;
                uint b_col = load_idx % BN;
                tile_b[load_idx] = matrix_b[b_row * num_cols_b + b_col];
            }
        }

        __syncthreads();

        matrix_a += BK;
        matrix_b += BK * num_cols_b;

        for (uint dot_idx = 0; dot_idx < BK; ++dot_idx)
        {
            for (uint i = 0; i < TM; ++i)
            {
                register_m[i] = tile_a[(thread_row * TM + i) * BK + dot_idx];
            }

            for (uint i = 0; i < TN; ++i)
            {
                register_n[i] = tile_b[dot_idx * BN + thread_col * TN + i];
            }

            for (uint res_idx_m = 0; res_idx_m < TM; ++res_idx_m)
            {
                for (uint res_idx_n = 0; res_idx_n < TN; ++res_idx_n)
                {
                    thread_results[res_idx_m * TN + res_idx_n] +=
                        register_m[res_idx_m] * register_n[res_idx_n];
                }
            }
        }

        __syncthreads();
    }

    for (uint res_idx_m = 0; res_idx_m < TM; ++res_idx_m)
    {
        for (uint res_idx_n = 0; res_idx_n < TN; ++res_idx_n)
        {
            const uint c_idx = (thread_row * TM + res_idx_m) * num_cols_b +
                               (thread_col * TN + res_idx_n);
            matrix_c[c_idx] = alpha * thread_results[res_idx_m * TN + res_idx_n] +
                              beta * matrix_c[c_idx];
        }
    }
}

// Template parameters must be defined at compile time via -D_BM, -D_BN, -D_BK, -D_TM -D_TN flags
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
#ifndef _TN
#define _TN 1
#endif

extern "C" float run_kernel(int M, int N, int K, int BM_arg, int BN_arg, int BK_arg, int TM_arg, int TN_arg,
                            unsigned int seed)
{
    if (BM_arg != _BM || BN_arg != _BN || BK_arg != _BK || TM_arg != _TM || TN_arg != _TN)
    {
        return -1.0f;
    }

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

    dim3 dimBlock(_BN / _TN, _BM / _TM);
    dim3 dimGrid((M + _BM - 1) / _BM, (N + _BN - 1) / _BN);
    const float alpha = 1.0f, beta = 0.0f;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    sgemm_blocktiling_2d_kernel<_BM, _BN, _BK, _TM, _TN>
        <<<dimGrid, dimBlock>>>(M, N, K, alpha, dA, dB, beta, dC);
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
