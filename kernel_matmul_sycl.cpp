#include <algorithm>
#include <random>
#include <sycl/sycl.hpp>
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
sycl::event sgemm_blocktiling_1d_kernel(sycl::queue &q, int num_rows_a,
                                        int num_cols_b, int num_cols_a,
                                        float alpha, const float *matrix_a,
                                        const float *matrix_b, float beta,
                                        float *matrix_c) {

  sycl::range<2> local_range(BM / TM, BN);
  sycl::range<2> global_range(((num_rows_a + BM - 1) / BM) * (BM / TM),
                              ((num_cols_b + BN - 1) / BN) * BN);

  return q.submit([&](sycl::handler &h) {
    sycl::local_accessor<float, 1> tile_a(BM * BK, h);
    sycl::local_accessor<float, 1> tile_b(BK * BN, h);

    h.parallel_for(
        sycl::nd_range<2>(global_range, local_range),
        [=](sycl::nd_item<2> item) {
          const uint block_row = item.get_group(0);
          const uint block_col = item.get_group(1);
          const uint thread_row = item.get_local_id(0);  // == threadIdx.x / BN
          const uint thread_col = item.get_local_id(1);  // == threadIdx.x % BN
          const uint tid = thread_row * BN + thread_col; // linear threadIdx.x

          const int global_row = block_row * BM + thread_row * TM;
          const int global_col = block_col * BN + thread_col;

          int a_offset = block_row * BM * num_cols_a;
          int b_offset = block_col * BN;
          int c_offset = block_row * BM * num_cols_b + block_col * BN;

          float thread_results[TM] = {0.0f};

          for (int tile_idx = 0; tile_idx < num_cols_a; tile_idx += BK) {
            const uint a_row = tid / BK;
            const uint a_col = tid % BK;
            tile_a[a_row * BK + a_col] =
                ((block_row * BM + a_row) < num_rows_a &&
                 (tile_idx + a_col) < num_cols_a)
                    ? matrix_a[a_offset + a_row * num_cols_a + a_col]
                    : 0.0f;

            const uint b_row = tid / BN;
            const uint b_col = tid % BN;
            tile_b[b_row * BN + b_col] =
                ((tile_idx + b_row) < num_cols_a &&
                 (block_col * BN + b_col) < num_cols_b)
                    ? matrix_b[b_offset + b_row * num_cols_b + b_col]
                    : 0.0f;

            sycl::group_barrier(item.get_group());

            a_offset += BK;
            b_offset += BK * num_cols_b;

            for (uint dot_idx = 0; dot_idx < BK; ++dot_idx) {
              float b_tmp = tile_b[dot_idx * BN + thread_col];
              for (uint res_idx = 0; res_idx < TM; ++res_idx) {
                thread_results[res_idx] +=
                    tile_a[(thread_row * TM + res_idx) * BK + dot_idx] * b_tmp;
              }
            }

            sycl::group_barrier(item.get_group());
          }

          for (uint res_idx = 0; res_idx < TM; ++res_idx) {
            int row = global_row + res_idx;
            if (row < num_rows_a && global_col < num_cols_b) {
              const int c_idx = c_offset +
                                (thread_row * TM + res_idx) * num_cols_b +
                                thread_col;
              matrix_c[c_idx] =
                  alpha * thread_results[res_idx] + beta * matrix_c[c_idx];
            }
          }
        });
  });
}

extern "C" float run_kernel(int M, int N, int K, int BM, int BN, int BK, int TM,
                            unsigned int seed) {
  sycl::queue q(sycl::gpu_selector_v,
                sycl::property::queue::enable_profiling{});

  auto gen = make_rng(seed);
  std::vector<float> A(M * K), B(K * N);
  fillWithRandom(A, gen);
  fillWithRandom(B, gen);

  float *dA = sycl::malloc_device<float>(M * K, q);
  float *dB = sycl::malloc_device<float>(K * N, q);
  float *dC = sycl::malloc_device<float>(M * N, q);

  q.memcpy(dA, A.data(), M * K * sizeof(float));
  q.memcpy(dB, B.data(), K * N * sizeof(float));
  q.memset(dC, 0, M * N * sizeof(float));
  q.wait();

  const float alpha = 1.0f, beta = 0.0f;

  sycl::event e;
  if (BM == 8 && BN == 8 && BK == 8 && TM == 1)
    e = sgemm_blocktiling_1d_kernel<8, 8, 8, 1>(q, M, N, K, alpha, dA, dB, beta,
                                                dC);
  else if (BM == 16 && BN == 16 && BK == 8 && TM == 2)
    e = sgemm_blocktiling_1d_kernel<16, 16, 8, 2>(q, M, N, K, alpha, dA, dB,
                                                  beta, dC);
  else if (BM == 32 && BN == 32 && BK == 8 && TM == 4)
    e = sgemm_blocktiling_1d_kernel<32, 32, 8, 4>(q, M, N, K, alpha, dA, dB,
                                                  beta, dC);
  else if (BM == 16 && BN == 16 && BK == 16 && TM == 1)
    e = sgemm_blocktiling_1d_kernel<16, 16, 16, 1>(q, M, N, K, alpha, dA, dB,
                                                   beta, dC);
  else if (BM == 32 && BN == 32 && BK == 16 && TM == 2)
    e = sgemm_blocktiling_1d_kernel<32, 32, 16, 2>(q, M, N, K, alpha, dA, dB,
                                                   beta, dC);
  else if (BM == 32 && BN == 32 && BK == 32 && TM == 1)
    e = sgemm_blocktiling_1d_kernel<32, 32, 32, 1>(q, M, N, K, alpha, dA, dB,
                                                   beta, dC);
  else {
    sycl::free(dA, q);
    sycl::free(dB, q);
    sycl::free(dC, q);
    return -1.0f;
  }
  e.wait();

  auto t_start =
      e.get_profiling_info<sycl::info::event_profiling::command_start>();
  auto t_end = e.get_profiling_info<sycl::info::event_profiling::command_end>();

  sycl::free(dA, q);
  sycl::free(dB, q);
  sycl::free(dC, q);

  return (t_end - t_start) * 1e-6f;
}
