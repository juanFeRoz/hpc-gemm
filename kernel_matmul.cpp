#include <chrono>
#include <sycl/sycl.hpp>
#include <vector>

using namespace sycl;

extern "C" float run_kernel(int M, int N, int K, int BM, int BN, int BK,
                            int TM) {

  queue q{property::queue::in_order()};

  std::vector<float> A(M * K, 1.0f);
  std::vector<float> B(K * N, 1.0f);
  std::vector<float> C(M * N, 0.0f);

  float *dA{malloc_device<float>(M * K, q)};
  float *dB{malloc_device<float>(K * N, q)};
  float *dC{malloc_device<float>(M * N, q)};

  q.memcpy(dA, A.data(), sizeof(float) * M * K);
  q.memcpy(dB, B.data(), sizeof(float) * K * N);
  q.memcpy(dC, C.data(), sizeof(float) * M * N);

  auto start = std::chrono::high_resolution_clock::now();

  int grid_M = (M + BM - 1) / BM;
  int grid_N = (N + BN - 1) / BN;

  int threads = (BM * BN) / TM;

  q.submit([&](handler &h) {
     local_accessor<float, 2> As(range<2>(BM, BK), h);
     local_accessor<float, 2> Bs(range<2>(BK, BN), h);

     h.parallel_for(nd_range<2>(range<2>(grid_M * 1, grid_N * threads),
                                range<2>(1, threads)),
                    [=](nd_item<2> it) {
                      int blockRow = it.get_group(0);
                      int blockCol = it.get_group(1);

                      int tid = it.get_local_id(1);

                      int threadRow = tid / BN;
                      int threadCol = tid % BN;

                      // cada hilo calcula TM resultados en fila
                      float acc[32]; // TM máximo
                      for (int i = 0; i < TM; i++)
                        acc[i] = 0.0f;

                      int rowBase = blockRow * BM;
                      int colBase = blockCol * BN;

                      for (int k0 = 0; k0 < K; k0 += BK) {

                        for (int i = 0; i < TM; i++) {
                          int r = rowBase + threadRow * TM + i;
                          int c = k0 + threadCol;

                          if (r < M && c < K)
                            As[threadRow * TM + i][threadCol] = dA[r * K + c];
                        }

                        int rB = k0 + threadRow;
                        int cB = colBase + threadCol;

                        if (rB < K && cB < N)
                          Bs[threadRow][threadCol] = dB[rB * N + cB];

                        it.barrier(access::fence_space::local_space);

                        for (int k = 0; k < BK; k++) {
                          float bval = Bs[k][threadCol];

                          for (int i = 0; i < TM; i++) {
                            acc[i] += As[threadRow * TM + i][k] * bval;
                          }
                        }

                        it.barrier(access::fence_space::local_space);
                      }

                      for (int i = 0; i < TM; i++) {
                        int r = rowBase + threadRow * TM + i;
                        int c = colBase + threadCol;

                        if (r < M && c < N)
                          dC[r * N + c] = acc[i];
                      }
                    });
   }).wait();

  auto end = std::chrono::high_resolution_clock::now();
  q.memcpy(C.data(), dC, sizeof(float) * M * N).wait();
  free(dA, q);
  free(dB, q);
  free(dC, q);
  return std::chrono::duration<float, std::milli>(end - start).count();
}
