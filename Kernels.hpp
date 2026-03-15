#ifndef KERNELS_HPP
#define KERNELS_HPP

#include "Matrix.hpp"
#include <omp.h>

namespace Kernels {
template <int BLOCK>
inline void gemmKernel(const double *a, const double *mb, double *c, int N,
                       int K) {
  for (int i = 0; i < BLOCK; ++i, c += N, a += K) {
    const double *b = mb;
    for (int k = 0; k < BLOCK; ++k, b += N) {
      for (int j = 0; j < BLOCK; ++j) {
        c[j] += a[k] * b[j];
      }
    }
  }
}
template <int BLOCK>
inline void matMulParallel(const Matrix<double> &A, const Matrix<double> &B,
                           Matrix<double> &C) {
  const int M = static_cast<int>(A.row());
  const int K = static_cast<int>(A.col());
  const int N = static_cast<int>(B.col());

#pragma omp parallel for
  for (int ib = 0; ib < M; ib += BLOCK) {
    for (int kb = 0; kb < K; kb += BLOCK) {
      for (int jb = 0; jb < N; jb += BLOCK) {
        const double *a = &A(ib, kb);
        const double *mb = &B(kb, jb);
        double *c = &C(ib, jb);

        gemmKernel<BLOCK>(a, mb, c, N, K);
      }
    }
  }
}
} // namespace Kernels

#endif
