#ifndef KERNELS_HPP
#define KERNELS_HPP

#include "Matrix.hpp"
#include <immintrin.h>

namespace Kernels {
template <int Nr, int Mr, int Kc>
inline void gemmKernel(const double *__restrict a, const double *__restrict b,
                       double *__restrict c, int N) {
  double carr[Nr * Mr] = {0.0};

  // Array local para registros (el compilador intentará poner esto en registros
  // AVX)
  for (int k{0}; k < Kc; ++k, b += Nr, a += Mr) {
    for (int i{0}; i < Mr; ++i) {
      for (int j{0}; j < Nr; ++j) {
        carr[i * Nr + j] += a[i] * b[j];
      }
    }
  }

  // Volcar el acumulador temporal a la matriz C original
  for (int i{0}; i < Mr; ++i) {
    for (int j{0}; j < Nr; ++j) {
      c[i * N + j] += carr[i * Nr + j];
    }
  }
}
void matMulParallel(const Matrix<double> &A, const Matrix<double> &B,
                    Matrix<double> &C);

} // namespace Kernels

#endif
