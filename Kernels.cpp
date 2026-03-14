#include "Kernels.hpp"
#include <omp.h>

namespace Kernels {
// Reorganiza un bloque de B para que sea contiguo
template <int Nr>
void packB(int Kc, int Nc, const double *src, int lda, double *dst) {
  for (int j = 0; j < Nc; j += Nr) {
    for (int k = 0; k < Kc; ++k) {
      for (int jj = 0; jj < Nr; ++jj) {
        *dst++ = src[k * lda + j + jj];
      }
    }
  }
}

// Reorganiza un bloque de A
template <int Mr>
void packA(int Mc, int Kc, const double *src, int lda, double *dst) {
  for (int i = 0; i < Mc; i += Mr) {
    for (int k = 0; k < Kc; ++k) {
      for (int ii = 0; ii < Mr; ++ii) {
        *dst++ = src[(i + ii) * lda + k];
      }
    }
  }
}
void matMulParallel(const Matrix<double> &A, const Matrix<double> &B,
                    Matrix<double> &C) {
  auto M = A.row();
  auto K = A.col();
  auto N = B.col();

  // Valores optimizados para arquitecturas modernas
  constexpr int Mc{180};
  constexpr int Nc{96};
  constexpr int Kc{240};
  constexpr int Nr{12};
  constexpr int Mr{4};

#pragma omp parallel
  {
    // Buffers temporales por cada hilo (viven en el stack o heap local del
    // hilo)
    double aPacked[Mc * Kc];
    double bPacked[Kc * Nc];

#pragma omp for collapse(2) schedule(static)
    for (size_t ib = 0; ib < M; ib += Mc) {
      for (size_t jb = 0; jb < N; jb += Nc) {
        for (size_t kb = 0; kb < K; kb += Kc) {
          packA<Mr>(Mc, Kc, &A(ib, kb), static_cast<int>(K), aPacked);
          packB<Nr>(Kc, Nc, &B(kb, jb), static_cast<int>(N), bPacked);

          for (size_t i2 = 0; i2 < Mc; i2 += Mr) {
            for (size_t j2 = 0; j2 < Nc; j2 += Nr) {
              gemmKernel<Nr, Mr, Kc>(&aPacked[i2 * Kc], &bPacked[j2 * Kc],
                                     &C(ib + i2, jb + j2), static_cast<int>(N));
            }
          }
        }
      }
    }
  }
}
} // namespace Kernels
