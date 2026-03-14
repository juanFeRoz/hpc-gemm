#include "Kernels.hpp"
#include "Matrix.hpp"
#include <cblas.h>
#include <chrono>
#include <cstddef>
#include <iostream>

void matMulBLAS(const Matrix<double> &A, const Matrix<double> &B,
                Matrix<double> &C) {
  const int M{static_cast<int>(A.row())};
  const int K{static_cast<int>(A.col())};
  const int N{static_cast<int>(B.col())};

  // dgemm: Double precision General Matrix Multiply
  cblas_dgemm(CblasRowMajor, // Layout
              CblasNoTrans,  // Transpose A?
              CblasNoTrans,  // Transpose B?
              M, N, K,       // Dimensions
              1.0,           // Alpha (scalar to multiply A*B)
              A.data(), K,   // Matrix A and its leading dimension (LDA)
              B.data(), N,   // Matrix B and its leading dimension (LDB)
              0.0,           // Beta (scalar to multiply C before adding)
              C.data(), N    // Matrix C and its leading dimension (LDC)
  );
}

int main() {
  size_t N = 2880;
  Matrix<double> A(N, N);
  Matrix<double> B(N, N);
  Matrix<double> C(N, N);

  A.fillRandom();
  B.fillRandom();

  auto start{std::chrono::high_resolution_clock::now()};
  matMulBLAS(A, B, C);
  auto end{std::chrono::high_resolution_clock::now()};

  std::chrono::duration<double> diff{end - start};
  double elapsed{diff.count()};
  std::cout << "Took: " << elapsed << " s." << '\n';

  double N_val = static_cast<double>(N);
  double total_flops{2 * N_val * N_val * N_val};
  double gflops{total_flops / (elapsed * 1e9)};
  std::cout << "Performance: " << gflops << " GFLOPS" << '\n';
}
