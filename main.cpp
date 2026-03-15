#include "Kernels.hpp"
#include "Matrix.hpp"
#include <chrono>
#include <cstddef>
#include <iostream>

int main() {
  size_t N = 2880;
  Matrix<double> A(N, N);
  Matrix<double> B(N, N);
  Matrix<double> C(N, N);

  A.fillRandom();
  B.fillRandom();

  auto start{std::chrono::high_resolution_clock::now()};
  Kernels::matMulParallel<64>(A, B, C);
  auto end{std::chrono::high_resolution_clock::now()};

  std::chrono::duration<double> diff{end - start};
  double elapsed{diff.count()};
  std::cout << "Took: " << elapsed << " s." << '\n';

  double N_val = static_cast<double>(N);
  double total_flops{2 * N_val * N_val * N_val};
  double gflops{total_flops / (elapsed * 1e9)};
  std::cout << "Performance: " << gflops << " GFLOPS" << '\n';
}
