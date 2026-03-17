#include "Kernels.hpp"
#include "Matrix.hpp"

int main() {
  size_t N = 2880;
  Matrix<double> A(N, N);
  Matrix<double> B(N, N);
  Matrix<double> C(N, N);

  A.fillRandom();
  B.fillRandom();

  Kernels::matMulParallel<64>(A, B, C);
}
