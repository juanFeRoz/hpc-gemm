#include <algorithm>
#include <chrono>
#include <cstddef>
#include <iostream>
#include <random>
#include <vector>

template <typename T> struct Matrix {
private:
  size_t rows;
  size_t cols;
  std::vector<T> data_vec;

public:
  Matrix(size_t n, size_t m) : rows(n), cols(m), data_vec(n * m, 0) {}

  size_t row() const { return rows; }
  size_t col() const { return cols; }

  T *data() { return data_vec.data(); }
  const T *data() const { return data_vec.data(); }

  T &operator()(size_t i, size_t j) { return data_vec[i * cols + j]; }
  const T &operator()(size_t i, size_t j) const {
    return data_vec[i * cols + j];
  }

  void fillRandom(T min = 0, T max = 1) {
    static std::mt19937 prng(std::random_device{}());
    std::uniform_real_distribution<T> dist(min, max);
    std::generate(data_vec.begin(), data_vec.end(),
                  [&]() { return dist(prng); });
  }
};

struct Timer {
  std::chrono::time_point<std::chrono::high_resolution_clock> start;
  Timer() { start = std::chrono::high_resolution_clock::now(); }
  ~Timer() {
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "Took: " << diff.count() << " s" << '\n';
  }
};

void matMulNaive(const Matrix<double> &A, const Matrix<double> &B,
                 Matrix<double> &C) {
  auto M = A.row();
  auto K = A.col();
  auto N = B.col();

  for (size_t i{0}; i < M; ++i) {
    for (size_t j{0}; j < N; ++j) {
      for (size_t k{0}; k < K; ++k) {
        C(i, j) += A(i, k) * B(k, j);
      }
    }
  }
}

int main() {
  size_t N = 2880;
  Matrix<double> A(N, N);
  Matrix<double> B(N, N);
  Matrix<double> C(N, N);

  A.fillRandom();
  B.fillRandom();

  {
    Timer t;
    matMulNaive(A, B, C);
  }
  return 0;
}
