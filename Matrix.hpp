#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <algorithm>
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

#endif
