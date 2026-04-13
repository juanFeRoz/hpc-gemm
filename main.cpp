#include <algorithm>
#include <array>
#include <iostream>
#include <random>
#include <sycl/sycl.hpp>

template <typename Container> void fillWithRandom(Container &c) {
  static std::random_device rd;
  static std::mt19937 gen(rd());
  using T = typename Container::value_type;
  std::uniform_real_distribution<T> dis(0.0, 1.0);
  std::generate(std::begin(c), std::end(c), [&]() { return dis(gen); });
}

int main() {
  constexpr int N = 512;
  std::array<float, N * N> A{};
  fillWithRandom(A);
  std::array<float, N * N> B{};
  fillWithRandom(B);

  sycl::queue q{sycl::property::queue::in_order()};

  float *dA{sycl::malloc_device<float>(N * N, q)};
  q.submit(
      [&](sycl::handler &h) { h.memcpy(dA, &A[0], N * N * sizeof(float)); });

  float *dB{sycl::malloc_device<float>(N * N, q)};
  q.submit(
      [&](sycl::handler &h) { h.memcpy(dB, &B[0], N * N * sizeof(float)); });

  float *dC{sycl::malloc_device<float>(N * N, q)};

  return 0;
}
