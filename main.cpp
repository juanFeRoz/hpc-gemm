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
  constexpr int N = 2048;
  std::vector<float> A(N * N);
  fillWithRandom(A);
  std::vector<float> B(N * N);
  fillWithRandom(B);

  sycl::queue q{sycl::property::queue::in_order()};

  float *dA{sycl::malloc_device<float>(N * N, q)};
  q.submit(
      [&](sycl::handler &h) { h.memcpy(dA, A.data(), N * N * sizeof(float)); });

  float *dB{sycl::malloc_device<float>(N * N, q)};
  q.submit(
      [&](sycl::handler &h) { h.memcpy(dB, B.data(), N * N * sizeof(float)); });

  float *dC{sycl::malloc_device<float>(N * N, q)};
  q.submit([&](sycl::handler &h) {
     h.parallel_for(sycl::range{N}, [=](sycl::id<1> idx) {
       size_t j{idx[0]};
       for (size_t i{0}; i < N; ++i) {
         float sum{0};
         for (size_t k{0}; k < N; ++k) {
           sum += dA[i * N + k] * dB[k * N + j];
         }
         dC[i * N + j] = sum;
       }
     });
   }).wait();

  std::vector<float> C(N * N);
  q.submit(
      [&](sycl::handler &h) { h.memcpy(C.data(), dC, N * N * sizeof(float)); });

  sycl::free(dA, q);
  sycl::free(dB, q);
  sycl::free(dC, q);

  return 0;
}
