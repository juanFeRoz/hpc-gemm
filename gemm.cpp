#include <algorithm>
#include <chrono>
#include <cstddef>
#include <immintrin.h>
#include <iostream>
#include <omp.h>
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

void matMulNaive_Order(const Matrix<double> &A, const Matrix<double> &B,
                       Matrix<double> &C) {
  auto M = A.row();
  auto K = A.col();
  auto N = B.col();

  for (size_t i{0}; i < M; ++i) {
    for (size_t k{0}; k < K; ++k) {
      for (size_t j{0}; j < N; ++j) {
        C(i, j) += A(i, k) * B(k, j);
      }
    }
  }
}

template <int BLOCK>
void block_kernel(const double *a, const double *mb, double *c, size_t N,
                  size_t K) {
  for (int i{0}; i < BLOCK; ++i, c += N, a += K) {
    const double *b{mb};
    for (int k{0}; k < BLOCK; ++k, b += N) {
      for (int j{0}; j < BLOCK; ++j) {
        c[j] += a[k] * b[j];
      }
    }
  }
}

void matMulNaive_Tile(const Matrix<double> &A, const Matrix<double> &B,
                      Matrix<double> &C) {
  auto M = A.row();
  auto K = A.col();
  auto N = B.col();

  constexpr int BLOCK = 64;

  for (size_t ib{0}; ib < M; ib += BLOCK) {
    for (size_t kb{0}; kb < K; kb += BLOCK) {
      for (size_t jb{0}; jb < N; jb += BLOCK) {
        const double *a{&A(ib, kb)};
        const double *mb{&B(kb, jb)};
        double *c{&C(ib, jb)};

        block_kernel<BLOCK>(a, mb, c, N, K);
      }
    }
  }
}

template <int BLOCK>
void avx_block_kernel(const double *a, const double *mb, double *c, size_t N,
                      size_t K) {
  // 256 bits / 64 bits por double = 4 doubles por registro
  constexpr int avx_doubles{4};

  for (int i{0}; i < BLOCK; ++i, c += N, a += K) {
    const double *b{mb};
    for (int k{0}; k < BLOCK; ++k, b += N) {
      // 1. Cargamos un solo valor de A y lo duplicamos en las 4 posiciones
      // del registro [a[k], a[k], a[k], a[k]]
      __m256d a_reg{_mm256_broadcast_sd(&a[k])};
      for (int j{0}; j < BLOCK; j += avx_doubles) {
        // 2. Cargamos 4 valores contiguos de B
        __m256d b_reg{_mm256_loadu_pd(&b[j])};
        // 3. Cargamos 4 valores contiguos de C
        __m256d c_reg{_mm256_loadu_pd(&c[j])};
        // 4. Fused Multiply-Add: c = (a * b) + c
        c_reg = _mm256_fmadd_pd(a_reg, b_reg, c_reg);
        // 5. Guardamos los 4 resultados de vuelta en la memoria de C
        _mm256_storeu_pd(&c[j], c_reg);
      }
    }
  }
}

void matMulNaive_AVX(const Matrix<double> &A, const Matrix<double> &B,
                     Matrix<double> &C) {
  auto M = A.row();
  auto K = A.col();
  auto N = B.col();

  constexpr int BLOCK = 64;

  for (size_t ib{0}; ib < M; ib += BLOCK) {
    for (size_t kb{0}; kb < K; kb += BLOCK) {
      for (size_t jb{0}; jb < N; jb += BLOCK) {
        const double *a{&A(ib, kb)};
        const double *mb{&B(kb, jb)};
        double *c{&C(ib, jb)};

        avx_block_kernel<BLOCK>(a, mb, c, N, K);
      }
    }
  }
}

template <int Nr, int Mr, int Kc, int Nc>
void avx_cache_micro_kernel(double *c, const double *a, const double *b,
                            size_t N, size_t K) {
  constexpr int avx_doubles = 4;

  for (int i{0}; i < Mr; ++i, c += N, a += K) {
    const double *b_ptr{b};
    for (int k{0}; k < Kc; ++k, b_ptr += N) {
      __m256d a_reg{_mm256_broadcast_sd(&a[k])};
      for (int j{0}; j < Nr; j += avx_doubles) {
        __m256d b_reg{_mm256_loadu_pd(&b_ptr[j])};
        __m256d c_reg{_mm256_loadu_pd(&c[j])};
        c_reg = _mm256_fmadd_pd(a_reg, b_reg, c_reg);
        _mm256_storeu_pd(&c[j], c_reg);
      }
    }
  }
}

void matMulAvxCache(const Matrix<double> &A, const Matrix<double> &B,
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

  for (size_t ib{0}; ib < M; ib += Mc) {
    for (size_t kb{0}; kb < K; kb += Kc) {
      for (size_t jb{0}; jb < N; jb += Nc) {
        const double *ma{&A(ib, kb)};
        const double *mb{&B(kb, jb)};
        double *mc{&C(ib, jb)};

        // Bucles para subdividir el macro-bloque en micro-bloques de registros
        for (size_t i2{0}; i2 < Mc; i2 += Mr) {
          for (size_t j2{0}; j2 < Nc; j2 += Nr) {
            const double *a_ptr = &ma[i2 * K];
            const double *b_ptr = &mb[j2];
            double *c_ptr = &mc[i2 * N + j2];

            // Llamada al micro-kernel vectorizado
            avx_cache_micro_kernel<Nr, Mr, Kc, Nc>(c_ptr, a_ptr, b_ptr, N, K);
          }
        }
      }
    }
  }
}

void matMulAvxCache_Parallel(const Matrix<double> &A, const Matrix<double> &B,
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

#pragma omp parallel for schedule(static) collapse(2)
  for (size_t ib = 0; ib < M; ib += Mc) {
    for (size_t kb = 0; kb < K; kb += Kc) {
      for (size_t jb = 0; jb < N; jb += Nc) {
        const double *ma{&A(ib, kb)};
        const double *mb{&B(kb, jb)};
        double *mc{&C(ib, jb)};

        // Bucles para subdividir el macro-bloque en micro-bloques de registros
        for (size_t i2 = 0; i2 < Mc; i2 += Mr) {
          for (size_t j2 = 0; j2 < Nc; j2 += Nr) {
            const double *a_ptr = &ma[i2 * K];
            const double *b_ptr = &mb[j2];
            double *c_ptr = &mc[i2 * N + j2];

            // Llamada al micro-kernel vectorizado
            avx_cache_micro_kernel<Nr, Mr, Kc, Nc>(c_ptr, a_ptr, b_ptr, N, K);
          }
        }
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

  auto start{std::chrono::high_resolution_clock::now()};
  matMulAvxCache_Parallel(A, B, C);
  auto end{std::chrono::high_resolution_clock::now()};

  std::chrono::duration<double> diff{end - start};
  double elapsed{diff.count()};
  std::cout << "Took: " << elapsed << " s." << '\n';

  double N_val = static_cast<double>(N);
  double total_flops{2 * N_val * N_val * N_val};
  double gflops{total_flops / (elapsed * 1e9)};
  std::cout << "Performance: " << gflops << " GFLOPS" << '\n';
}
