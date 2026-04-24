# Matrix Multiplication Autotuning Benchmark

This repository contains a small autotuning and benchmarking framework for matrix-matrix multiplication kernels implemented for SYCL and CUDA (HIP). It compiles parameterized kernels, benchmarks them, and runs several autotuning strategies to find performant tiling parameters.

## Project Structure

- autotune.py — autotuning utilities and search strategies (Bayesian, Random, Grid).
- test_and_compare.py — compile, benchmark, autotune, compare backends and strategies.
- wrapper.py — ctypes wrapper to call compiled shared libraries from Python.
- kernel_matmul_sycl.cpp — SYCL kernel implementation (parameterized via -D flags).
- kernel_matmul_cuda.hip — CUDA/HIP kernel implementation (parameterized via -D flags).
- Makefile — build convenience rules for compiling kernels to shared libraries.

## Requirements

- Python 3.8+
- optuna (for Bayesian tuning)
- numpy, matplotlib
- A working SYCL toolchain (`acpp`/oneAPI as used in the Makefile) for SYCL builds
- HIP/CUDA toolchain (`hipcc`) for CUDA builds

Install Python deps:

```bash
python3 -m pip install optuna numpy matplotlib
```

## Quickstart

1. Build and run the end-to-end benchmark and tuning pipeline:

```bash
python3 test_and_compare.py
```

This will:
- Benchmark CUDA and SYCL defaults
- Run multiple autotuning strategies on SYCL (Bayesian, Random, Grid)
- Test the best-found configuration on CUDA and SYCL
- Save comparison plots

Generated files:
- `kernel_comparison.png` — comparison plot of execution times
- `kernel_summary.png` — detailed summary visualization
- `benchmark_results.json` — full numeric results and metadata

## Autotuning

Autotuning logic resides in `autotune.py`. Available strategies:

- `bayesian_search(...)` — Optuna (recommended when trials are limited)
- `random_search(...)` — random sampling baseline
- `grid_search(...)` — exhaustive grid (only feasible for small spaces)

Search space parameters 
- `BM` (block M) ∈ {32, 64, 128}
- `BN` (block N) ∈ {32, 64, 128}
- `BK` (block K) ∈ {8, 16, 32}
- `TM` (thread multiplier) ∈ {1, 2, 4}

Constraints (enforced by `is_valid_config`):
- threads_per_block = (BM/TM) × BN ≤ 1024

Customize the search space and strategy parameters directly inside `autotune.py`

## Building kernels manually

Use the Makefile to compile parameterized shared libraries

```bash
make kernel_matmul_sycl.so BM=32 BN=32 BK=32 TM=4
make kernel_matmul_cuda.so BM=32 BN=32 BK=32 TM=4
```

Adjust compilers in the Makefile if your environment uses different toolchain commands.

## Interpreting results

- Lower execution time is better. The JSON contains per-configuration timings and throughput (GFLOP/s).
- Compare `CUDA (default)` vs `SYCL (default)` vs tuned results to assess portability of tuned parameters.

## Tips & Troubleshooting

- If compilation fails, check your SYCL/CUDA toolchain paths and the Makefile flags.
- Autotuning can be time-consuming — reduce `num_trials` or run `random_search` for faster (but less optimal) results.
- If many configurations are skipped, widen your search constraints or revise `is_valid_config` to match your device limits.