#!/usr/bin/env python3
"""
Comprehensive test script to benchmark CUDA and SYCL kernels,
autotune SYCL, and compare performance before and after tuning.
"""

import subprocess
import json
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import argparse
from autotune import bayesian_search, random_search, compile_config, benchmark_config

parser = argparse.ArgumentParser(description="Benchmark and autotune matrix multiplication kernels")
parser.add_argument('--M', type=int, default=512, help='Matrix M dimension')
parser.add_argument('--N', type=int, default=512, help='Matrix N dimension')  
parser.add_argument('--K', type=int, default=512, help='Matrix K dimension')
args = parser.parse_args()

M, N, K = args.M, args.N, args.K

DEFAULT_BM = 32
DEFAULT_BN = 32
DEFAULT_BK = 8
DEFAULT_TM = 1
DEFAULT_TN = 1
DEFAULT_RUNS = 10
DEFAULT_TRIALS = 15


def compile_and_test(bm, bn, bk, tm, tn, backend, kernel="matmul", num_runs=DEFAULT_RUNS):
    """Compile and test a specific configuration."""
    print(f"\n{'='*70}")
    print(f"Testing {backend.upper()} {kernel.upper()} with BM={bm}, BN={bn}, BK={bk}, TM={tm}, TN={tn}")
    print(f"{'='*70}")
    
    try:
        if not compile_config(bm, bn, bk, tm, tn, backend=backend, kernel=kernel):
            print(f"Compilation failed!")
            return None
        
        print(f"Compilation successful")
        
        stats, success = benchmark_config(M, N, K, bm, bn, bk, tm, tn, backend=backend, kernel=kernel, num_runs=num_runs)
        
        if not success or stats is None:
            print(f"Benchmark failed!")
            return None
        
        avg_time = stats["mean"]
        std_ms = stats["std"]
        ci_half = stats["ci_half"]
        n = stats["n"]
        if kernel == "stencil":
            throughput = (6 * M * N) / (avg_time * 1e6)
        else:
            throughput = (2 * M * N * K) / (avg_time * 1e6)
        print(f"Average time: {avg_time:.4f} ± {ci_half:.4f} ms (n={n})")
        print(f"Throughput: {throughput:.2f} GFLOP/s")
        return {
            "time_ms": avg_time,
            "std_ms": std_ms,
            "ci_half": ci_half,
            "n": n,
            "throughput_gflops": throughput,
            "BM": bm,
            "BN": bn,
            "BK": bk,
            "TM": tm,
            "TN": tn,
            "kernel": kernel
        }
            
    except subprocess.TimeoutExpired:
        print(f"Timeout!")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None


def main():
    print("\n" + "="*70)
    print("KERNEL BENCHMARKING AND AUTOTUNING")
    print("="*70)
    print(f"Problem size: {M}x{N}x{K}")
    
    # Step 1: Test CUDA with defaults
    print(f"\n{'#'*70}")
    print("STEP 1: Testing CUDA with default parameters")
    print(f"{'#'*70}")
    cuda_default = compile_and_test(DEFAULT_BM, DEFAULT_BN, DEFAULT_BK, DEFAULT_TM, DEFAULT_TN, "cuda", num_runs=DEFAULT_RUNS)
    
    if cuda_default is None:
        print("\nCUDA test failed. Aborting.")
        return
    
    # Step 2: Test SYCL with defaults
    print(f"\n{'#'*70}")
    print("STEP 2: Testing SYCL with default parameters (before tuning)")
    print(f"{'#'*70}")
    sycl_default = compile_and_test(DEFAULT_BM, DEFAULT_BN, DEFAULT_BK, DEFAULT_TM, DEFAULT_TN, "sycl", num_runs=DEFAULT_RUNS)
    
    if sycl_default is None:
        print("\nSYCL test failed. Aborting.")
        return

    # Step 3: Test stencil benchmark on both backends
    print(f"\n{'#'*70}")
    print("STEP 3: Testing 2D stencil kernel with default parameters")
    print(f"{'#'*70}")
    cuda_stencil = compile_and_test(DEFAULT_BM, DEFAULT_BN, DEFAULT_BK, DEFAULT_TM, DEFAULT_TN, "cuda", kernel="stencil", num_runs=DEFAULT_RUNS)
    sycl_stencil = compile_and_test(DEFAULT_BM, DEFAULT_BN, DEFAULT_BK, DEFAULT_TM, DEFAULT_TN, "sycl", kernel="stencil", num_runs=DEFAULT_RUNS)

    if cuda_stencil is None or sycl_stencil is None:
        print("\nStencil test failed. Aborting.")
        return

    # Step 4: Autotune SYCL matmul with multiple strategies
    print(f"\n{'#'*70}")
    print("STEP 4: Autotuning SYCL matmul kernel with multiple strategies")
    print(f"{'#'*70}\n")
    print("NOTE: Using repeated measurements for publishable benchmarks, and convergence baseline graphs will be generated.")
    
    tuning_results = {}
    
    # Bayesian search
    print("3a) Running Bayesian Optimization...")
    bayesian_results = bayesian_search(M, N, K, backend="sycl", num_runs=DEFAULT_RUNS, num_trials=DEFAULT_TRIALS)
    bayesian_results_sorted = sorted(bayesian_results, key=lambda x: x["time_ms"])
    tuning_results["bayesian"] = {
        "results": bayesian_results_sorted,
        "best": bayesian_results_sorted[0] if bayesian_results_sorted else None
    }
    
    # Random search
    print("\n3b) Running Random Search...")
    random_results = random_search(M, N, K, backend="sycl", num_runs=DEFAULT_RUNS, num_samples=DEFAULT_TRIALS)
    random_results_sorted = sorted(random_results, key=lambda x: x["time_ms"])
    tuning_results["random"] = {
        "results": random_results_sorted,
        "best": random_results_sorted[0] if random_results_sorted else None
    }

    # Find overall best configuration among search strategies only
    def cumulative_best(times):
        best = float('inf')
        best_so_far = []
        for t in times:
            best = min(best, t)
            best_so_far.append(best)
        return best_so_far

    bayesian_best_so_far = cumulative_best([entry["time_ms"] for entry in bayesian_results])
    random_best_so_far = cumulative_best([entry["time_ms"] for entry in random_results])

    all_bests = [(strategy, data["best"]) for strategy, data in tuning_results.items() if strategy in ["bayesian", "random"] and data["best"]]
    if not all_bests:
        print("No valid tuning results found. Aborting.")
        return
    overall_best = min(all_bests, key=lambda x: x[1]["time_ms"])
    best_strategy, best_config = overall_best
    
    print(f"\n{'='*70}")
    print("TUNING STRATEGIES COMPARISON:")
    print(f"{'='*70}")
    for strategy, data in tuning_results.items():
        if data["best"]:
            label = strategy.upper()
            print(f"{label:12} - Time: {data['best']['time_ms']:8.4f} ms, Throughput: {data['best']['throughput_gflops']:6.2f} GFLOP/s")
    
    print(f"\nBEST OVERALL: {best_strategy.upper()}")
    print(f"  BM={best_config['BM']}, BN={best_config['BN']}, BK={best_config['BK']}, TM={best_config['TM']}, TN={best_config.get('TN', 1)}")
    print(f"  Time: {best_config['time_ms']:.4f} ms")
    print(f"  Throughput: {best_config['throughput_gflops']:.2f} GFLOP/s")
    print(f"{'='*70}\n")
    
    # Step 5: Test tuned matmul on both backends
    print(f"\n{'#'*70}")
    print("STEP 5: Testing matmul with best tuned parameters")
    print(f"{'#'*70}")
    sycl_tuned = compile_and_test(best_config['BM'], best_config['BN'], 
                                   best_config['BK'], best_config['TM'], best_config.get('TN', 1), "sycl", kernel="matmul", num_runs=DEFAULT_RUNS)
    
    if sycl_tuned is None:
        print("\nTuned SYCL matmul test failed!")
        return
    
    print(f"\n{'#'*70}")
    print("Testing CUDA with best tuned matmul parameters")
    print(f"{'#'*70}")
    cuda_matmul_tuned = compile_and_test(best_config['BM'], best_config['BN'], 
                                   best_config['BK'], best_config['TM'], best_config.get('TN', 1), "cuda", kernel="matmul", num_runs=DEFAULT_RUNS)
    
    if cuda_matmul_tuned is None:
        print("\nTuned CUDA matmul test failed!")
        cuda_matmul_tuned = {"time_ms": None, "throughput_gflops": None}
    
    # Step 6: Autotune SYCL stencil kernel
    print(f"\n{'#'*70}")
    print("STEP 6: Autotuning SYCL stencil kernel with multiple strategies")
    print(f"{'#'*70}\n")
    
    stencil_tuning_results = {}
    stencil_best_strategy = None
    stencil_best_config = None
    sycl_stencil_tuned = None
    cuda_stencil_tuned = None
    
    print("6a) Running Bayesian Optimization for stencil...")
    bayesian_stencil = bayesian_search(M, N, K, backend="sycl", num_runs=DEFAULT_RUNS, num_trials=DEFAULT_TRIALS, kernel="stencil")
    bayesian_stencil_sorted = sorted(bayesian_stencil, key=lambda x: x["time_ms"])
    stencil_tuning_results["bayesian"] = {
        "results": bayesian_stencil_sorted,
        "best": bayesian_stencil_sorted[0] if bayesian_stencil_sorted else None
    }
    
    print("\n6b) Running Random Search for stencil...")
    random_stencil = random_search(M, N, K, backend="sycl", num_runs=DEFAULT_RUNS, num_samples=DEFAULT_TRIALS, kernel="stencil")
    random_stencil_sorted = sorted(random_stencil, key=lambda x: x["time_ms"])
    stencil_tuning_results["random"] = {
        "results": random_stencil_sorted,
        "best": random_stencil_sorted[0] if random_stencil_sorted else None
    }
    
    stencil_all_bests = [(strategy, data["best"]) for strategy, data in stencil_tuning_results.items() if strategy in ["bayesian", "random"] and data["best"]]
    if not stencil_all_bests:
        print("No valid stencil tuning results found.")
        stencil_best_config = None
    else:
        stencil_best_strategy, stencil_best_config = min(stencil_all_bests, key=lambda x: x[1]["time_ms"])
        
        print(f"\n{'='*70}")
        print("STENCIL TUNING STRATEGIES COMPARISON:")
        print(f"{'='*70}")
        for strategy, data in stencil_tuning_results.items():
            if data["best"]:
                label = strategy.upper()
                print(f"{label:12} - Time: {data['best']['time_ms']:8.4f} ms, Throughput: {data['best']['throughput_gflops']:6.2f} GFLOP/s")
        
        print(f"\nBEST STENCIL OVERALL: {stencil_best_strategy.upper()}")
        print(f"  BM={stencil_best_config['BM']}, BN={stencil_best_config['BN']}, BK={stencil_best_config['BK']}, TM={stencil_best_config['TM']}, TN={stencil_best_config.get('TN', 1)}")
        print(f"  Time: {stencil_best_config['time_ms']:.4f} ms")
        print(f"  Throughput: {stencil_best_config['throughput_gflops']:.2f} GFLOP/s")
        print(f"{'='*70}\n")

    stencil_bayesian_raw = bayesian_stencil
    stencil_random_raw = random_stencil
    
    # Step 7: Test tuned stencil on both backends
    print(f"\n{'#'*70}")
    print("STEP 7: Testing stencil with best tuned parameters")
    print(f"{'#'*70}")
    
    if stencil_best_config:
        sycl_stencil_tuned = compile_and_test(stencil_best_config['BM'], stencil_best_config['BN'], 
                                       stencil_best_config['BK'], stencil_best_config['TM'], stencil_best_config.get('TN', 1), 
                                       "sycl", kernel="stencil", num_runs=DEFAULT_RUNS)
        
        if sycl_stencil_tuned is None:
            print("\nTuned SYCL stencil test failed!")
            sycl_stencil_tuned = sycl_stencil
        
        print(f"\n{'#'*70}")
        print("Testing CUDA with best tuned stencil parameters")
        print(f"{'#'*70}")
        cuda_stencil_tuned = compile_and_test(stencil_best_config['BM'], stencil_best_config['BN'], 
                                       stencil_best_config['BK'], stencil_best_config['TM'], stencil_best_config.get('TN', 1), 
                                       "cuda", kernel="stencil", num_runs=DEFAULT_RUNS)
        
        if cuda_stencil_tuned is None:
            print("\nTuned CUDA stencil test failed!")
            cuda_stencil_tuned = {"time_ms": None, "throughput_gflops": None}
    else:
        print("Skipping stencil tuned tests (no valid tuning results)")
        sycl_stencil_tuned = sycl_stencil
        cuda_stencil_tuned = cuda_stencil
    
    # Step 8: Generate comparison graphs
    print(f"\n{'#'*70}")
    print("STEP 8: Generating comparison graphs")
    print(f"{'#'*70}")
    
    # Calculate improvements
    cuda_default_time = cuda_default["time_ms"]
    sycl_default_time = sycl_default["time_ms"]
    sycl_matmul_tuned_time = sycl_tuned["time_ms"]
    cuda_matmul_tuned_time = cuda_matmul_tuned["time_ms"] if cuda_matmul_tuned["time_ms"] else cuda_default_time
    
    print(f"\nFINAL PERFORMANCE SUMMARY:")
    print(f"{'='*70}")
    print(f"MATMUL KERNELS:")
    print(f"  CUDA (default):     {cuda_default_time:8.4f} ms  ({cuda_default['throughput_gflops']:6.2f} GFLOP/s)")
    print(f"  CUDA (tuned):       {cuda_matmul_tuned_time:8.4f} ms  ({cuda_matmul_tuned['throughput_gflops']:6.2f} GFLOP/s)")
    print(f"  SYCL (default):     {sycl_default_time:8.4f} ms  ({sycl_default['throughput_gflops']:6.2f} GFLOP/s)")
    print(f"  SYCL (tuned):       {sycl_matmul_tuned_time:8.4f} ms  ({sycl_tuned['throughput_gflops']:6.2f} GFLOP/s)")
    print(f"\nSTENCIL KERNELS:")
    print(f"  CUDA (default):     {cuda_stencil['time_ms']:8.4f} ms  ({cuda_stencil['throughput_gflops']:6.2f} GFLOP/s)")
    print(f"  CUDA (tuned):       {cuda_stencil_tuned['time_ms']:8.4f} ms  ({cuda_stencil_tuned['throughput_gflops']:6.2f} GFLOP/s)")
    print(f"  SYCL (default):     {sycl_stencil['time_ms']:8.4f} ms  ({sycl_stencil['throughput_gflops']:6.2f} GFLOP/s)")
    print(f"  SYCL (tuned):       {sycl_stencil_tuned['time_ms']:8.4f} ms  ({sycl_stencil_tuned['throughput_gflops']:6.2f} GFLOP/s)")
    print(f"{'='*70}\n")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Matmul: Time comparison
    ax0 = axes[0, 0]
    configs_matmul = ["CUDA\ndefault", "CUDA\ntuned", "SYCL\ndefault", "SYCL\ntuned"]
    times_matmul = [cuda_default_time, cuda_matmul_tuned_time, sycl_default_time, sycl_matmul_tuned_time]
    colors_matmul = ["#FF6B6B", "#FF8E72", "#4ECDC4", "#45B7D1"]
    
    ci_errors_matmul = [cuda_default.get('ci_half', 0.0), cuda_matmul_tuned.get('ci_half', 0.0), sycl_default.get('ci_half', 0.0), sycl_tuned.get('ci_half', 0.0)]
    bars0 = ax0.bar(configs_matmul, times_matmul, yerr=ci_errors_matmul, capsize=8, color=colors_matmul, alpha=0.8, edgecolor='black', linewidth=2)
    ax0.set_ylabel("Execution Time (ms)", fontsize=12, fontweight='bold')
    ax0.set_title("MatMul: Execution Time Comparison", fontsize=14, fontweight='bold')
    ax0.grid(axis='y', alpha=0.3, linestyle='--')
    
    for bar, time in zip(bars0, times_matmul):
        height = bar.get_height()
        ax0.text(bar.get_x() + bar.get_width()/2., height,
                f'{time:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Matmul: Throughput comparison
    ax1 = axes[0, 1]
    throughputs_matmul = [cuda_default["throughput_gflops"], cuda_matmul_tuned["throughput_gflops"],
                       sycl_default["throughput_gflops"], sycl_tuned["throughput_gflops"]]
    
    bars1 = ax1.bar(configs_matmul, throughputs_matmul, color=colors_matmul, alpha=0.8, edgecolor='black', linewidth=2)
    ax1.set_ylabel("Throughput (GFLOP/s)", fontsize=12, fontweight='bold')
    ax1.set_title("MatMul: Throughput Comparison", fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    for bar, throughput in zip(bars1, throughputs_matmul):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{throughput:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Stencil: Time comparison
    ax2 = axes[1, 0]
    configs_stencil = ["CUDA\ndefault", "CUDA\ntuned", "SYCL\ndefault", "SYCL\ntuned"]
    times_stencil = [cuda_stencil["time_ms"], cuda_stencil_tuned["time_ms"] if cuda_stencil_tuned["time_ms"] else cuda_stencil["time_ms"], 
                     sycl_stencil["time_ms"], sycl_stencil_tuned["time_ms"]]
    colors_stencil = ["#FFD93D", "#FFC869", "#6BCB77", "#4D96FF"]
    
    ci_errors_stencil = [cuda_stencil.get('ci_half', 0.0), cuda_stencil_tuned.get('ci_half', 0.0), sycl_stencil.get('ci_half', 0.0), sycl_stencil_tuned.get('ci_half', 0.0)]
    bars2 = ax2.bar(configs_stencil, times_stencil, yerr=ci_errors_stencil, capsize=8, color=colors_stencil, alpha=0.8, edgecolor='black', linewidth=2)
    ax2.set_ylabel("Execution Time (ms)", fontsize=12, fontweight='bold')
    ax2.set_title("Stencil: Execution Time Comparison", fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    for bar, time in zip(bars2, times_stencil):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{time:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Stencil: Throughput comparison
    ax3 = axes[1, 1]
    throughputs_stencil = [cuda_stencil["throughput_gflops"], 
                           cuda_stencil_tuned["throughput_gflops"] if cuda_stencil_tuned["throughput_gflops"] else cuda_stencil["throughput_gflops"],
                           sycl_stencil["throughput_gflops"], 
                           sycl_stencil_tuned["throughput_gflops"] if sycl_stencil_tuned["throughput_gflops"] else sycl_stencil["throughput_gflops"]]
    
    bars3 = ax3.bar(configs_stencil, throughputs_stencil, color=colors_stencil, alpha=0.8, edgecolor='black', linewidth=2)
    ax3.set_ylabel("Throughput (GFLOP/s)", fontsize=12, fontweight='bold')
    ax3.set_title("Stencil: Throughput Comparison", fontsize=14, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    
    for bar, throughput in zip(bars3, throughputs_stencil):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{throughput:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.suptitle(f"Kernel Benchmarks: MatMul and Stencil ({M}×{N}×{K})", 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig("kernel_comparison.png", dpi=300, bbox_inches='tight')
    print("Saved comparison plot: kernel_comparison.png")

    # Convergence baseline for matmul tuning
    fig_matmul_conv, ax_matmul = plt.subplots(figsize=(8, 5))
    if bayesian_best_so_far:
        ax_matmul.plot(range(1, len(bayesian_best_so_far) + 1), bayesian_best_so_far,
                     marker='o', linestyle='-', color='#45B7D1', label='Bayesian')
    if random_best_so_far:
        ax_matmul.plot(range(1, len(random_best_so_far) + 1), random_best_so_far,
                     marker='s', linestyle='--', color='#96CEB4', label='Random')
    ax_matmul.set_xlabel('Evaluation #', fontsize=12, fontweight='bold')
    ax_matmul.set_ylabel('Best Time So Far (ms)', fontsize=12, fontweight='bold')
    ax_matmul.set_title('MatMul Search Convergence', fontsize=14, fontweight='bold')
    ax_matmul.grid(True, alpha=0.3, linestyle='--')
    ax_matmul.legend()
    plt.tight_layout()
    plt.savefig('convergence_matmul.png', dpi=300, bbox_inches='tight')
    print('Saved MatMul convergence plot: convergence_matmul.png')
    
    # Convergence baseline for stencil tuning
    fig_stencil_conv, ax_stencil = plt.subplots(figsize=(8, 5))
    if stencil_all_bests:
        if stencil_bayesian_raw:
            stencil_bayesian_best_so_far = cumulative_best([entry["time_ms"] for entry in stencil_bayesian_raw])
            ax_stencil.plot(range(1, len(stencil_bayesian_best_so_far) + 1), stencil_bayesian_best_so_far,
                         marker='o', linestyle='-', color='#FFD93D', label='Bayesian')
        if stencil_random_raw:
            stencil_random_best_so_far = cumulative_best([entry["time_ms"] for entry in stencil_random_raw])
            ax_stencil.plot(range(1, len(stencil_random_best_so_far) + 1), stencil_random_best_so_far,
                         marker='s', linestyle='--', color='#6BCB77', label='Random')
    ax_stencil.set_xlabel('Evaluation #', fontsize=12, fontweight='bold')
    ax_stencil.set_ylabel('Best Time So Far (ms)', fontsize=12, fontweight='bold')
    ax_stencil.set_title('Stencil Search Convergence', fontsize=14, fontweight='bold')
    ax_stencil.grid(True, alpha=0.3, linestyle='--')
    ax_stencil.legend()
    plt.tight_layout()
    plt.savefig('convergence_stencil.png', dpi=300, bbox_inches='tight')
    print('Saved Stencil convergence plot: convergence_stencil.png')
    
    # Calculate improvements
    cuda_default_time = cuda_default["time_ms"]
    cuda_matmul_tuned_time = cuda_matmul_tuned["time_ms"]
    sycl_default_time = sycl_default["time_ms"]
    sycl_matmul_tuned_time = sycl_tuned["time_ms"]
    
    sycl_vs_cuda_default = (cuda_default_time - sycl_default_time) / cuda_default_time * 100
    sycl_matmul_tuned_vs_cuda_default = (cuda_default_time - sycl_matmul_tuned_time) / cuda_default_time * 100
    cuda_matmul_tuned_vs_default = (cuda_default_time - cuda_matmul_tuned_time) / cuda_default_time * 100
    
    # Save comprehensive results
    results_summary = {
        "problem_size": {"M": M, "N": N, "K": K},
        "matmul_kernels": {
            "cuda_default": cuda_default,
            "cuda_tuned": cuda_matmul_tuned,
            "sycl_default": sycl_default,
            "sycl_tuned": sycl_tuned,
            "best_config": best_config,
            "best_strategy": best_strategy,
            "tuning_strategies": {
                "bayesian": {
                    "best_time_ms": tuning_results["bayesian"]["best"]["time_ms"] if tuning_results["bayesian"]["best"] else None,
                    "best_throughput_gflops": tuning_results["bayesian"]["best"]["throughput_gflops"] if tuning_results["bayesian"]["best"] else None,
                },
                "random": {
                    "best_time_ms": tuning_results["random"]["best"]["time_ms"] if tuning_results["random"]["best"] else None,
                    "best_throughput_gflops": tuning_results["random"]["best"]["throughput_gflops"] if tuning_results["random"]["best"] else None,
                }
            },
            "improvements": {
                "sycl_default_vs_cuda_default_percent": sycl_vs_cuda_default,
                "sycl_tuned_vs_cuda_default_percent": sycl_matmul_tuned_vs_cuda_default,
                "cuda_tuned_vs_default_percent": cuda_matmul_tuned_vs_default,
                "sycl_tuning_improvement_percent": (sycl_default_time - sycl_matmul_tuned_time) / sycl_default_time * 100,
            }
        },
        "stencil_kernels": {
            "cuda_default": cuda_stencil,
            "cuda_tuned": cuda_stencil_tuned,
            "sycl_default": sycl_stencil,
            "sycl_tuned": sycl_stencil_tuned,
            "best_config": stencil_best_config,
            "best_strategy": stencil_best_strategy if stencil_best_config else None,
            "tuning_strategies": {
                "bayesian": {
                    "best_time_ms": stencil_tuning_results["bayesian"]["best"]["time_ms"] if stencil_tuning_results["bayesian"]["best"] else None,
                    "best_throughput_gflops": stencil_tuning_results["bayesian"]["best"]["throughput_gflops"] if stencil_tuning_results["bayesian"]["best"] else None,
                },
                "random": {
                    "best_time_ms": stencil_tuning_results["random"]["best"]["time_ms"] if stencil_tuning_results["random"]["best"] else None,
                    "best_throughput_gflops": stencil_tuning_results["random"]["best"]["throughput_gflops"] if stencil_tuning_results["random"]["best"] else None,
                }
            }
        }
    }
    
    with open("benchmark_results.json", "w") as f:
        json.dump(results_summary, f, indent=2)
    print("Saved results to benchmark_results.json")
    
    print(f"\n{'='*70}")
    print("All tests and comparisons completed successfully!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
