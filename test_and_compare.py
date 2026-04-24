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
from autotune import bayesian_search, random_search, grid_search, compile_config, benchmark_config

M, N, K = 2048, 2048, 2048

DEFAULT_BM = 32
DEFAULT_BN = 32
DEFAULT_BK = 8
DEFAULT_TM = 1


def compile_and_test(bm, bn, bk, tm, backend, num_runs=5):
    """Compile and test a specific configuration."""
    print(f"\n{'='*70}")
    print(f"Testing {backend.upper()} with BM={bm}, BN={bn}, BK={bk}, TM={tm}")
    print(f"{'='*70}")
    
    try:
        if not compile_config(bm, bn, bk, tm, backend=backend):
            print(f"Compilation failed!")
            return None
        
        print(f"Compilation successful")
        
        avg_time, success = benchmark_config(bm, bn, bk, tm, backend=backend, num_runs=num_runs)
        
        if not success or avg_time is None:
            print(f"Benchmark failed!")
            return None
        
        throughput = (2 * M * N * K) / (avg_time * 1e6)  # GFLOP/s
        print(f"Average time: {avg_time:.4f} ms")
        print(f"Throughput: {throughput:.2f} GFLOP/s")
        return {
            "time_ms": avg_time,
            "throughput_gflops": throughput,
            "BM": bm,
            "BN": bn,
            "BK": bk,
            "TM": tm
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
    cuda_default = compile_and_test(DEFAULT_BM, DEFAULT_BN, DEFAULT_BK, DEFAULT_TM, "cuda", num_runs=5)
    
    if cuda_default is None:
        print("\nCUDA test failed. Aborting.")
        return
    
    # Step 2: Test SYCL with defaults
    print(f"\n{'#'*70}")
    print("STEP 2: Testing SYCL with default parameters (before tuning)")
    print(f"{'#'*70}")
    sycl_default = compile_and_test(DEFAULT_BM, DEFAULT_BN, DEFAULT_BK, DEFAULT_TM, "sycl", num_runs=5)
    
    if sycl_default is None:
        print("\nSYCL test failed. Aborting.")
        return
    
    # Step 3: Autotune SYCL with multiple strategies
    print(f"\n{'#'*70}")
    print("STEP 3: Autotuning SYCL kernel with multiple strategies")
    print(f"{'#'*70}\n")
    
    tuning_results = {}
    
    # Bayesian search
    print("3a) Running Bayesian Optimization...")
    bayesian_results = bayesian_search(backend="sycl", num_runs=2, num_trials=15)
    bayesian_results_sorted = sorted(bayesian_results, key=lambda x: x["time_ms"])
    tuning_results["bayesian"] = {
        "results": bayesian_results_sorted,
        "best": bayesian_results_sorted[0] if bayesian_results_sorted else None
    }
    
    # Random search
    print("\n3b) Running Random Search...")
    random_results = random_search(backend="sycl", num_runs=2, num_samples=15)
    random_results_sorted = sorted(random_results, key=lambda x: x["time_ms"])
    tuning_results["random"] = {
        "results": random_results_sorted,
        "best": random_results_sorted[0] if random_results_sorted else None
    }
    
    # Grid search
    print("\n3c) Running Grid Search...")
    grid_results = grid_search(backend="sycl", num_runs=2)
    grid_results_sorted = sorted(grid_results, key=lambda x: x["time_ms"])
    tuning_results["grid"] = {
        "results": grid_results_sorted,
        "best": grid_results_sorted[0] if grid_results_sorted else None
    }
    
    # Find overall best configuration
    all_bests = [(strategy, data["best"]) for strategy, data in tuning_results.items() if data["best"]]
    overall_best = min(all_bests, key=lambda x: x[1]["time_ms"])
    best_strategy, best_config = overall_best
    
    print(f"\n{'='*70}")
    print("TUNING STRATEGIES COMPARISON:")
    print(f"{'='*70}")
    for strategy, data in tuning_results.items():
        if data["best"]:
            print(f"{strategy.upper():12} - Time: {data['best']['time_ms']:8.4f} ms, Throughput: {data['best']['throughput_gflops']:6.2f} GFLOP/s")
    
    print(f"\nBEST OVERALL: {best_strategy.upper()}")
    print(f"  BM={best_config['BM']}, BN={best_config['BN']}, BK={best_config['BK']}, TM={best_config['TM']}")
    print(f"  Time: {best_config['time_ms']:.4f} ms")
    print(f"  Throughput: {best_config['throughput_gflops']:.2f} GFLOP/s")
    print(f"{'='*70}\n")
    
    # Step 4: Test SYCL with best tuned parameters
    print(f"\n{'#'*70}")
    print("STEP 4: Testing SYCL with best tuned parameters")
    print(f"{'#'*70}")
    sycl_tuned = compile_and_test(best_config['BM'], best_config['BN'], 
                                   best_config['BK'], best_config['TM'], "sycl", num_runs=5)
    
    if sycl_tuned is None:
        print("\nTuned SYCL test failed!")
        return
    
    # Step 5: Test CUDA with best tuned parameters
    print(f"\n{'#'*70}")
    print("STEP 5: Testing CUDA with best tuned SYCL parameters")
    print(f"{'#'*70}")
    cuda_tuned = compile_and_test(best_config['BM'], best_config['BN'], 
                                   best_config['BK'], best_config['TM'], "cuda", num_runs=5)
    
    if cuda_tuned is None:
        print("\nTuned CUDA test failed!")
        cuda_tuned = {"time_ms": None, "throughput_gflops": None}
    
    # Step 6: Generate comparison graphs
    print(f"\n{'#'*70}")
    print("STEP 6: Generating comparison graphs")
    print(f"{'#'*70}")
    
    # Calculate improvements
    cuda_default_time = cuda_default["time_ms"]
    sycl_default_time = sycl_default["time_ms"]
    sycl_tuned_time = sycl_tuned["time_ms"]
    cuda_tuned_time = cuda_tuned["time_ms"] if cuda_tuned["time_ms"] else cuda_default_time
    
    print(f"\nFINAL PERFORMANCE SUMMARY:")
    print(f"{'='*70}")
    print(f"CUDA (default):     {cuda_default_time:8.4f} ms  ({cuda_default['throughput_gflops']:6.2f} GFLOP/s)")
    print(f"CUDA (tuned):       {cuda_tuned_time:8.4f} ms  ({cuda_tuned['throughput_gflops']:6.2f} GFLOP/s)")
    print(f"SYCL (default):     {sycl_default_time:8.4f} ms  ({sycl_default['throughput_gflops']:6.2f} GFLOP/s)")
    print(f"SYCL (tuned):       {sycl_tuned_time:8.4f} ms  ({sycl_tuned['throughput_gflops']:6.2f} GFLOP/s)")
    print(f"{'='*70}\n")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Tuning strategy comparison
    ax1 = axes[0]
    strategies = []
    strategy_times = []
    strategy_colors_map = {"bayesian": "#45B7D1", "random": "#96CEB4", "grid": "#FFEAA7"}
    
    for strategy in ["bayesian", "random", "grid"]:
        if tuning_results[strategy]["best"]:
            strategies.append(strategy.capitalize())
            strategy_times.append(tuning_results[strategy]["best"]["time_ms"])
    
    bars1 = ax1.bar(strategies, strategy_times, 
                     color=[strategy_colors_map[s.lower()] for s in strategies], 
                     alpha=0.8, edgecolor='black', linewidth=2)
    ax1.set_ylabel("Best Time (ms)", fontsize=12, fontweight='bold')
    ax1.set_title("Tuning Strategy Comparison", fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    for bar, time in zip(bars1, strategy_times):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{time:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # Plot 2: Time comparison (all configurations)
    ax2 = axes[1]
    configs_all = ["CUDA\ndefault", "CUDA\ntuned", "SYCL\ndefault", "SYCL\ntuned"]
    times_all = [cuda_default_time, cuda_tuned_time, sycl_default_time, sycl_tuned_time]
    colors_all = ["#FF6B6B", "#FF8E72", "#4ECDC4", "#45B7D1"]
    
    bars2 = ax2.bar(configs_all, times_all, color=colors_all, alpha=0.8, edgecolor='black', linewidth=2)
    ax2.set_ylabel("Execution Time (ms)", fontsize=12, fontweight='bold')
    ax2.set_title("Execution Time Comparison", fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    for bar, time in zip(bars2, times_all):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{time:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Plot 3: Throughput comparison
    ax3 = axes[2]
    throughputs_all = [cuda_default["throughput_gflops"], cuda_tuned["throughput_gflops"],
                       sycl_default["throughput_gflops"], sycl_tuned["throughput_gflops"]]
    
    bars3 = ax3.bar(configs_all, throughputs_all, color=colors_all, alpha=0.8, edgecolor='black', linewidth=2)
    ax3.set_ylabel("Throughput (GFLOP/s)", fontsize=12, fontweight='bold')
    ax3.set_title("Throughput Comparison", fontsize=14, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    
    for bar, throughput in zip(bars3, throughputs_all):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{throughput:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    plt.suptitle(f"Matrix Multiplication Kernels Comprehensive Comparison ({M}×{N}×{K})", 
                 fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig("kernel_comparison.png", dpi=300, bbox_inches='tight')
    print("Saved comparison plot: kernel_comparison.png")
    
    # Create detailed summary table
    fig, ax = plt.subplots(figsize=(14, 10))
    
    sycl_vs_cuda_default = (sycl_default_time - cuda_default_time) / cuda_default_time * 100
    sycl_tuned_vs_cuda_default = (sycl_tuned_time - cuda_default_time) / cuda_default_time * 100
    cuda_tuned_vs_default = (cuda_tuned_time - cuda_default_time) / cuda_default_time * 100
    
    summary_data = [
        ["Metric", "CUDA Default", "CUDA Tuned", "SYCL Default", "SYCL Tuned"],
        ["Time (ms)", f"{cuda_default_time:.4f}", f"{cuda_tuned_time:.4f}", 
         f"{sycl_default_time:.4f}", f"{sycl_tuned_time:.4f}"],
        ["Throughput (GFLOP/s)", f"{cuda_default['throughput_gflops']:.2f}", 
         f"{cuda_tuned['throughput_gflops']:.2f}",
         f"{sycl_default['throughput_gflops']:.2f}", f"{sycl_tuned['throughput_gflops']:.2f}"],
        ["vs CUDA Default", "—", f"{cuda_tuned_vs_default:+.1f}%", 
         f"{sycl_vs_cuda_default:+.1f}%", f"{sycl_tuned_vs_cuda_default:+.1f}%"],
        ["", "", "", "", ""],
        ["Tuning Strategy Comparison", "", "", "", ""],
        ["Best Bayesian", f"{tuning_results['bayesian']['best']['time_ms']:.2f} ms", "", "", ""],
        ["Best Random", f"{tuning_results['random']['best']['time_ms']:.2f} ms", "", "", ""],
        ["Best Grid", f"{tuning_results['grid']['best']['time_ms']:.2f} ms", "", "", ""],
        ["Overall Winner", f"{best_strategy.upper()}", "", "", ""],
        ["", "", "", "", ""],
        ["Best Configuration", "", "", "", ""],
        ["BM", f"{best_config['BM']}", "", "", ""],
        ["BN", f"{best_config['BN']}", "", "", ""],
        ["BK", f"{best_config['BK']}", "", "", ""],
        ["TM", f"{best_config['TM']}", "", "", ""],
    ]
    
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=summary_data, cellLoc='center', loc='center',
                     colWidths=[0.2, 0.2, 0.2, 0.2, 0.2])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.2)
    
    # Header row
    for i in range(5):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Section headers
    for row in [5, 11]:
        for i in range(5):
            table[(row, i)].set_facecolor('#e0e0e0')
            table[(row, i)].set_text_props(weight='bold')
    
    # Data rows background
    for row in [1, 2, 3]:
        for i in range(5):
            table[(row, i)].set_facecolor('#f9f9f9')
    
    plt.title("Detailed Performance Summary - All Strategies", fontsize=14, fontweight='bold', pad=20)
    plt.savefig("kernel_summary.png", dpi=300, bbox_inches='tight')
    print("Saved detailed summary: kernel_summary.png")
    
    # Save comprehensive results
    results_summary = {
        "problem_size": {"M": M, "N": N, "K": K},
        "cuda_default": cuda_default,
        "cuda_tuned": cuda_tuned,
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
            },
            "grid": {
                "best_time_ms": tuning_results["grid"]["best"]["time_ms"] if tuning_results["grid"]["best"] else None,
                "best_throughput_gflops": tuning_results["grid"]["best"]["throughput_gflops"] if tuning_results["grid"]["best"] else None,
            }
        },
        "improvements": {
            "sycl_default_vs_cuda_default_percent": sycl_vs_cuda_default,
            "sycl_tuned_vs_cuda_default_percent": sycl_tuned_vs_cuda_default,
            "cuda_tuned_vs_default_percent": cuda_tuned_vs_default,
            "sycl_tuning_improvement_percent": (sycl_default_time - sycl_tuned_time) / sycl_default_time * 100,
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
