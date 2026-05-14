#!/usr/bin/env python3
"""
Run multiple independent autotuning experiments for statistical comparison.
Orchestrates Bayesian vs Random search convergence analysis.
"""

import json
import os
import argparse
from pathlib import Path
import sys
from autotune import bayesian_search, random_search
from statistical_analysis import compare_algorithms, print_statistical_report


def run_tuning_experiment(M, N, K, run_idx, trials=30, seed=None):
    """
    Run one complete tuning experiment (matmul only).
    Returns convergence data for both Bayesian and Random search.
    """
    if seed is None:
        seed = run_idx * 1000
    
    print(f"\n{'='*80}")
    print(f"EXPERIMENT {run_idx + 1}: Bayesian Search")
    print(f"{'='*80}")
    
    # Run Bayesian search
    try:
        bayesian_results = bayesian_search(
            M, N, K,
            backend="sycl",
            num_runs=5,
            num_trials=trials,
            kernel="matmul",
            seed=seed
        )
        bayesian_times = [cfg["time_ms"] for cfg in bayesian_results]
        print(f"✓ Bayesian run completed: {len(bayesian_results)} configs evaluated")
    except Exception as e:
        print(f"ERROR: Bayesian search failed for run {run_idx}: {e}")
        return None, None
    
    print(f"\n{'='*80}")
    print(f"EXPERIMENT {run_idx + 1}: Random Search")
    print(f"{'='*80}")
    
    # Run Random search with same seed
    try:
        random_results = random_search(
            M, N, K,
            backend="sycl",
            num_runs=5,
            num_samples=trials,
            kernel="matmul",
            seed=seed
        )
        random_times = [cfg["time_ms"] for cfg in random_results]
        print(f"✓ Random run completed: {len(random_results)} configs evaluated")
    except Exception as e:
        print(f"ERROR: Random search failed for run {run_idx}: {e}")
        return bayesian_times, None
    
    return bayesian_times, random_times


def compute_best_so_far(times):
    """Compute cumulative best for convergence tracking."""
    best = float('inf')
    best_so_far = []
    for t in times:
        best = min(best, t)
        best_so_far.append(best)
    return best_so_far


def main():
    parser = argparse.ArgumentParser(description="Run multiple independent tuning experiments")
    parser.add_argument('--M', type=int, default=512, help='Matrix M dimension')
    parser.add_argument('--N', type=int, default=512, help='Matrix N dimension')
    parser.add_argument('--K', type=int, default=512, help='Matrix K dimension')
    parser.add_argument('--num-runs', type=int, default=5, help='Number of independent runs (default: 5)')
    parser.add_argument('--trials', type=int, default=40, help='Trials per run (default: 40)')
    parser.add_argument('--output', default="convergence_analysis.json", help="Output file for convergence data")
    
    args = parser.parse_args()
    
    M, N, K = args.M, args.N, args.K
    
    print("\n" + "="*80)
    print(f"STATISTICAL ANALYSIS: BAYESIAN vs RANDOM SEARCH")
    print(f"Problem size: {M}x{N}x{K}")
    print(f"Independent runs: {args.num_runs}")
    print(f"Trials per run: {args.trials}")
    print("="*80)
    
    all_bayesian = []
    all_random = []
    convergence_data = {
        "problem_size": {"M": M, "N": N, "K": K},
        "num_runs": args.num_runs,
        "trials_per_run": args.trials,
        "bayesian_runs": [],
        "random_runs": []
    }
    
    for run_idx in range(args.num_runs):
        bay_times, ran_times = run_tuning_experiment(
            M, N, K, run_idx,
            trials=args.trials,
            seed=run_idx * 1000
        )
        
        if bay_times:
            all_bayesian.append(bay_times)
            bay_best_so_far = compute_best_so_far(bay_times)
            convergence_data["bayesian_runs"].append({
                "run_idx": run_idx,
                "times": bay_times,
                "best_time_ms": min(bay_times),
                "mean_time_ms": sum(bay_times) / len(bay_times),
                "best_so_far": bay_best_so_far
            })
            print(f"\n✓ Bayesian Run {run_idx+1}: best = {min(bay_times):.4f} ms")
        else:
            print(f"\n✗ Bayesian Run {run_idx+1}: FAILED")
        
        if ran_times:
            all_random.append(ran_times)
            ran_best_so_far = compute_best_so_far(ran_times)
            convergence_data["random_runs"].append({
                "run_idx": run_idx,
                "times": ran_times,
                "best_time_ms": min(ran_times),
                "mean_time_ms": sum(ran_times) / len(ran_times),
                "best_so_far": ran_best_so_far
            })
            print(f"✓ Random Run {run_idx+1}:   best = {min(ran_times):.4f} ms")
        else:
            print(f"✗ Random Run {run_idx+1}:   FAILED")
    
    # Save convergence data
    with open(args.output, 'w') as f:
        json.dump(convergence_data, f, indent=2)
    print(f"\nSaved convergence data to: {args.output}")
    
    # Clean up temp files
    for i in range(args.num_runs):
        for prefix in ["_temp_bayesian_run", "_temp_random_run"]:
            try:
                os.remove(f"{prefix}{i}.json")
            except:
                pass
    
    # Prepare data for statistical analysis
    if all_bayesian and all_random:
        # Extract best from each run
        bayesian_best = [min(run) for run in all_bayesian]
        random_best = [min(run) for run in all_random]
        
        # Run statistical analysis directly
        print("\n" + "="*80)
        print("Running statistical analysis...")
        print("="*80)
        
        try:
            results = compare_algorithms(bayesian_best, random_best)
            print_statistical_report(results)
            
            with open("statistical_results.json", 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Statistical results saved to: statistical_results.json")
        except Exception as e:
            print(f"ERROR running statistical analysis: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\nInsufficient data for statistical analysis")
        return 1
    
    print("\n" + "="*80)
    print("Analysis complete! Results saved to:")
    print(f"  - Convergence data: {args.output}")
    print(f"  - Statistical results: statistical_results.json")
    print("="*80 + "\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
