#!/usr/bin/env python3
import subprocess
import json
import sys
import re
import os
import random
import math
import statistics
import optuna
from pathlib import Path

M, N, K = 512, 512, 512
DEFAULT_SEED = 42

search_spaces = {
    "matmul": {
        "BM": [32, 64, 128],
        "BN": [32, 64, 128],
        "BK": [8, 16, 32],
        "TM": [1, 2, 4],
        "TN": [1, 2, 4]
    },
}


def all_valid_configs(kernel="matmul"):
    space = search_spaces.get(kernel, search_spaces["matmul"])
    return [
        f"{bm}_{bn}_{bk}_{tm}_{tn}"
        for bm in space["BM"]
        for bn in space["BN"]
        for bk in space["BK"]
        for tm in space["TM"]
        for tn in space["TN"]
        if is_valid_config(bm, bn, bk, tm, tn, kernel=kernel)[0]
    ]


def compile_config(bm, bn, bk, tm, tn=1, backend="sycl", kernel="matmul"):
    try:
        env = os.environ.copy()
        env['BM'] = str(bm)
        env['BN'] = str(bn)
        env['BK'] = str(bk)
        env['TM'] = str(tm)
        env['TN'] = str(tn)
        
        subprocess.run(["rm", "-f", f"kernel_{kernel}_{backend}.o", f"kernel_{kernel}_{backend}.so"], capture_output=True)
        
        result = subprocess.run(
            ["make", f"kernel_{kernel}_{backend}.so", f"BM={bm}", f"BN={bn}", f"BK={bk}", f"TM={tm}", f"TN={tn}" ],
            capture_output=True,
            text=True,
            timeout=60,
            env=env
        )
        
        if result.returncode != 0:
            print(f"  [COMPILE ERROR] {result.stderr[-500:]}")
            return False
        return True
    except subprocess.TimeoutExpired:
        return False
    except Exception as e:
        return False

def benchmark_config(M, N, K, bm, bn, bk, tm, tn=1, backend="sycl", kernel="matmul", num_runs=20, warmup_runs=3, seed=DEFAULT_SEED):
    try:
        if not compile_config(bm, bn, bk, tm, tn, backend=backend, kernel=kernel):
            return None, False
        
        script_name = f"_benchmark_temp_{kernel}_{bm}_{bn}_{bk}_{tm}_{tn}.py"
        benchmark_code = f"""
import sys
import json
import math
import statistics
sys.path.insert(0, '.')
from wrapper import run

M, N, K = {M}, {N}, {K}
seed = {seed}
for i in range({warmup_runs}):
    _ = run(M, N, K, {bm}, {bn}, {bk}, {tm}, {tn}, backend=\"{backend}\", kernel=\"{kernel}\", seed=seed + i)

times = []
for i in range({num_runs}):
    elapsed = run(M, N, K, {bm}, {bn}, {bk}, {tm}, {tn}, backend=\"{backend}\", kernel=\"{kernel}\", seed=seed + {warmup_runs} + i)
    times.append(elapsed)

n = len(times)
avg = statistics.mean(times)
median = statistics.median(times)
std = statistics.stdev(times) if n > 1 else 0.0

t_critical = {{1:12.706, 2:4.303, 3:3.182, 4:2.776, 5:2.571,
               6:2.447, 7:2.365, 8:2.306, 9:2.262, 10:2.228,
               11:2.201, 12:2.179, 13:2.160, 14:2.145, 15:2.131}}

t_coeff = t_critical.get(n, 2.0)
ci_half = t_coeff * std / math.sqrt(n) if n > 1 else 0.0

stats = {{
    "mean": avg,
    "median": median,
    "std": std,
    "ci_half": ci_half,
    "n": n,
    "times": times
}}
print(json.dumps(stats))
"""
        
        with open(script_name, 'w') as f:
            f.write(benchmark_code)
        
        result = subprocess.run(
            ["python3", script_name],
            capture_output=True,
            text=True,
            timeout=120
        )
        
        try:
            os.remove(script_name)
        except:
            pass
        
        if result.returncode != 0:
            return None, False
        
        try:
            stats = json.loads(result.stdout.strip().split('\n')[-1])
            if stats.get("median", 0.0) <= 0.0:
                return None, False
            return stats, True
        except Exception:
            return None, False
            
    except subprocess.TimeoutExpired:
        return None, False
    except Exception as e:
        return None, False

def is_valid_config(bm, bn, bk, tm, tn=1, kernel="matmul"):
    if tm <= 0 or tn <= 0:
        return False, "TM and TN must be positive"
    if bm % tm != 0 or bn % tn != 0:
        return False, "BM must be divisible by TM and BN must be divisible by TN"

    threads_per_block = (bm // tm) * (bn // tn)
    if threads_per_block <= 0 or threads_per_block > 1024:
        return False, f"threads={threads_per_block} out of range"

    if kernel == "matmul":
        shared_memory = (bm * bk + bk * bn) * 4
        if shared_memory > 48 * 1024:
            return False, f"shared_mem={shared_memory/1024:.1f}KB > 96KB"

    return True, None

def random_search(M, N, K, backend="cuda", num_runs=3, num_samples=20, kernel="matmul", seed=DEFAULT_SEED):
    results = []
    
    print(f"\n{'='*80}")
    print(f"Random Search Auto-tuning {backend.upper()} {kernel.upper()} kernel")
    print(f"{'='*80}")
    print(f"Problem size: {M}x{N}x{K}")
    print(f"Samples to evaluate: {num_samples}\n")
    
    random.seed(seed)
    
    valid_configs = all_valid_configs(kernel=kernel)
    if not valid_configs:
        print("No valid configurations available in search space.")
        return results

    num_samples = min(num_samples, len(valid_configs))
    sampled_configs = random.sample(valid_configs, num_samples)

    for sample_idx, config in enumerate(sampled_configs, start=1):
        bm, bn, bk, tm, tn = map(int, config.split('_'))
        stats, success = benchmark_config(
            M, N, K, bm, bn, bk, tm, tn,
            backend=backend,
            kernel=kernel,
            num_runs=num_runs,
            seed=seed
        )

        if success and stats:
            median_time = stats["median"]
            mean_time = stats["mean"]
            ci_half = stats["ci_half"]
            if kernel == "stencil":
                throughput = (6 * M * N) / (median_time * 1e6)
            else:
                throughput = (2 * M * N * K) / (median_time * 1e6)
            results.append({
                "BM": bm, "BN": bn, "BK": bk, "TM": tm, "TN": tn,
                "time_ms": median_time,
                "mean_time_ms": mean_time,
                "ci_half": ci_half,
                "throughput_gflops": throughput
            })
            print(
                f"[{sample_idx:3d}/{num_samples}] BM={bm:3d}, BN={bn:3d}, BK={bk:2d}, TM={tm}, TN={tn}: "
                f"{median_time:8.4f} ms (median), mean={mean_time:.4f} ± {ci_half:.4f} ms ({throughput:6.2f} GFLOP/s)"
            )
        else:
            print(f"[{sample_idx:3d}/{num_samples}] BM={bm:3d}, BN={bn:3d}, BK={bk:2d}, TM={tm}, TN={tn}: FAILED")
    
    return results

def bayesian_search(M, N, K, backend="sycl", num_runs=3, num_trials=30, kernel="matmul", seed=DEFAULT_SEED):
    print(f"\n{'='*80}")
    print(f"Bayesian Optimization Auto-tuning {backend.upper()} {kernel.upper()} kernel (Optuna)")
    print(f"{'='*80}")
    print(f"Problem size: {M}x{N}x{K}")
    print(f"Trials to evaluate: {num_trials}\n")

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    results = []
    trial_count = [0]

    def objective(trial):
        valid_configs = all_valid_configs(kernel=kernel)
        if not valid_configs:
            raise RuntimeError("No valid configurations available for Bayesian search.")

        config = trial.suggest_categorical('CONFIG', valid_configs)
        bm, bn, bk, tm, tn = map(int, config.split('_'))

        trial_count[0] += 1
        idx = trial_count[0]

        stats, success = benchmark_config(M, N, K, bm, bn, bk, tm, tn, backend=backend, kernel=kernel, num_runs=num_runs, seed=seed)

        if success and stats and stats["median"] > 0:
            median_time = stats["median"]
            mean_time = stats["mean"]
            ci_half = stats["ci_half"]
            if kernel == "stencil":
                throughput = (6 * M * N) / (median_time * 1e6)
            else:
                throughput = (2 * M * N * K) / (median_time * 1e6)
            results.append({
                "BM": bm, "BN": bn, "BK": bk, "TM": tm, "TN": tn,
                "time_ms": median_time,
                "mean_time_ms": mean_time,
                "ci_half": ci_half,
                "throughput_gflops": throughput,
                "trial": idx
            })
            print(f"[{idx:3d}/{num_trials}] BM={bm:3d}, BN={bn:3d}, BK={bk:2d}, TM={tm}, TN={tn}: {median_time:8.4f} ms (median), mean={mean_time:.4f} ± {ci_half:.4f} ms ({throughput:6.2f} GFLOP/s)")
            return median_time
        else:
            print(f"[{idx:3d}/{num_trials}] BM={bm:3d}, BN={bn:3d}, BK={bk:2d}, TM={tm}, TN={tn}: FAILED")
            return float('inf')

    sampler = optuna.samplers.TPESampler(
        seed=seed,
        n_startup_trials=min(10, num_trials // 3),  
    )
    study = optuna.create_study(sampler=sampler, direction="minimize")

    # Warm-start: give Optuna the corners of the valid space so it doesn't waste
    # startup trials on obviously invalid regions
    space = search_spaces.get(kernel, search_spaces["matmul"])
    valid_configs = [
        (bm, bn, bk, tm, tn)
        for bm in space["BM"]
        for bn in space["BN"]
        for bk in space["BK"]
        for tm in space["TM"]
        for tn in space["TN"]
        if is_valid_config(bm, bn, bk, tm, tn, kernel=kernel)[0]
    ]
    print(f"Valid configurations in search space: {len(valid_configs)}/{len(space['BM'])*len(space['BN'])*len(space['BK'])*len(space['TM'])*len(space['TN'])}\n")

    if not valid_configs:
        return results

    study.optimize(objective, n_trials=num_trials, show_progress_bar=False)

    if study.best_value != float('inf'):
        print(f"\nBest trial: Trial #{study.best_trial.number + 1}")
        print(f"Best value: {study.best_value:.4f} ms")

    return results

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Auto-tune kernel block sizes")
    parser.add_argument("--backend", default="sycl", choices=["cuda", "sycl"], help="Backend to tune (default: sycl)")
    parser.add_argument("--kernel", default="matmul", choices=["matmul", "stencil"], help="Kernel to tune (default: matmul)")
    parser.add_argument("--runs", type=int, default=10, help="Number of runs per configuration (default: 10)")
    parser.add_argument("--strategy", default="random", choices=["random", "bayesian"], 
                       help="Search strategy (default: random)")
    parser.add_argument("--samples", type=int, default=20, help="Number of samples for random search (default: 20)")
    parser.add_argument("--trials", type=int, default=30, help="Number of trials for Bayesian optimization (default: 30)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for reproducibility")
    parser.add_argument("--output", default="autotune_results.json", help="Output file for results")
    args = parser.parse_args()

    if args.strategy == "random":
        results = random_search(
            M, N, K,   
            backend=args.backend,
            num_runs=args.runs,
            num_samples=args.samples,
            kernel=args.kernel,
            seed=args.seed
        )
    elif args.strategy == "bayesian":
        results = bayesian_search(
            M, N, K,   
            backend=args.backend,
            num_runs=args.runs,
            num_trials=args.trials,
            kernel=args.kernel,
            seed=args.seed
        )
    else:
        raise ValueError(f"Unsupported strategy: {args.strategy}")

    if results:
        results_sorted = sorted(results, key=lambda x: x["time_ms"])
        
        for rank, config in enumerate(results_sorted[:10], 1):
            print(f"{rank:<6} {config['BM']:<6} {config['BN']:<6} {config['BK']:<6} {config['TM']:<6} {config.get('TN', 1):<6} "
                  f"{config['time_ms']:<15.4f} {config['throughput_gflops']:<12.2f}")
        
        print("\n" + "="*80)
        best = results_sorted[0]
        print(f"\nBEST CONFIGURATION:")
        print(f"  BM={best['BM']}, BN={best['BN']}, BK={best['BK']}, TM={best['TM']}, TN={best.get('TN', 1)}")
        print(f"  Time: {best['time_ms']:.4f} ms")
        print(f"  Throughput: {best['throughput_gflops']:.2f} GFLOP/s")
        print(f"\nTo use this configuration for compilation:")
        print(f"  make clean && make BM={best['BM']} BN={best['BN']} BK={best['BK']} TM={best['TM']} TN={best.get('TN', 1)}")
        
        with open(args.output, 'w') as f:
            json.dump(results_sorted, f, indent=2)
        print(f"\nDetailed results saved to: {args.output}")
    else:
        print("\nNo valid configurations found!")

if __name__ == "__main__":
    main()
