#!/usr/bin/env python3
import subprocess
import json
import sys
import re
import os
import random
import optuna
from pathlib import Path

M, N, K = 2048, 2048, 2048

search_space = {
    "BM": [32, 64, 128],
    "BN": [32, 64, 128],
    "BK": [8, 16, 32],
    "TM": [1, 2, 4]
}

def compile_config(bm, bn, bk, tm, backend="sycl"):
    try:
        env = os.environ.copy()
        env['BM'] = str(bm)
        env['BN'] = str(bn)
        env['BK'] = str(bk)
        env['TM'] = str(tm)
        
        subprocess.run(["rm", "-f", f"kernel_matmul_{backend}.o", f"kernel_matmul_{backend}.so"], capture_output=True)
        
        result = subprocess.run(
            ["make", f"kernel_matmul_{backend}.so", f"BM={bm}", f"BN={bn}", f"BK={bk}", f"TM={tm}" ],
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

def benchmark_config(bm, bn, bk, tm, backend="sycl", num_runs=3):
    try:
        if not compile_config(bm, bn, bk, tm, backend=backend):
            return None, False
        
        script_name = f"_benchmark_temp_{bm}_{bn}_{bk}_{tm}.py"
        benchmark_code = f"""
import sys
sys.path.insert(0, '.')
from wrapper import run

M, N, K = {M}, {N}, {K}
times = []
for i in range({num_runs}):
    elapsed = run(M, N, K, {bm}, {bn}, {bk}, {tm}, backend="{backend}", seed=42)
    times.append(elapsed)

avg_time = sum(times[1:]) / len(times[1:]) if {num_runs} > 1 else times[0]
print(f"{{avg_time:.4f}}")
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
            avg_time = float(result.stdout.strip().split('\n')[-1])
            return avg_time, True
        except:
            return None, False
            
    except subprocess.TimeoutExpired:
        return None, False
    except Exception as e:
        return None, False

def is_valid_config(bm, bn, bk, tm):
    threads_per_block = (bm // tm) * bn
    if threads_per_block > 1024:
        return False, f"threads={threads_per_block} > 1024"
    
    shared_memory = (bm * bk + bk * bn) * 4
    if shared_memory > 96 * 1024:
        return False, f"shared_mem={shared_memory/1024:.1f}KB > 96KB"
    
    return True, None

def random_search(backend="cuda", num_runs=3, num_samples=20):
    results = []
    
    print(f"\n{'='*80}")
    print(f"Random Search Auto-tuning {backend.upper()} kernel")
    print(f"{'='*80}")
    print(f"Problem size: {M}x{N}x{K}")
    print(f"Samples to evaluate: {num_samples}\n")
    
    evaluated = set()
    sample_idx = 0
    
    while sample_idx < num_samples:
        bm = random.choice(search_space["BM"])
        bn = random.choice(search_space["BN"])
        bk = random.choice(search_space["BK"])
        tm = random.choice(search_space["TM"])
        
        config_tuple = (bm, bn, bk, tm)
        
        if config_tuple in evaluated:
            continue
        
        evaluated.add(config_tuple)
        sample_idx += 1
        
        is_valid, reason = is_valid_config(bm, bn, bk, tm)
        if not is_valid:
            print(f"[{sample_idx:3d}/{num_samples}] BM={bm:3d}, BN={bn:3d}, BK={bk:2d}, TM={tm}: SKIPPED ({reason})")
            continue
        
        avg_time, success = benchmark_config(bm, bn, bk, tm, backend=backend, num_runs=num_runs)
        
        if success and avg_time:
            throughput = (2 * M * N * K) / (avg_time * 1e6)
            results.append({
                "BM": bm, "BN": bn, "BK": bk, "TM": tm,
                "time_ms": avg_time,
                "throughput_gflops": throughput
            })
            print(f"[{sample_idx:3d}/{num_samples}] BM={bm:3d}, BN={bn:3d}, BK={bk:2d}, TM={tm}: {avg_time:8.4f} ms ({throughput:6.2f} GFLOP/s)")
        else:
            print(f"[{sample_idx:3d}/{num_samples}] BM={bm:3d}, BN={bn:3d}, BK={bk:2d}, TM={tm}: FAILED")
    
    return results

def bayesian_search(backend="sycl", num_runs=3, num_trials=30):
    print(f"\n{'='*80}")
    print(f"Bayesian Optimization Auto-tuning {backend.upper()} kernel (Optuna)")
    print(f"{'='*80}")
    print(f"Problem size: {M}x{N}x{K}")
    print(f"Trials to evaluate: {num_trials}\n")

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    results = []
    trial_count = [0]
    evaluated = {} 

    def objective(trial):
        bm = trial.suggest_categorical('BM', search_space["BM"])
        bn = trial.suggest_categorical('BN', search_space["BN"])
        bk = trial.suggest_categorical('BK', search_space["BK"])
        tm = trial.suggest_categorical('TM', search_space["TM"])

        trial_count[0] += 1
        idx = trial_count[0]
        config = (bm, bn, bk, tm)

        if config in evaluated:
            cached = evaluated[config]
            label = f"{cached:.4f} ms (cached)" if cached != float('inf') else "SKIPPED (cached)"
            print(f"[{idx:3d}/{num_trials}] BM={bm:3d}, BN={bn:3d}, BK={bk:2d}, TM={tm}: {label}")
            return cached

        is_valid, reason = is_valid_config(bm, bn, bk, tm)
        if not is_valid:
            print(f"[{idx:3d}/{num_trials}] BM={bm:3d}, BN={bn:3d}, BK={bk:2d}, TM={tm}: SKIPPED ({reason})")
            evaluated[config] = float('inf')
            return float('inf')

        avg_time, success = benchmark_config(bm, bn, bk, tm, backend=backend, num_runs=num_runs)

        if success and avg_time and avg_time > 0:
            throughput = (2 * M * N * K) / (avg_time * 1e6)
            evaluated[config] = avg_time
            results.append({
                "BM": bm, "BN": bn, "BK": bk, "TM": tm,
                "time_ms": avg_time,
                "throughput_gflops": throughput,
                "trial": idx
            })
            print(f"[{idx:3d}/{num_trials}] BM={bm:3d}, BN={bn:3d}, BK={bk:2d}, TM={tm}: {avg_time:8.4f} ms ({throughput:6.2f} GFLOP/s)")
            return avg_time
        else:
            evaluated[config] = float('inf')
            print(f"[{idx:3d}/{num_trials}] BM={bm:3d}, BN={bn:3d}, BK={bk:2d}, TM={tm}: FAILED")
            return float('inf')

    sampler = optuna.samplers.TPESampler(
        seed=42,
        n_startup_trials=min(10, num_trials // 3),  # random phase scales with budget
        constant_liar=True,     # discourages re-suggesting 
        multivariate=True,      # models parameter interactions (BM+TM co-vary)
    )
    study = optuna.create_study(sampler=sampler, direction="minimize")

    # Warm-start: give Optuna the corners of the valid space so it doesn't waste
    # startup trials on obviously invalid regions
    valid_configs = [
        (bm, bn, bk, tm)
        for bm in search_space["BM"]
        for bn in search_space["BN"]
        for bk in search_space["BK"]
        for tm in search_space["TM"]
        if is_valid_config(bm, bn, bk, tm)[0]
    ]
    print(f"Valid configurations in search space: {len(valid_configs)}/{len(search_space['BM'])*len(search_space['BN'])*len(search_space['BK'])*len(search_space['TM'])}\n")

    study.optimize(objective, n_trials=num_trials, show_progress_bar=False)

    if study.best_value != float('inf'):
        print(f"\nBest trial: Trial #{study.best_trial.number + 1}")
        print(f"Best value: {study.best_value:.4f} ms")

    return results

def grid_search(backend="sycl", num_runs=3):
    results = []
    total_configs = 1
    for param, values in search_space.items():
        total_configs *= len(values)
    
    config_idx = 0
    print(f"\n{'='*80}")
    print(f"Grid Search Auto-tuning {backend.upper()} kernel")
    print(f"{'='*80}")
    print(f"Problem size: {M}x{N}x{K}")
    print(f"Total configurations to test: {total_configs}\n")
    
    for bm in search_space["BM"]:
        for bn in search_space["BN"]:
            for bk in search_space["BK"]:
                for tm in search_space["TM"]:
                    config_idx += 1
                    
                    is_valid, reason = is_valid_config(bm, bn, bk, tm)
                    if not is_valid:
                        print(f"[{config_idx:3d}/{total_configs}] BM={bm:3d}, BN={bn:3d}, BK={bk:2d}, TM={tm}: SKIPPED ({reason})")
                        continue
                    
                    avg_time, success = benchmark_config(bm, bn, bk, tm, backend=backend, num_runs=num_runs)
                    
                    if success and avg_time:
                        throughput = (2 * M * N * K) / (avg_time * 1e6)
                        results.append({
                            "BM": bm, "BN": bn, "BK": bk, "TM": tm,
                            "time_ms": avg_time,
                            "throughput_gflops": throughput
                        })
                        print(f"[{config_idx:3d}/{total_configs}] BM={bm:3d}, BN={bn:3d}, BK={bk:2d}, TM={tm}: {avg_time:8.4f} ms ({throughput:6.2f} GFLOP/s)")
                    else:
                        print(f"[{config_idx:3d}/{total_configs}] BM={bm:3d}, BN={bn:3d}, BK={bk:2d}, TM={tm}: FAILED")
    
    return results

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Auto-tune matrix multiplication kernel block sizes")
    parser.add_argument("--backend", default="sycl", choices=["cuda", "sycl"], help="Backend to tune (default: sycl)")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs per configuration (default: 3)")
    parser.add_argument("--strategy", default="random", choices=["random", "bayesian", "grid"], 
                       help="Search strategy (default: random)")
    parser.add_argument("--samples", type=int, default=20, help="Number of samples for random search (default: 20)")
    parser.add_argument("--trials", type=int, default=30, help="Number of trials for Bayesian optimization (default: 30)")
    parser.add_argument("--output", default="autotune_results.json", help="Output file for results")
    
    args = parser.parse_args()
    
    if args.strategy == "random":
        results = random_search(backend=args.backend, num_runs=args.runs, num_samples=args.samples)
    elif args.strategy == "bayesian":
        results = bayesian_search(backend=args.backend, num_runs=args.runs, num_trials=args.trials)
    else:  # grid
        results = grid_search(backend=args.backend, num_runs=args.runs)
    
    if results:
        results_sorted = sorted(results, key=lambda x: x["time_ms"])
        
        print("\n" + "="*80)
        print("TOP 10 CONFIGURATIONS (sorted by performance)")
        print("="*80)
        print(f"{'Rank':<6} {'BM':<6} {'BN':<6} {'BK':<6} {'TM':<6} {'Time (ms)':<15} {'GFLOP/s':<12}")
        print("-"*80)
        
        for rank, config in enumerate(results_sorted[:10], 1):
            print(f"{rank:<6} {config['BM']:<6} {config['BN']:<6} {config['BK']:<6} {config['TM']:<6} "
                  f"{config['time_ms']:<15.4f} {config['throughput_gflops']:<12.2f}")
        
        print("\n" + "="*80)
        best = results_sorted[0]
        print(f"\nBEST CONFIGURATION:")
        print(f"  BM={best['BM']}, BN={best['BN']}, BK={best['BK']}, TM={best['TM']}")
        print(f"  Time: {best['time_ms']:.4f} ms")
        print(f"  Throughput: {best['throughput_gflops']:.2f} GFLOP/s")
        print(f"\nTo use this configuration for compilation:")
        print(f"  make clean && make BM={best['BM']} BN={best['BN']} BK={best['BK']} TM={best['TM']}")
        
        with open(args.output, 'w') as f:
            json.dump(results_sorted, f, indent=2)
        print(f"\nDetailed results saved to: {args.output}")
    else:
        print("\nNo valid configurations found!")

if __name__ == "__main__":
    main()