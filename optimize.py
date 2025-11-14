import optuna
from optuna.samplers import TPESampler 
import optuna.visualization as vis
import hip_gemm_module 
import sys

def objective(trial: optuna.trial.Trial):
    """
    Optimiza el Block Size (X, Y) para el simpleGEMM kernel.
    """
    
    # Rango de optimización: Potencias de 2 son estándar for block dimensions
    block_dims = [4, 8, 16, 32] 
    
    # 1. Sugerir variables independientes para las dimensiones del bloque
    block_x = trial.suggest_categorical("BLOCK_X", block_dims)
    block_y = trial.suggest_categorical("BLOCK_Y", block_dims)
    
    # Restricción: Total threads per block must not exceed 1024 (GPU limit)
    if (block_x * block_y) > 1024:
        raise optuna.exceptions.TrialPruned(f"Block size too large: {block_x}x{block_y} > 1024")
    
    print(f"  Trial: Block Size={block_x}x{block_y}")

    # 2. Llamar a la función C++ con los dos parámetros (Block X, Block Y)
    try:
        kernel_time_ms = hip_gemm_module.run_gemm_trial(block_x, block_y)
        return kernel_time_ms

    except Exception as e:
        print(f"  Trial failed: {e}", file=sys.stderr)
        raise optuna.exceptions.TrialPruned()


if __name__ == "__main__":
    sampler = TPESampler(seed=42)
    
    study = optuna.create_study(
        direction="minimize", 
        sampler=sampler
    )
    
    print("Starting Basic GEMM Block Size Optimization (Block X & Y)...")
    
    # Run a reasonable number of trials to find the best configuration
    study.optimize(objective, n_trials=30) 

    # 3. Reportar Resultados
    print("\n--- Optimization Complete ---")
    print(f"Best Time: {study.best_trial.value:.3f} ms")
    print("Best Parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    fig = vis.plot_parallel_coordinate(study, params=['BLOCK_X', 'BLOCK_Y'])
    fig.show()