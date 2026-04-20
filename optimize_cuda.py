import optuna
from wrapper import run

def objective(trial):
    configs = [
        {"BM": 8,  "BN": 8,  "BK": 8,  "TM": 1},
        {"BM": 16, "BN": 16, "BK": 8,  "TM": 2},
        {"BM": 32, "BN": 32, "BK": 8,  "TM": 4},
        {"BM": 16, "BN": 16, "BK": 16, "TM": 1},
        {"BM": 32, "BN": 32, "BK": 16, "TM": 2},
        {"BM": 32, "BN": 32, "BK": 32, "TM": 1},
    ]
    
    # Let Optuna pick one of the valid indices
    idx = trial.suggest_int("config_idx", 0, len(configs) - 1)
    c = configs[idx]
    
    M, N, K = 512, 512, 512
    res = run(M, N, K, c["BM"], c["BN"], c["BK"], c["TM"], backend="cuda")
    
    if res < 0:
        return float("inf")
    return res

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)
print("Best params:", study.best_params)
print("Best time (ms):", study.best_value)
