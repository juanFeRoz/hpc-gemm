import optuna
from wrapper import run

def objective(trial):
    M, N, K = 512, 512, 512
    BM = trial.suggest_categorical("BM", [8, 16, 32])
    BN = trial.suggest_categorical("BN", [8, 16, 32])
    BK = trial.suggest_categorical("BK", [8, 16, 32])
    TM = trial.suggest_categorical("TM", [1, 2, 4, 8])

    if BM % TM != 0:
        return float("inf")
    if BM != BK * TM:   
        return float("inf")
    if BN != BK * TM:  
        return float("inf")
    if BK > K:
        return float("inf")

    return run(M, N, K, BM, BN, BK, TM, backend="cuda")

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)
print("Best params:", study.best_params)
print("Best time (ms):", study.best_value)
