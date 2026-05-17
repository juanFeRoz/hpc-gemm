#!/bin/bash
#SBATCH --job-name=gemm_autotune
#SBATCH --output=logs/autotune_%j.out
#SBATCH --error=logs/autotune_%j.err
#SBATCH --partition=GPU                
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00               

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hpc-gemm

REPO_DIR="$HOME/hpc-gemm"             
cd "$REPO_DIR" || { echo "ERROR: repo dir not found"; exit 1; }

# ---------------------------------------------------------------------------
# Output layout
# ---------------------------------------------------------------------------
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="$REPO_DIR/results_${TIMESTAMP}"
mkdir -p "$RESULTS_DIR" logs

echo "======================================================================"
echo "GEMM Autotuning Experiment"
echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $SLURMD_NODENAME"
echo "Results dir: $RESULTS_DIR"
echo "Started    : $(date)"
echo "======================================================================"

# ---------------------------------------------------------------------------
# Matrix sizes to benchmark (N means NxNxN square matmul)
# ---------------------------------------------------------------------------
SIZES=(512 1024 2048 4096)

# ---------------------------------------------------------------------------
# Run pipeline for each size
# ---------------------------------------------------------------------------
for SIZE in "${SIZES[@]}"; do
    SIZE_DIR="$RESULTS_DIR/size_${SIZE}"
    mkdir -p "$SIZE_DIR"

    echo ""
    echo "======================================================================"
    echo "Matrix size: ${SIZE}x${SIZE}x${SIZE}   $(date)"
    echo "======================================================================"

    python3 -u run_statistical_experiments.py \
        --M "$SIZE" \
        --N "$SIZE" \
        --K "$SIZE" \
        --num-runs 20 \
        --trials 40 \
        --output "$SIZE_DIR/convergence.json" \
        2>&1 | tee "$SIZE_DIR/run.log"

    EXIT_CODE=${PIPESTATUS[0]}

    # statistical_results.json is written to CWD by the script
    [ -f statistical_results.json ] && mv statistical_results.json "$SIZE_DIR/"
    [ -f autotune_results.json ]    && mv autotune_results.json    "$SIZE_DIR/"

    if [ $EXIT_CODE -ne 0 ]; then
        echo "WARNING: size ${SIZE} exited with code $EXIT_CODE — continuing to next size"
    else
        echo "Completed size ${SIZE}x${SIZE}   $(date)"
    fi
done

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "======================================================================"
echo "All sizes finished: $(date)"
echo "Results saved to  : $RESULTS_DIR"
echo ""
echo "Per-size layout:"
for SIZE in "${SIZES[@]}"; do
    echo "  $RESULTS_DIR/size_${SIZE}/"
    echo "    convergence.json       — best-so-far curves per run"
    echo "    statistical_results.json — Bayesian vs Random significance test"
    echo "    run.log                — full stdout/stderr for debugging"
done
echo "======================================================================"
