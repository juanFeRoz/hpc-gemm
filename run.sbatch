#!/bin/bash
#SBATCH --job-name=autotuning_bench
#SBATCH --partition=gpu_titan
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=bench_%j.log
#SBATCH --error=bench_%j.err

set -euo pipefail

module load adaptivecpp/nvidia
module load cuda

# Build
make clean
make

# Python env - create only if it doesn't exist yet
if [ ! -d "./venv" ]; then
    python3 -m venv ./venv
fi
source venv/bin/activate
pip install -q optuna

python3 optimize_sycl.py
python3 optimize_cuda.py
