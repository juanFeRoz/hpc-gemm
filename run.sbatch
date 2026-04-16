#!/bin/bash
#SBATCH --job-name=autotuning_bench
#SBATCH --partition=gpu_titan
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --output=bench_%j.log

module load adaptivecpp/nvidia

make clean
make

python3 -m venv ./venv 
source venv/bin/activate

python3 optimize.py
