#!/bin/bash
#SBATCH --job-name=cuda_make_bench
#SBATCH --partition=legacy
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --output=make_output_%j.log

module load apptainer/1.3.6
CONTAINER="$HOME/containers/cuda8.sif"

apptainer exec --nv $CONTAINER make clean
apptainer exec --nv $CONTAINER make

EXE="./dgemm"

if [ -f "$EXE" ]; then
    echo "--- Ejecutando Profiling de $EXE ---"
    apptainer exec --nv $CONTAINER nvprof $EXE
else
    echo "Error: El ejecutable $EXE no fue encontrado. Revisa el log de compilación."
    exit 1
fi
