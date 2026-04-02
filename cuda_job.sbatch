#!/bin/bash
#SBATCH --job-name=cuda_bench
#SBATCH --partition=legacy
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --output=bench_%j.log

module load apptainer/1.3.6
CONTAINER="$HOME/containers/cuda8.sif"

apptainer exec --nv $CONTAINER make clean
apptainer exec --nv $CONTAINER make

EXE="./dgemm"

if [ ! -f "$EXE" ]; then
    echo "Error: $EXE not found."
    exit 1
fi

SIZES=(512 1024 2048 4096)
TILES=(8 16 32)

for n in "${SIZES[@]}"; do
    for t in "${TILES[@]}"; do
        echo "==========================================="
        echo "PROFILING: N=$n | TILE=$t"
        echo "==========================================="
        
        apptainer exec --nv $CONTAINER nvprof $EXE $n $t
        
        echo -e "\n"
    done
done
