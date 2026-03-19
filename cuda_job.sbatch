#!/bin/bash
#SBATCH --job-name=cuda_fermi_bench    # Nombre del trabajo
#SBATCH --partition=legacy             # Partición obligatoria para GPUs antiguas
#SBATCH --nodes=1                      # Usar un solo nodo
#SBATCH --ntasks-per-node=1            # Una tarea por nodo
#SBATCH --gres=gpu:1                   # Solicitar 1 GPU
#SBATCH --output=resultado_%j.log      # Archivo de salida (%j es el ID del trabajo)
#SBATCH --error=error_%j.err           # Archivo de errores

module load apptainer

CONTAINER="$HOME/containers/cuda8.sif"
SOURCE="hello.cu"
EXE="hello"

echo "Iniciando trabajo en el nodo: $SLURM_NODELIST"
echo "Fecha: $(date)"

echo "--- Compilando para Fermi (sm_20) ---"
apptainer exec --nv $CONTAINER nvcc -arch=sm_20 $SOURCE -o $EXE

if [ $? -ne 0 ]; then
    echo "Error en la compilación. Abortando."
    exit 1
fi

echo "--- Ejecutando Benchmark con nvprof ---"
apptainer exec --nv $CONTAINER nvprof ./$EXE

echo "Trabajo finalizado con éxito."
