# Optimización de Kernel GEMM Naive con HIP y Optuna

Este repositorio contiene un proyecto de **Optimización de Parámetros de Kernel** que utiliza la librería **Optuna** (Bayesian Optimization) para encontrar el tamaño de bloque óptimo (`BLOCK_X`, `BLOCK_Y`) para un kernel de Multiplicación General de Matrices (**GEMM**) simple (naive) escrito en **HIP C++**.

El objetivo es demostrar cómo el **diseño de la cuadrícula de hilos** impacta drásticamente el rendimiento, incluso en un kernel no optimizado.

## Objetivo de la Optimización

El proyecto busca el par de dimensiones de bloque $(\text{BLOCK\_X}, \text{BLOCK\_Y})$ que minimice el tiempo de ejecución del kernel `simpleGEMM` para una matriz fija de $64 \times 64$.

### Ecuación

El kernel implementa la multiplicación matricial estándar:

$$C = A \times B$$

-----

## Estructura del Repositorio

```
.
├── hip_gemm_wrapper.cpp  # Kernel HIP C++ y función host con Pybind11.
├── setup.py              # Script para compilar el módulo C++ a Python.
├── optimize.py           # Script principal de Optuna para la optimización.
```

-----

## Requisitos

Asegúrate de tener instalado el **HIP SDK** (parte de ROCm) y los paquetes Python necesarios:

1.  **ROCm (con `hipcc`):** Entorno de compilación configurado.
2.  **Python 3.x**
3.  **Librerías Python:**
    ```bash
    pip install optuna pybind11 plotly setuptools
    ```

-----

## Guía de Uso y Compilación

Sigue estos pasos en tu entorno Linux con soporte ROCm activo.

### 1\. Compilar el Módulo C++

Utiliza `setup.py` para compilar el código HIP C++ en un módulo Python (`hip_gemm_module.so`).

** Solución de Errores de Compilación (Importante):**
Para evitar conflictos de banderas de seguridad del compilador (`-fcf-protection`), limpia las variables de entorno de `CFLAGS`/`CXXFLAGS` antes de compilar:

```bash
# Limpiar banderas de compilación problemáticas
export CFLAGS=""
export CXXFLAGS=""

# Ejecutar la compilación con setup.py
python3 setup.py build_ext --inplace
```

Esto generará el archivo `hip_gemm_module.<suffix>.so` en el directorio actual.

### 2\. Ejecutar la Optimización de Optuna

Ejecuta el script principal `optimize.py`. Optuna utilizará el **Muestreador TPE (Bayesian Optimization)** para explorar el espacio de búsqueda.

```bash
python3 optimize.py
```

El script imprimirá el mejor tiempo encontrado y los parámetros correspondientes.

-----

## Resultados Típicos e Interpretación

Después de la ejecución, Optuna mostrará la combinación óptima de parámetros.

**Hallazgos Clave (Basado en el Plot adjunto):**

  * **Óptimo Encontrado:** El mejor rendimiento se logra con una configuración de bloque **asimétrica**, resultando en $\text{BLOCK\_X} = 4$ y $\text{BLOCK\_Y} = 32$ (o viceversa).
  * **Threads Totales:** La configuración óptima es **128 threads/bloque**.
  * **Peor Rendimiento:** Configuraciones grandes como $32 \times 32$ (1024 threads/bloque) fueron las más lentas, demostrando la ineficiencia de usar demasiados hilos sin optimización de memoria compartida (LDS).

Para una interpretación visual detallada, puedes generar el gráfico de Coordenadas Paralelas (requiere el objeto `study` generado por `optimize.py`):

```python
# Dentro de un script o entorno interactivo:
import optuna.visualization as vis
# Suponiendo que el objeto 'study' está cargado o definido.
fig = vis.plot_parallel_coordinate(study, params=['BLOCK_X', 'BLOCK_Y'])
fig.show()
```
