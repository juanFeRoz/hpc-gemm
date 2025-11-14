# Optimización de Kernel GEMM Naive con HIP y Optuna

Este repositorio contiene un proyecto de **optimización de parámetros de kernel** que utiliza la librería **Optuna** (Bayesian Optimization) para encontrar el tamaño de bloque óptimo (`BLOCK_X`, `BLOCK_Y`) para un kernel simple (naive) de Multiplicación General de Matrices (**GEMM**) escrito en **HIP C++**.

El objetivo es demostrar cómo el **diseño de la cuadrícula de hilos** impacta drásticamente el rendimiento, incluso en un kernel no optimizado.

## Objetivo de la Optimización

El proyecto busca el par de dimensiones de bloque `BLOCK_X`, `BLOCK_Y` que minimice el tiempo de ejecución del kernel `simpleGEMM` para una matriz fija de $64 \times 64$.

### Ecuación

El kernel implementa la multiplicación matricial estándar:

$$C = A \times B$$

---

## Estructura del Repositorio

```

.
├── hip_gemm_wrapper.cpp  # Kernel HIP C++ y función host con Pybind11.
├── setup.py              # Script para compilar el módulo C++ a Python.
├── optimize.py           # Script principal de Optuna para la optimización.

````

---

## Requisitos

Asegúrate de tener instalado el **HIP SDK** (parte de ROCm) y los paquetes de Python necesarios:

1. **ROCm (con `hipcc`):** Entorno de compilación configurado.
2. **Python 3.x**
3. **Librerías de Python:**
    ```bash
    pip install optuna pybind11 plotly setuptools
    ```

---

## Guía de Uso y Compilación

Sigue estos pasos en tu entorno Linux con soporte ROCm activo.

### 1. Compilar el Módulo C++

Utiliza `setup.py` para compilar el código HIP C++ en un módulo de Python (`hip_gemm_module.so`).

**Solución de errores de compilación (importante):**  
Para evitar conflictos por banderas de seguridad del compilador (`-fcf-protection`), limpia las variables de entorno `CFLAGS` y `CXXFLAGS` antes de compilar:

```bash
# Limpiar banderas de compilación problemáticas
export CFLAGS=""
export CXXFLAGS=""

# Ejecutar la compilación con setup.py
python3 setup.py build_ext --inplace
````

Esto generará el archivo `hip_gemm_module.<suffix>.so` en el directorio actual.

### 2. Ejecutar la Optimización con Optuna

Ejecuta el script principal `optimize.py`. Optuna utilizará el **muestreador TPE (Bayesian Optimization)** para explorar el espacio de búsqueda.

```bash
python3 optimize.py
```

El script imprimirá el mejor tiempo encontrado y los parámetros correspondientes.

---

## Resultados Típicos e Interpretación

Después de la ejecución, Optuna mostrará la combinación óptima de parámetros.

Para una interpretación visual detallada, puedes generar el gráfico de Coordenadas Paralelas (requiere el objeto `study` generado por `optimize.py`):

```python
# Dentro de un script o entorno interactivo:
import optuna.visualization as vis
# Suponiendo que el objeto 'study' está cargado o definido.
fig = vis.plot_parallel_coordinate(study, params=['BLOCK_X', 'BLOCK_Y'])
fig.show()
```
