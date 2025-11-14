# setup.py
import os
from setuptools import setup, Extension
import pybind11

# Define the ROCm include directory (Standard location)
ROCM_PATH = '/opt/rocm'
HIPBLAS_INCLUDE = os.path.join(ROCM_PATH, 'include', 'hipblas')

# Define the compiler to use. We must use hipcc for HIP code.
os.environ['CC'] = 'hipcc'
os.environ['CXX'] = 'hipcc'

# Determine the extension suffix (e.g., .so or .pyd)
EXT_SUFFIX = os.getenv('PYTHON_EXT_SUFFIX')

# Define the C++ extension module
ext_modules = [
    Extension(
        'hip_gemm_module',  
        ['hip_gemm_wrapper.cpp'],  
        
        # Compiler settings
        language='c++',
        include_dirs=[
            # Standard pybind11 includes
            pybind11.get_include(),
            pybind11.get_include(True),
            HIPBLAS_INCLUDE,
        ],
        extra_compile_args=[
            '-O3', 
            '-std=c++17', 
            '-fPIC',
        ],

        extra_link_args=[
            '-lhipblas',
            
        ]
    ),
]

setup(
    name='hip_gemm_module',
    version='1.0',
    ext_modules=ext_modules,
    # This ensures setuptools can handle C++ extensions correctly
    zip_safe=False, 
)