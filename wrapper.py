import ctypes

_ARGTYPES = [
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_uint
]
_RESTYPE = ctypes.c_float

def _load(path):
    lib = ctypes.CDLL(path)
    lib.run_kernel.argtypes = _ARGTYPES
    lib.run_kernel.restype = _RESTYPE
    return lib

_libs = {
    "sycl": _load("./kernel_matmul_sycl.so"),
    "cuda": _load("./kernel_matmul_cuda.so"),
}

def run(M, N, K, BM, BN, BK, TM, backend="sycl", seed="42"):
    if backend not in _libs:
        raise ValueError(f"Unknown backend '{backend}'. Choose from: {list(_libs)}")
    return _libs[backend].run_kernel(M, N, K, BM, BN, BK, TM, seed)
