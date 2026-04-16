import ctypes

lib = ctypes.CDLL("./kernel_matmul.so")

lib.run_kernel.argtypes = [
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int
]
lib.run_kernel.restype = ctypes.c_float

def run(M, N, K, BM, BN, BK, TM):
    return lib.run_kernel(M, N, K, BM, BN, BK, TM)
