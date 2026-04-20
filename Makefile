SYCL_KERNEL := kernel_matmul_sycl.cpp
CUDA_KERNEL := kernel_matmul_cuda.cu

SYCL_OBJ := kernel_matmul_sycl.o
SYCL_LIB := kernel_matmul_sycl.so

CUDA_OBJ := kernel_matmul_cuda.o
CUDA_LIB := kernel_matmul_cuda.so

ACPP_FLAGS := -O3 -fPIC

all: $(SYCL_LIB) $(CUDA_LIB)

# Link SYCL
$(SYCL_LIB): $(SYCL_OBJ)
	acpp $(ACPP_FLAGS) -shared -o kernel_matmul_sycl.so kernel_matmul_sycl.o

# Compile SYCL
$(SYCL_OBJ): $(SYCL_KERNEL)
	acpp $(ACPP_FLAGS) -c kernel_matmul_sycl.cpp -o kernel_matmul_sycl.o

# Link CUDA
$(CUDA_LIB): $(CUDA_OBJ)
	nvcc -ccbin /usr/bin/gcc -shared -o kernel_matmul_cuda.so kernel_matmul_cuda.o

# Compile CUDA
$(CUDA_OBJ): $(CUDA_KERNEL)
	nvcc -O3 -ccbin /usr/bin/gcc -Xcompiler -fPIC -c kernel_matmul_cuda.cu -o kernel_matmul_cuda.o

clean:
	rm -f $(SYCL_OBJ) $(SYCL_LIB) $(CUDA_OBJ) $(CUDA_LIB)
