SYCL_KERNEL := kernel_matmul_sycl.cpp
CUDA_KERNEL := kernel_matmul_cuda.cu

SYCL_OBJ := $(SYCL_KERNEL:.cpp=.o)
SYCL_LIB := $(SYCL_KERNEL:.cpp=.so)

CUDA_OBJ := $(CUDA_KERNEL:.cu=.o)
CUDA_LIB := $(CUDA_KERNEL:.cu=.so)

ACPP_FLAGS := -O3 -fPIC

all: $(SYCL_LIB) $(CUDA_LIB)

$(SYCL_LIB): $(SYCL_OBJ)
	acpp $(ACPP_FLAGS) -shared -o $@ $

$(SYCL_OBJ): $(SYCL_KERNEL)
	acpp $(ACPP_FLAGS) -c $< -o $@

$(CUDA_LIB): $(CUDA_OBJ)
	nvcc -shared -o $@ $

$(CUDA_OBJ): $(CUDA_KERNEL)
	nvcc -O3 -Xcompiler -fPIC -c $< -o $@

clean:
	rm -f $(SYCL_OBJ) $(SYCL_LIB) $(CUDA_OBJ) $(CUDA_LIB)
