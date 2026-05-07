ACPP_FLAGS := -O3
HIP_FLAGS := -O3

# Allow template parameters to be set via environment or command line
BM ?= 32
BN ?= 32
BK ?= 32
TM ?= 1
TN ?= 1

# Append template parameter defines to compiler flags
ACPP_FLAGS += -D_BM=$(BM) -D_BN=$(BN) -D_BK=$(BK) -D_TM=$(TM) -D_TN=$(TN)
HIP_FLAGS += -D_BM=$(BM) -D_BN=$(BN) -D_BK=$(BK) -D_TM=$(TM) -D_TN=$(TN)

# Default libraries built by `make all`
all: kernel_matmul_sycl.so kernel_matmul_cuda.so kernel_stencil_sycl.so kernel_stencil_cuda.so

kernel_%_sycl.so: kernel_%_sycl.o
	acpp $(ACPP_FLAGS) -shared -o $@ $^

kernel_%_sycl.o: kernel_%_sycl.cpp
	acpp $(ACPP_FLAGS) -fPIC -c $< -o $@

kernel_%_cuda.so: kernel_%_cuda.o
	hipcc $(HIP_FLAGS) -shared -o $@ $^

kernel_%_cuda.o: kernel_%_cuda.hip
	hipcc $(HIP_FLAGS) -Xcompiler -fPIC -c $< -o $@

clean:
	rm -f kernel_*_sycl.o kernel_*_sycl.so kernel_*_cuda.o kernel_*_cuda.so
