SYCL_KERNEL := kernel_matmul_sycl.cpp
HIP_KERNEL := kernel_matmul_cuda.hip

SYCL_OBJ := kernel_matmul_sycl.o
SYCL_LIB := kernel_matmul_sycl.so

HIP_OBJ := kernel_matmul_cuda.o
HIP_LIB := kernel_matmul_cuda.so

ACPP_FLAGS := -O3 
HIP_FLAGS := -O3

# Allow template parameters to be set via environment or command line
BM ?= 32
BN ?= 32
BK ?= 32
TM ?= 1

# Append template parameter defines to compiler flags
ACPP_FLAGS += -D_BM=$(BM) -D_BN=$(BN) -D_BK=$(BK) -D_TM=$(TM)
HIP_FLAGS += -D_BM=$(BM) -D_BN=$(BN) -D_BK=$(BK) -D_TM=$(TM)

all: $(SYCL_LIB) $(HIP_LIB)

$(SYCL_LIB): $(SYCL_OBJ)
	acpp $(ACPP_FLAGS) -shared -o $(SYCL_LIB) $(SYCL_OBJ)

$(SYCL_OBJ): $(SYCL_KERNEL)
	acpp $(ACPP_FLAGS) -fPIC -c $(SYCL_KERNEL) -o $(SYCL_OBJ)

$(HIP_LIB): $(HIP_OBJ)
	hipcc $(HIP_FLAGS) -shared -o $(HIP_LIB) $(HIP_OBJ)

$(HIP_OBJ): $(HIP_KERNEL)
	hipcc $(HIP_FLAGS) -Xcompiler -fPIC -c $(HIP_KERNEL) -o $(HIP_OBJ)

clean:
	rm -f $(SYCL_OBJ) $(SYCL_LIB) $(HIP_OBJ) $(HIP_LIB)