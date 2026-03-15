NVCC      := nvcc
NVCCFLAGS := -O3 -Xcompiler -Wall

GPU_ARCH := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -d '.')
ifeq ($(GPU_ARCH),)
    GPU_ARCH := 80
endif
ARCH := -gencode arch=compute_$(GPU_ARCH),code=sm_$(GPU_ARCH)

TARGET  := sgemm_naive
SRC     := main.cu
HEADERS := kernels.cuh
OBJ     := main.o

all: $(TARGET)

$(TARGET): $(OBJ)
	$(NVCC) $(NVCCFLAGS) $(ARCH) $(OBJ) -o $(TARGET)

$(OBJ): $(SRC) $(HEADERS)
	$(NVCC) $(NVCCFLAGS) $(ARCH) -c $(SRC) -o $(OBJ)

clean:
	rm -f $(TARGET) $(OBJ) run_gemm.sh

.PHONY: all clean
