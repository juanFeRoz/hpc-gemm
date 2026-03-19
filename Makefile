NVCC      := nvcc
NVCCFLAGS := -O3 -Xcompiler -Wall -std=c++11

GPU_ARCH := 20

ARCH := -gencode arch=compute_$(GPU_ARCH),code=sm_$(GPU_ARCH)

TARGET  := dgemm
SRC     := main.cu
HEADERS := Kernels.cuh
OBJ     := main.o

all: $(TARGET)

$(TARGET): $(OBJ)
	$(NVCC) $(NVCCFLAGS) $(ARCH) $(OBJ) -o $(TARGET)

$(OBJ): $(SRC) $(HEADERS)
	$(NVCC) $(NVCCFLAGS) $(ARCH) -c $(SRC) -o $(OBJ)

clean:
	rm -f $(TARGET) $(OBJ) 

.PHONY: all clean
