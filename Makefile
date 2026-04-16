kernel_matmul.so: kernel_matmul.o
	acpp -shared -o kernel_matmul.so kernel_matmul.o

kernel_matmul.o: kernel_matmul.cpp
	acpp -O3 -fPIC -c kernel_matmul.cpp -o kernel_matmul.o 

clean:
	rm -f kernel_matmul.o kernel_matmul.so
