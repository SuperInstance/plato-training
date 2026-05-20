# PLATO Forge — CUDA compilation
# Target: RTX 4050 Laptop GPU (sm_89, Ada Lovelace)

NVCC = /usr/local/cuda-12.6/bin/nvcc
NVCC_FLAGS = -arch=sm_89 -O2 -Xcompiler -fPIC -shared
CC = gcc
CFLAGS = -O2 -Wall -fPIC

all: forge_kernels.o libplato_forge.so

forge_kernels.o: forge_kernels.cu plato_forge.h
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

libplato_forge.so: forge_kernels.o
	$(NVCC) -shared -o $@ $< -lcudart

clean:
	rm -f *.o *.so

.PHONY: all clean
