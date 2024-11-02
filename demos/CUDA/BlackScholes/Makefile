# Paths and Compiler Settings
LLVM_PATH = /usr/lib/llvm-17
CLANG = $(LLVM_PATH)/bin/clang
CUDA_PATH ?= /usr/local/cuda-11.8
CLAD_PATH = $(CURDIR)/../../..
CLAD_PLUGIN = $(CLAD_PATH)/build/./lib/clad.so

# Compiler flags
CXXFLAGS = -std=c++17 -Xclang -add-plugin -Xclang clad
CXXFLAGS += -Xclang -load -Xclang $(CLAD_PLUGIN)
CXXFLAGS += -I$(CLAD_PATH)/include -I$(CURDIR)/helper -I$(CUDA_PATH)/include 

CUDA_FLAGS = --cuda-path=$(CUDA_PATH) --cuda-gpu-arch=sm_60

# Linker flags
LDFLAGS = -L$(CUDA_PATH)/lib64 -lcudart_static -ldl -lrt -pthread -lm -lstdc++

all: build

.SILENT: build run clean clobber BlackScholes BlackScholes.o BlackScholes_gold.o

build: BlackScholes

BlackScholes.o:BlackScholes.cu
	$(CLANG) $(CXXFLAGS) -o $@ -c $< $(CUDA_FLAGS)

BlackScholes_gold.o:BlackScholes_gold.cpp
	$(CLANG) $(CXXFLAGS) -o $@ -c $<

BlackScholes: BlackScholes.o BlackScholes_gold.o
	$(CLANG) $(CXXFLAGS) -o $@ BlackScholes.o BlackScholes_gold.o $(LDFLAGS)

run: build
	./BlackScholes

testrun: build

clean:
	rm -f BlackScholes BlackScholes.o BlackScholes_gold.o

clobber: clean