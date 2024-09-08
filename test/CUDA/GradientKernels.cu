// RUN: %cladclang_cuda -I%S/../../include  %s -fsyntax-only \
// RUN: %cudasmlevel --cuda-path=%cudapath  -Xclang -verify 2>&1 | %filecheck %s

// RUN: %cladclang_cuda -I%S/../../include %s -xc++ %cudasmlevel \
// RUN: --cuda-path=%cudapath -L/usr/local/cuda/lib64 -lcudart_static \
// RUN: -L%cudapath/lib64/stubs \
// RUN: -ldl -lrt -pthread -lm -lstdc++ -lcuda -lnvrtc

// REQUIRES: cuda-runtime

// expected-no-diagnostics

// XFAIL: clang-15

#include "clad/Differentiator/Differentiator.h"

__global__ void kernel(int *a) {
  *a *= *a;
}

// CHECK:    void kernel_grad(int *a, int *_d_a) {
//CHECK-NEXT:    int _t0 = *a;
//CHECK-NEXT:    *a *= *a;
//CHECK-NEXT:    {
//CHECK-NEXT:        *a = _t0;
//CHECK-NEXT:        int _r_d0 = *_d_a;
//CHECK-NEXT:        *_d_a = 0;
//CHECK-NEXT:        *_d_a += _r_d0 * *a;
//CHECK-NEXT:       *_d_a += *a * _r_d0;
//CHECK-NEXT:    }
//CHECK-NEXT: }

void fake_kernel(int *a) {
  *a *= *a;
}

__global__ void add_kernel(int *out, int *in) {
  int index = threadIdx.x;
  out[index] += in[index];
}

// CHECK:    void add_kernel_grad(int *out, int *in, int *_d_out, int *_d_in) {
//CHECK-NEXT:    int _d_index = 0;
//CHECK-NEXT:     int index0 = threadIdx.x;
//CHECK-NEXT:     int _t0 = out[index0];
//CHECK-NEXT:     out[index0] += in[index0];
//CHECK-NEXT:     {
//CHECK-NEXT:         out[index0] = _t0;
//CHECK-NEXT:         int _r_d0 = _d_out[index0];
//CHECK-NEXT:         _d_in[index0] += _r_d0;
//CHECK-NEXT:     }
//CHECK-NEXT: }

__global__ void add_kernel_2(int *out, int *in) {
  out[threadIdx.x] += in[threadIdx.x];
}

// CHECK:    void add_kernel_2_grad(int *out, int *in, int *_d_out, int *_d_in) {
//CHECK-NEXT:     int _t0 = out[threadIdx.x];
//CHECK-NEXT:     out[threadIdx.x] += in[threadIdx.x];
//CHECK-NEXT:     {
//CHECK-NEXT:         out[threadIdx.x] = _t0;
//CHECK-NEXT:         int _r_d0 = _d_out[threadIdx.x];
//CHECK-NEXT:         _d_in[threadIdx.x] += _r_d0;
//CHECK-NEXT:     }
//CHECK-NEXT: }

__global__ void add_kernel_3(int *out, int *in) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  out[index] += in[index];
}

// CHECK:    void add_kernel_3_grad(int *out, int *in, int *_d_out, int *_d_in) {
//CHECK-NEXT:    unsigned int _t1 = blockIdx.x;
//CHECK-NEXT:    unsigned int _t0 = blockDim.x;
//CHECK-NEXT:    int _d_index = 0;
//CHECK-NEXT:    int index0 = threadIdx.x + _t1 * _t0;
//CHECK-NEXT:    int _t2 = out[index0];
//CHECK-NEXT:    out[index0] += in[index0];
//CHECK-NEXT:    {
//CHECK-NEXT:        out[index0] = _t2;
//CHECK-NEXT:        int _r_d0 = _d_out[index0];
//CHECK-NEXT:        _d_in[index0] += _r_d0;
//CHECK-NEXT:    }
//CHECK-NEXT:}

int main(void) {
  int *a = (int*)malloc(sizeof(int));
  *a = 2;
  int *d_a;
  cudaMalloc(&d_a, sizeof(int));
  cudaMemcpy(d_a, a, sizeof(int), cudaMemcpyHostToDevice);

  int *asquare = (int*)malloc(sizeof(int));
  *asquare = 1;
  int *d_square;
  cudaMalloc(&d_square, sizeof(int));
  cudaMemcpy(d_square, asquare, sizeof(int), cudaMemcpyHostToDevice);

  auto test = clad::gradient(kernel);
  dim3 grid(1);
  dim3 block(1);
  cudaStream_t cudaStream;
  cudaStreamCreate(&cudaStream);
  test.execute_kernel(grid, block, 0, cudaStream, d_a, d_square);

  cudaDeviceSynchronize();

  cudaMemcpy(asquare, d_square, sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(a, d_a, sizeof(int), cudaMemcpyDeviceToHost);
  printf("a = %d, d(a^2)/da = %d\n", *a, *asquare); // CHECK-EXEC: a = 2, a^2 = 4

  auto error = clad::gradient(fake_kernel);
  error.execute_kernel(grid, block, d_a, d_square); // CHECK-EXEC: Use execute() for non-global CUDA kernels

  test.execute(d_a, d_square); // CHECK-EXEC: Use execute_kernel() for global CUDA kernels

  cudaMemset(d_a, 5, 1); // first byte is set to 5
  cudaMemset(d_square, 1, 1);

  test.execute_kernel(grid, block, d_a, d_square); 
  cudaDeviceSynchronize();

  cudaMemcpy(asquare, d_square, sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(a, d_a, sizeof(int), cudaMemcpyDeviceToHost);
  printf("a = %d, d(a^2)/da = %d\n", *a, *asquare); // CHECK-EXEC: a = 5, a^2 = 10

  int *dummy_in, *dummy_out;
  cudaMalloc(&dummy_in, sizeof(int));
  cudaMalloc(&dummy_out, sizeof(int));

  int *out = (int*)malloc(10 * sizeof(int));
  for(int i = 0; i < 10; i++) {
    out[i] = 5;
  }
  int *d_out;
  cudaMalloc(&d_out, 10 * sizeof(int));
  cudaMemcpy(d_out, out, 10 * sizeof(int), cudaMemcpyHostToDevice);

  int *d_in;
  cudaMalloc(&d_in, 10 * sizeof(int));

  auto add = clad::gradient(add_kernel, "in, out");
  add.execute_kernel(dim3(1), dim3(10, 1, 1), dummy_out, dummy_in, d_out, d_in);
  cudaDeviceSynchronize();

  int *res = (int*)malloc(10 * sizeof(int));
  cudaMemcpy(res, d_in, 10 * sizeof(int), cudaMemcpyDeviceToHost);
  for(int i = 0; i < 10; i++) {
   if (res[i] != 5) {
      std::cerr << "wrong result of add_kernel_grad at index " << i << std::endl;
      return 1;
    }
  }

  cudaMemset(d_in, 0, 10 * sizeof(int));
  auto add_2 = clad::gradient(add_kernel_2, "in, out");
  add_2.execute_kernel(dim3(1), dim3(10, 1, 1), dummy_out, dummy_in, d_out, d_in);
  cudaDeviceSynchronize();

  cudaMemcpy(res, d_in, 10 * sizeof(int), cudaMemcpyDeviceToHost);
  for(int i = 0; i < 10; i++) {
    if (res[i] != 5) {
      std::cerr << "wrong result of add_kernel_2_grad at index " << i << std::endl;
      return 1;
    }
  }

  cudaMemset(d_in, 0, 10 * sizeof(int));
  auto add_3 = clad::gradient(add_kernel_3, "in, out");
  add_3.execute_kernel(dim3(10), dim3(1), dummy_out, dummy_in, d_out, d_in);
  cudaDeviceSynchronize();

  cudaMemcpy(res, d_in, 10 * sizeof(int), cudaMemcpyDeviceToHost);
  for(int i = 0; i < 10; i++) {
    if (res[i] != 5) {
      std::cerr << "wrong result of add_kernel_3_grad at index " << i << std::endl;
      return 1;
    }
  }

  return 0;
}