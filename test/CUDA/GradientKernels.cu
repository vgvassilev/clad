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
  test.execute_kernel(grid, block, 0, nullptr, d_a, d_square);

  cudaDeviceSynchronize();

  cudaMemcpy(asquare, d_square, sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(a, d_a, sizeof(int), cudaMemcpyDeviceToHost);
  printf("a = %d, a^2 = %d\n", *a, *asquare);

  return 0;
}