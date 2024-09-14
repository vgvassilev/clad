// RUN: %cladclang_cuda -I%S/../../include -fsyntax-only \
// RUN:     --cuda-gpu-arch=%cudaarch --cuda-path=%cudapath  -Xclang -verify \
// RUN:     %s 2>&1 | %filecheck %s
//
// RUN: %cladclang_cuda -I%S/../../include --cuda-path=%cudapath \
// RUN:     --cuda-gpu-arch=%cudaarch %cudaldflags -oGradientKernels.out %s
//
// RUN: ./GradientKernels.out | %filecheck_exec %s
//
// REQUIRES: cuda-runtime
//
// expected-no-diagnostics
//
// CHECK-NOT: {{.*error|warning|note:.*}}

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
  printf("a = %d, a^2 = %d\n", *a, *asquare); // CHECK-EXEC: a = 2, a^2 = 4

  auto error = clad::gradient(fake_kernel);
  error.execute_kernel(grid, block, d_a, d_square); // CHECK-EXEC: Use execute() for non-global CUDA kernels

  test.execute(d_a, d_square); // CHECK-EXEC: Use execute_kernel() for global CUDA kernels

  cudaMemset(d_a, 5, 1); // first byte is set to 5
  cudaMemset(d_square, 1, 1);

  test.execute_kernel(grid, block, d_a, d_square); 
  cudaDeviceSynchronize();

  cudaMemcpy(asquare, d_square, sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(a, d_a, sizeof(int), cudaMemcpyDeviceToHost);
  printf("a = %d, a^2 = %d\n", *a, *asquare); // CHECK-EXEC: a = 5, a^2 = 10

  return 0;
}
