// RUN: %cladclang_cuda -I%S/../../include --cuda-path=%cudapath \
// RUN: --cuda-gpu-arch=%cudaarch %cudaldflags -oActivityTest.out \
// RUN: -Xclang -plugin-arg-clad -Xclang -enable-va -Xclang -verify %s 2>&1 | %filecheck %s
//
// RUN: %if cuda-runtime %{ %cudarun ./ActivityTest.out | %filecheck_exec %s %}
//
// REQUIRES: cuda-compile
//
// expected-no-diagnostics

#include <iostream>
#include "clad/Differentiator/Differentiator.h"
#include <cuda.h>

__global__ void func(double* out, double x) {
    double val = x;
    dim3 t = threadIdx;
    val = val + t.x + threadIdx.x;
    val = val + blockIdx.x + blockDim.x + gridDim.x;
    if (threadIdx.x == 0) {
        val = val * 2.0;
    }
    out[threadIdx.x] = val;
}

// CHECK-LABEL: void func_grad(double *out, double x, double *_d_out, double *_d_x) {
// CHECK-NOT: _d_threadIdx
// CHECK: double val = x;
// CHECK: dim3 t = threadIdx;
// CHECK-NEXT: val = val + t.x + threadIdx.x;
// CHECK: val = val + blockIdx.x + blockDim.x + gridDim.x;
// CHECK: if
// CHECK: val = val * 2.;
// CHECK: }

__global__ void square(double* out, const double* in) {
    out[0] = in[0] * in[0];
}

double func1(double* in) {
  double* tmp = nullptr;
  cudaMalloc(&tmp, sizeof(double));
  square<<<1, 1>>>(tmp, in);
  double res = 0;
  cudaMemcpy(&res, tmp, sizeof(double), cudaMemcpyDeviceToHost);
  cudaFree(tmp);
  return res * 2;
}

// CHECK: void func1_grad(double *in, double *_d_in) {
// CHECK-NEXT:     double *_d_tmp = nullptr;
// CHECK-NEXT:     double *tmp = nullptr;
// CHECK-NEXT:     cudaMalloc(&_d_tmp, sizeof(double));
// CHECK-NEXT:     cudaMemset(_d_tmp, 0, sizeof(double));
// CHECK-NEXT:     cudaMalloc(&tmp, sizeof(double));
// CHECK-NEXT:     square<<<1, 1>>>(tmp, in);
// CHECK-NEXT:     double _d_res = 0.;
// CHECK-NEXT:     double res = 0;
// CHECK-NEXT:     cudaMemcpy(&res, tmp, sizeof(double), cudaMemcpyDeviceToHost);
// CHECK-NEXT:     _d_res += 1 * 2;
// CHECK-NEXT:     {
// CHECK-NEXT:         {{.*}} _r0 = {{.*}};
// CHECK-NEXT:         cudaMemcpyKind _r1 = static_cast<cudaMemcpyKind>(0U);
// CHECK-NEXT:         clad::custom_derivatives::cudaMemcpy_pullback(&res, tmp, sizeof(double), cudaMemcpyDeviceToHost, {{.*}}, &_d_res, _d_tmp, &_r0, &_r1);
// CHECK-NEXT:     }
// CHECK-NEXT:     square_pullback<<<1, 1>>>(tmp, in, _d_tmp, _d_in);
// CHECK-NEXT:     cudaFree(tmp);
// CHECK-NEXT:     cudaFree(_d_tmp);
// CHECK-NEXT: }

// Only the launch makes tmp varied here: memory from cudaMalloc is varied
// already. Compiled only, as a kernel cannot write host stack memory.
double func2(double* in) {
  double tmp[1] = {0};
  square<<<1, 1>>>(tmp, in);
  return tmp[0] * 2;
}

// CHECK: void func2_grad(double *in, double *_d_in) {
// CHECK-NEXT:     double _d_tmp[1] = {0};
// CHECK-NEXT:     double tmp[1] = {0};
// CHECK-NEXT:     square<<<1, 1>>>(tmp, in);
// CHECK-NEXT:     _d_tmp[0] += 1 * 2;
// CHECK-NEXT:     square_pullback<<<1, 1>>>(tmp, in, _d_tmp, _d_in);
// CHECK-NEXT: }

int main() {
    double *d_out, *d_out_d, *d_x;
    cudaMalloc(&d_out, sizeof(double) * 2);
    cudaMalloc(&d_out_d, sizeof(double) * 2);
    cudaMalloc(&d_x, sizeof(double));

    double seed_out[2] = {0.0, 1.0};
    double seed_x = 0;
    
    cudaMemcpy(d_out_d, seed_out, sizeof(double) * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, &seed_x, sizeof(double), cudaMemcpyHostToDevice);

    auto df = clad::gradient(func);
    auto d_func1 = clad::gradient(func1);
    auto d_func2 = clad::gradient(func2);
    df.execute_kernel(dim3(1), dim3(2), d_out, 5.0, d_out_d, d_x);
    cudaDeviceSynchronize();

    double grad_x;
    cudaMemcpy(&grad_x, d_x, sizeof(double), cudaMemcpyDeviceToHost);
    std::cout << "Grad x: " << grad_x << std::endl;
    // CHECK-EXEC: Grad x: 1
    
    cudaFree(d_out);
    cudaFree(d_out_d);
    cudaFree(d_x);
    return 0;
}