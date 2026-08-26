// RUN: %cladclang_cuda -I%S/../../include --cuda-path=%cudapath \
// RUN: --cuda-gpu-arch=%cudaarch %cudaldflags -oActivityTest.out \
// RUN: -Xclang -verify %s 2>&1 | %filecheck %s
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
    if (threadIdx.x == 0) {
        val = val * 2.0;
    }
    out[threadIdx.x] = val;
}

// CHECK-LABEL: void func_grad(double *out, double x, double *_d_out, double *_d_x) {
// CHECK-NOT: _d_threadIdx
// CHECK: double val = x;
// CHECK: dim3 t = threadIdx;
// CHECK: val = val + t.x + threadIdx.x;
// CHECK: if
// CHECK: val = val * 2.;
// CHECK: }

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