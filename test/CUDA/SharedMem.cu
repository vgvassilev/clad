// RUN: %cladclang_cuda -I%S/../../include --cuda-path=%cudapath \
// RUN:     --cuda-gpu-arch=%cudaarch %cudaldflags -oReverseModeSharedMem.out \
// RUN:     -Xclang -verify %s 2>&1 | %filecheck %s
//
// RUN: %cudarun ./ReverseModeSharedMem.out | %filecheck_exec %s
//
// REQUIRES: cuda-runtime
//
// expected-no-diagnostics

#include <iostream>
#include "clad/Differentiator/Differentiator.h"
#include <cuda.h>

__global__ void func(int* vec_d, int val) {
    __shared__ int sharedMem[1];
    sharedMem[0] = val * 3;
    __syncthreads();
    vec_d[0] = sharedMem[0] * 5 + vec_d[0];
}

// CHECK: void func_grad(int *vec_d, int val, int *_d_vec_d, int *_d_val) {
// CHECK-NEXT:     static int _d_sharedMem[1] __attribute__((shared));
// CHECK-NEXT:     clad::zero_init(_d_sharedMem);
// CHECK-NEXT:     static int sharedMem[1] __attribute__((shared));
// CHECK-NEXT:     sharedMem[0] = val * 3;
// CHECK-NEXT:     __syncthreads();
// CHECK-NEXT:     vec_d[0] = sharedMem[0] * 5 + vec_d[0];
// CHECK-NEXT:     {
// CHECK-NEXT:         int _r_d1 = _d_vec_d[0];
// CHECK-NEXT:         _d_vec_d[0] = 0;
// CHECK-NEXT:         atomicAdd(&_d_sharedMem[0], _r_d1 * 5);
// CHECK-NEXT:         atomicAdd(&_d_vec_d[0], _r_d1);
// CHECK-NEXT:     }
// CHECK-NEXT:     __syncthreads();
// CHECK-NEXT:     {
// CHECK-NEXT:         int _r_d0 = _d_sharedMem[0];
// CHECK-NEXT:         _d_sharedMem[0] = 0;
// CHECK-NEXT:         atomicAdd(_d_val, _r_d0 * 3);
// CHECK-NEXT:     }
// CHECK-NEXT: }

__global__ void func1(int* vec1_d, int val1) {
    extern __shared__ int sharedMem1[];
    sharedMem1[0] = val1 * 3;
    __syncthreads();
    vec1_d[0] = sharedMem1[0] * 5 + vec1_d[0];
}

// CHECK: void func1_grad(int *vec1_d, int val1, int *_d_vec1_d, int *_d_val1) {
// CHECK-NEXT:     int *_d_sharedMem1 = (int *)((char *)sharedMem1 + clad::get_dynamic_smem_size() / 2);
// CHECK-NEXT:     _d_sharedMem1[0] = 0;
// CHECK-NEXT:     extern int sharedMem1[] __attribute__((shared));
// CHECK-NEXT:     sharedMem1[0] = val1 * 3;
// CHECK-NEXT:     __syncthreads();
// CHECK-NEXT:     vec1_d[0] = sharedMem1[0] * 5 + vec1_d[0];
// CHECK-NEXT:     {
// CHECK-NEXT:         int _r_d1 = _d_vec1_d[0];
// CHECK-NEXT:         _d_vec1_d[0] = 0;
// CHECK-NEXT:         _d_sharedMem1[0] += _r_d1 * 5;
// CHECK-NEXT:         atomicAdd(&_d_vec1_d[0], _r_d1);
// CHECK-NEXT:     }
// CHECK-NEXT:     __syncthreads();
// CHECK-NEXT:     {
// CHECK-NEXT:         int _r_d0 = _d_sharedMem1[0];
// CHECK-NEXT:         _d_sharedMem1[0] = 0;
// CHECK-NEXT:         atomicAdd(_d_val1, _r_d0 * 3);
// CHECK-NEXT:     }
// CHECK-NEXT: }

__global__ void scalar_func(int* vec2_d, int val2) {
    __shared__ int x;
    x = val2 * 3;
    __syncthreads();
    vec2_d[0] = x * 5;
}

// CHECK: void scalar_func_grad(int *vec2_d, int val2, int *_d_vec2_d, int *_d_val2) {
// CHECK-NEXT:     static int _d_x __attribute__((shared));
// CHECK-NEXT:     clad::zero_init(_d_x);
// CHECK-NEXT:     static int x __attribute__((shared));
// CHECK-NEXT:     x = val2 * 3;
// CHECK-NEXT:     __syncthreads();
// CHECK-NEXT:     vec2_d[0] = x * 5;
// CHECK-NEXT:     {
// CHECK-NEXT:         int _r_d0 = _d_vec2_d[0];
// CHECK-NEXT:         _d_vec2_d[0] = 0;
// CHECK-NEXT:         atomicAdd(&_d_x, _r_d0 * 5);
// CHECK-NEXT:     }
// CHECK-NEXT:     __syncthreads();
// CHECK-NEXT:     {
// CHECK-NEXT:         atomicAdd(_d_val2, _d_x * 3);
// CHECK-NEXT:         _d_x = 0;
// CHECK-NEXT:     }
// CHECK-NEXT: }

int main() {

    int *d_vec, *d_vec_adj, *d_val_adj;
    cudaMalloc(&d_vec, sizeof(int));
    cudaMalloc(&d_vec_adj, sizeof(int));
    cudaMalloc(&d_val_adj, sizeof(int));

    int seed = 1, seed1 = 0, grad, grad1;

    cudaMemcpy(d_vec_adj, &seed, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_val_adj, &seed1, sizeof(int), cudaMemcpyHostToDevice);
    
    auto dfunc = clad::gradient(func);
    dfunc.execute_kernel(dim3(1), dim3(1), d_vec, 90, d_vec_adj, d_val_adj);
    cudaDeviceSynchronize();

    cudaMemcpy(&grad, d_val_adj, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&grad1, d_vec_adj, sizeof(int), cudaMemcpyDeviceToHost);
    
    std::cout << "Grad dvec/dval: " << grad << std::endl;
    // CHECK-EXEC: Grad dvec/dval: 15
    std::cout << "Grad dvec/dvec_d[0]: " << grad1 << std::endl;
    // CHECK-EXEC-NEXT: Grad dvec/dvec_d[0]: 1

    cudaMemcpy(d_vec_adj, &seed, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_val_adj, &seed1, sizeof(int), cudaMemcpyHostToDevice);

    auto dfunc1 = clad::gradient(func1);
    auto* kernel_ptr = dfunc1.getFunctionPtr();
    kernel_ptr<<<dim3(1), dim3(1), 2 * sizeof(int)>>>(d_vec, 90, d_vec_adj, d_val_adj);
    cudaDeviceSynchronize();

    cudaMemcpy(&grad, d_val_adj, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&grad1, d_vec_adj, sizeof(int), cudaMemcpyDeviceToHost);
    
    std::cout << "Grad dvec/dval: " << grad << std::endl;
    // CHECK-EXEC-NEXT: Grad dvec/dval: 15
    std::cout << "Grad dvec/dvec1_d[0]: " << grad1 << std::endl;
    // CHECK-EXEC-NEXT: Grad dvec/dvec1_d[0]: 1

    cudaMemcpy(d_vec_adj, &seed, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_val_adj, &seed1, sizeof(int), cudaMemcpyHostToDevice);

    auto d_scalar = clad::gradient(scalar_func);
    d_scalar.execute_kernel(dim3(1), dim3(1), d_vec, 9, d_vec_adj, d_val_adj);
    cudaDeviceSynchronize();

    cudaMemcpy(&grad, d_val_adj, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&grad1, d_vec_adj, sizeof(int), cudaMemcpyDeviceToHost);
    
    std::cout << "Grad dvec/dval: " << grad << std::endl;
    // CHECK-EXEC-NEXT: Grad dvec/dval: 15
    std::cout << "Grad dvec/dvec2_d[0]: " << grad1 << std::endl;
    // CHECK-EXEC-NEXT: Grad dvec/dvec2_d[0]: 0

    cudaFree(d_vec);
    cudaFree(d_vec_adj);
    cudaFree(d_val_adj);

    return 0;
}
