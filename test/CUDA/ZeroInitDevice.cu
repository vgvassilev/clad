// RUN: %cladclang_cuda -I%S/../../include --cuda-path=%cudapath \
// RUN:     --cuda-gpu-arch=%cudaarch %cudaldflags -oZeroInitDevice.out \
// RUN:     -Xclang -verify %s 2>&1 | %filecheck %s
//
// RUN: %cudarun ./ZeroInitDevice.out | %filecheck_exec %s
//
// RUN: %cladclang_cuda -I%S/../../include --cuda-path=%cudapath \
// RUN:     --cuda-gpu-arch=%cudaarch -fsyntax-only %s -DEXPECT_DIAG \
// RUN:     -Xclang -verify=device -Xclang -verify-ignore-unexpected=note
//
// REQUIRES: cuda-runtime
//
// expected-no-diagnostics

#include <iostream>
#include "clad/Differentiator/Differentiator.h"
#include <cuda.h>

struct CustomStruct {
    double x;
    double y;
};

//Non trivial ctor will fallback to use of loop 
struct CustomStruct1{
    double x;
    double y;
    __host__ __device__ CustomStruct1(double x, double y){
        this->x=x;
        this->y=y;
    }
};
__host__ __device__ double dummy_func(double x) { return x * 3.0; }

// CHECK: void dummy_func_grad(double x, double *_d_x) {
// CHECK-NEXT:     *_d_x += 1 * 3.;
// CHECK-NEXT: }

__global__ void zero_init_device(double* out) {
    CustomStruct s;
    s.x = 10.0;
    s.y = 20.0;
    
    clad::zero_init(s);

    out[0] = s.x;
    out[1] = s.y;
}

__global__ void zero_init_device1(double* out1){
    CustomStruct1 s1(1.0,2.0);

    clad::zero_init(s1);

    out1[0]=s1.x;
    out1[1]=s1.y;
}

#ifdef EXPECT_DIAG
struct CustomStruct2 {
    double x;
    __host__ __device__ CustomStruct2() : x(0) {}
    __host__ __device__ ~CustomStruct2() {} 
};

__global__ void test_diag() {
    CustomStruct2 s;
    clad::zero_init(s); // device-error@* {{Clad device fallback zero_init requires trivially destructible types}}
}
#endif

int main() {
    auto grad = clad::gradient(dummy_func, "x");
    
    double* d_out;
    cudaMalloc(&d_out, 2*sizeof(double));
    
    zero_init_device<<<1, 1>>>(d_out);
    cudaDeviceSynchronize();
    
    double h_out[2];
    cudaMemcpy(h_out, d_out, 2*sizeof(double), cudaMemcpyDeviceToHost);
    
    std::cout << "s.x: " << h_out[0] << std::endl;
    // CHECK-EXEC: s.x: 0
    std::cout << "s.y: " << h_out[1] << std::endl;
    // CHECK-EXEC-NEXT: s.y: 0
    
    double* d_out1;
    cudaMalloc(&d_out1,2*sizeof(double));

    zero_init_device1<<<1,1>>>(d_out1);
    cudaDeviceSynchronize();

    double h_out1[2];
    cudaMemcpy(h_out1, d_out1, 2*sizeof(double), cudaMemcpyDeviceToHost);

    std::cout << "s1.x: " << h_out1[0] << std::endl;
    // CHECK-EXEC: s1.x: 0
    std::cout << "s1.y: " << h_out1[1] << std::endl;
    // CHECK-EXEC-NEXT: s1.y: 0

    cudaFree(d_out);
    cudaFree(d_out1);
    return 0;
}
