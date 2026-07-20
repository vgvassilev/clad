// RUN: %cladclang_cuda -I%S/../../include --cuda-path=%cudapath \
// RUN:     --cuda-gpu-arch=%cudaarch %cudaldflags -o RestoreTrackerDevice.out %s
//
// RUN: %cudarun ./RestoreTrackerDevice.out | %filecheck_exec %s
//
// REQUIRES: cuda-runtime
#include "clad/Differentiator/Differentiator.h"
#include <iostream>

__global__ void test_kernel() {
    clad::restore_tracker tracker;
    
    double val = 3.14;
    tracker.store(val);
    val = 0.0;
    tracker.restore();
    
    if (val == 3.14) {
        printf("Working on device!\n");
    }
}

int main() {
    
    test_kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
    
    // CHECK-EXEC: Working on device!
    return 0;
}
