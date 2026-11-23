// A tape that outgrows its inline buffer takes its slabs from the device
// heap. The run below needs one slab per thread and fits the default heap.
//
// The other half is not run here: once the heap is used up the tape prints
// "clad: the tape ran out of device heap" and traps, and a trap takes the
// CUDA context down with it. Launching fill_tape with 4096 threads shows it.

// RUN: %cladclang_cuda -I%S/../../include --cuda-path=%cudapath \
// RUN:     --cuda-gpu-arch=%cudaarch %cudaldflags -o TapeDeviceHeap.out %s
//
// RUN: %if cuda-runtime %{ %cudarun ./TapeDeviceHeap.out | %filecheck_exec %s %}
//
// REQUIRES: cuda-compile
#include "clad/Differentiator/Differentiator.h"

#include <cstdio>

// Each thread pushes past the 64 inline elements, so it needs a slab.
__global__ void fill_tape(double* out, int n) {
  clad::tape<double> t = {};
  for (int i = 0; i < n; ++i)
    clad::push(t, (double)i);
  double sum = 0;
  for (int i = 0; i < n; ++i)
    sum += clad::pop(t);
  out[blockIdx.x * blockDim.x + threadIdx.x] = sum;
}

int main() {
  const int threads = 128;
  double* out = nullptr;
  cudaMalloc(&out, threads * sizeof(double));
  fill_tape<<<1, threads>>>(out, 100);
  bool ok = cudaDeviceSynchronize() == cudaSuccess;

  double last = 0;
  if (ok)
    cudaMemcpy(&last, out + threads - 1, sizeof(double),
               cudaMemcpyDeviceToHost);
  // 4950 = 0 + 1 + ... + 99
  printf("%s %g\n", ok ? "ok" : "failed", last); // CHECK-EXEC: ok 4950
  cudaFree(out);
  return 0;
}
