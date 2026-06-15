// RUN: %cladclang_cuda -fsyntax-only -I%S/../../include --cuda-path=%cudapath \
// RUN:     --cuda-gpu-arch=%cudaarch -Xclang -verify %s
//
// REQUIRES: cuda-runtime
// expected-no-diagnostics

#include "clad/Differentiator/Tape.h"

struct TrackedValue {
    double val;
    __host__ __device__ TrackedValue() : val(0.0) {}
    
    __host__ __device__ ~TrackedValue() {}
};

template <class T>
__device__ void exercise_tape(clad::tape_impl<T>& t) {
  t.emplace_back();
  
  auto& back = t.back();
  (void)back;
  
  // This explicitly triggers the `destroy_element` template 
  // via `pop_back()`.
  t.pop_back(); 
}

__global__ void my_kernel() {
  clad::tape_impl<TrackedValue> t;
  exercise_tape(t);
}
