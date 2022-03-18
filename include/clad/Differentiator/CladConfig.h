// This file defines configuration that is shared by various files

#ifndef CLAD_CONFIG_H
#define CLAD_CONFIG_H

#include <cstdlib>
#include <memory>

// Define CUDA_HOST_DEVICE attribute for adding CUDA support to
// clad functions
#ifdef __CUDACC__
#define CUDA_HOST_DEVICE __host__ __device__
#else
#define CUDA_HOST_DEVICE
#endif

// Define trap function that is a CUDA compatible replacement for
// exit(int code) function
#ifdef __CUDACC__
__device__ void trap(int code) {
  asm("trap;");
}
__host__ void trap(int code) {
  exit(code);
}
#else
void trap(int code) {
  exit(code);
}
#endif

#ifdef  __CUDACC__
template<typename T>
__device__ T* clad_addressof(T& r) {
  return __builtin_addressof(r);
}
template<typename T>
__host__ T* clad_addressof(T& r) {
  return std::addressof(r);
}
#else
template<typename T>
T* clad_addressof(T& r) {
  return std::addressof(r);
}
#endif

#endif // CLAD_CONFIG_H
