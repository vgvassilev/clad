// RUN: %cladclang_cuda -I%S/../../include %s -xc++ %cudasmlevel \
// RUN: --cuda-path=%cudapath -L/usr/local/cuda/lib64 -lcudart_static \
// RUN: -ldl -lrt -pthread -lm -lstdc++ -oForwardMode.out 2>&1 | %filecheck %s

// RUN: ./ForwardMode.out

// REQUIRES: cuda-runtime

// expected-no-diagnostics

#include "clad/Differentiator/Differentiator.h"
#include "../TestUtils.h"

__global__ void add(double *a, double *b, double *c, int n) {
  int idx = threadIdx.x;
  if (idx < n)
    c[idx] = a[idx] + b[idx];
}

// CHECK: void add_pushforward(double *a, double *b, double *c, int n, double *_d_a, double *_d_b, double *_d_c, int _d_n) __attribute__((global)) {
// CHECK-NEXT:     int _d_idx = 0;
// CHECK-NEXT:     int idx = threadIdx.x;
// CHECK-NEXT:     if (idx < n) {
// CHECK-NEXT:         _d_c[idx] = _d_a[idx] + _d_b[idx];
// CHECK-NEXT:         c[idx] = a[idx] + b[idx];
// CHECK-NEXT:     }
// CHECK-NEXT: }

double fn1(double i, double j) {
  double a[500] = {};
  double b[500] = {};
  double c[500] = {};
  int n = 500;

  for (int idx=0; idx<500; ++idx) {
    a[idx] = 7;
    b[idx] = 9;
  }

  double *device_a = nullptr;
  double *device_b = nullptr;
  double *device_c = nullptr;

  cudaMalloc(&device_a, n * sizeof(double));
  cudaMalloc(&device_b, n * sizeof(double));
  cudaMalloc(&device_c, n * sizeof(double));

  cudaMemcpy(device_a, a, n * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(device_b, b, n * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(device_c, c, n * sizeof(double), cudaMemcpyHostToDevice);

  add<<<1, 700>>>(device_a, device_b, device_c, n);

  cudaDeviceSynchronize();

  cudaMemcpy(c, device_c, n * sizeof(double), cudaMemcpyDeviceToHost);

  double sum = 0;
  for (int idx=0; idx<n; ++idx)
    sum += c[idx];
  
  return sum * i + 2 * sum * j;
}

// CHECK: double fn1_darg0(double i, double j) {
// CHECK-NEXT:     double _d_i = 1;
// CHECK-NEXT:     double _d_j = 0;
// CHECK-NEXT:     double _d_a[500] = {};
// CHECK-NEXT:     double a[500] = {};
// CHECK-NEXT:     double _d_b[500] = {};
// CHECK-NEXT:     double b[500] = {};
// CHECK-NEXT:     double _d_c[500] = {};
// CHECK-NEXT:     double c[500] = {};
// CHECK-NEXT:     int _d_n = 0;
// CHECK-NEXT:     int n = 500;
// CHECK-NEXT:     {
// CHECK-NEXT:         int _d_idx = 0;
// CHECK-NEXT:         for (int idx = 0; idx < 500; ++idx) {
// CHECK-NEXT:             _d_a[idx] = 0;
// CHECK-NEXT:             a[idx] = 7;
// CHECK-NEXT:             _d_b[idx] = 0;
// CHECK-NEXT:             b[idx] = 9;
// CHECK-NEXT:         }
// CHECK-NEXT:     }
// CHECK-NEXT:     double *_d_device_a = nullptr;
// CHECK-NEXT:     double *device_a = nullptr;
// CHECK-NEXT:     double *_d_device_b = nullptr;
// CHECK-NEXT:     double *device_b = nullptr;
// CHECK-NEXT:     double *_d_device_c = nullptr;
// CHECK-NEXT:     double *device_c = nullptr;
// CHECK-NEXT:     unsigned long _t0 = sizeof(double);
// CHECK-NEXT:     ValueAndPushforward<cudaError_t, cudaError_t> _t1 = clad::custom_derivatives::cudaMalloc_pushforward(&device_a, n * _t0, &_d_device_a, _d_n * _t0 + n * sizeof(double));
// CHECK-NEXT:     unsigned long _t2 = sizeof(double);
// CHECK-NEXT:     ValueAndPushforward<cudaError_t, cudaError_t> _t3 = clad::custom_derivatives::cudaMalloc_pushforward(&device_b, n * _t2, &_d_device_b, _d_n * _t2 + n * sizeof(double));
// CHECK-NEXT:     unsigned long _t4 = sizeof(double);
// CHECK-NEXT:     ValueAndPushforward<cudaError_t, cudaError_t> _t5 = clad::custom_derivatives::cudaMalloc_pushforward(&device_c, n * _t4, &_d_device_c, _d_n * _t4 + n * sizeof(double));
// CHECK-NEXT:     unsigned long _t6 = sizeof(double);
// CHECK-NEXT:     ValueAndPushforward<cudaError_t, cudaError_t> _t7 = clad::custom_derivatives::cudaMemcpy_pushforward(device_a, a, n * _t6, cudaMemcpyHostToDevice, _d_device_a, _d_a, _d_n * _t6 + n * sizeof(double));
// CHECK-NEXT:     unsigned long _t8 = sizeof(double);
// CHECK-NEXT:     ValueAndPushforward<cudaError_t, cudaError_t> _t9 = clad::custom_derivatives::cudaMemcpy_pushforward(device_b, b, n * _t8, cudaMemcpyHostToDevice, _d_device_b, _d_b, _d_n * _t8 + n * sizeof(double));
// CHECK-NEXT:     unsigned long _t10 = sizeof(double);
// CHECK-NEXT:     ValueAndPushforward<cudaError_t, cudaError_t> _t11 = clad::custom_derivatives::cudaMemcpy_pushforward(device_c, c, n * _t10, cudaMemcpyHostToDevice, _d_device_c, _d_c, _d_n * _t10 + n * sizeof(double));
// CHECK-NEXT:     add_pushforward<<<1, 700>>>(device_a, device_b, device_c, n, _d_device_a, _d_device_b, _d_device_c, _d_n);
// CHECK-NEXT:     ValueAndPushforward<int, int> _t12 = clad::custom_derivatives::cudaDeviceSynchronize_pushforward();
// CHECK-NEXT:     unsigned long _t13 = sizeof(double);
// CHECK-NEXT:     ValueAndPushforward<cudaError_t, cudaError_t> _t14 = clad::custom_derivatives::cudaMemcpy_pushforward(c, device_c, n * _t13, cudaMemcpyDeviceToHost, _d_c, _d_device_c, _d_n * _t13 + n * sizeof(double));
// CHECK-NEXT:     double _d_sum = 0;
// CHECK-NEXT:     double sum = 0;
// CHECK-NEXT:     {
// CHECK-NEXT:         int _d_idx = 0;
// CHECK-NEXT:         for (int idx = 0; idx < n; ++idx) {
// CHECK-NEXT:             _d_sum += _d_c[idx];
// CHECK-NEXT:             sum += c[idx];
// CHECK-NEXT:         }
// CHECK-NEXT:     }
// CHECK-NEXT:     double _t15 = 2 * sum;
// CHECK-NEXT:     return _d_sum * i + sum * _d_i + (0 * sum + 2 * _d_sum) * j + _t15 * _d_j;
// CHECK-NEXT: }

int main() {
  INIT_DIFFERENTIATE(fn1, "i");

  TEST_DIFFERENTIATE(fn1, 3, 5);  // CHECK-EXEC: 8000.00
}