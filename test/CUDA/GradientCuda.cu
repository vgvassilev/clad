// The Test checks whether a clad gradient can be successfully be generated on
// the device having all the dependencies also as device functions.

// RUN: %cladclang_cuda -I%S/../../include  %s -fsyntax-only \
// RUN: %cudasmlevel --cuda-path=%cudapath  -Xclang -verify 2>&1 | FileCheck %s

// RUN: %cladclang_cuda -I%S/../../include %s -xc++ %cudasmlevel \
// RUN: --cuda-path=%cudapath -L/usr/local/cuda/lib64 -lcudart_static \
// RUN: -ldl -lrt -pthread -lm -lstdc++

// REQUIRES: cuda-runtime

// expected-no-diagnostics

// XFAIL: clang-15

#include "clad/Differentiator/Differentiator.h"
#include <array>

#define N 3

__device__ __host__ double gauss(double* x, double* p, double sigma, int dim) {
   double t = 0;
   for (int i = 0; i< dim; i++)
       t += (x[i] - p[i]) * (x[i] - p[i]);
   t = -t / (2*sigma*sigma);
   return std::pow(2*M_PI, -dim/2.0) * std::pow(sigma, -0.5) * std::exp(t);
}


// CHECK:    void gauss_grad_1(double *x, double *p, double sigma, int dim, double *_d_p) __attribute__((device)) __attribute__((host)) {
//CHECK-NEXT:     double _d_sigma = 0;
//CHECK-NEXT:     int _d_dim = 0;
//CHECK-NEXT:     double _d_t = 0;
//CHECK-NEXT:     unsigned long _t0;
//CHECK-NEXT:     int _d_i = 0;
//CHECK-NEXT:     int i = 0;
//CHECK-NEXT:     clad::tape<double> _t1 = {};
//CHECK-NEXT:     double _t2;
//CHECK-NEXT:     double _t3;
//CHECK-NEXT:     double _t4;
//CHECK-NEXT:     double _t5;
//CHECK-NEXT:     double _t6;
//CHECK-NEXT:     double t = 0;
//CHECK-NEXT:     _t0 = 0;
//CHECK-NEXT:     for (i = 0; i < dim; i++) {
//CHECK-NEXT:         _t0++;
//CHECK-NEXT:         clad::push(_t1, t);
//CHECK-NEXT:         t += (x[i] - p[i]) * (x[i] - p[i]);
//CHECK-NEXT:     }
//CHECK-NEXT:     _t2 = t;
//CHECK-NEXT:     _t3 = (2 * sigma * sigma);
//CHECK-NEXT:     t = -t / _t3;
//CHECK-NEXT:     _t6 = std::pow(2 * 3.1415926535897931, -dim / 2.);
//CHECK-NEXT:     _t5 = std::pow(sigma, -0.5);
//CHECK-NEXT:     _t4 = std::exp(t);
//CHECK-NEXT:     goto _label0;
//CHECK-NEXT:   _label0:
//CHECK-NEXT:     {
//CHECK-NEXT:         double _r1 = 0;
//CHECK-NEXT:         double _r2 = 0;
//CHECK-NEXT:         clad::custom_derivatives{{(::std)?}}::pow_pullback(2 * 3.1415926535897931, -dim / 2., 1 * _t4 * _t5, &_r1, &_r2);
//CHECK-NEXT:         _d_dim += -_r2 / 2.;
//CHECK-NEXT:         double _r3 = 0;
//CHECK-NEXT:         double _r4 = 0;
//CHECK-NEXT:         clad::custom_derivatives{{(::std)?}}::pow_pullback(sigma, -0.5, _t6 * 1 * _t4, &_r3, &_r4);
//CHECK-NEXT:         _d_sigma += _r3;
//CHECK-NEXT:         double _r5 = 0;
//CHECK-NEXT:         _r5 += _t6 * _t5 * 1 * clad::custom_derivatives::exp_pushforward(t, 1.).pushforward;
//CHECK-NEXT:         _d_t += _r5;
//CHECK-NEXT:     }
//CHECK-NEXT:     {
//CHECK-NEXT:         t = _t2;
//CHECK-NEXT:         double _r_d1 = _d_t;
//CHECK-NEXT:         _d_t -= _r_d1;
//CHECK-NEXT:         _d_t += -_r_d1 / _t3;
//CHECK-NEXT:         double _r0 = _r_d1 * --t / (_t3 * _t3);
//CHECK-NEXT:         _d_sigma += 2 * _r0 * sigma;
//CHECK-NEXT:         _d_sigma += 2 * sigma * _r0;
//CHECK-NEXT:     }
//CHECK-NEXT:     for (; _t0; _t0--) {
//CHECK-NEXT:         i--;
//CHECK-NEXT:         t = clad::pop(_t1);
//CHECK-NEXT:         double _r_d0 = _d_t;
//CHECK-NEXT:         _d_p[i] += -_r_d0 * (x[i] - p[i]);
//CHECK-NEXT:         _d_p[i] += -(x[i] - p[i]) * _r_d0;
//CHECK-NEXT:     }
//CHECK-NEXT: }

__global__ void compute(double* d_x, double* d_p, int n, double* d_result) {
  auto gauss_g = clad::gradient(gauss, "p");
  gauss_g.execute(d_x, d_p, 2.0, n, d_result);
}

int main(void) {
  double *x, *d_x;
  double *p, *d_p;

  x = (double*)malloc(N * sizeof(double));
  p = (double*)malloc(N * sizeof(double));
  for (int i = 0; i < N; i++) {
    x[i] = 2.0;
    p[i] = 1.0;
  }

  cudaMalloc(&d_x, N * sizeof(double));
  cudaMemcpy(d_x, x, N * sizeof(double), cudaMemcpyHostToDevice);
  cudaMalloc(&d_p, N * sizeof(double));
  cudaMemcpy(d_p, p, N * sizeof(double), cudaMemcpyHostToDevice);
  std::array<double, N> result{0};
  double *d_result;

  cudaMalloc(&d_result, N * sizeof(double));

  compute<<<1, 1>>>(d_x, d_p, N, d_result);
  cudaDeviceSynchronize();

  cudaMemcpy(result.data(), d_result, N * sizeof(double), cudaMemcpyDeviceToHost);
  printf("%f,%f,%f\n", result[0], result[1], result[2]);

  std::array<double, N> result_cpu{0};
  auto gauss_g = clad::gradient(gauss, "p");
  gauss_g.execute(x, p, 2.0, N, result_cpu.data());

  if (result != result_cpu) {
    printf("Results are not equal\n");
    return 1;
  }

}