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

#define N 3

__device__ __host__ double gauss(double* x, double* p, double sigma, int dim) {
   double t = 0;
   for (int i = 0; i< dim; i++)
       t += (x[i] - p[i]) * (x[i] - p[i]);
   t = -t / (2*sigma*sigma);
   return std::pow(2*M_PI, -dim/2.0) * std::pow(sigma, -0.5) * std::exp(t);
}

auto gauss_g = clad::gradient(gauss, "p");

// CHECK:    void gauss_grad_1(double *x, double *p, double sigma, int dim, clad::array_ref<double> _d_p) __attribute__((device)) __attribute__((host)) {
//CHECK-NEXT:     double _d_sigma = 0;
//CHECK-NEXT:     int _d_dim = 0;
//CHECK-NEXT:     double _d_t = 0;
//CHECK-NEXT:     unsigned long _t0;
//CHECK-NEXT:     int _d_i = 0;
//CHECK-NEXT:     clad::tape<double> _t1 = {};
//CHECK-NEXT:     clad::tape<int> _t2 = {};
//CHECK-NEXT:     clad::tape<int> _t4 = {};
//CHECK-NEXT:     clad::tape<double> _t6 = {};
//CHECK-NEXT:     clad::tape<int> _t7 = {};
//CHECK-NEXT:     clad::tape<int> _t9 = {};
//CHECK-NEXT:     double _t11;
//CHECK-NEXT:     double _t12;
//CHECK-NEXT:     double _t13;
//CHECK-NEXT:     double _t14;
//CHECK-NEXT:     double _t15;
//CHECK-NEXT:     double _t16;
//CHECK-NEXT:     double _t17;
//CHECK-NEXT:     double _t18;
//CHECK-NEXT:     double _t19;
//CHECK-NEXT:     double _t20;
//CHECK-NEXT:     double _t21;
//CHECK-NEXT:     double _t22;
//CHECK-NEXT:     double _t23;
//CHECK-NEXT:     double t = 0;
//CHECK-NEXT:     _t0 = 0;
//CHECK-NEXT:     for (int i = 0; i < dim; i++) {
//CHECK-NEXT:         _t0++;
//CHECK-NEXT:         t += clad::push(_t6, (x[clad::push(_t2, i)] - p[clad::push(_t4, i)])) * clad::push(_t1, (x[clad::push(_t7, i)] - p[clad::push(_t9, i)]));
//CHECK-NEXT:     }
//CHECK-NEXT:     _t12 = -t;
//CHECK-NEXT:     _t14 = sigma;
//CHECK-NEXT:     _t15 = 2 * _t14;
//CHECK-NEXT:     _t13 = sigma;
//CHECK-NEXT:     _t11 = (_t15 * _t13);
//CHECK-NEXT:     t = _t12 / _t11;
//CHECK-NEXT:     _t18 = 2 * 3.1415926535897931;
//CHECK-NEXT:     _t19 = -dim / 2.;
//CHECK-NEXT:     _t20 = std::pow(_t18, _t19);
//CHECK-NEXT:     _t21 = sigma;
//CHECK-NEXT:     _t17 = std::pow(_t21, -0.5);
//CHECK-NEXT:     _t22 = _t20 * _t17;
//CHECK-NEXT:     _t23 = t;
//CHECK-NEXT:     _t16 = std::exp(_t23);
//CHECK-NEXT:     {
//CHECK-NEXT:         double _r8 = 1 * _t16;
//CHECK-NEXT:         double _r9 = _r8 * _t17;
//CHECK-NEXT:         double _grad0 = 0.;
//CHECK-NEXT:         double _grad1 = 0.;
//CHECK-NEXT:         clad::custom_derivatives{{(::std)?}}::pow_pullback(_t18, _t19, _r9, &_grad0, &_grad1);
//CHECK-NEXT:         double _r10 = _grad0;
//CHECK-NEXT:         double _r11 = _r10 * 3.1415926535897931;
//CHECK-NEXT:         double _r12 = _grad1;
//CHECK-NEXT:         double _r13 = _r12 / 2.;
//CHECK-NEXT:         _d_dim += -_r13;
//CHECK-NEXT:         double _r14 = _t20 * _r8;
//CHECK-NEXT:         double _grad2 = 0.;
//CHECK-NEXT:         double _grad3 = 0.;
//CHECK-NEXT:         clad::custom_derivatives{{(::std)?}}::pow_pullback(_t21, -0.5, _r14, &_grad2, &_grad3);
//CHECK-NEXT:         double _r15 = _grad2;
//CHECK-NEXT:         _d_sigma += _r15;
//CHECK-NEXT:         double _r16 = _grad3;
//CHECK-NEXT:         double _r17 = _t22 * 1;
//CHECK-NEXT:         double _r18 = _r17 * clad::custom_derivatives::exp_pushforward(_t23, 1.).pushforward;
//CHECK-NEXT:         _d_t += _r18;
//CHECK-NEXT:     }
//CHECK-NEXT:     {
//CHECK-NEXT:         double _r_d1 = _d_t;
//CHECK-NEXT:         double _r2 = _r_d1 / _t11;
//CHECK-NEXT:         _d_t += -_r2;
//CHECK-NEXT:         double _r3 = _r_d1 * -_t12 / (_t11 * _t11);
//CHECK-NEXT:         double _r4 = _r3 * _t13;
//CHECK-NEXT:         double _r5 = _r4 * _t14;
//CHECK-NEXT:         double _r6 = 2 * _r4;
//CHECK-NEXT:         _d_sigma += _r6;
//CHECK-NEXT:         double _r7 = _t15 * _r3;
//CHECK-NEXT:         _d_sigma += _r7;
//CHECK-NEXT:         _d_t -= _r_d1;
//CHECK-NEXT:     }
//CHECK-NEXT:     for (; _t0; _t0--) {
//CHECK-NEXT:         double _r_d0 = _d_t;
//CHECK-NEXT:         _d_t += _r_d0;
//CHECK-NEXT:         double _r0 = _r_d0 * clad::pop(_t1);
//CHECK-NEXT:         int _t3 = clad::pop(_t2);
//CHECK-NEXT:         int _t5 = clad::pop(_t4);
//CHECK-NEXT:         _d_p[_t5] += -_r0;
//CHECK-NEXT:         double _r1 = clad::pop(_t6) * _r_d0;
//CHECK-NEXT:         int _t8 = clad::pop(_t7);
//CHECK-NEXT:         int _t10 = clad::pop(_t9);
//CHECK-NEXT:         _d_p[_t10] += -_r1;
//CHECK-NEXT:         _d_t -= _r_d0;
//CHECK-NEXT:     }
//CHECK-NEXT: }

__global__ void compute(decltype(gauss_g) grad, double* d_x, double* d_p, int n, double* d_result) {
  grad.execute(d_x, d_p, 2.0, n, d_result);
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
  double *result, *d_result;

  result = (double*)malloc(N * sizeof(double));
  cudaMalloc(&d_result, N * sizeof(double));

  compute<<<1, 1>>>(gauss_g, d_x, d_p, N, d_result);
  cudaDeviceSynchronize();

  cudaMemcpy(result, d_result, N * sizeof(double), cudaMemcpyDeviceToHost);
  printf("%f,%f,%f\n", result[0], result[1], result[2]);
}
