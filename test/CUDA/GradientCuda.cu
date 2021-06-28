// The Test checks whether a clad gradient can be successfully be generated on the device having all the dependencies also as device functions 

// RUN: %cladclang_cuda -I%S/../../include  %s -fsyntax-only  --cuda-path=%cudapath  -Xclang -verify 2>&1 | FileCheck %s

// REQUIRES: cuda-runtime

// expected-no-diagnostics

//#include <iostream>
#include "clad/Differentiator/Differentiator.h"

#define N 3

__device__ __host__ double gaus(double* x, double* p, double sigma, int dim) {
   double t = 0;
   for (int i = 0; i< dim; i++)
       t += (x[i] - p[i]) * (x[i] - p[i]);
   t = -t / (2*sigma*sigma);
   return std::pow(2*M_PI, -dim/2.0) * std::pow(sigma, -0.5) * std::exp(t);
}

__device__ __host__ void
gaus_grad_1(double* x, double* p, double sigma, int dim, double* _d_p);

auto gaus_g = clad::gradient(gaus, "p");

// CHECK:    void gaus_grad_1(double *x, double *p, double sigma, int dim,
// double *_d_p) __attribute__((device)) __attribute__((host)) { CHECK-NEXT:
// double _d_t = 0; CHECK-NEXT:       unsigned long _t0; CHECK-NEXT:       int
// _d_i = 0; CHECK-NEXT:       clad::tape<double> _t1 = {}; CHECK-NEXT:
// clad::tape<int> _t2 = {}; CHECK-NEXT:       clad::tape<int> _t3 = {};
// CHECK-NEXT:       clad::tape<double> _t4 = {};
// CHECK-NEXT:       clad::tape<int> _t5 = {};
// CHECK-NEXT:       clad::tape<int> _t6 = {};
// CHECK-NEXT:       double _t7;
// CHECK-NEXT:       double _t8;
// CHECK-NEXT:       double _t9;
// CHECK-NEXT:       double _t10;
// CHECK-NEXT:       double _t11;
// CHECK-NEXT:       double _t12;
// CHECK-NEXT:       double _t13;
// CHECK-NEXT:       double _t14;
// CHECK-NEXT:       double _t15;
// CHECK-NEXT:       double _t16;
// CHECK-NEXT:       double _t17;
// CHECK-NEXT:       double _t18;
// CHECK-NEXT:       double _t19;
// CHECK-NEXT:       double t = 0;
// CHECK-NEXT:       _t0 = 0;
// CHECK-NEXT:       for (int i = 0; i < dim; i++) {
// CHECK-NEXT:           _t0++;
// CHECK-NEXT:           t += clad::push(_t4, (x[clad::push(_t2, i)] -
// p[clad::push(_t3, i)])) * clad::push(_t1, (x[clad::push(_t5, i)] -
// p[clad::push(_t6, i)])); CHECK-NEXT:       } CHECK-NEXT:       _t8 = -t;
// CHECK-NEXT:       _t10 = sigma;
// CHECK-NEXT:       _t11 = 2 * _t10;
// CHECK-NEXT:       _t9 = sigma;
// CHECK-NEXT:       _t7 = (_t11 * _t9);
// CHECK-NEXT:       t = _t8 / _t7;
// CHECK-NEXT:       _t14 = 2 * 3.1415926535897931;
// CHECK-NEXT:       _t15 = -dim / 2.;
// CHECK-NEXT:       _t16 = std::pow(_t14, _t15);
// CHECK-NEXT:       _t17 = sigma;
// CHECK-NEXT:       _t13 = std::pow(_t17, -0.5);
// CHECK-NEXT:       _t18 = _t16 * _t13;
// CHECK-NEXT:       _t19 = t;
// CHECK-NEXT:       _t12 = std::exp(_t19);
// CHECK-NEXT:       double gaus_return = _t18 * _t12;
// CHECK-NEXT:       goto _label0;
// CHECK-NEXT:     _label0:
// CHECK-NEXT:       {
// CHECK-NEXT:           double _r8 = 1 * _t12;
// CHECK-NEXT:           double _r9 = _r8 * _t13;
// CHECK-NEXT:           double _grad0 = 0.;
// CHECK-NEXT:           double _grad1 = 0.;
// CHECK-NEXT:           custom_derivatives::pow_grad(_t14, _t15, &_grad0,
// &_grad1); CHECK-NEXT:           double _r10 = _r9 * _grad0; CHECK-NEXT: double
// _r11 = _r10 * 3.1415926535897931; CHECK-NEXT:           double _r12 = _r9 *
// _grad1; CHECK-NEXT:           double _r13 = _r12 / 2.; CHECK-NEXT: double _r14
// = _t16 * _r8; CHECK-NEXT:           double _grad2 = 0.; CHECK-NEXT: double
// _grad3 = 0.; CHECK-NEXT:           custom_derivatives::pow_grad(_t17, -0.5,
// &_grad2, &_grad3); CHECK-NEXT:           double _r15 = _r14 * _grad2;
// CHECK-NEXT:           double _r16 = _r14 * _grad3;
// CHECK-NEXT:           double _r17 = _t18 * 1;
// CHECK-NEXT:           double _r18 = _r17 *
// custom_derivatives::exp_darg0(_t19); CHECK-NEXT:           _d_t += _r18;
// CHECK-NEXT:       }
// CHECK-NEXT:       {
// CHECK-NEXT:           double _r_d1 = _d_t;
// CHECK-NEXT:           double _r2 = _r_d1 / _t7;
// CHECK-NEXT:           _d_t += -_r2;
// CHECK-NEXT:           double _r3 = _r_d1 * -_t8 / (_t7 * _t7);
// CHECK-NEXT:           double _r4 = _r3 * _t9;
// CHECK-NEXT:           double _r5 = _r4 * _t10;
// CHECK-NEXT:           double _r6 = 2 * _r4;
// CHECK-NEXT:           double _r7 = _t11 * _r3;
// CHECK-NEXT:           _d_t -= _r_d1;
// CHECK-NEXT:       }
// CHECK-NEXT:       for (; _t0; _t0--) {
// CHECK-NEXT:           double _r_d0 = _d_t;
// CHECK-NEXT:           _d_t += _r_d0;
// CHECK-NEXT:           double _r0 = _r_d0 * clad::pop(_t1);
// CHECK-NEXT:           _d_p[clad::pop(_t3)] += -_r0;
// CHECK-NEXT:           double _r1 = clad::pop(_t4) * _r_d0;
// CHECK-NEXT:           _d_p[clad::pop(_t6)] += -_r1;
// CHECK-NEXT:           _d_t -= _r_d0;
// CHECK-NEXT:       }
// CHECK-NEXT:   }

__global__ void compute(double* d_x, double* d_p, int n) {
  gaus_grad_1(d_x, d_x, 2.0, n, d_p);
}

int main(void) {
    double *x, *d_x;

    x = (double*)malloc(N*sizeof(double));
    for (int i = 0; i < N; i++) {
        x[i] = 2.0;
    }

    cudaMalloc(&d_x, N*sizeof(double));
    cudaMemcpy(d_x, x, N*sizeof(double), cudaMemcpyHostToDevice);

    double *result, *d_result;

    result = (double*)malloc(N*sizeof(double));
    cudaMalloc(&d_result, N*sizeof(double));

    compute<<<1, 1>>>(d_x, d_result, N);
    cudaMemcpy(result, d_result, N*sizeof(double), cudaMemcpyDeviceToHost);
}