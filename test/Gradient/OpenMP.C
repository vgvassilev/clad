// RUN: %cladclang %s -I%S/../../include -fopenmp -fsyntax-only -oOpenMP.out 2>&1 | %filecheck %s

#include "clad/Differentiator/Differentiator.h"

double fn1(const double *x, int n) {
  double total = 0.0;

  #pragma omp parallel for reduction(+:total)
  for (int i = 1; i < n; i++) {
    total += x[i];
  }

  return total;
}

// CHECK:  void fn1_grad(const double *x, int n, double *_d_x, int *_d_n) {
// CHECK-NEXT:      double _d_total = 0.;
// CHECK-NEXT:      double total = 0.;
// CHECK-NEXT:      #pragma omp parallel reduction(+: total)
// CHECK-NEXT:          {
// CHECK-NEXT:              int _t_chunklo0 = 0;
// CHECK-NEXT:              int _t_chunkhi0 = 0;
// CHECK-NEXT:              clad::GetStaticSchedule(1, n, 1, &_t_chunklo0, &_t_chunkhi0);
// CHECK-NEXT:              for (int i = _t_chunklo0; i <= _t_chunkhi0; i += 1) {
// CHECK-NEXT:                  total += x[i];
// CHECK-NEXT:              }
// CHECK-NEXT:          }
// CHECK-NEXT:      _d_total += 1;
// CHECK-NEXT:      #pragma omp parallel firstprivate(_d_total)
// CHECK-NEXT:          {
// CHECK-NEXT:              int _t_chunklo1 = 0;
// CHECK-NEXT:              int _t_chunkhi1 = 0;
// CHECK-NEXT:              clad::GetStaticSchedule(1, n, 1, &_t_chunklo1, &_t_chunkhi1);
// CHECK-NEXT:              for (int i = _t_chunkhi1; i >= _t_chunklo1; i -= 1) {
// CHECK-NEXT:                  {
// CHECK-NEXT:                      double _r_d0 = _d_total;
// CHECK-NEXT:                      _d_x[i] += _r_d0;
// CHECK-NEXT:                  }
// CHECK-NEXT:              }
// CHECK-NEXT:          }
// CHECK-NEXT:  }

void fn2(const double *x, int n, double *y) {
  #pragma omp parallel for
  for (int i = 1; i < n; i++) {
    double t = x[i] * x[i];
    y[i] = t * t;
  }
}

// CHECK:  void fn2_grad(const double *x, int n, double *y, double *_d_x, int *_d_n, double *_d_y) {
// CHECK-NEXT:      static clad::tape<double> _t0 = {};
// CHECK-NEXT:      #pragma omp threadprivate(_t0);
// CHECK-NEXT:      static double _d_t = 0.;
// CHECK-NEXT:      #pragma omp threadprivate(_d_t);
// CHECK-NEXT:      static double t = 0.;
// CHECK-NEXT:      #pragma omp threadprivate(t);
// CHECK-NEXT:      #pragma omp parallel
// CHECK-NEXT:          {
// CHECK-NEXT:              int _t_chunklo0 = 0;
// CHECK-NEXT:              int _t_chunkhi0 = 0;
// CHECK-NEXT:              clad::GetStaticSchedule(1, n, 1, &_t_chunklo0, &_t_chunkhi0);
// CHECK-NEXT:              for (int i = _t_chunklo0; i <= _t_chunkhi0; i += 1) {
// CHECK-NEXT:                  clad::push(_t0, t) , t = x[i] * x[i];
// CHECK-NEXT:                  y[i] = t * t;
// CHECK-NEXT:              }
// CHECK-NEXT:          }
// CHECK-NEXT:      #pragma omp parallel
// CHECK-NEXT:          {
// CHECK-NEXT:              int _t_chunklo1 = 0;
// CHECK-NEXT:              int _t_chunkhi1 = 0;
// CHECK-NEXT:              clad::GetStaticSchedule(1, n, 1, &_t_chunklo1, &_t_chunkhi1);
// CHECK-NEXT:              for (int i = _t_chunkhi1; i >= _t_chunklo1; i -= 1) {
// CHECK-NEXT:                  {
// CHECK-NEXT:                      double _r_d0 = _d_y[i];
// CHECK-NEXT:                      _d_y[i] = 0.;
// CHECK-NEXT:                      _d_t += _r_d0 * t;
// CHECK-NEXT:                      _d_t += t * _r_d0;
// CHECK-NEXT:                  }
// CHECK-NEXT:                  {
// CHECK-NEXT:                      _d_x[i] += _d_t * x[i];
// CHECK-NEXT:                      _d_x[i] += x[i] * _d_t;
// CHECK-NEXT:                      _d_t = 0.;
// CHECK-NEXT:                      t = clad::pop(_t0);
// CHECK-NEXT:                  }
// CHECK-NEXT:              }
// CHECK-NEXT:          }
// CHECK-NEXT:  }

int main() {
  auto fn1_grad = clad::gradient(fn1);
  auto fn2_grad = clad::gradient(fn2);
  return 0;
}