// RUN: %cladclang %s -I%S/../../include -fopenmp -oOpenMP.out 2>&1 | %filecheck %s
// RUN: ./OpenMP.out | %filecheck_exec %s
// REQUIRES: OpenMP

#include "clad/Differentiator/Differentiator.h"

double fn1(const double *x, int n) {
  double total = 0.0;

  #pragma omp parallel for reduction(+:total)
  for (int i = 1; i < n; i++) {
    total += x[i];
  }

  return total;
}

// CHECK:      void fn1_grad(const double *x, int n, double *_d_x, int *_d_n) {
// CHECK-NEXT:          double _d_total = 0.;
// CHECK-NEXT:          double total = 0.;
// CHECK-NEXT:          #pragma omp parallel reduction(+: total)
// CHECK-NEXT:              {
// CHECK-NEXT:                  int _t_chunklo0 = 0;
// CHECK-NEXT:                  int _t_chunkhi0 = 0;
// CHECK-NEXT:                  clad::GetStaticSchedule(1, n - 1, 1, &_t_chunklo0, &_t_chunkhi0);
// CHECK-NEXT:                  for (int i = _t_chunklo0; i <= _t_chunkhi0; i += 1) {
// CHECK-NEXT:                      total += x[i];
// CHECK-NEXT:                  }
// CHECK-NEXT:              }
// CHECK-NEXT:          _d_total += 1;
// CHECK-NEXT:          #pragma omp parallel private(total) firstprivate(_d_total)
// CHECK-NEXT:              {
// CHECK-NEXT:                  int _t_chunklo1 = 0;
// CHECK-NEXT:                  int _t_chunkhi1 = 0;
// CHECK-NEXT:                  clad::GetStaticSchedule(1, n - 1, 1, &_t_chunklo1, &_t_chunkhi1);
// CHECK-NEXT:                  for (int i = _t_chunkhi1; i >= _t_chunklo1; i -= 1) {
// CHECK-NEXT:                      {
// CHECK-NEXT:                          double _r_d0 = _d_total;
// CHECK-NEXT:                          _d_x[i] += _r_d0;
// CHECK-NEXT:                      }
// CHECK-NEXT:                  }
// CHECK-NEXT:              }
// CHECK-NEXT:      }

void fn2(const double *x, int n, double *y) {
  #pragma omp parallel for
  for (int i = 1; i <= n; i++) {
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

// Test different loop initialization forms
void fn3(const double *x, int n, double *y) {
  int i;
  #pragma omp parallel for
  for (i = 0; i < n; i++) {
    y[i] = x[i] * 2.0;
  }
}

// CHECK:  void fn3_grad(const double *x, int n, double *y, double *_d_x, int *_d_n, double *_d_y) {
// CHECK-NEXT:      int _d_i = 0;
// CHECK-NEXT:      int i;
// CHECK-NEXT:      #pragma omp parallel
// CHECK-NEXT:          {
// CHECK-NEXT:              int _t_chunklo0 = 0;
// CHECK-NEXT:              int _t_chunkhi0 = 0;
// CHECK-NEXT:              clad::GetStaticSchedule(0, n - 1, 1, &_t_chunklo0, &_t_chunkhi0);
// CHECK-NEXT:              for (int i0 = _t_chunklo0; i0 <= _t_chunkhi0; i0 += 1) {
// CHECK-NEXT:                  y[i0] = x[i0] * 2.;
// CHECK-NEXT:              }
// CHECK-NEXT:          }
// CHECK-NEXT:      #pragma omp parallel
// CHECK-NEXT:          {
// CHECK-NEXT:              int _t_chunklo1 = 0;
// CHECK-NEXT:              int _t_chunkhi1 = 0;
// CHECK-NEXT:              clad::GetStaticSchedule(0, n - 1, 1, &_t_chunklo1, &_t_chunkhi1);
// CHECK-NEXT:              for (int i0 = _t_chunkhi1; i0 >= _t_chunklo1; i0 -= 1) {
// CHECK-NEXT:                  {
// CHECK-NEXT:                      double _r_d0 = _d_y[i0];
// CHECK-NEXT:                      _d_y[i0] = 0.;
// CHECK-NEXT:                      _d_x[i0] += _r_d0 * 2.;
// CHECK-NEXT:                  }
// CHECK-NEXT:              }
// CHECK-NEXT:          }
// CHECK-NEXT:  }

// Test loop with <= condition
void fn4(const double *x, int n, double *y) {
  #pragma omp parallel for
  for (int i = 1; i <= n; i++) {
    y[i] = x[i] + x[i];
  }
}

// CHECK:  void fn4_grad(const double *x, int n, double *y, double *_d_x, int *_d_n, double *_d_y) {
// CHECK-NEXT:      #pragma omp parallel
// CHECK-NEXT:          {
// CHECK-NEXT:              int _t_chunklo0 = 0;
// CHECK-NEXT:              int _t_chunkhi0 = 0;
// CHECK-NEXT:              clad::GetStaticSchedule(1, n, 1, &_t_chunklo0, &_t_chunkhi0);
// CHECK-NEXT:              for (int i = _t_chunklo0; i <= _t_chunkhi0; i += 1) {
// CHECK-NEXT:                  y[i] = x[i] + x[i];
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
// CHECK-NEXT:                      _d_x[i] += _r_d0;
// CHECK-NEXT:                      _d_x[i] += _r_d0;
// CHECK-NEXT:                  }
// CHECK-NEXT:              }
// CHECK-NEXT:          }
// CHECK-NEXT:  }

// Test loop with reversed condition (ub > var)
void fn5(const double *x, int n, double *y) {
  #pragma omp parallel for
  for (int i = 0; n > i; i++) {
    y[i] = x[i] * x[i];
  }
}

// CHECK:  void fn5_grad(const double *x, int n, double *y, double *_d_x, int *_d_n, double *_d_y) {
// CHECK-NEXT:      #pragma omp parallel
// CHECK-NEXT:          {
// CHECK-NEXT:              int _t_chunklo0 = 0;
// CHECK-NEXT:              int _t_chunkhi0 = 0;
// CHECK-NEXT:              clad::GetStaticSchedule(0, n - 1, 1, &_t_chunklo0, &_t_chunkhi0);
// CHECK-NEXT:              for (int i = _t_chunklo0; i <= _t_chunkhi0; i += 1) {
// CHECK-NEXT:                  y[i] = x[i] * x[i];
// CHECK-NEXT:              }
// CHECK-NEXT:          }
// CHECK-NEXT:      #pragma omp parallel
// CHECK-NEXT:          {
// CHECK-NEXT:              int _t_chunklo1 = 0;
// CHECK-NEXT:              int _t_chunkhi1 = 0;
// CHECK-NEXT:              clad::GetStaticSchedule(0, n - 1, 1, &_t_chunklo1, &_t_chunkhi1);
// CHECK-NEXT:              for (int i = _t_chunkhi1; i >= _t_chunklo1; i -= 1) {
// CHECK-NEXT:                  {
// CHECK-NEXT:                      double _r_d0 = _d_y[i];
// CHECK-NEXT:                      _d_y[i] = 0.;
// CHECK-NEXT:                      _d_x[i] += _r_d0 * x[i];
// CHECK-NEXT:                      _d_x[i] += x[i] * _r_d0;
// CHECK-NEXT:                  }
// CHECK-NEXT:              }
// CHECK-NEXT:          }
// CHECK-NEXT:  }

// Test loop with reversed condition (ub >= var)
void fn6(const double *x, int n, double *y) {
  #pragma omp parallel for
  for (int i = 1; n >= i; i++) {
    y[i] = x[i] * 3.0;
  }
}

// CHECK:  void fn6_grad(const double *x, int n, double *y, double *_d_x, int *_d_n, double *_d_y) {
// CHECK-NEXT:      #pragma omp parallel
// CHECK-NEXT:          {
// CHECK-NEXT:              int _t_chunklo0 = 0;
// CHECK-NEXT:              int _t_chunkhi0 = 0;
// CHECK-NEXT:              clad::GetStaticSchedule(1, n, 1, &_t_chunklo0, &_t_chunkhi0);
// CHECK-NEXT:              for (int i = _t_chunklo0; i <= _t_chunkhi0; i += 1) {
// CHECK-NEXT:                  y[i] = x[i] * 3.;
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
// CHECK-NEXT:                      _d_x[i] += _r_d0 * 3.;
// CHECK-NEXT:                  }
// CHECK-NEXT:              }
// CHECK-NEXT:          }
// CHECK-NEXT:  }

// Test decrement loop with --i
void fn7(const double *x, int n, double *y) {
  #pragma omp parallel for
  for (int i = n; i > 0; --i) {
    y[i] = x[i] * x[i];
  }
}

// CHECK:  void fn7_grad(const double *x, int n, double *y, double *_d_x, int *_d_n, double *_d_y) {
// CHECK-NEXT:      #pragma omp parallel
// CHECK-NEXT:          {
// CHECK-NEXT:              int _t_chunklo0 = 0;
// CHECK-NEXT:              int _t_chunkhi0 = 0;
// CHECK-NEXT:              clad::GetStaticSchedule(n, 0 + 1, -1, &_t_chunklo0, &_t_chunkhi0);
// CHECK-NEXT:              for (int i = _t_chunklo0; i >= _t_chunkhi0; i += -1) {
// CHECK-NEXT:                  y[i] = x[i] * x[i];
// CHECK-NEXT:              }
// CHECK-NEXT:          }
// CHECK-NEXT:      #pragma omp parallel
// CHECK-NEXT:          {
// CHECK-NEXT:              int _t_chunklo1 = 0;
// CHECK-NEXT:              int _t_chunkhi1 = 0;
// CHECK-NEXT:              clad::GetStaticSchedule(n, 0 + 1, -1, &_t_chunklo1, &_t_chunkhi1);
// CHECK-NEXT:              for (int i = _t_chunkhi1; i <= _t_chunklo1; i -= -1) {
// CHECK-NEXT:                  {
// CHECK-NEXT:                      double _r_d0 = _d_y[i];
// CHECK-NEXT:                      _d_y[i] = 0.;
// CHECK-NEXT:                      _d_x[i] += _r_d0 * x[i];
// CHECK-NEXT:                      _d_x[i] += x[i] * _r_d0;
// CHECK-NEXT:                  }
// CHECK-NEXT:              }
// CHECK-NEXT:          }
// CHECK-NEXT:  }

// Test decrement loop with i--
void fn8(const double *x, int n, double *y) {
  #pragma omp parallel for
  for (int i = n; i >= 1; i--) {
    y[i] = x[i] + 1.0;
  }
}

// CHECK:  void fn8_grad(const double *x, int n, double *y, double *_d_x, int *_d_n, double *_d_y) {
// CHECK-NEXT:      #pragma omp parallel
// CHECK-NEXT:          {
// CHECK-NEXT:              int _t_chunklo0 = 0;
// CHECK-NEXT:              int _t_chunkhi0 = 0;
// CHECK-NEXT:              clad::GetStaticSchedule(n, 1, -1, &_t_chunklo0, &_t_chunkhi0);
// CHECK-NEXT:              for (int i = _t_chunklo0; i >= _t_chunkhi0; i += -1) {
// CHECK-NEXT:                  y[i] = x[i] + 1.;
// CHECK-NEXT:              }
// CHECK-NEXT:          }
// CHECK-NEXT:      #pragma omp parallel
// CHECK-NEXT:          {
// CHECK-NEXT:              int _t_chunklo1 = 0;
// CHECK-NEXT:              int _t_chunkhi1 = 0;
// CHECK-NEXT:              clad::GetStaticSchedule(n, 1, -1, &_t_chunklo1, &_t_chunkhi1);
// CHECK-NEXT:              for (int i = _t_chunkhi1; i <= _t_chunklo1; i -= -1) {
// CHECK-NEXT:                  {
// CHECK-NEXT:                      double _r_d0 = _d_y[i];
// CHECK-NEXT:                      _d_y[i] = 0.;
// CHECK-NEXT:                      _d_x[i] += _r_d0;
// CHECK-NEXT:                  }
// CHECK-NEXT:              }
// CHECK-NEXT:          }
// CHECK-NEXT:  }

// Test loop with += stride
void fn9(const double *x, int n, double *y) {
  #pragma omp parallel for
  for (int i = 0; i < n; i += 2) {
    y[i] = x[i] * 2.0;
  }
}

// CHECK:  void fn9_grad(const double *x, int n, double *y, double *_d_x, int *_d_n, double *_d_y) {
// CHECK-NEXT:      #pragma omp parallel
// CHECK-NEXT:          {
// CHECK-NEXT:              int _t_chunklo0 = 0;
// CHECK-NEXT:              int _t_chunkhi0 = 0;
// CHECK-NEXT:              clad::GetStaticSchedule(0, n - 1, 2, &_t_chunklo0, &_t_chunkhi0);
// CHECK-NEXT:              for (int i = _t_chunklo0; i <= _t_chunkhi0; i += 2) {
// CHECK-NEXT:                  y[i] = x[i] * 2.;
// CHECK-NEXT:              }
// CHECK-NEXT:          }
// CHECK-NEXT:      #pragma omp parallel
// CHECK-NEXT:          {
// CHECK-NEXT:              int _t_chunklo1 = 0;
// CHECK-NEXT:              int _t_chunkhi1 = 0;
// CHECK-NEXT:              clad::GetStaticSchedule(0, n - 1, 2, &_t_chunklo1, &_t_chunkhi1);
// CHECK-NEXT:              for (int i = _t_chunkhi1; i >= _t_chunklo1; i -= 2) {
// CHECK-NEXT:                  {
// CHECK-NEXT:                      double _r_d0 = _d_y[i];
// CHECK-NEXT:                      _d_y[i] = 0.;
// CHECK-NEXT:                      _d_x[i] += _r_d0 * 2.;
// CHECK-NEXT:                  }
// CHECK-NEXT:              }
// CHECK-NEXT:          }
// CHECK-NEXT:  }

// Test loop with -= stride
void fn10(const double *x, int n, double *y) {
  #pragma omp parallel for
  for (int i = n; i > 0; i -= 3) {
    y[i] = x[i] * 4.0;
  }
}

// CHECK:  void fn10_grad(const double *x, int n, double *y, double *_d_x, int *_d_n, double *_d_y) {
// CHECK-NEXT:      #pragma omp parallel
// CHECK-NEXT:          {
// CHECK-NEXT:              int _t_chunklo0 = 0;
// CHECK-NEXT:              int _t_chunkhi0 = 0;
// CHECK-NEXT:              clad::GetStaticSchedule(n, 0 + 1, -3, &_t_chunklo0, &_t_chunkhi0);
// CHECK-NEXT:              for (int i = _t_chunklo0; i >= _t_chunkhi0; i += -3) {
// CHECK-NEXT:                  y[i] = x[i] * 4.;
// CHECK-NEXT:              }
// CHECK-NEXT:          }
// CHECK-NEXT:      #pragma omp parallel
// CHECK-NEXT:          {
// CHECK-NEXT:              int _t_chunklo1 = 0;
// CHECK-NEXT:              int _t_chunkhi1 = 0;
// CHECK-NEXT:              clad::GetStaticSchedule(n, 0 + 1, -3, &_t_chunklo1, &_t_chunkhi1);
// CHECK-NEXT:              for (int i = _t_chunkhi1; i <= _t_chunklo1; i -= -3) {
// CHECK-NEXT:                  {
// CHECK-NEXT:                      double _r_d0 = _d_y[i];
// CHECK-NEXT:                      _d_y[i] = 0.;
// CHECK-NEXT:                      _d_x[i] += _r_d0 * 4.;
// CHECK-NEXT:                  }
// CHECK-NEXT:              }
// CHECK-NEXT:          }
// CHECK-NEXT:  }

// Test loop with i = i + stride
void fn11(const double *x, int n, double *y) {
  #pragma omp parallel for
  for (int i = 0; i < n; i = i + 2) {
    y[i] = x[i] * x[i];
  }
}

// CHECK:  void fn11_grad(const double *x, int n, double *y, double *_d_x, int *_d_n, double *_d_y) {
// CHECK-NEXT:      #pragma omp parallel
// CHECK-NEXT:          {
// CHECK-NEXT:              int _t_chunklo0 = 0;
// CHECK-NEXT:              int _t_chunkhi0 = 0;
// CHECK-NEXT:              clad::GetStaticSchedule(0, n - 1, 2, &_t_chunklo0, &_t_chunkhi0);
// CHECK-NEXT:              for (int i = _t_chunklo0; i <= _t_chunkhi0; i += 2) {
// CHECK-NEXT:                  y[i] = x[i] * x[i];
// CHECK-NEXT:              }
// CHECK-NEXT:          }
// CHECK-NEXT:      #pragma omp parallel
// CHECK-NEXT:          {
// CHECK-NEXT:              int _t_chunklo1 = 0;
// CHECK-NEXT:              int _t_chunkhi1 = 0;
// CHECK-NEXT:              clad::GetStaticSchedule(0, n - 1, 2, &_t_chunklo1, &_t_chunkhi1);
// CHECK-NEXT:              for (int i = _t_chunkhi1; i >= _t_chunklo1; i -= 2) {
// CHECK-NEXT:                  {
// CHECK-NEXT:                      double _r_d0 = _d_y[i];
// CHECK-NEXT:                      _d_y[i] = 0.;
// CHECK-NEXT:                      _d_x[i] += _r_d0 * x[i];
// CHECK-NEXT:                      _d_x[i] += x[i] * _r_d0;
// CHECK-NEXT:                  }
// CHECK-NEXT:              }
// CHECK-NEXT:          }
// CHECK-NEXT:  }

// Test loop with i = stride + i
void fn12(const double *x, int n, double *y) {
  #pragma omp parallel for
  for (int i = 0; i < n; i = 3 + i) {
    y[i] = x[i] * 5.0;
  }
}

// CHECK:  void fn12_grad(const double *x, int n, double *y, double *_d_x, int *_d_n, double *_d_y) {
// CHECK-NEXT:      #pragma omp parallel
// CHECK-NEXT:          {
// CHECK-NEXT:              int _t_chunklo0 = 0;
// CHECK-NEXT:              int _t_chunkhi0 = 0;
// CHECK-NEXT:              clad::GetStaticSchedule(0, n - 1, 3, &_t_chunklo0, &_t_chunkhi0);
// CHECK-NEXT:              for (int i = _t_chunklo0; i <= _t_chunkhi0; i += 3) {
// CHECK-NEXT:                  y[i] = x[i] * 5.;
// CHECK-NEXT:              }
// CHECK-NEXT:          }
// CHECK-NEXT:      #pragma omp parallel
// CHECK-NEXT:          {
// CHECK-NEXT:              int _t_chunklo1 = 0;
// CHECK-NEXT:              int _t_chunkhi1 = 0;
// CHECK-NEXT:              clad::GetStaticSchedule(0, n - 1, 3, &_t_chunklo1, &_t_chunkhi1);
// CHECK-NEXT:              for (int i = _t_chunkhi1; i >= _t_chunklo1; i -= 3) {
// CHECK-NEXT:                  {
// CHECK-NEXT:                      double _r_d0 = _d_y[i];
// CHECK-NEXT:                      _d_y[i] = 0.;
// CHECK-NEXT:                      _d_x[i] += _r_d0 * 5.;
// CHECK-NEXT:                  }
// CHECK-NEXT:              }
// CHECK-NEXT:          }
// CHECK-NEXT:  }

// Test loop with i = i - stride
void fn13(const double *x, int n, double *y) {
  #pragma omp parallel for
  for (int i = n; i > 0; i = i - 2) {
    y[i] = x[i] * x[i];
  }
}

// CHECK:  void fn13_grad(const double *x, int n, double *y, double *_d_x, int *_d_n, double *_d_y) {
// CHECK-NEXT:      #pragma omp parallel
// CHECK-NEXT:          {
// CHECK-NEXT:              int _t_chunklo0 = 0;
// CHECK-NEXT:              int _t_chunkhi0 = 0;
// CHECK-NEXT:              clad::GetStaticSchedule(n, 0 + 1, -2, &_t_chunklo0, &_t_chunkhi0);
// CHECK-NEXT:              for (int i = _t_chunklo0; i >= _t_chunkhi0; i += -2) {
// CHECK-NEXT:                  y[i] = x[i] * x[i];
// CHECK-NEXT:              }
// CHECK-NEXT:          }
// CHECK-NEXT:      #pragma omp parallel
// CHECK-NEXT:          {
// CHECK-NEXT:              int _t_chunklo1 = 0;
// CHECK-NEXT:              int _t_chunkhi1 = 0;
// CHECK-NEXT:              clad::GetStaticSchedule(n, 0 + 1, -2, &_t_chunklo1, &_t_chunkhi1);
// CHECK-NEXT:              for (int i = _t_chunkhi1; i <= _t_chunklo1; i -= -2) {
// CHECK-NEXT:                  {
// CHECK-NEXT:                      double _r_d0 = _d_y[i];
// CHECK-NEXT:                      _d_y[i] = 0.;
// CHECK-NEXT:                      _d_x[i] += _r_d0 * x[i];
// CHECK-NEXT:                      _d_x[i] += x[i] * _r_d0;
// CHECK-NEXT:                  }
// CHECK-NEXT:              }
// CHECK-NEXT:          }
// CHECK-NEXT:  }

// Test private clause
void fn14(const double *x, int n, double *y) {
  double temp = 0.0;
  #pragma omp parallel for private(temp)
  for (int i = 0; i < n; i++) {
    temp = x[i] * x[i];
    y[i] = temp;
  }
}

// CHECK:  void fn14_grad(const double *x, int n, double *y, double *_d_x, int *_d_n, double *_d_y) {
// CHECK-NEXT:      double _d_temp = 0.;
// CHECK-NEXT:      double temp = 0.;
// CHECK-NEXT:      #pragma omp parallel private(temp)
// CHECK-NEXT:          {
// CHECK-NEXT:              int _t_chunklo0 = 0;
// CHECK-NEXT:              int _t_chunkhi0 = 0;
// CHECK-NEXT:              clad::GetStaticSchedule(0, n - 1, 1, &_t_chunklo0, &_t_chunkhi0);
// CHECK-NEXT:              for (int i = _t_chunklo0; i <= _t_chunkhi0; i += 1) {
// CHECK-NEXT:                  temp = x[i] * x[i];
// CHECK-NEXT:                  y[i] = temp;
// CHECK-NEXT:              }
// CHECK-NEXT:          }
// CHECK-NEXT:      #pragma omp parallel private(temp) private(_d_temp)
// CHECK-NEXT:          {
// CHECK-NEXT:              int _t_chunklo1 = 0;
// CHECK-NEXT:              int _t_chunkhi1 = 0;
// CHECK-NEXT:              clad::GetStaticSchedule(0, n - 1, 1, &_t_chunklo1, &_t_chunkhi1);
// CHECK-NEXT:              for (int i = _t_chunkhi1; i >= _t_chunklo1; i -= 1) {
// CHECK-NEXT:                  {
// CHECK-NEXT:                      double _r_d1 = _d_y[i];
// CHECK-NEXT:                      _d_y[i] = 0.;
// CHECK-NEXT:                      _d_temp += _r_d1;
// CHECK-NEXT:                  }
// CHECK-NEXT:                  {
// CHECK-NEXT:                      double _r_d0 = _d_temp;
// CHECK-NEXT:                      _d_temp = 0.;
// CHECK-NEXT:                      _d_x[i] += _r_d0 * x[i];
// CHECK-NEXT:                      _d_x[i] += x[i] * _r_d0;
// CHECK-NEXT:                  }
// CHECK-NEXT:              }
// CHECK-NEXT:          }
// CHECK-NEXT:  }

// Test firstprivate clause
void fn15(const double *x, int n, double *y) {
  double scale = 2.0;
  #pragma omp parallel for firstprivate(scale)
  for (int i = 0; i < n; i++) {
    y[i] = x[i] * scale;
  }
}

// CHECK:  void fn15_grad(const double *x, int n, double *y, double *_d_x, int *_d_n, double *_d_y) {
// CHECK-NEXT:      double _d_scale = 0.;
// CHECK-NEXT:      double scale = 2.;
// CHECK-NEXT:      #pragma omp parallel firstprivate(scale)
// CHECK-NEXT:          {
// CHECK-NEXT:              int _t_chunklo0 = 0;
// CHECK-NEXT:              int _t_chunkhi0 = 0;
// CHECK-NEXT:              clad::GetStaticSchedule(0, n - 1, 1, &_t_chunklo0, &_t_chunkhi0);
// CHECK-NEXT:              for (int i = _t_chunklo0; i <= _t_chunkhi0; i += 1) {
// CHECK-NEXT:                  y[i] = x[i] * scale;
// CHECK-NEXT:              }
// CHECK-NEXT:          }
// CHECK-NEXT:      #pragma omp parallel firstprivate(scale) reduction(+: _d_scale)
// CHECK-NEXT:          {
// CHECK-NEXT:              int _t_chunklo1 = 0;
// CHECK-NEXT:              int _t_chunkhi1 = 0;
// CHECK-NEXT:              clad::GetStaticSchedule(0, n - 1, 1, &_t_chunklo1, &_t_chunkhi1);
// CHECK-NEXT:              for (int i = _t_chunkhi1; i >= _t_chunklo1; i -= 1) {
// CHECK-NEXT:                  {
// CHECK-NEXT:                      double _r_d0 = _d_y[i];
// CHECK-NEXT:                      _d_y[i] = 0.;
// CHECK-NEXT:                      _d_x[i] += _r_d0 * scale;
// CHECK-NEXT:                      _d_scale += x[i] * _r_d0;
// CHECK-NEXT:                  }
// CHECK-NEXT:              }
// CHECK-NEXT:          }
// CHECK-NEXT:  }

// Test shared clause
void fn16(const double *x, int n, double *y) {
  double sum = 0.0;
  #pragma omp parallel for shared(sum)
  for (int i = 0; i < n; i++) {
    y[i] = x[i] * x[i];
  }
}

// CHECK:  void fn16_grad(const double *x, int n, double *y, double *_d_x, int *_d_n, double *_d_y) {
// CHECK-NEXT:      double _d_sum = 0.;
// CHECK-NEXT:      double sum = 0.;
// CHECK-NEXT:      #pragma omp parallel shared(sum)
// CHECK-NEXT:          {
// CHECK-NEXT:              int _t_chunklo0 = 0;
// CHECK-NEXT:              int _t_chunkhi0 = 0;
// CHECK-NEXT:              clad::GetStaticSchedule(0, n - 1, 1, &_t_chunklo0, &_t_chunkhi0);
// CHECK-NEXT:              for (int i = _t_chunklo0; i <= _t_chunkhi0; i += 1) {
// CHECK-NEXT:                  y[i] = x[i] * x[i];
// CHECK-NEXT:              }
// CHECK-NEXT:          }
// CHECK-NEXT:      #pragma omp parallel shared(sum) reduction(+: _d_sum)
// CHECK-NEXT:          {
// CHECK-NEXT:              int _t_chunklo1 = 0;
// CHECK-NEXT:              int _t_chunkhi1 = 0;
// CHECK-NEXT:              clad::GetStaticSchedule(0, n - 1, 1, &_t_chunklo1, &_t_chunkhi1);
// CHECK-NEXT:              for (int i = _t_chunkhi1; i >= _t_chunklo1; i -= 1) {
// CHECK-NEXT:                  {
// CHECK-NEXT:                      double _r_d0 = _d_y[i];
// CHECK-NEXT:                      _d_y[i] = 0.;
// CHECK-NEXT:                      _d_x[i] += _r_d0 * x[i];
// CHECK-NEXT:                      _d_x[i] += x[i] * _r_d0;
// CHECK-NEXT:                  }
// CHECK-NEXT:              }
// CHECK-NEXT:          }
// CHECK-NEXT:  }

// Test multiple clauses combined
void fn17(const double *x, int n, double *y) {
  double temp = 0.0;
  double scale = 3.0;
  
  #pragma omp parallel for private(temp) firstprivate(scale)
  for (int i = 0; i < n; i++) {
    temp = x[i] * scale;
    y[i] = temp * temp;
  }
}

// CHECK:  void fn17_grad(const double *x, int n, double *y, double *_d_x, int *_d_n, double *_d_y) {
// CHECK-NEXT:      static clad::tape<double> _t0 = {};
// CHECK-NEXT:      #pragma omp threadprivate(_t0);
// CHECK-NEXT:      double _d_temp = 0.;
// CHECK-NEXT:      double temp = 0.;
// CHECK-NEXT:      double _d_scale = 0.;
// CHECK-NEXT:      double scale = 3.;
// CHECK-NEXT:      #pragma omp parallel private(temp) firstprivate(scale)
// CHECK-NEXT:          {
// CHECK-NEXT:              int _t_chunklo0 = 0;
// CHECK-NEXT:              int _t_chunkhi0 = 0;
// CHECK-NEXT:              clad::GetStaticSchedule(0, n - 1, 1, &_t_chunklo0, &_t_chunkhi0);
// CHECK-NEXT:              for (int i = _t_chunklo0; i <= _t_chunkhi0; i += 1) {
// CHECK-NEXT:                  clad::push(_t0, temp);
// CHECK-NEXT:                  temp = x[i] * scale;
// CHECK-NEXT:                  y[i] = temp * temp;
// CHECK-NEXT:              }
// CHECK-NEXT:              clad::push(_t0, temp);
// CHECK-NEXT:          }
// CHECK-NEXT:      #pragma omp parallel private(temp) private(_d_temp) firstprivate(scale) reduction(+: _d_scale)
// CHECK-NEXT:          {
// CHECK-NEXT:              temp = clad::pop(_t0);
// CHECK-NEXT:              int _t_chunklo1 = 0;
// CHECK-NEXT:              int _t_chunkhi1 = 0;
// CHECK-NEXT:              clad::GetStaticSchedule(0, n - 1, 1, &_t_chunklo1, &_t_chunkhi1);
// CHECK-NEXT:              for (int i = _t_chunkhi1; i >= _t_chunklo1; i -= 1) {
// CHECK-NEXT:                  {
// CHECK-NEXT:                      double _r_d1 = _d_y[i];
// CHECK-NEXT:                      _d_y[i] = 0.;
// CHECK-NEXT:                      _d_temp += _r_d1 * temp;
// CHECK-NEXT:                      _d_temp += temp * _r_d1;
// CHECK-NEXT:                  }
// CHECK-NEXT:                  {
// CHECK-NEXT:                      temp = clad::pop(_t0);
// CHECK-NEXT:                      double _r_d0 = _d_temp;
// CHECK-NEXT:                      _d_temp = 0.;
// CHECK-NEXT:                      _d_x[i] += _r_d0 * scale;
// CHECK-NEXT:                      _d_scale += x[i] * _r_d0;
// CHECK-NEXT:                  }
// CHECK-NEXT:              }
// CHECK-NEXT:          }
// CHECK-NEXT:  }

// Test nested structure in loop body
void fn18(const double *x, int n, double *y) {
  #pragma omp parallel for
  for (int i = 0; i < n; i++) {
    double a = x[i];
    double b = a * a;
    if (b > 0) {
      y[i] = b + a;
    } else {
      y[i] = b - a;
    }
  }
}

// CHECK:  void fn18_grad(const double *x, int n, double *y, double *_d_x, int *_d_n, double *_d_y) {
// CHECK-NEXT:      static clad::tape<double> _t0 = {};
// CHECK-NEXT:      #pragma omp threadprivate(_t0);
// CHECK-NEXT:      static double _d_a = 0.;
// CHECK-NEXT:      #pragma omp threadprivate(_d_a);
// CHECK-NEXT:      static double a = 0.;
// CHECK-NEXT:      #pragma omp threadprivate(a);
// CHECK-NEXT:      static double _d_b = 0.;
// CHECK-NEXT:      #pragma omp threadprivate(_d_b);
// CHECK-NEXT:      static double b = 0.;
// CHECK-NEXT:      #pragma omp threadprivate(b);
// CHECK-NEXT:      static clad::tape<bool> _cond0 = {};
// CHECK-NEXT:      #pragma omp threadprivate(_cond0);
// CHECK-NEXT:      #pragma omp parallel
// CHECK-NEXT:          {
// CHECK-NEXT:              int _t_chunklo0 = 0;
// CHECK-NEXT:              int _t_chunkhi0 = 0;
// CHECK-NEXT:              clad::GetStaticSchedule(0, n - 1, 1, &_t_chunklo0, &_t_chunkhi0);
// CHECK-NEXT:              for (int i = _t_chunklo0; i <= _t_chunkhi0; i += 1) {
// CHECK-NEXT:                  clad::push(_t0, a) , a = x[i];
// CHECK-NEXT:                  b = a * a;
// CHECK-NEXT:                  {
// CHECK-NEXT:                      clad::push(_cond0, b > 0);
// CHECK-NEXT:                      if (clad::back(_cond0)) {
// CHECK-NEXT:                          y[i] = b + a;
// CHECK-NEXT:                      } else {
// CHECK-NEXT:                          y[i] = b - a;
// CHECK-NEXT:                      }
// CHECK-NEXT:                  }
// CHECK-NEXT:              }
// CHECK-NEXT:          }
// CHECK-NEXT:      #pragma omp parallel
// CHECK-NEXT:          {
// CHECK-NEXT:              int _t_chunklo1 = 0;
// CHECK-NEXT:              int _t_chunkhi1 = 0;
// CHECK-NEXT:              clad::GetStaticSchedule(0, n - 1, 1, &_t_chunklo1, &_t_chunkhi1);
// CHECK-NEXT:              for (int i = _t_chunkhi1; i >= _t_chunklo1; i -= 1) {
// CHECK-NEXT:                  {
// CHECK-NEXT:                      if (clad::back(_cond0)) {
// CHECK-NEXT:                          {
// CHECK-NEXT:                              double _r_d0 = _d_y[i];
// CHECK-NEXT:                              _d_y[i] = 0.;
// CHECK-NEXT:                              _d_b += _r_d0;
// CHECK-NEXT:                              _d_a += _r_d0;
// CHECK-NEXT:                          }
// CHECK-NEXT:                      } else {
// CHECK-NEXT:                          {
// CHECK-NEXT:                              double _r_d1 = _d_y[i];
// CHECK-NEXT:                              _d_y[i] = 0.;
// CHECK-NEXT:                              _d_b += _r_d1;
// CHECK-NEXT:                              _d_a += -_r_d1;
// CHECK-NEXT:                          }
// CHECK-NEXT:                      }
// CHECK-NEXT:                      clad::pop(_cond0);
// CHECK-NEXT:                  }
// CHECK-NEXT:                  {
// CHECK-NEXT:                      _d_a += _d_b * a;
// CHECK-NEXT:                      _d_a += a * _d_b;
// CHECK-NEXT:                      _d_b = 0.;
// CHECK-NEXT:                  }
// CHECK-NEXT:                  {
// CHECK-NEXT:                      _d_x[i] += _d_a;
// CHECK-NEXT:                      _d_a = 0.;
// CHECK-NEXT:                      a = clad::pop(_t0);
// CHECK-NEXT:                  }
// CHECK-NEXT:              }
// CHECK-NEXT:          }
// CHECK-NEXT:  }

// Test complex loop bounds
void fn19(const double *x, int start, int end, double *y) {
  #pragma omp parallel for
  for (int i = start + 1; i < end - 1; i++) {
    y[i] = x[i] * x[i];
  }
}

// CHECK:  void fn19_grad(const double *x, int start, int end, double *y, double *_d_x, int *_d_start, int *_d_end, double *_d_y) {
// CHECK-NEXT:      #pragma omp parallel
// CHECK-NEXT:          {
// CHECK-NEXT:              int _t_chunklo0 = 0;
// CHECK-NEXT:              int _t_chunkhi0 = 0;
// CHECK-NEXT:              clad::GetStaticSchedule(start + 1, end - 1 - 1, 1, &_t_chunklo0, &_t_chunkhi0);
// CHECK-NEXT:              for (int i = _t_chunklo0; i <= _t_chunkhi0; i += 1) {
// CHECK-NEXT:                  y[i] = x[i] * x[i];
// CHECK-NEXT:              }
// CHECK-NEXT:          }
// CHECK-NEXT:      #pragma omp parallel
// CHECK-NEXT:          {
// CHECK-NEXT:              int _t_chunklo1 = 0;
// CHECK-NEXT:              int _t_chunkhi1 = 0;
// CHECK-NEXT:              clad::GetStaticSchedule(start + 1, end - 1 - 1, 1, &_t_chunklo1, &_t_chunkhi1);
// CHECK-NEXT:              for (int i = _t_chunkhi1; i >= _t_chunklo1; i -= 1) {
// CHECK-NEXT:                  {
// CHECK-NEXT:                      double _r_d0 = _d_y[i];
// CHECK-NEXT:                      _d_y[i] = 0.;
// CHECK-NEXT:                      _d_x[i] += _r_d0 * x[i];
// CHECK-NEXT:                      _d_x[i] += x[i] * _r_d0;
// CHECK-NEXT:                  }
// CHECK-NEXT:              }
// CHECK-NEXT:          }
// CHECK-NEXT:  }

template <size_t N>
void reset(double (&arr)[N], double val = 0) {
  for (size_t i = 0; i < N; ++i)
    arr[i] = val;
}

int main() {
  double x[] = {2, 3, 4, 5}, dx[4] = {0};
  int dn = 0;
  auto fn1_grad = clad::gradient(fn1);
  fn1_grad.execute(x, 4, dx, &dn);
  printf("{%.2f, %.2f, %.2f, %.2f}\n", dx[0], dx[1], dx[2], dx[3]); // CHECK-EXEC: {0.00, 1.00, 1.00, 1.00}

  reset(dx);
  double y[4] = {0}, dy[4] = {1, 1, 1, 1};
  auto fn2_grad = clad::gradient(fn2);
  fn2_grad.execute(x, 3, y, dx, &dn, dy);
  printf("{%.2f, %.2f, %.2f, %.2f}\n", dx[0], dx[1], dx[2], dx[3]); // CHECK-EXEC: {0.00, 108.00, 256.00, 500.00}

  reset(dx); reset(dy, 1);
  auto fn3_grad = clad::gradient(fn3);
  fn3_grad.execute(x, 4, y, dx, &dn, dy);
  printf("{%.2f, %.2f, %.2f, %.2f}\n", dx[0], dx[1], dx[2], dx[3]); // CHECK-EXEC: {2.00, 2.00, 2.00, 2.00}

  reset(dx); reset(dy, 1);
  auto fn4_grad = clad::gradient(fn4);
  fn4_grad.execute(x, 3, y, dx, &dn, dy);
  printf("{%.2f, %.2f, %.2f, %.2f}\n", dx[0], dx[1], dx[2], dx[3]); // CHECK-EXEC: {0.00, 2.00, 2.00, 2.00}
  
  reset(dx); reset(dy, 1);
  auto fn5_grad = clad::gradient(fn5);
  fn5_grad.execute(x, 4, y, dx, &dn, dy);
  printf("{%.2f, %.2f, %.2f, %.2f}\n", dx[0], dx[1], dx[2], dx[3]); // CHECK-EXEC: {4.00, 6.00, 8.00, 10.00}

  reset(dx); reset(dy, 1);
  auto fn6_grad = clad::gradient(fn6);
  fn6_grad.execute(x, 3, y, dx, &dn, dy);
  printf("{%.2f, %.2f, %.2f, %.2f}\n", dx[0], dx[1], dx[2], dx[3]); // CHECK-EXEC: {0.00, 3.00, 3.00, 3.00}
  
  reset(dx); reset(dy, 1);
  auto fn7_grad = clad::gradient(fn7);
  fn7_grad(x, 3, y, dx, &dn, dy);
  printf("{%.2f, %.2f, %.2f, %.2f}\n", dx[0], dx[1], dx[2], dx[3]); // CHECK-EXEC: {0.00, 6.00, 8.00, 10.00}
  
  reset(dx); reset(dy, 1);
  auto fn8_grad = clad::gradient(fn8);
  fn8_grad.execute(x, 3, y, dx, &dn, dy);
  printf("{%.2f, %.2f, %.2f, %.2f}\n", dx[0], dx[1], dx[2], dx[3]); // CHECK-EXEC: {0.00, 1.00, 1.00, 1.00}
  
  reset(dx); reset(dy, 1);
  auto fn9_grad = clad::gradient(fn9);
  fn9_grad.execute(x, 4, y, dx, &dn, dy);
  printf("{%.2f, %.2f, %.2f, %.2f}\n", dx[0], dx[1], dx[2], dx[3]); // CHECK-EXEC: {2.00, 0.00, 2.00, 0.00}
  
  reset(dx); reset(dy, 1);
  auto fn10_grad = clad::gradient(fn10);
  fn10_grad.execute(x, 3, y, dx, &dn, dy);
  printf("{%.2f, %.2f, %.2f, %.2f}\n", dx[0], dx[1], dx[2], dx[3]); // CHECK-EXEC: {0.00, 0.00, 0.00, 4.00}
  
  reset(dx); reset(dy, 1);
  auto fn11_grad = clad::gradient(fn11);
  fn11_grad.execute(x, 4, y, dx, &dn, dy);
  printf("{%.2f, %.2f, %.2f, %.2f}\n", dx[0], dx[1], dx[2], dx[3]); // CHECK-EXEC: {4.00, 0.00, 8.00, 0.00}
  
  reset(dx); reset(dy, 1);
  auto fn12_grad = clad::gradient(fn12);
  fn12_grad.execute(x, 4, y, dx, &dn, dy);
  printf("{%.2f, %.2f, %.2f, %.2f}\n", dx[0], dx[1], dx[2], dx[3]); // CHECK-EXEC: {5.00, 0.00, 0.00, 5.00}
  
  reset(dx); reset(dy, 1);
  auto fn13_grad = clad::gradient(fn13);
  fn13_grad.execute(x, 3, y, dx, &dn, dy);
  printf("{%.2f, %.2f, %.2f, %.2f}\n", dx[0], dx[1], dx[2], dx[3]); // CHECK-EXEC: {0.00, 6.00, 0.00, 10.00}
  
  reset(dx); reset(dy, 1);
  auto fn14_grad = clad::gradient(fn14);
  fn14_grad.execute(x, 4, y, dx, &dn, dy);
  printf("{%.2f, %.2f, %.2f, %.2f}\n", dx[0], dx[1], dx[2], dx[3]); // CHECK-EXEC: {4.00, 6.00, 8.00, 10.00}
  
  reset(dx); reset(dy, 1);
  auto fn15_grad = clad::gradient(fn15);
  fn15_grad(x, 4, y, dx, &dn, dy);
  printf("{%.2f, %.2f, %.2f, %.2f}\n", dx[0], dx[1], dx[2], dx[3]); // CHECK-EXEC: {2.00, 2.00, 2.00, 2.00}
    
  reset(dx); reset(dy, 1);
  auto fn16_grad = clad::gradient(fn16);
  fn16_grad.execute(x, 4, y, dx, &dn, dy);
  printf("{%.2f, %.2f, %.2f, %.2f}\n", dx[0], dx[1], dx[2], dx[3]); // CHECK-EXEC: {4.00, 6.00, 8.00, 10.00}
   
  reset(dx); reset(dy, 1);
  auto fn17_grad = clad::gradient(fn17);
  fn17_grad.execute(x, 4, y, dx, &dn, dy);
  printf("{%.2f, %.2f, %.2f, %.2f}\n", dx[0], dx[1], dx[2], dx[3]); // CHECK-EXEC: {36.00, 54.00, 72.00, 90.00}
   
  reset(dx); reset(dy, 1);
  auto fn18_grad = clad::gradient(fn18);
  fn18_grad.execute(x, 4, y, dx, &dn, dy);
  printf("{%.2f, %.2f, %.2f, %.2f}\n", dx[0], dx[1], dx[2], dx[3]); // CHECK-EXEC: {5.00, 7.00, 9.00, 11.00}
  
  reset(dx); reset(dy, 1);
  auto fn19_grad = clad::gradient(fn19);
  fn19_grad.execute(x, 0, 4, y, dx, &dn, &dn, dy);
  printf("{%.2f, %.2f, %.2f, %.2f}\n", dx[0], dx[1], dx[2], dx[3]); // CHECK-EXEC: {0.00, 6.00, 8.00, 0.00}
  return 0;
}