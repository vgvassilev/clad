// RUN: %cladclang %s -I%S/../../include -oLambdas.out 2>&1 | %filecheck %s
// RUN: ./Lambdas.out | %filecheck_exec %s
// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -enable-tbr %s -I%S/../../include -oLambdas.out
// RUN: ./Lambdas.out | %filecheck_exec %s
// CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"

double f1(double i, double j) {
  auto _f = [] (double t) {
    return t*t + 1.0;
  };
  return i + _f(j);
}

// CHECK:     inline void operator_call_pullback(double t, double _d_y, double *_d_t) const;
// CHECK-NEXT:     void f1_grad(double i, double j, double *_d_i, double *_d_j) {
// CHECK-NEXT:         auto _f = []{{ ?}}(double t) {
// CHECK-NEXT:             return t * t + 1.;
// CHECK-NEXT:         }{{;?}}
// CHECK:         {
// CHECK-NEXT:             *_d_i += 1;
// CHECK-NEXT:             double _r0 = 0;
// CHECK-NEXT:             _f.operator_call_pullback(j, 1, &_r0);
// CHECK-NEXT:             *_d_j += _r0;
// CHECK-NEXT:         }
// CHECK-NEXT:     }

double f2(double i, double j) {
  auto _f = [] (double t, double k) {
    return t + k;
  };
  double x = _f(i + j, i);
  return x;
}

// CHECK:     inline void operator_call_pullback(double t, double k, double _d_y, double *_d_t, double *_d_k) const;
// CHECK-NEXT:     void f2_grad(double i, double j, double *_d_i, double *_d_j) {
// CHECK-NEXT:             auto _f = []{{ ?}}(double t, double k) {
// CHECK-NEXT:                 return t + k;
// CHECK-NEXT:             }{{;?}}
// CHECK:        double _d_x = 0;
// CHECK-NEXT:             double x = operator()(i + j, i);
// CHECK-NEXT:             _d_x += 1;
// CHECK-NEXT:             {
// CHECK-NEXT:                 double _r0 = 0;
// CHECK-NEXT:                 double _r1 = 0;
// CHECK-NEXT:                 _f.operator_call_pullback(i + j, i, _d_x, &_r0, &_r1);
// CHECK-NEXT:                 *_d_i += _r0;
// CHECK-NEXT:                 *_d_j += _r0;
// CHECK-NEXT:                 *_d_i += _r1;
// CHECK-NEXT:             }
// CHECK-NEXT:         }


int main() {
  auto df1 = clad::gradient(f1);
  double di = 0, dj = 0;
  df1.execute(3, 4, &di, &dj);
  printf("%.2f %.2f\n", di, dj);              // CHECK-EXEC: 1.00 8.00

  auto df2 = clad::gradient(f2);
  di = 0, dj = 0;
  df2.execute(3, 4, &di, &dj);
  printf("%.2f %.2f\n", di, dj);              // CHECK-EXEC: 2.00 1.00
}

// CHECK:     inline void operator_call_pullback(double t, double _d_y, double *_d_t) const {
// CHECK-NEXT:         {
// CHECK-NEXT:             *_d_t += _d_y * t;
// CHECK-NEXT:             *_d_t += t * _d_y;
// CHECK-NEXT:         }
// CHECK-NEXT:     }
// CHECK-NEXT:     inline void operator_call_pullback(double t, double k, double _d_y, double *_d_t, double *_d_k) const {
// CHECK-NEXT:         {
// CHECK-NEXT:             *_d_t += _d_y;
// CHECK-NEXT:             *_d_k += _d_y;
// CHECK-NEXT:         }
// CHECK-NEXT:     }