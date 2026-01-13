// RUN: %cladclang %s -I%S/../../include -oLambdas.out 2>&1 | %filecheck %s
// RUN: ./Lambdas.out | %filecheck_exec %s
// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -disable-tbr -Xclang -plugin-arg-clad -Xclang -disable-va %s -I%S/../../include -oLambdas.out
// RUN: ./Lambdas.out | %filecheck_exec %s
// UNSUPPORTED: clang-11, clang-12, clang-13, clang-14, clang-15, clang-16

#include "clad/Differentiator/Differentiator.h"

double f1(double i, double j) {
  auto _f = [] (double t) {
    return t*t + 1.0;
  };
  return i + _f(j);
}

// CHECK: void f1_grad(double i, double j, double *_d_i, double *_d_j) {
// CHECK-NEXT:     auto _f0 = [](double t) {
// CHECK-NEXT:         return t * t + 1.;
// CHECK-NEXT:     };
// CHECK-NEXT:     auto _d__f = [](double t, double _d_y, double *_d_t) {
// CHECK-NEXT:         {
// CHECK-NEXT:             *_d_t += _d_y * t;
// CHECK-NEXT:             *_d_t += t * _d_y;
// CHECK-NEXT:         }
// CHECK-NEXT:     };
// CHECK-NEXT:     {
// CHECK-NEXT:         *_d_i += 1;
// CHECK-NEXT:         double _r0 = 0.;
// CHECK-NEXT:         _d__f(j, 1, &_r0);
// CHECK-NEXT:         *_d_j += _r0;
// CHECK-NEXT:     }
// CHECK-NEXT: }

double f2(double i, double j) {
  auto _f = [] (double t, double k) {
    double a = t*k;
    return a;
  };
  double x = _f(i + j, i);
  return x;
}

// CHECK: void f2_grad(double i, double j, double *_d_i, double *_d_j) {
// CHECK-NEXT:     auto _f0 = [](double t, double k) {
// CHECK-NEXT:         double a = t * k;
// CHECK-NEXT:         return a;
// CHECK-NEXT:     };
// CHECK-NEXT:     auto _d__f = [](double t, double k, double _d_y, double *_d_t, double *_d_k) {
// CHECK-NEXT:         double _d_a = 0.;
// CHECK-NEXT:         double a = t * k;
// CHECK-NEXT:         _d_a += _d_y;
// CHECK-NEXT:         {
// CHECK-NEXT:             *_d_t += _d_a * k;
// CHECK-NEXT:             *_d_k += t * _d_a;
// CHECK-NEXT:         }
// CHECK-NEXT:     };
// CHECK-NEXT:     double _d_x = 0.;
// CHECK-NEXT:     double x = _f0(i + j, i);
// CHECK-NEXT:     _d_x += 1;
// CHECK-NEXT:     {
// CHECK-NEXT:         double _r0 = 0.;
// CHECK-NEXT:         double _r1 = 0.;
// CHECK-NEXT:         _d__f(i + j, i, _d_x, &_r0, &_r1);
// CHECK-NEXT:         *_d_i += _r0;
// CHECK-NEXT:         *_d_j += _r0;
// CHECK-NEXT:         *_d_i += _r1;
// CHECK-NEXT:     }
// CHECK-NEXT: }


int main() {
  auto df1 = clad::gradient(f1);
  double di = 0, dj = 0;
  df1.execute(3, 4, &di, &dj);
  printf("%.2f %.2f\n", di, dj);              // CHECK-EXEC: 1.00 8.00

  auto df2 = clad::gradient(f2);
  di = 0, dj = 0;
  df2.execute(1, 1, &di, &dj);
  printf("%.2f %.2f\n", di, dj);              // CHECK-EXEC: 3.00 1.00
}
