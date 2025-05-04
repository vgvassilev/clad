// RUN: %cladclang %s -I%S/../../include -oTimingsReport.out -ftime-report 2>&1 | %filecheck %s
// RUN: env CLAD_ENABLE_TIMING=1 %cladclang %s -I%S/../../include -oTimingsReport.out 2>&1 | %filecheck -check-prefix=CHECK_TIMING_ENV %s
// RUN: %cladclang %s -I%S/../../include -fsyntax-only -DGLOBAL \
// RUN:            -Xclang -plugin-arg-clad -Xclang -disable-tbr -Xclang \
// RUN:             -print-stats -Xclang -verify 2>&1 | %filecheck -check-prefix=CHECK_STATS %s
// RUN: %cladclang %s -I%S/../../include -fsyntax-only -Xclang -print-stats 2>&1 | %filecheck -check-prefix=CHECK_STATS_TBR %s

#include "clad/Differentiator/Differentiator.h"
// CHECK: Clad AST Generation Timing Report
// CHECK: Clad Analysis Timing Report
// CHECK_TIMING_ENV: Clad AST Generation Timing Report
// CHECK_TIMING_ENV: TBR func
// CHECK_STATS: *** INFORMATION ABOUT THE DIFF REQUESTS
// CHECK_STATS-NEXT: <double nested1(double c)>[name=nested1, order=1, mode=pushforward, args='']: #0 (source), (done)
// CHECK_STATS-NEXT: <double test1(double x, double y)>[name=test1, order=1, mode=forward, args='"x"']: #1 (source), (done)
// CHECK_STATS-NEXT: <double nested2(double z, double j)>[name=nested2, order=1, mode=pullback, args='z,j']: #2 (source), (done)
// CHECK_STATS-NEXT: <double test2(double a, double b)>[name=test2, order=1, mode=reverse, args='']: #3 (source), (done)
// CHECK_STATS-NEXT: <double addArrImpl(double *arr)>[name=addArrImpl, order=1, mode=pullback, args='arr']: #4 (source), (done)
// CHECK_STATS-NEXT: <double addArr(double *arr)>[name=addArr, order=1, mode=reverse, args='"arr[0:1]"']: #5 (source), (done)
// CHECK_STATS-NEXT: <float func(float *a)>[name=func, order=1, mode=reverse, args='']: #6 (source), (done)
// CHECK_STATS-NEXT: <double g = 3.1400000000000001>[name=, order=1, mode=unknown, args='']: #7 (source), (done)
// CHECK_STATS-NEXT: <double global_fn(double x)>[name=global_fn, order=1, mode=reverse, args='']: #8 (source), (done)
// CHECK_STATS-NEXT: <constexpr double constexpr_fn(double x, double y)>[name=constexpr_fn, order=1, mode=reverse, args='']: #9 (source), (done)

// CHECK_STATS_TBR: <double test1(double x, double y)>[name=test1, order=1, mode=forward, args='"x"', tbr]: #1 (source), (done)

#ifdef GLOBAL
double g = 3.14; // expected-warning {{The gradient utilizes a global variable 'g'}}
double global_fn(double x) {
  return g * x;
}
#endif // GLOBAL

constexpr double constexpr_fn(double x, double y) { return x * y; }

float func(float* a) {
  float sum = 0;
  for (int i = 0; i < 3; i++)
    sum += a[i];
  return sum;
}

double nested1(double c) {
  double t = 1;
  for (int i = 0; i < 3; i++)
    t *= c;
  return t; // == c^3
}

double nested2(double z, double j){
  return 4*z*z;
}

double test1(double x, double y) {
  return 2*y*nested1(y) * 3 * x * nested1(x);
}

double test2(double a, double b) {
  return 3*a*a + b * nested2(a, b) + a * b;
}

double addArrImpl(double *arr) {
  return arr[0] + arr[1] + arr[2] + arr[3];
}

double addArr(double *arr) {
  return addArrImpl(arr);
}

int main() {
  auto d_fn_1 = clad::differentiate(test1, "x");
  double dp = -1, dq = -1;
  auto f_grad = clad::gradient(test2);
  f_grad.execute(3, 4, &dp, &dq);
  printf("Result is = %f\n", d_fn_1.execute(3,4));
  printf("Result is = %f %f\n", dp, dq);
  clad::gradient(addArr, "arr[0:1]");
  clad::gradient(func);
#ifdef GLOBAL
  clad::gradient(global_fn);
#endif // GLOBAL
  clad::gradient(constexpr_fn);
  return 0;
}
