// RUN: %cladclang %s -I%S/../../include -oTimingsReport.out -ftime-report 2>&1 | %filecheck %s
// RUN: %cladclang %s -I%S/../../include -fsyntax-only -Xclang -print-stats 2>&1 | %filecheck -check-prefix=CHECK_STATS %s
// RUN: %cladclang %s -I%S/../../include -fsyntax-only -Xclang -plugin-arg-clad -Xclang -enable-tbr -Xclang -print-stats 2>&1 | %filecheck -check-prefix=CHECK_STATS_TBR %s

#include "clad/Differentiator/Differentiator.h"
// CHECK: Timers for Clad Funcs
// CHECK_STATS: *** INFORMATION ABOUT THE DIFF REQUESTS
// CHECK_STATS-NEXT: <double test1(double x, double y)>[name=test1, order=1, mode=forward]: #0 (source), (done)
// CHECK_STATS-NEXT: <double test2(double a, double b)>[name=test2, order=1, mode=reverse]: #1 (source), (done)
// CHECK_STATS-NEXT: <double nested1(double c)>[name=nested1, order=1, mode=pushforward]: #2, (done)
// CHECK_STATS-NEXT: <double nested2(double z)>[name=nested2, order=1, mode=pullback]: #3, (done)
// CHECK_STATS-NEXT: 0 -> 2
// CHECK_STATS-NEXT: 1 -> 3

// CHECK_STATS_TBR: <double test1(double x, double y)>[name=test1, order=1, mode=forward, tbr]: #0 (source), (done)

double nested1(double c){
  return c*3*c;
}

double nested2(double z){
  return 4*z*z;
}

double test1(double x, double y) {
  return 2*y*nested1(y) * 3 * x * nested1(x);
}

double test2(double a, double b) {
  return 3*a*a + b * nested2(a) + a * b;
}

int main() {
  auto d_fn_1 = clad::differentiate(test1, "x");
  double dp = -1, dq = -1;
  auto f_grad = clad::gradient(test2);
  f_grad.execute(3, 4, &dp, &dq);
  printf("Result is = %f\n", d_fn_1.execute(3,4));
  printf("Result is = %f %f\n", dp, dq);
  return 0;
}
