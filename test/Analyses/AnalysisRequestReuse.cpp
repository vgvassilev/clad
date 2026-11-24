// RUN: %cladclang %s -I%S/../../include -oAnalysisRequestReuse.out 2>&1 \
// RUN:   | %filecheck %s
// RUN: ./AnalysisRequestReuse.out | %filecheck_exec %s

#include "clad/Differentiator/Differentiator.h"

#include <cstdio>

double plainFirst(double x, double y) {
  double a = y * y;
  double b = x * x;
  return a + b;
}

double enabledFirst(double x, double y) {
  double a = y * y;
  double b = x * x;
  return a + b;
}

// The plain request keeps the adjoint of a; the following activity-analysis
// request prunes it.
// CHECK-LABEL: void plainFirst_grad_0(double x, double y, double *_d_x) {
// CHECK: double _d_a = 0.;
// CHECK-LABEL: void plainFirst_grad_0_analysis_2212(double x, double y, double *_d_x) {
// CHECK-NOT: _d_a

// In the opposite request order, the activity-analysis request prunes the
// adjoint and the following plain request keeps it.
// CHECK-LABEL: void enabledFirst_grad_0(double x, double y, double *_d_x) {
// CHECK-NOT: _d_a
// CHECK-LABEL: void enabledFirst_grad_0_analysis_2180(double x, double y, double *_d_x) {
// CHECK: double _d_a = 0.;

int main() {
  auto plain = clad::gradient(plainFirst, "x");
  auto enabled = clad::gradient<clad::opts::enable_va>(plainFirst, "x");

  auto enabledBefore =
      clad::gradient<clad::opts::enable_va>(enabledFirst, "x");
  auto plainAfter = clad::gradient(enabledFirst, "x");

  std::printf("%d %d\n", plain.getFunctionPtr() != enabled.getFunctionPtr(),
              enabledBefore.getFunctionPtr() != plainAfter.getFunctionPtr());
  // CHECK-EXEC: 1 1
}
