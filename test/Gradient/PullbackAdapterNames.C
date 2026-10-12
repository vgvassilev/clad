// RUN: %cladclang %s -I%S/../../include -o %t 2>&1 | %filecheck %s
// RUN: %t | %filecheck_exec %s
#include "clad/Differentiator/Differentiator.h"
#include <cstdio>

// Reserve names across primals and adjoints, including a numeric suffix.
// CHECK: void colliding_pullback_0_2(double x, double _d_x, double _, double _d_y, double *_d_x0, double *_d_) {
double colliding(double x, double _d_x, double _) { return x * _d_x + _; }
// CHECK: void unnamed_pullback_1(double arg, double y, double _d_y0, double *_d_y) {
double unnamed(double, double y) { return y * y; }
// The public adapter must allow a safe pointer qualification conversion,
// without erasing the public primal type or consuming the ignored x slot.
// CHECK: void qualified_custom_pullback_1(double *x, double y, double _d_y0, double *_d_y) {
double qualified_custom(double* x, double y) { return *x * y; }
namespace clad { namespace custom_derivatives {
void qualified_custom_pullback(const double* x, double y, double seed, double* d_y) {
  *d_y += *x * seed;
}
}}
int main() {
  auto names=clad::pullback(colliding,"x,_");
  double dx=0, ignored=42, du=0;
  names.execute(2.,3.,4.,2.,&dx,&ignored,&du);
  std::printf("Names: %.0f %.0f %.0f\n",dx,ignored,du);
  // CHECK-EXEC: Names: 6 42 2
  auto no_name=clad::pullback(unnamed,"y");
  double dy=0;
  no_name.execute(7.,3.,2.,nullptr,&dy);
  std::printf("Unnamed: %.0f\n",dy);
  // CHECK-EXEC: Unnamed: 12
  auto custom=clad::pullback(qualified_custom,"y");
  double x=3;dy=1;
  custom.execute(&x,4.,2.,nullptr,&dy);
  std::printf("Custom qualification: %.0f\n",dy);
  // CHECK-EXEC: Custom qualification: 7
}
