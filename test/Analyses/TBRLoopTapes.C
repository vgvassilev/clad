// What to-be-recorded analysis changes about a loop: the record a loop body
// would otherwise keep, one value per iteration.
//
// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -fdisable-analysis=tbr %s \
// RUN:   -I%S/../../include -oTBRLoopTapes.out 2>&1 \
// RUN:   | %filecheck --check-prefix=CHECK-OFF %s
// RUN: ./TBRLoopTapes.out | %filecheck_exec %s
//
// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -fenable-analysis=tbr %s \
// RUN:   -I%S/../../include -oTBRLoopTapes.out 2>&1 \
// RUN:   | %filecheck --check-prefix=CHECK-TBR %s
// RUN: ./TBRLoopTapes.out | %filecheck_exec %s

#include "clad/Differentiator/Differentiator.h"

#include <cstdio>

// The pullback of `+` reads neither operand, so no iteration's y is wanted in
// reverse and the record holding them goes away. The trip count that drives
// the reverse sweep is not the record's business and stays either way.
double sum(double x) {
  double y = 0;
  for (int i = 0; i < 3; ++i)
    y = y + x * i;
  return y;
}

// CHECK-OFF-LABEL: void sum_grad(double x, double *_d_x) {
// CHECK-OFF: double _t1[3];
// CHECK-OFF: _t1[i] = y;
// CHECK-OFF: for (_t0 = 3{{U|UL|ULL}}; _t0; _t0--) {
// CHECK-OFF: y = _t1[i];

// CHECK-TBR-LABEL: void sum_grad(double x, double *_d_x) {
// CHECK-TBR: for (_t0 = 3{{U|UL|ULL}}; _t0; _t0--) {
// CHECK-TBR-NOT: double _t1[
// CHECK-TBR-NOT: clad::tape<double>
// CHECK-TBR-NOT: clad::push

// The pullback of `*` reads both operands, so every iteration's y is wanted
// and the record stays. Without this, an analysis that dropped every record
// would still pass the case above.
double product(double x) {
  double y = 1;
  for (int i = 0; i < 3; ++i)
    y = y * x;
  return y;
}

// CHECK-OFF-LABEL: void product_grad(double x, double *_d_x) {
// CHECK-OFF: double _t1[3];
// CHECK-OFF: _t1[i] = y;

// CHECK-TBR-LABEL: void product_grad(double x, double *_d_x) {
// CHECK-TBR: double _t1[3];
// CHECK-TBR: _t1[i] = y;

int main() {
  auto s = clad::gradient(sum);
  auto p = clad::gradient(product);
  double d = 0;
  s.execute(3, &d);
  printf("%.2f\n", d); // CHECK-EXEC: 3.00
  d = 0;
  p.execute(3, &d);
  printf("%.2f\n", d); // CHECK-EXEC: 27.00
}
