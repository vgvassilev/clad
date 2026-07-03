// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -enable-ua -Xclang -plugin-arg-clad -Xclang -disable-tbr %s -I%S/../../include -oUsefulNonTermination.out
// RUN: ./UsefulNonTermination.out | %filecheck_exec %s

#include "clad/Differentiator/Differentiator.h"
#include "../TestUtils.h"

// A loop in each arm of an if/else. Under -enable-ua this hung forever because
// UsefulAnalyzer's back-edge termination used one never-cleared, function-wide
// m_LoopMem set that could only ever converge for a single loop.
// CHECK: double foo_darg1(bool cond, double x, double y) {
// CHECK-NEXT:     bool _d_cond = 0;
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     double _d_y = 0;
// CHECK-NEXT:     double _d_r = 0;
// CHECK-NEXT:     double r = 0;
// CHECK-NEXT:     if (cond) {
// CHECK-NEXT:         for (int i = 0; i < 3; i++) {
// CHECK-NEXT:             _d_r += _d_x;
// CHECK-NEXT:             r += x;
// CHECK-NEXT:         }
// CHECK-NEXT:     } else {
// CHECK-NEXT:         for (int i = 0; i < 3; i++) {
// CHECK-NEXT:             _d_r += _d_y;
// CHECK-NEXT:             r += y;
// CHECK-NEXT:         }
// CHECK-NEXT:     }
// CHECK-NEXT:     return _d_r;
// CHECK-NEXT: }
double foo(bool cond, double x, double y) {
  double r = 0;
  if (cond) {
    for (int i = 0; i < 3; i++)
      r += x;
  } else {
    for (int i = 0; i < 3; i++)
      r += y;
  }
  return r;
}

int main() {
  INIT_DIFFERENTIATE_UA(foo, "x");

  TEST_DIFFERENTIATE(foo, true, 3, 5);  // CHECK-EXEC: {3.00}
  TEST_DIFFERENTIATE(foo, false, 3, 5); // CHECK-EXEC: {0.00}
  return 0;
}
