// RUN: %cladclang %s -I%S/../../include -oTBR.out 2>&1 | %filecheck %s
// RUN: ./TBR.out | %filecheck_exec %s
// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -disable-tbr %s -I%S/../../include -oTBR.out
// RUN: ./TBR.out | %filecheck_exec %s

#include "clad/Differentiator/Differentiator.h"

#include "../TestUtils.h"

double mat4[4];
clad::array_ref<double> mat4_ref(mat4, 4);

// A hessian's inner reverse pass is scheduled lazily by
// HandleNestedDiffRequest and never walked by the static planner, so it only
// runs TBR if PlanNestedRequest built it an AnalysisDeclContext. Here that
// shows up in the loop: `s = s + a` is linear, so neither s nor _d_s has to be
// taped, while `a = a * x` is not and both a and _d_a do.
double linearAndNonlinear(double x, double y) {
  double a = x * y;
  double s = a;
  for (int i = 0; i < 3; ++i) {
    s = s + a;
    a = a * x;
  }
  return s * a;
}

// CHECK: void linearAndNonlinear_darg0_grad(double x, double y, double *_d_x, double *_d_y) {
// Two tapes, for the nonlinear assignment only.
// CHECK: clad::tape<double> _t1 = {};
// CHECK-NEXT: clad::tape<double> _t2 = {};
// CHECK-NOT: clad::tape<double> _t3 = {};
// CHECK: for (i = 0; i < 3; ++i) {
// CHECK-NEXT: _d_s = _d_s + _d_a;
// CHECK-NEXT: s = s + a;
// CHECK-NEXT: clad::push(_t1, _d_a);
// CHECK-NEXT: _d_a = _d_a * x + a * _d_x0;
// CHECK-NEXT: clad::push(_t2, a);
// CHECK-NEXT: a = a * x;
// CHECK-NEXT: }

int main() {
  INIT_HESSIAN(linearAndNonlinear);

  TEST_HESSIAN(linearAndNonlinear, 1, 1.5, 2.5, mat4_ref);
  // CHECK-EXEC: {3786.33, 879.61, 879.61, 87.33}
}
