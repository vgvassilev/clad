// RUN: %cladclang %s -I%S/../../include -o %t 2>&1 | %filecheck %s
// RUN: %t | %filecheck_exec %s

#include "clad/Differentiator/Differentiator.h"
#include "clad/Differentiator/STLBuiltins.h"

#include <cstdio>
#include <vector>

struct Box {
  double value;
  // A non-trivial destructor makes Clang bind prvalues of this type in a
  // CXXBindTemporaryExpr.
  ~Box() {}

  double get() const { return value; }
};

Box make_box(double x) { return {x * x}; }

double temporary_method(double x) { return make_box(x).get(); }

double vector_temporary_method(double x) {
  return x + static_cast<double>(std::vector<double>().size());
}

// CHECK: void temporary_method_grad(double x, double *_d_x) {
// CHECK:     make_box(x).get_pullback(1, &{{.*}});
// CHECK:     make_box_pullback(x, {{.*}}, &{{.*}});
// CHECK: }

// CHECK: void vector_temporary_method_grad(double x, double *_d_x) {
// CHECK:     *_d_x += 1;
// CHECK: }

int main() {
  auto temporary_method_grad = clad::gradient(temporary_method);
  double d_x = 0.0;
  temporary_method_grad.execute(3.0, &d_x);
  std::printf("box=%.1f\n", d_x); // CHECK-EXEC: box=6.0

  auto vector_temporary_method_grad = clad::gradient(vector_temporary_method);
  d_x = 0.0;
  vector_temporary_method_grad.execute(3.0, &d_x);
  std::printf("vector=%.1f\n", d_x); // CHECK-EXEC: vector=1.0
}
