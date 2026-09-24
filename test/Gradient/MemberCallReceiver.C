// RUN: %cladclang %s -I%S/../../include -o %t 2>&1 | %filecheck %s
// RUN: %t | %filecheck_exec %s
// RUN: %cladclang %s -I%S/../../include -Xclang -plugin-arg-clad -Xclang -disable-tbr -o %t.no-tbr
// RUN: %t.no-tbr | %filecheck_exec %s

#include "clad/Differentiator/Differentiator.h"

extern "C" int printf(const char*, ...);

int made = 0, called = 0;

struct Box {
  double value;
  double* storage;

  double scaled() const {
    ++called;
    return 3 * value;
  }
  template <typename T> double convert(T factor) const {
    ++called;
    return factor * value;
  }
};

Box make_box(double x) {
  ++made;
  return {x * x, nullptr};
}

struct CopyOnly {
  double value = 0;
  double* storage = nullptr;
  CopyOnly() = default;
  explicit CopyOnly(double x) : value(x) {}
  CopyOnly(const CopyOnly&) = default;
  CopyOnly(CopyOnly&&) = delete;
  double read() const { return value; }
};

CopyOnly make_copy_only(double x) { return CopyOnly(x * x); }

namespace clad {
namespace custom_derivatives {
// Supply the receiver's primal and adjoint without default reverse-forward
// generation, so this test isolates member-call reconstruction.
clad::ValueAndAdjoint<Box, Box> make_box_reverse_forw(double x, double) {
  return {make_box(x), {}};
}
clad::ValueAndAdjoint<CopyOnly, CopyOnly>
make_copy_only_reverse_forw(double x, double) {
  return {make_copy_only(x), {}};
}
void make_box_pullback(double x, Box d_output, double* d_x) {
  *d_x += 2 * x * d_output.value;
}
void make_copy_only_pullback(double x, CopyOnly d_output, double* d_x) {
  *d_x += 2 * x * d_output.value;
}
namespace class_functions {
void read_pullback(const CopyOnly* /*self*/, double d_output, CopyOnly* d_self) {
  d_self->value += d_output;
}
void scaled_pullback(const Box* /*self*/, double d_output, Box* d_self) {
  d_self->value += 3 * d_output;
}
template <typename T>
void convert_pullback(const Box* self, T factor, double d_output, Box* d_self,
                      T* d_factor) {
  d_self->value += factor * d_output;
  *d_factor += self->value * d_output;
}
} // namespace class_functions
} // namespace custom_derivatives
} // namespace clad

double temporary(double x) { return make_box(x).scaled() * x; }
double nested(double x) { return make_box(make_box(x).scaled()).scaled() * x; }
double templated(double x) { return make_box(x).convert<double>(4) * x; }
double lvalue(double x) {
  auto box = make_box(x);
  return box.scaled() * x;
}
double qualified(double x) {
  return make_box(x).Box::scaled() * x;
}
double parenthesized(double x) { return (make_box(x)).scaled() * x; }
double dereferenced(double x) {
  Box box{x, nullptr};
  Box* ptr = &box;
  return (*ptr).scaled() * x;
}
double loop(double x) {
  double result = 0;
  for (int i = 0; i < 3; ++i)
    result += make_box(x + i).scaled() * x;
  return result;
}
double branch(double x) {
  return make_box(x > 0 ? x : -x).scaled() * x;
}
double copy_only(double x) { return make_copy_only(x).read() * x; }

// CHECK-LABEL: void temporary_grad(
// CHECK: = clad::custom_derivatives::make_box_reverse_forw(x,
// CHECK-NOT: make_box(
// CHECK: .scaled();
// CHECK-NOT: make_box(
// CHECK: make_box_pullback(x,
// CHECK-LABEL: void lvalue_grad(
// CHECK: make_box_pullback(x,
// The const method adds an implicit cast around the original parentheses.
// CHECK-LABEL: void dereferenced_grad(
// CHECK: = (*ptr).scaled();
// CHECK-LABEL: void loop_grad(
// CHECK: make_box_pullback(
// CHECK-LABEL: void copy_only_grad(
// CHECK: make_copy_only_pullback(x,

template <typename Gradient>
void check(const char* name, Gradient gradient, double x = 3) {
  made = called = 0;
  double dx = 0;
  gradient.execute(x, &dx);
  printf("%s: %.0f %d %d\n", name, dx, made, called);
}

int main() {
  check("temporary", clad::gradient(temporary));
  // CHECK-EXEC: temporary: 81 1 1
  check("nested", clad::gradient(nested));
  // The inner scaled() is replayed for the outer make_box_pullback argument;
  // both make_box receivers must still be evaluated only once.
  // CHECK-EXEC-NEXT: nested: 10935 2 3
  check("templated", clad::gradient(templated));
  // CHECK-EXEC-NEXT: templated: 108 1 1
  check("lvalue", clad::gradient(lvalue));
  // CHECK-EXEC-NEXT: lvalue: 81 1 1
  check("qualified", clad::gradient(qualified));
  // CHECK-EXEC-NEXT: qualified: 81 1 1
  check("parenthesized", clad::gradient(parenthesized));
  // CHECK-EXEC-NEXT: parenthesized: 81 1 1
  check("dereferenced", clad::gradient(dereferenced));
  // CHECK-EXEC-NEXT: dereferenced: 18 0 1
  check("loop", clad::gradient(loop));
  // CHECK-EXEC-NEXT: loop: 366 3 3
  check("branch", clad::gradient(branch));
  // CHECK-EXEC-NEXT: branch: 81 1 1
  check("branch", clad::gradient(branch), -3);
  // CHECK-EXEC-NEXT: branch: 81 1 1
  check("copy_only", clad::gradient(copy_only));
  // CHECK-EXEC-NEXT: copy_only: 27 0 0
}
