// RUN: %cladclang %s -I%S/../../include -oPullbackEdgeCases.out 2>&1 | %filecheck %s
// RUN: ./PullbackEdgeCases.out | %filecheck_exec %s

#include "clad/Differentiator/Differentiator.h"
#include <cstdio>

// The public function type drops top-level cv-qualifiers on value parameters.
// A partial adapter must derive its adjoint slots from that function type,
// not from the locally qualified ParmVarDecl. Otherwise execute() calls it
// through a mismatched function pointer (caught by -fsanitize=function).
double qualified(double x, volatile double y) { return x * y; }
using QualifiedValue = const volatile double;
double qualified_alias(double x, QualifiedValue y) { return x * y; }
double qualified_reference(const volatile double& x, double y) { return x * y; }


// A named rvalue-reference argument is an lvalue; the adapter must forward it
// as an rvalue when calling the selected implementation.
double rvalue_input(double&& x, double y) { return x * y; }

// A user-defined tag with libc++'s internal spelling is still an ordinary
// differentiable record; only std's tag is excluded at the public boundary.
namespace user_tags {
struct __nat { double value; };
}
double ordinary_tag(user_tags::__nat tag, double x) { return tag.value * x; }


// Generated names must be unique even when a primal uses Clad's prefixes.
double seed_name(double _d_y) { return _d_y * _d_y; }
// CHECK: void seed_name_pullback(double _d_y, double _d_y0, double *_d_d_y) {

struct Object {
  double factor;
  double method(double _d_this) const { return factor * _d_this; }
  // CHECK: void method_pullback(double _d_this, double _d_y, Object *_d_this0, double *_d_d_this) const {

  double custom(double x) const noexcept { return factor * x * x; }
  double custom_partial(double x, double y) const { return factor * x * y; }
  double rvalue_method(double x, double y) && { return factor * x * y; }
};

namespace clad {
namespace custom_derivatives {
namespace class_functions {
void custom_pullback(const Object* self, double x, double seed, Object* d_self,
                     double* d_x) noexcept {
  d_self->factor += x * x * seed;
  *d_x += 2 * self->factor * x * seed;
}
void custom_partial_pullback(const Object* self, double x, double y, double seed,
                             Object* d_self, double* d_y) {
  d_self->factor += x * y * seed;
  *d_y += self->factor * x * seed;
}
} // namespace class_functions
} // namespace custom_derivatives
} // namespace clad

// Passive adjoint slots must stay positional and must not be dereferenced.
double passive(double x, double y __attribute__((annotate("non_differentiable")))) {
  return x * y;
}

int main() {
  auto pb_qualified = clad::pullback(qualified, "x");
  double dx = 1, ignored = 42;
  pb_qualified.execute(2., 3., 2., &dx, &ignored);
  std::printf("Qualified: %.0f %.0f\n", dx, ignored);
  // CHECK-EXEC: Qualified: 7 42

  auto pb_alias = clad::pullback(qualified_alias, "x");
  dx = 0;
  pb_alias.execute(2., 3., 2., &dx, nullptr);
  std::printf("Qualified alias: %.0f\n", dx);
  // CHECK-EXEC: Qualified alias: 6

  auto pb_reference = clad::pullback(qualified_reference, "x");
  volatile double input = 2, d_input = 0;
  pb_reference.execute(input, 3., 2., &d_input, nullptr);
  std::printf("Qualified reference: %.0f\n", static_cast<double>(d_input));
  // CHECK-EXEC: Qualified reference: 6

  auto pb_rvalue = clad::pullback(rvalue_input, "x");
  dx = 0;
  pb_rvalue.execute(2., 3., 2., &dx, nullptr);
  std::printf("Rvalue input: %.0f\n", dx);
  // CHECK-EXEC: Rvalue input: 6

  auto pb_seed = clad::pullback(seed_name);
  dx = 0;
  pb_seed.execute(3., 2., &dx);
  std::printf("Seed name: %.0f\n", dx);
  // CHECK-EXEC: Seed name: 12

  Object object{2}, d_object{0};
  auto pb_method = clad::pullback(&Object::method);
  dx = 0;
  pb_method.execute(object, 3., 2., &d_object, &dx);
  std::printf("Object name: %.0f %.0f\n", dx, d_object.factor);
  // CHECK-EXEC: Object name: 4 6

  auto pb_custom = clad::pullback(&Object::custom);
  dx = 0;
  d_object.factor = 0;
  pb_custom.execute(object, 3., 2., &d_object, &dx);
  std::printf("Custom noexcept member: %.0f %.0f\n", dx, d_object.factor);
  // CHECK-EXEC: Custom noexcept member: 24 18

  auto pb_custom_partial = clad::pullback(&Object::custom_partial, "y");
  double dy = 0;
  d_object.factor = 0;
  pb_custom_partial.execute(object, 3., 4., 2., &d_object, nullptr, &dy);
  std::printf("Custom partial member: %.0f %.0f\n", dy, d_object.factor);
  // CHECK-EXEC: Custom partial member: 12 24

  auto pb_rvalue_method = clad::pullback(&Object::rvalue_method, "y");
  dy = 0;
  d_object.factor = 0;
  pb_rvalue_method.execute(static_cast<Object&&>(object), 3., 4., 2.,
                           &d_object, nullptr, &dy);
  std::printf("Rvalue method: %.0f %.0f\n", dy, d_object.factor);
  // CHECK-EXEC: Rvalue method: 12 24

  auto pb_tag = clad::pullback(ordinary_tag);
  user_tags::__nat tag{2}, d_tag{0};
  dx = 0;
  pb_tag.execute(tag, 3., 2., &d_tag, &dx);
  std::printf("Ordinary tag: %.0f %.0f\n", dx, d_tag.value);
  // CHECK-EXEC: Ordinary tag: 4 6

  auto pb_passive = clad::pullback(passive);
  dx = 0;
  pb_passive.execute(2., 3., 2., &dx, &ignored);
  std::printf("Passive: %.0f %.0f\n", dx, ignored);
  // CHECK-EXEC: Passive: 6 42
  pb_passive.execute(2., 3., 2., &dx, nullptr);
  std::printf("Passive null: %.0f\n", dx);
  // CHECK-EXEC: Passive null: 12
}
