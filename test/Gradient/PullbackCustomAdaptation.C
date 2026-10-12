// RUN: %cladclang %s -I%S/../../include -o %t 2>&1 | %filecheck %s
// RUN: %t | %filecheck_exec %s

#include "clad/Differentiator/Differentiator.h"
#include <cstdio>
#include <cassert>

// 1. Free function with user-provided full custom pullback
double f_custom(double x, double y) {
  return x * y;
}

namespace clad {
namespace custom_derivatives {
void f_custom_pullback(double x, double y, double seed, double* d_x, double* d_y) {
  *d_x += 100.0 * seed;
  *d_y += 1000.0 * seed;
}
} // namespace custom_derivatives
} // namespace clad

// CHECK: inline void f_custom_pullback_1(double x, double y, double seed, double *_d_x, double *_d_y) {
// CHECK:     double _scratch_00 = 0.;
// CHECK:     clad::custom_derivatives::f_custom_pullback(x, y, seed, _d_x ? _d_x : &_scratch_00, _d_y);
// CHECK: }

// 2. Nested call: function calling f_custom
double f_nested_call(double a, double b) {
  return f_custom(a, b);
}

// 3. Member function with user-provided full custom pullback
struct CustomStruct {
  double factor;
  double compute(double x) const {
    return factor * x;
  }
};

namespace clad {
namespace custom_derivatives {
namespace class_functions {
void compute_pullback(const CustomStruct* self, double x, double seed,
                      CustomStruct* d_self, double* d_x) {
  d_self->factor += 7.0 * seed;
  *d_x += 70.0 * seed;
}
} // namespace class_functions
} // namespace custom_derivatives
} // namespace clad

int main() {
  // Test 1: Full custom pullback
  auto pb_full = clad::pullback(f_custom);
  double dx_full = 0.0, dy_full = 0.0;
  pb_full.execute(2.0, 3.0, 1.0, &dx_full, &dy_full);
  printf("Full custom: dx=%.0f dy=%.0f\n", dx_full, dy_full);
  // CHECK-EXEC: Full custom: dx=100 dy=1000

  // Test 2: Partial pullback ("y") adapting full custom pullback
  auto pb_part_y = clad::pullback(f_custom, "y");
  double dy_part = 0.0;
  pb_part_y.execute(2.0, 3.0, 1.0, nullptr, &dy_part);
  printf("Partial custom y: dy=%.0f\n", dy_part);
  // CHECK-EXEC: Partial custom y: dy=1000

  // Test 3: Partial pullback ("x") adapting full custom pullback with non-unit seed & accumulation
  auto pb_part_x = clad::pullback(f_custom, "x");
  double dx_part = 5.0;
  pb_part_x.execute(2.0, 3.0, 2.5, &dx_part, nullptr);
  printf("Partial custom x (scaled seed + init): dx=%.0f\n", dx_part);
  // CHECK-EXEC: Partial custom x (scaled seed + init): dx=255

  // Test 4: Nested pullback respects custom derivative
  auto pb_nested = clad::pullback(f_nested_call);
  double da_nested = 0.0, db_nested = 0.0;
  pb_nested.execute(2.0, 3.0, 1.0, &da_nested, &db_nested);
  printf("Nested custom: da=%.0f db=%.0f\n", da_nested, db_nested);
  // CHECK-EXEC: Nested custom: da=100 db=1000

  // Test 5: Member function full custom pullback
  CustomStruct cs{3.0}, d_cs{0.0};
  auto pb_mem_full = clad::pullback(&CustomStruct::compute);
  double dx_mem_full = 0.0;
  pb_mem_full.execute(cs, 4.0, 1.0, &d_cs, &dx_mem_full);
  printf("Member full custom: d_factor=%.0f dx=%.0f\n", d_cs.factor, dx_mem_full);
  // CHECK-EXEC: Member full custom: d_factor=7 dx=70

  // Test 6: Member function partial custom pullback w.r.t "x" with nullable d_this
  CustomStruct d_cs_null{0.0};
  auto pb_mem_part = clad::pullback(&CustomStruct::compute, "x");
  double dx_mem_part = 10.0;
  pb_mem_part.execute(cs, 4.0, 2.0, nullptr, &dx_mem_part);
  printf("Member partial custom nullable d_this: dx=%.0f\n", dx_mem_part);
  // CHECK-EXEC: Member partial custom nullable d_this: dx=150

  return 0;
}
