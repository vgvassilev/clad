// RUN: %cladclang %s -I%S/../../include -oConstantFolding.out 2>&1 | %filecheck %s
// RUN: ./ConstantFolding.out | %filecheck_exec %s

// Pins the algebraic folding of forward-mode derivative expressions and the
// store-elision that goes with it: an operand read only by a dropped
// product-rule term must not be kept alive as a dead temporary.

#include "clad/Differentiator/Differentiator.h"

extern "C" int printf(const char* fmt, ...);

// Both tangents are literal zeros: the product rule collapses to a plain zero
// and neither operand is stored.
double fn_mul_both_zero(const double* arr) {
  return arr[0] * arr[1];
}
// CHECK: double fn_mul_both_zero_darg0_2(const double *arr) {
// CHECK-NEXT:     return 0.;
// CHECK-NEXT: }

// Same for the quotient rule: 0 / (R * R) folds away the denominator, so not
// even the divisor is stored.
double fn_div_both_zero(const double* arr) {
  return arr[0] / arr[1];
}
// CHECK: double fn_div_both_zero_darg0_2(const double *arr) {
// CHECK-NEXT:     return 0.;
// CHECK-NEXT: }

// The right tangent is zero, so the `L * dR` term is dropped and the left
// primal (x + x) is not stored into a temporary.
double fn_mul_skip_store(double x) {
  return (x + x) * 3.14;
}
// CHECK: double fn_mul_skip_store_darg0(double x) {
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     return (_d_x + _d_x) * 3.1400000000000001;
// CHECK-NEXT: }

// Division by a constant: the numerator keeps only the `dL * R` term.
double fn_div_skip_store(double x) {
  return (x + x) / 3.14;
}
// CHECK: double fn_div_skip_store_darg0(double x) {
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     return ((_d_x + _d_x) * 3.1400000000000001) / 9.8596000000000004;
// CHECK-NEXT: }

// `_d_x * 1.` must not fold to the bare lvalue `_d_x`: the value category of
// a tangent decides how a call it is passed to is differentiated.
double fn_keep_value_category(double x) {
  return x * 1.0;
}
// CHECK: double fn_keep_value_category_darg0(double x) {
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     return _d_x * 1.;
// CHECK-NEXT: }

// A tangent with a side effect is never treated as a constant zero, even when
// it evaluates to one: dropping it would drop the assignment.
double fn_side_effect_tangent(double x) {
  double y = 1;
  return (y = 0.0) * x;
}
// CHECK: double fn_side_effect_tangent_darg0(double x) {
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     double _d_y = 0;
// CHECK-NEXT:     double y = 1;
// CHECK-NEXT:     double &_t0 = (y = 0.);
// CHECK-NEXT:     return (_d_y = 0.) * x + _t0 * _d_x;
// CHECK-NEXT: }

// A parenthesized conditional stays parenthesized when the fold hands it back
// as an operand; without the parentheses the dump would reparse differently.
double fn_paren_cond(double x, bool c) {
  return (c ? x : 2.0) * 3.0;
}
// CHECK: double fn_paren_cond_darg0(double x, bool c) {
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     bool _d_c = 0;
// CHECK-NEXT:     return (c ? _d_x : 0.) * 3.;
// CHECK-NEXT: }

// sizeof is a target-dependent constant: it must survive the fold spelled as
// written, not literalized to the host's value (`x * 8UL`).
double fn_sizeof_opaque(double x) {
  return x * sizeof(double);
}
// CHECK: double fn_sizeof_opaque_darg0(double x) {
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     unsigned {{(int|long)}} _t0 = sizeof(double);
// CHECK-NEXT:     return _d_x * _t0 + x * sizeof(double);
// CHECK-NEXT: }

// Same for the identity rules: sizeof(char) evaluates to 1 on the host, but
// `x * sizeof(char)` must not fold to `x` on its account.
double fn_sizeof_one(double x) {
  return x * sizeof(char);
}
// CHECK: double fn_sizeof_one_darg0(double x) {
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     unsigned {{(int|long)}} _t0 = sizeof(char);
// CHECK-NEXT:     return _d_x * _t0 + x * sizeof(char);
// CHECK-NEXT: }

// A literal operand keeps its source spelling: re-synthesizing it at the
// implicitly converted type would print `x *= 2` as `x *= 2.`.
double fn_keep_literal_spelling(double x) {
  return x *= 2;
}
// CHECK: double fn_keep_literal_spelling_darg0(double x) {
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     return _d_x = _d_x * 2;
// CHECK-NEXT: }

// A constant that folds to a boolean must be synthesized as a bool literal:
// an IntegerLiteral of type bool crashes clang's statement printer.
double fn_bool_constant(double x, bool b) {
  return ((b = (2 > 1)), x * x);
}
// CHECK: double fn_bool_constant_darg0(double x, bool b) {
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     bool _d_b = 0;
// CHECK-NEXT:     return (((_d_b = false) , (b = true)) , (_d_x * x + x * _d_x));
// CHECK-NEXT: }

int main() {
  double arr[3] = {2.0, 4.0, 8.0};

  auto d_mul = clad::differentiate(fn_mul_both_zero, "arr[2]");
  printf("{%.2f}\n", d_mul.execute(arr)); // CHECK-EXEC: {0.00}

  auto d_div = clad::differentiate(fn_div_both_zero, "arr[2]");
  printf("{%.2f}\n", d_div.execute(arr)); // CHECK-EXEC: {0.00}

  auto d_mul_skip = clad::differentiate(fn_mul_skip_store, "x");
  printf("{%.2f}\n", d_mul_skip.execute(5)); // CHECK-EXEC: {6.28}

  auto d_div_skip = clad::differentiate(fn_div_skip_store, "x");
  printf("{%.2f}\n", d_div_skip.execute(5)); // CHECK-EXEC: {0.64}

  auto d_keep = clad::differentiate(fn_keep_value_category, "x");
  printf("{%.2f}\n", d_keep.execute(5)); // CHECK-EXEC: {1.00}

  auto d_side = clad::differentiate(fn_side_effect_tangent, "x");
  printf("{%.2f}\n", d_side.execute(5)); // CHECK-EXEC: {0.00}

  auto d_cond = clad::differentiate(fn_paren_cond, "x");
  printf("{%.2f}\n", d_cond.execute(5, true)); // CHECK-EXEC: {3.00}
  printf("{%.2f}\n", d_cond.execute(5, false)); // CHECK-EXEC: {0.00}

  auto d_sizeof = clad::differentiate(fn_sizeof_opaque, "x");
  printf("{%.2f}\n", d_sizeof.execute(5)); // CHECK-EXEC: {48.00}

  auto d_sizeof_one = clad::differentiate(fn_sizeof_one, "x");
  printf("{%.2f}\n", d_sizeof_one.execute(5)); // CHECK-EXEC: {6.00}

  auto d_literal = clad::differentiate(fn_keep_literal_spelling, "x");
  printf("{%.2f}\n", d_literal.execute(5)); // CHECK-EXEC: {2.00}

  auto d_bool = clad::differentiate(fn_bool_constant, "x");
  printf("{%.2f}\n", d_bool.execute(5, false)); // CHECK-EXEC: {10.00}
  return 0;
}
