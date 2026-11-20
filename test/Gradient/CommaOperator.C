// RUN: %cladclang %s -I%S/../../include -Wno-unused-value -oCommaOperator.out 2>&1 | %filecheck %s
// RUN: ./CommaOperator.out | %filecheck_exec %s

#include "clad/Differentiator/Differentiator.h"
#include <cstdio>

// Test 1: Basic comma expression where result is used by enclosing expression.
// Evaluates to b * c.
// Expected derivatives: da = 0, db = c, dc = b.
double comma_basic(double a, double b, double c) {
  return (a, b) * c;
}

// CHECK: void comma_basic_grad(double a, double b, double c, double *_d_a, double *_d_b, double *_d_c) {
// CHECK-NEXT:     {
// CHECK-NEXT:         *_d_b += 1 * c;
// CHECK-NEXT:         *_d_a += 0;
// CHECK-NEXT:         *_d_c += b * 1;
// CHECK-NEXT:     }
// CHECK-NEXT: }

// Test 2: Left operand of comma expression has side effects.
// Evaluates to b * c, with a += 1.0 executed first in the forward pass.
// Expected derivatives: da = 0, db = c, dc = b.
double comma_side_effect(double a, double b, double c) {
  return (a += 1.0, b) * c;
}

// CHECK: void comma_side_effect_grad(double a, double b, double c, double *_d_a, double *_d_b, double *_d_c) {
// CHECK-NEXT:     double _t0 = (a += 1. , b);
// CHECK-NEXT:     {
// CHECK-NEXT:         *_d_b += 1 * c;
// CHECK-NEXT:         *_d_a += 0;
// CHECK-NEXT:         *_d_c += _t0 * 1;
// CHECK-NEXT:     }
// CHECK-NEXT: }

// Test 3: Comma expression yielding an lvalue target for assignment.
// (a, b) = c assigns to b; a remains unchanged.
// Returns b * 2.0 (which equals c * 2.0).
// Expected derivatives: da = 0, db = 0, dc = 2.
double comma_lvalue_assign(double a, double b, double c) {
  (a, b) = c;
  return b * 2.0;
}

// CHECK: void comma_lvalue_assign_grad(double a, double b, double c, double *_d_a, double *_d_b, double *_d_c) {
// CHECK-NEXT:     (a , b) = c;
// CHECK-NEXT:     *_d_b += 1 * 2.;
// CHECK-NEXT:     {
// CHECK-NEXT:         double _r_d0 = *_d_b;
// CHECK-NEXT:         *_d_b = 0.;
// CHECK-NEXT:         *_d_c += _r_d0;
// CHECK-NEXT:         *_d_a += 0;
// CHECK-NEXT:     }
// CHECK-NEXT: }

// Test 4: Comma expression nested within another arithmetic operation with itself.
// Evaluates to (b) * (b) = b^2.
// Expected derivatives: da = 0, db = 2 * b.
double comma_nested_self(double a, double b) {
  return (a, b) * (a, b);
}

// CHECK: void comma_nested_self_grad(double a, double b, double *_d_a, double *_d_b) {
// CHECK-NEXT:     {
// CHECK-NEXT:         *_d_b += 1 * (a , b);
// CHECK-NEXT:         *_d_a += 0;
// CHECK-NEXT:         *_d_b += b * 1;
// CHECK-NEXT:         *_d_a += 0;
// CHECK-NEXT:     }
// CHECK-NEXT: }

// Test 5: Nested / chained comma expressions ((a, b), c) * d.
// Evaluates to c * d.
// Expected derivatives: da = 0, db = 0, dc = d, dd = c.
double comma_nested_chain(double a, double b, double c, double d) {
  return ((a, b), c) * d;
}

// CHECK: void comma_nested_chain_grad(double a, double b, double c, double d, double *_d_a, double *_d_b, double *_d_c, double *_d_d) {
// CHECK-NEXT:     {
// CHECK-NEXT:         *_d_c += 1 * d;
// CHECK-NEXT:         *_d_b += 0;
// CHECK-NEXT:         *_d_a += 0;
// CHECK-NEXT:         *_d_d += c * 1;
// CHECK-NEXT:     }
// CHECK-NEXT: }

// Test 6: Comma expression yielding an lvalue target for assignment within an enclosing expression.
// Evaluates to ((a, b) = c) * d = c * d.
// Expected derivatives: da = 0, db = 0, dc = d, dd = c.
double comma_lvalue_expr(double a, double b, double c, double d) {
  return ((a, b) = c) * d;
}

// CHECK: void comma_lvalue_expr_grad(double a, double b, double c, double d, double *_d_a, double *_d_b, double *_d_c, double *_d_d) {
// CHECK-NEXT:     double _t0 = ((a , b) = c);
// CHECK-NEXT:     {
// CHECK-NEXT:         *_d_b += 1 * d;
// CHECK-NEXT:         double _r_d0 = *_d_b;
// CHECK-NEXT:         *_d_b = 0.;
// CHECK-NEXT:         *_d_c += _r_d0;
// CHECK-NEXT:         *_d_a += 0;
// CHECK-NEXT:         *_d_d += _t0 * 1;
// CHECK-NEXT:     }
// CHECK-NEXT: }

// Test 7: Nested comma expressions yielding an lvalue target for assignment.
// (a, (b, c)) = d assigns to c; a and b remain unchanged.
// Returns c * 3.0 (which equals d * 3.0).
// Expected derivatives: da = 0, db = 0, dc = 0, dd = 3.
double comma_lvalue_chain(double a, double b, double c, double d) {
  (a, (b, c)) = d;
  return c * 3.0;
}

// CHECK: void comma_lvalue_chain_grad(double a, double b, double c, double d, double *_d_a, double *_d_b, double *_d_c, double *_d_d) {
// CHECK-NEXT:     (a , (b , c)) = d;
// CHECK-NEXT:     *_d_c += 1 * 3.;
// CHECK-NEXT:     {
// CHECK-NEXT:         double _r_d0 = *_d_c;
// CHECK-NEXT:         *_d_c = 0.;
// CHECK-NEXT:         *_d_d += _r_d0;
// CHECK-NEXT:         *_d_b += 0;
// CHECK-NEXT:         *_d_a += 0;
// CHECK-NEXT:     }
// CHECK-NEXT: }

double comma_store(double x) {
  double a[3] = {x, x * x, 2 * x};
  double s = 0, t = 1;
  int j = 0;
  s += a[j] * x, t *= a[j++] * s;
  return s * t;
}

// CHECK: void comma_store_grad(double x, double *_d_x) {
// CHECK-NEXT:     double _d_a[3] = {0};
// CHECK-NEXT:     double a[3] = {x, x * x, 2 * x};
// CHECK-NEXT:     double _d_s = 0., _d_t = 0.;
// CHECK-NEXT:     double s = 0, t = 1;
// CHECK-NEXT:     int _d_j = 0;
// CHECK-NEXT:     int j = 0;
// CHECK-NEXT:     double _t2 = s;
// CHECK-NEXT:     s += a[j] * x;
// CHECK-NEXT:     double _t0 = t;
// CHECK-NEXT:     double _t1 = a[j++];
// CHECK-NEXT:     t *= _t1 * s;
// CHECK-NEXT:     {
// CHECK-NEXT:         _d_s += 1 * t;
// CHECK-NEXT:         _d_t += s * 1;
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         t = _t0;
// CHECK-NEXT:         double _r_d0 = _d_t;
// CHECK-NEXT:         _d_t = 0.;
// CHECK-NEXT:         _d_t += _r_d0 * (_t1 * s);
// CHECK-NEXT:         j--;
// CHECK-NEXT:         _d_a[j] += t * _r_d0 * s;
// CHECK-NEXT:         _d_s += _t1 * t * _r_d0;
// CHECK-NEXT:         _d_s += 0;
// CHECK-NEXT:         s = _t2;
// CHECK-NEXT:         double _r_d1 = _d_s;
// CHECK-NEXT:         _d_a[j] += _r_d1 * x;
// CHECK-NEXT:         *_d_x += a[j] * _r_d1;
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         *_d_x += _d_a[0];
// CHECK-NEXT:         *_d_x += _d_a[1] * x;
// CHECK-NEXT:         *_d_x += x * _d_a[1];
// CHECK-NEXT:         *_d_x += 2 * _d_a[2];
// CHECK-NEXT:     }
// CHECK-NEXT: }

double comma_body(double x) {
  double a[3] = {x, x * x, 2 * x};
  double s = 0, t = 1;
  int j = 0;
  for (int i = 0; i < 2; ++i) {
    (s += a[j] * x, t *= a[j++] * s), 0;
  }
  return s * t;
}

double comma_store_unaffected(double x) {
  double a[3] = {x, x * x, 2 * x};
  double s = x, t = 1;
  int j = 0, k = 0;
  (k++ + 1), t *= a[j++] * s;
  return s * t + a[k];
}

double comma_right_chain(double x) {
  double a[4] = {x, x * x, 2 * x, 3 * x}, t = x;
  int j = 1;
  (j = 2), ((j = 1), t *= a[j++] * x);
  return t;
}

double comma_subscript(double x) {
  int j = 0, k = 0;
  double a[2] = {x, x * x}, s = 1;
  double result = (j = 0, a)[(j++, (s *= a[k++] * x, 0))];
  return s + a[j] + result;
}

double comma_callee(double* a, int& j, double x) {
  j = 1, a[j] *= x;
  return a[0] + a[1];
}

// CHECK: inline double comma_callee_reverse_forw(double *a, int &j, double x, double *_d_a, int &_d_j, double _d_x, clad::restore_tracker &_tracker0) {
// CHECK-NEXT:     _tracker0.store(j);
// CHECK-NEXT:     j = 1;
// CHECK-NEXT:     _tracker0.store(a[j]);
// CHECK-NEXT:     a[j] *= x;
// CHECK-NEXT:     return a[0] + a[1];
// CHECK-NEXT: }

double comma_call(double x) {
  double a[3] = {x, x * x, 2 * x};
  int j = 0;
  double r = comma_callee(a, j, x);
  return r * a[1];
}

int main() {
  {
    auto df = clad::gradient(comma_basic);
    double da = 0, db = 0, dc = 0;
    df.execute(10.0, 5.0, 2.0, &da, &db, &dc);
    printf("comma_basic: {%.2f, %.2f, %.2f}\n", da, db, dc);
    // CHECK-EXEC: comma_basic: {0.00, 2.00, 5.00}
  }

  {
    auto df = clad::gradient(comma_side_effect);
    double da = 0, db = 0, dc = 0;
    df.execute(10.0, 5.0, 2.0, &da, &db, &dc);
    printf("comma_side_effect: {%.2f, %.2f, %.2f}\n", da, db, dc);
    // CHECK-EXEC: comma_side_effect: {0.00, 2.00, 5.00}
  }

  {
    auto df = clad::gradient(comma_lvalue_assign);
    double da = 0, db = 0, dc = 0;
    df.execute(10.0, 5.0, 3.0, &da, &db, &dc);
    printf("comma_lvalue_assign: {%.2f, %.2f, %.2f}\n", da, db, dc);
    // CHECK-EXEC: comma_lvalue_assign: {0.00, 0.00, 2.00}
  }

  {
    auto df = clad::gradient(comma_nested_self);
    double da = 0, db = 0;
    df.execute(10.0, 3.0, &da, &db);
    printf("comma_nested_self: {%.2f, %.2f}\n", da, db);
    // CHECK-EXEC: comma_nested_self: {0.00, 6.00}
  }

  {
    auto df = clad::gradient(comma_nested_chain);
    double da = 0, db = 0, dc = 0, dd = 0;
    df.execute(10.0, 5.0, 3.0, 2.0, &da, &db, &dc, &dd);
    printf("comma_nested_chain: {%.2f, %.2f, %.2f, %.2f}\n", da, db, dc, dd);
    // CHECK-EXEC: comma_nested_chain: {0.00, 0.00, 2.00, 3.00}
  }

  {
    auto df = clad::gradient(comma_lvalue_expr);
    double da = 0, db = 0, dc = 0, dd = 0;
    df.execute(10.0, 5.0, 3.0, 2.0, &da, &db, &dc, &dd);
    printf("comma_lvalue_expr: {%.2f, %.2f, %.2f, %.2f}\n", da, db, dc, dd);
    // CHECK-EXEC: comma_lvalue_expr: {0.00, 0.00, 2.00, 3.00}
  }

  {
    auto df = clad::gradient(comma_lvalue_chain);
    double da = 0, db = 0, dc = 0, dd = 0;
    df.execute(10.0, 5.0, 3.0, 2.0, &da, &db, &dc, &dd);
    printf("comma_lvalue_chain: {%.2f, %.2f, %.2f, %.2f}\n", da, db, dc, dd);
    // CHECK-EXEC: comma_lvalue_chain: {0.00, 0.00, 0.00, 3.00}
  }

  {
    auto df = clad::gradient(comma_store);
    double dx = 0;
    df.execute(-2.0, &dx);
    printf("comma_store: %.2f\n", dx);
    // CHECK-EXEC: comma_store: 80.00
  }

  {
    auto df = clad::gradient(comma_body);
    double dx = 0;
    df.execute(-2.0, &dx);
    printf("comma_body: %.2f\n", dx);
    // CHECK-EXEC: comma_body: 3328.00
  }

  {
    auto df = clad::gradient(comma_store_unaffected);
    double dx = 0;
    df.execute(-2.0, &dx);
    printf("comma_store_unaffected: %.2f\n", dx);
    // CHECK-EXEC: comma_store_unaffected: 8.00
  }

  {
    auto df = clad::gradient(comma_right_chain);
    double dx = 0;
    df.execute(-2.0, &dx);
    printf("comma_right_chain: %.2f\n", dx);
    // CHECK-EXEC: comma_right_chain: -32.00
  }

  {
    auto df = clad::gradient(comma_subscript);
    double dx = 0;
    df.execute(-2.0, &dx);
    printf("comma_subscript: %.2f\n", dx);
    // CHECK-EXEC: comma_subscript: -7.00
  }

  {
    auto df = clad::gradient(comma_call);
    double dx = 0;
    df.execute(-2.0, &dx);
    printf("comma_call: %.2f\n", dx);
    // CHECK-EXEC: comma_call: -224.00
  }

  return 0;
}
