// RUN: %cladclang %s -I%S/../../include -oDiscreteAssignments.out 2>&1 | %filecheck %s
// RUN: ./DiscreteAssignments.out | %filecheck_exec %s
// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -disable-tbr %s -I%S/../../include -oDiscreteAssignments.out
// RUN: ./DiscreteAssignments.out | %filecheck_exec %s

// `%=`, `&=`, `|=`, `^=`, `<<=` and `>>=` are piecewise constant in both
// operands: the derivative stops at them, and the value they leave behind is
// what later uses of the left side see.

#include "clad/Differentiator/Differentiator.h"

#include <cstdio>

double f_rem(double x, int y) {
  int n = 10;
  n %= y;
  return x * n;
}

// CHECK: void f_rem_grad_0(double x, int y, double *_d_x) {
// CHECK-NEXT:     int _d_y = 0;
// CHECK-NEXT:     int _d_n = 0;
// CHECK-NEXT:     int n = 10;
// CHECK-NEXT:     n %= y;
// CHECK-NEXT:     {
// CHECK-NEXT:         *_d_x += 1 * n;
// CHECK-NEXT:         _d_n += x * 1;
// CHECK-NEXT:     }
// CHECK-NEXT:     _d_n = 0;
// CHECK-NEXT: }

double f_and(double x, int y) {
  int n = 7;
  n &= y;
  return x * n;
}

// CHECK: void f_and_grad_0(double x, int y, double *_d_x) {
// CHECK-NEXT:     int _d_y = 0;
// CHECK-NEXT:     int _d_n = 0;
// CHECK-NEXT:     int n = 7;
// CHECK-NEXT:     n &= y;
// CHECK-NEXT:     {
// CHECK-NEXT:         *_d_x += 1 * n;
// CHECK-NEXT:         _d_n += x * 1;
// CHECK-NEXT:     }
// CHECK-NEXT:     _d_n = 0;
// CHECK-NEXT: }

double f_or(double x, int y) {
  int n = 3;
  n |= y;
  return x * n;
}

// CHECK: void f_or_grad_0(double x, int y, double *_d_x) {
// CHECK-NEXT:     int _d_y = 0;
// CHECK-NEXT:     int _d_n = 0;
// CHECK-NEXT:     int n = 3;
// CHECK-NEXT:     n |= y;
// CHECK-NEXT:     {
// CHECK-NEXT:         *_d_x += 1 * n;
// CHECK-NEXT:         _d_n += x * 1;
// CHECK-NEXT:     }
// CHECK-NEXT:     _d_n = 0;
// CHECK-NEXT: }

double f_xor(double x, int y) {
  int n = 5;
  n ^= y;
  return x * n;
}

// CHECK: void f_xor_grad_0(double x, int y, double *_d_x) {
// CHECK-NEXT:     int _d_y = 0;
// CHECK-NEXT:     int _d_n = 0;
// CHECK-NEXT:     int n = 5;
// CHECK-NEXT:     n ^= y;
// CHECK-NEXT:     {
// CHECK-NEXT:         *_d_x += 1 * n;
// CHECK-NEXT:         _d_n += x * 1;
// CHECK-NEXT:     }
// CHECK-NEXT:     _d_n = 0;
// CHECK-NEXT: }

double f_shl(double x, int y) {
  int n = 3;
  n <<= y;
  return x * n;
}

// CHECK: void f_shl_grad_0(double x, int y, double *_d_x) {
// CHECK-NEXT:     int _d_y = 0;
// CHECK-NEXT:     int _d_n = 0;
// CHECK-NEXT:     int n = 3;
// CHECK-NEXT:     n <<= y;
// CHECK-NEXT:     {
// CHECK-NEXT:         *_d_x += 1 * n;
// CHECK-NEXT:         _d_n += x * 1;
// CHECK-NEXT:     }
// CHECK-NEXT:     _d_n = 0;
// CHECK-NEXT: }

double f_shr(double x, int y) {
  int n = 20;
  n >>= y;
  return x * n;
}

// CHECK: void f_shr_grad_0(double x, int y, double *_d_x) {
// CHECK-NEXT:     int _d_y = 0;
// CHECK-NEXT:     int _d_n = 0;
// CHECK-NEXT:     int n = 20;
// CHECK-NEXT:     n >>= y;
// CHECK-NEXT:     {
// CHECK-NEXT:         *_d_x += 1 * n;
// CHECK-NEXT:         _d_n += x * 1;
// CHECK-NEXT:     }
// CHECK-NEXT:     _d_n = 0;
// CHECK-NEXT: }

// The left side is read before and after the assignment, every iteration.
double f_loop(double x, int count) {
  double r = x;
  int mask = 15;
  for (int i = 0; i < count; ++i) {
    mask %= 4;
    r += x * mask;
    mask += 5;
  }
  return r;
}

// CHECK: void f_loop_grad_0(double x, int count, double *_d_x) {
// CHECK-NEXT:     int _d_count = 0;
// CHECK-NEXT:     int _d_i = 0;
// CHECK-NEXT:     int i = 0;
// CHECK-NEXT:     clad::tape<int> _t1 = {};
// CHECK-NEXT:     double _d_r = 0.;
// CHECK-NEXT:     double r = x;
// CHECK-NEXT:     int _d_mask = 0;
// CHECK-NEXT:     int mask = 15;
// CHECK-NEXT:     unsigned long _t0;
// CHECK-NEXT:     for (i = 0; i < count; ++i) {
// CHECK-NEXT:         mask %= 4;
// CHECK-NEXT:         r += x * mask;
// CHECK-NEXT:         clad::push(_t1, mask);
// CHECK-NEXT:         mask += 5;
// CHECK-NEXT:     }
// CHECK-NEXT:     _d_r += 1;
// CHECK-NEXT:     for (_t0 = count > 0 ? (unsigned long)count : 0UL; _t0; _t0--) {
// CHECK-NEXT:         mask = clad::pop(_t1);
// CHECK-NEXT:         {
// CHECK-NEXT:             *_d_x += _d_r * mask;
// CHECK-NEXT:             _d_mask += x * _d_r;
// CHECK-NEXT:         }
// CHECK-NEXT:         _d_mask = 0;
// CHECK-NEXT:     }
// CHECK-NEXT:     *_d_x += _d_r;
// CHECK-NEXT: }

// The assignment's result feeds the product.
double f_nested(double x, int y) {
  int n = 10;
  return x * (n %= y);
}

// CHECK: void f_nested_grad_0(double x, int y, double *_d_x) {
// CHECK-NEXT:     int _d_y = 0;
// CHECK-NEXT:     int _d_n = 0;
// CHECK-NEXT:     int n = 10;
// CHECK-NEXT:     double _t0 = (n %= y);
// CHECK-NEXT:     {
// CHECK-NEXT:         *_d_x += 1 * _t0;
// CHECK-NEXT:         _d_n += x * 1;
// CHECK-NEXT:         _d_n = 0;
// CHECK-NEXT:     }
// CHECK-NEXT: }

// The assignment's result is the right side of another compound assignment.
double f_under_mul_assign(double x, int y) {
  double r = x;
  int n = 12;
  r *= (n ^= y);
  return r;
}

// CHECK: void f_under_mul_assign_grad_0(double x, int y, double *_d_x) {
// CHECK-NEXT:     int _d_y = 0;
// CHECK-NEXT:     double _d_r = 0.;
// CHECK-NEXT:     double r = x;
// CHECK-NEXT:     int _d_n = 0;
// CHECK-NEXT:     int n = 12;
// CHECK-NEXT:     double _t0 = r;
// CHECK-NEXT:     r *= (n ^= y);
// CHECK-NEXT:     _d_r += 1;
// CHECK-NEXT:     {
// CHECK-NEXT:         r = _t0;
// CHECK-NEXT:         double _r_d0 = _d_r;
// CHECK-NEXT:         _d_r = 0.;
// CHECK-NEXT:         _d_r += _r_d0 * n;
// CHECK-NEXT:         _d_n += r * _r_d0;
// CHECK-NEXT:         _d_n = 0;
// CHECK-NEXT:     }
// CHECK-NEXT:     *_d_x += _d_r;
// CHECK-NEXT: }

// An element whose adjoint is not a plain variable.
double f_element(double x, int y) {
  int a[2] = {10, 20};
  int i = 1;
  a[i] %= y;
  return x * a[i];
}

// CHECK: void f_element_grad_0(double x, int y, double *_d_x) {
// CHECK-NEXT:     int _d_y = 0;
// CHECK-NEXT:     int _d_a[2] = {0};
// CHECK-NEXT:     int a[2] = {10, 20};
// CHECK-NEXT:     int _d_i = 0;
// CHECK-NEXT:     int i = 1;
// CHECK-NEXT:     a[i] %= y;
// CHECK-NEXT:     {
// CHECK-NEXT:         *_d_x += 1 * a[i];
// CHECK-NEXT:         _d_a[i] += x * 1;
// CHECK-NEXT:     }
// CHECK-NEXT:     _d_a[i] = 0;
// CHECK-NEXT: }

// A bit-field, which no reference can bind to.
struct Flags {
  int bits : 8;
};
double f_bitfield(double x, int mask) {
  Flags s;
  s.bits = 12;
  s.bits &= mask;
  return x * s.bits;
}

// CHECK: void f_bitfield_grad_0(double x, int mask, double *_d_x) {
// CHECK-NEXT:     int _d_mask = 0;
// CHECK-NEXT:     Flags _d_s = {0};
// CHECK-NEXT:     Flags s;
// CHECK-NEXT:     s.bits = 12;
// CHECK-NEXT:     s.bits &= mask;
// CHECK-NEXT:     {
// CHECK-NEXT:         *_d_x += 1 * s.bits;
// CHECK-NEXT:         _d_s.bits += x * 1;
// CHECK-NEXT:     }
// CHECK-NEXT:     _d_s.bits = 0;
// CHECK-NEXT:     {
// CHECK-NEXT:         int _r_d0 = _d_s.bits;
// CHECK-NEXT:         _d_s.bits = 0;
// CHECK-NEXT:     }
// CHECK-NEXT: }

#define TEST(F, ...)                                                           \
  {                                                                            \
    double dx = 0;                                                             \
    clad::gradient(F, "x").execute(__VA_ARGS__, &dx);                          \
    printf(#F " %g\n", dx);                                                    \
  }

int main() {
  TEST(f_rem, 1.5, 3);              // CHECK-EXEC: f_rem 1
  TEST(f_and, 1.5, 11);             // CHECK-EXEC: f_and 3
  TEST(f_or, 1.5, 4);               // CHECK-EXEC: f_or 7
  TEST(f_xor, 1.5, 12);             // CHECK-EXEC: f_xor 9
  TEST(f_shl, 1.5, 2);              // CHECK-EXEC: f_shl 12
  TEST(f_shr, 1.5, 2);              // CHECK-EXEC: f_shr 5
  TEST(f_loop, 1.5, 3);             // CHECK-EXEC: f_loop 5
  TEST(f_nested, 1.5, 3);           // CHECK-EXEC: f_nested 1
  TEST(f_under_mul_assign, 1.5, 5); // CHECK-EXEC: f_under_mul_assign 9
  TEST(f_element, 1.5, 3);          // CHECK-EXEC: f_element 2
  TEST(f_bitfield, 1.5, 10);        // CHECK-EXEC: f_bitfield 8
}
