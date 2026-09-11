// RUN: %cladclang %s -I%S/../../include -oCompoundBitwiseOps.out 2>&1 | %filecheck %s
// RUN: ./CompoundBitwiseOps.out | %filecheck_exec %s

#include "clad/Differentiator/Differentiator.h"
#include <iostream>

// 1. %= operator
double f_rem(double x, int y) {
  int n = 10;
  n %= y;
  return x * n;
}

// CHECK-LABEL: f_rem_grad
// CHECK-NEXT:     int _d_y = 0;
// CHECK-NEXT:     int _d_n = 0;
// CHECK-NEXT:     int n = 10;
// CHECK-NEXT:     (n %= y);
// CHECK-NEXT:     {
// CHECK-NEXT:         *_d_x += 1 * n;
// CHECK-NEXT:         _d_n += x * 1;
// CHECK-NEXT:     }
// CHECK-NEXT:     _d_n = 0;
// CHECK-NEXT: }


// 2. &= operator
double f_and(double x, int y) {
  int n = y;
  n &= 7;
  return x * n;
}

// CHECK-LABEL: f_and_grad
// CHECK-NEXT:     int _d_y = 0;
// CHECK-NEXT:     int _d_n = 0;
// CHECK-NEXT:     int n = y;
// CHECK-NEXT:     (n &= 7);
// CHECK-NEXT:     {
// CHECK-NEXT:         *_d_x += 1 * n;
// CHECK-NEXT:         _d_n += x * 1;
// CHECK-NEXT:     }
// CHECK-NEXT:     _d_n = 0;
// CHECK-NEXT:     _d_y += _d_n;
// CHECK-NEXT: }


// 3. |= operator
double f_or(double x, int y) {
  int n = y;
  n |= 3;
  return x * n;
}

// CHECK-LABEL: f_or_grad
// CHECK-NEXT:     int _d_y = 0;
// CHECK-NEXT:     int _d_n = 0;
// CHECK-NEXT:     int n = y;
// CHECK-NEXT:     (n |= 3);
// CHECK-NEXT:     {
// CHECK-NEXT:         *_d_x += 1 * n;
// CHECK-NEXT:         _d_n += x * 1;
// CHECK-NEXT:     }
// CHECK-NEXT:     _d_n = 0;
// CHECK-NEXT:     _d_y += _d_n;
// CHECK-NEXT: }


// 4. ^= operator
double f_xor(double x, int y) {
  int n = y;
  n ^= 5;
  return x * n;
}

// CHECK-LABEL: f_xor_grad
// CHECK-NEXT:     int _d_y = 0;
// CHECK-NEXT:     int _d_n = 0;
// CHECK-NEXT:     int n = y;
// CHECK-NEXT:     (n ^= 5);
// CHECK-NEXT:     {
// CHECK-NEXT:         *_d_x += 1 * n;
// CHECK-NEXT:         _d_n += x * 1;
// CHECK-NEXT:     }
// CHECK-NEXT:     _d_n = 0;
// CHECK-NEXT:     _d_y += _d_n;
// CHECK-NEXT: }


// 5. <<= operator
double f_shl(double x, int y) {
  int n = y;
  n <<= 2;
  return x * n;
}

// CHECK-LABEL: f_shl_grad
// CHECK-NEXT:     int _d_y = 0;
// CHECK-NEXT:     int _d_n = 0;
// CHECK-NEXT:     int n = y;
// CHECK-NEXT:     (n <<= 2);
// CHECK-NEXT:     {
// CHECK-NEXT:         *_d_x += 1 * n;
// CHECK-NEXT:         _d_n += x * 1;
// CHECK-NEXT:     }
// CHECK-NEXT:     _d_n = 0;
// CHECK-NEXT:     _d_y += _d_n;
// CHECK-NEXT: }


// 6. >>= operator
double f_shr(double x, int y) {
  int n = y;
  n >>= 1;
  return x * n;
}

// CHECK-LABEL: f_shr_grad
// CHECK-NEXT:     int _d_y = 0;
// CHECK-NEXT:     int _d_n = 0;
// CHECK-NEXT:     int n = y;
// CHECK-NEXT:     (n >>= 1);
// CHECK-NEXT:     {
// CHECK-NEXT:         *_d_x += 1 * n;
// CHECK-NEXT:         _d_n += x * 1;
// CHECK-NEXT:     }
// CHECK-NEXT:     _d_n = 0;
// CHECK-NEXT:     _d_y += _d_n;
// CHECK-NEXT: }


// 7. Compound assignment inside a loop
double f_loop(double x, int n) {
  double res = x;
  int mask = 15;
  for (int i = 0; i < n; ++i) {
    mask %= 4;
    res += x * mask;
    mask += 5;
  }
  return res;
}

// CHECK-LABEL: f_loop_grad
// CHECK-NEXT:     int _d_n = 0;
// CHECK-NEXT:     int _d_i = 0;
// CHECK-NEXT:     int i = 0;
// CHECK-NEXT:     clad::tape<int> _t1 = {};
// CHECK-NEXT:     double _d_res = 0.;
// CHECK-NEXT:     double res = x;
// CHECK-NEXT:     int _d_mask = 0;
// CHECK-NEXT:     int mask = 15;
// CHECK-NEXT:     unsigned {{int|long}} _t0 = 0;
// CHECK-NEXT:     for (i = 0; i < n; ++i) {
// CHECK-NEXT:         _t0++;
// CHECK-NEXT:         (mask %= 4);
// CHECK-NEXT:         res += x * mask;
// CHECK-NEXT:         clad::push(_t1, mask);
// CHECK-NEXT:         mask += 5;
// CHECK-NEXT:     }
// CHECK-NEXT:     _d_res += 1;
// CHECK-NEXT:     for (; _t0; _t0--) {
// CHECK-NEXT:         mask = clad::pop(_t1);
// CHECK-NEXT:         {
// CHECK-NEXT:             *_d_x += _d_res * mask;
// CHECK-NEXT:             _d_mask += x * _d_res;
// CHECK-NEXT:         }
// CHECK-NEXT:         _d_mask = 0;
// CHECK-NEXT:     }
// CHECK-NEXT:     *_d_x += _d_res;
// CHECK-NEXT: }


// 8. Nested %= expression
double f_nested_rem(double x, int y) {
  int n = 10;
  return x * (n %= y);
}

// CHECK-LABEL: f_nested_rem_grad
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


// 9. Nested &= expression
double f_nested_and(double x, int y) {
  int n = 11;
  return x * (n &= y);
}

// CHECK-LABEL: f_nested_and_grad
// CHECK-NEXT:     int _d_y = 0;
// CHECK-NEXT:     int _d_n = 0;
// CHECK-NEXT:     int n = 11;
// CHECK-NEXT:     double _t0 = (n &= y);
// CHECK-NEXT:     {
// CHECK-NEXT:         *_d_x += 1 * _t0;
// CHECK-NEXT:         _d_n += x * 1;
// CHECK-NEXT:         _d_n = 0;
// CHECK-NEXT:     }
// CHECK-NEXT: }


int next(int& y) {
  return ++y;
}

// 10. Nested compound assignment with RHS side effect
double f_nested_side_effect(double x, int y) {
  int n = 10;
  return x * (n %= next(y));
}

// CHECK-LABEL: f_nested_side_effect_grad
// CHECK-NEXT:     int _d_y = 0;
// CHECK-NEXT:     int _d_n = 0;
// CHECK-NEXT:     int n = 10;
// CHECK-NEXT:     int _t1 = y;
// CHECK-NEXT:     double _t0 = (n %= next(y));
// CHECK-NEXT:     {
// CHECK-NEXT:         *_d_x += 1 * _t0;
// CHECK-NEXT:         _d_n += x * 1;
// CHECK-NEXT:         _d_n = 0;
// CHECK-NEXT:         y = _t1;
// CHECK-NEXT:         next_pullback(y, 0, &_d_y);
// CHECK-NEXT:     }
// CHECK-NEXT: }


// 11. Compound assignment inside a loop with RHS side effect
double f_loop_side_effect(double x, int n, int y) {
  double res = x;
  int mask = 15;
  for (int i = 0; i < n; ++i) {
    mask %= next(y);
    res += x * mask;
  }
  return res;
}

// CHECK-LABEL: f_loop_side_effect_grad
// CHECK-NEXT:     int _d_n = 0;
// CHECK-NEXT:     int _d_y = 0;
// CHECK-NEXT:     int _d_i = 0;
// CHECK-NEXT:     int i = 0;
// CHECK-NEXT:     clad::tape<int> _t1 = {};
// CHECK-NEXT:     clad::tape<int> _t2 = {};
// CHECK-NEXT:     double _d_res = 0.;
// CHECK-NEXT:     double res = x;
// CHECK-NEXT:     int _d_mask = 0;
// CHECK-NEXT:     int mask = 15;
// CHECK-NEXT:     unsigned {{int|long}} _t0 = 0;
// CHECK-NEXT:     for (i = 0; i < n; ++i) {
// CHECK-NEXT:         _t0++;
// CHECK-NEXT:         clad::push(_t1, mask);
// CHECK-NEXT:         clad::push(_t2, y);
// CHECK-NEXT:         (mask %= next(y));
// CHECK-NEXT:         res += x * mask;
// CHECK-NEXT:     }
// CHECK-NEXT:     _d_res += 1;
// CHECK-NEXT:     for (; _t0; _t0--) {
// CHECK-NEXT:         {
// CHECK-NEXT:             *_d_x += _d_res * mask;
// CHECK-NEXT:             _d_mask += x * _d_res;
// CHECK-NEXT:         }
// CHECK-NEXT:         {
// CHECK-NEXT:             mask = clad::pop(_t1);
// CHECK-NEXT:             _d_mask = 0;
// CHECK-NEXT:             y = clad::pop(_t2);
// CHECK-NEXT:             next_pullback(y, 0, &_d_y);
// CHECK-NEXT:         }
// CHECK-NEXT:     }
// CHECK-NEXT:     *_d_x += _d_res;
// CHECK-NEXT: }


// 12. Nested XOR under parent compound assignment (*=)
double nested_xor_mul_assign(double x, int y) {
  double result = x;
  int n = 12;
  result *= (n ^= y);
  return result;
}

// CHECK-LABEL: nested_xor_mul_assign_grad
// CHECK-NEXT:     int _d_y = 0;
// CHECK-NEXT:     int _t1;
// CHECK-NEXT:     double _d_result = 0.;
// CHECK-NEXT:     double result = x;
// CHECK-NEXT:     int _d_n = 0;
// CHECK-NEXT:     int n = 12;
// CHECK-NEXT:     double _t0 = result;
// CHECK-NEXT:     result *= (_t1 = n ^= y , n);
// CHECK-NEXT:     _d_result += 1;
// CHECK-NEXT:     {
// CHECK-NEXT:         result = _t0;
// CHECK-NEXT:         double _r_d0 = _d_result;
// CHECK-NEXT:         _d_result = 0.;
// CHECK-NEXT:         _d_result += _r_d0 * _t1;
// CHECK-NEXT:         _d_n += result * _r_d0;
// CHECK-NEXT:         _d_n = 0;
// CHECK-NEXT:     }
// CHECK-NEXT:     *_d_x += _d_result;
// CHECK-NEXT: }


// 13. Nested left shift under parent compound assignment (*=)
double nested_shl_mul_assign(double x, int shift) {
  double result = x;
  int n = 3;
  result *= (n <<= shift);
  return result;
}

// CHECK-LABEL: nested_shl_mul_assign_grad
// CHECK-NEXT:     int _d_shift = 0;
// CHECK-NEXT:     int _t1;
// CHECK-NEXT:     double _d_result = 0.;
// CHECK-NEXT:     double result = x;
// CHECK-NEXT:     int _d_n = 0;
// CHECK-NEXT:     int n = 3;
// CHECK-NEXT:     double _t0 = result;
// CHECK-NEXT:     result *= (_t1 = n <<= shift , n);
// CHECK-NEXT:     _d_result += 1;
// CHECK-NEXT:     {
// CHECK-NEXT:         result = _t0;
// CHECK-NEXT:         double _r_d0 = _d_result;
// CHECK-NEXT:         _d_result = 0.;
// CHECK-NEXT:         _d_result += _r_d0 * _t1;
// CHECK-NEXT:         _d_n += result * _r_d0;
// CHECK-NEXT:         _d_n = 0;
// CHECK-NEXT:     }
// CHECK-NEXT:     *_d_x += _d_result;
// CHECK-NEXT: }



// 14. Nested right shift under parent compound assignment (*=)
double nested_shr_mul_assign(double x, int shift) {
  double result = x;
  int n = 20;
  result *= (n >>= shift);
  return result;
}

// CHECK-LABEL: nested_shr_mul_assign_grad
// CHECK-NEXT:     int _d_shift = 0;
// CHECK-NEXT:     int _t1;
// CHECK-NEXT:     double _d_result = 0.;
// CHECK-NEXT:     double result = x;
// CHECK-NEXT:     int _d_n = 0;
// CHECK-NEXT:     int n = 20;
// CHECK-NEXT:     double _t0 = result;
// CHECK-NEXT:     result *= (_t1 = n >>= shift , n);
// CHECK-NEXT:     _d_result += 1;
// CHECK-NEXT:     {
// CHECK-NEXT:         result = _t0;
// CHECK-NEXT:         double _r_d0 = _d_result;
// CHECK-NEXT:         _d_result = 0.;
// CHECK-NEXT:         _d_result += _r_d0 * _t1;
// CHECK-NEXT:         _d_n += result * _r_d0;
// CHECK-NEXT:         _d_n = 0;
// CHECK-NEXT:     }
// CHECK-NEXT:     *_d_x += _d_result;
// CHECK-NEXT: }



// 15. Narrowing conversion (unsigned char)
double nested_narrow_shift(double x) {
  double result = x;
  unsigned char n = 200;
  result *= (n <<= 1);
  return result;
}

// CHECK-LABEL: nested_narrow_shift_grad
// CHECK-NEXT:     unsigned char _t1;
// CHECK-NEXT:     double _d_result = 0.;
// CHECK-NEXT:     double result = x;
// CHECK-NEXT:     unsigned char _d_n = 0;
// CHECK-NEXT:     unsigned char n = 200;
// CHECK-NEXT:     double _t0 = result;
// CHECK-NEXT:     result *= (_t1 = n <<= 1 , n);
// CHECK-NEXT:     _d_result += 1;
// CHECK-NEXT:     {
// CHECK-NEXT:         result = _t0;
// CHECK-NEXT:         double _r_d0 = _d_result;
// CHECK-NEXT:         _d_result = 0.;
// CHECK-NEXT:         _d_result += _r_d0 * _t1;
// CHECK-NEXT:         _d_n += result * _r_d0;
// CHECK-NEXT:         _d_n = 0;
// CHECK-NEXT:     }
// CHECK-NEXT:     *_d_x += _d_result;
// CHECK-NEXT: }



// 16. Loop-aware preservation for discrete compound assignment
double nested_xor_loop(double x, int count, int y) {
  double result = x;
  int n = 12;
  for (int i = 0; i < count; ++i)
    result *= (n ^= y);
  return result;
}

// CHECK-LABEL: nested_xor_loop_grad
// CHECK-NEXT:     int _d_count = 0;
// CHECK-NEXT:     int _d_y = 0;
// CHECK-NEXT:     int _d_i = 0;
// CHECK-NEXT:     int i = 0;
// CHECK-NEXT:     clad::tape<double> _t1 = {};
// CHECK-NEXT:     clad::tape<int> _t2 = {};
// CHECK-NEXT:     clad::tape<int> _t3 = {};
// CHECK-NEXT:     double _d_result = 0.;
// CHECK-NEXT:     double result = x;
// CHECK-NEXT:     int _d_n = 0;
// CHECK-NEXT:     int n = 12;
// CHECK-NEXT:     unsigned {{int|long}} _t0 = 0;
// CHECK-NEXT:     for (i = 0; i < count; ++i) {
// CHECK-NEXT:         _t0++;
// CHECK-NEXT:         clad::push(_t1, result);
// CHECK-NEXT:         clad::push(_t2, n);
// CHECK-NEXT:         result *= (clad::push(_t3, n ^= y) , n);
// CHECK-NEXT:     }
// CHECK-NEXT:     _d_result += 1;
// CHECK-NEXT:     for (; _t0; _t0--) {
// CHECK-NEXT:         result = clad::pop(_t1);
// CHECK-NEXT:         double _r_d0 = _d_result;
// CHECK-NEXT:         _d_result = 0.;
// CHECK-NEXT:         _d_result += _r_d0 * clad::back(_t3);
// CHECK-NEXT:         _d_n += result * _r_d0;
// CHECK-NEXT:         n = clad::pop(_t2);
// CHECK-NEXT:         _d_n = 0;
// CHECK-NEXT:         clad::pop(_t3);
// CHECK-NEXT:     }
// CHECK-NEXT:     *_d_x += _d_result;
// CHECK-NEXT: }


// 17. Bit-field LHS: reference binding is ill-formed ([class.bit]/5),
// so the generated code must not bind T& to the result.
struct Bits { int flags : 8; };
double f_bitfield(double x, int mask) {
  Bits b;
  b.flags = 12;
  b.flags &= mask;
  return x * b.flags;
}

// CHECK-LABEL: f_bitfield_grad
// CHECK-NOT: int &{{.*}} = b.flags
// CHECK-NEXT:     int _d_mask = 0;
// CHECK-NEXT:     Bits _d_b = {0};
// CHECK-NEXT:     Bits b;
// CHECK-NEXT:     b.flags = 12;
// CHECK-NEXT:     (b.flags &= mask);
// CHECK-NEXT:     {
// CHECK-NEXT:         *_d_x += 1 * b.flags;
// CHECK-NEXT:         _d_b.flags += x * 1;
// CHECK-NEXT:     }
// CHECK-NEXT:     _d_b.flags = 0;
// CHECK-NEXT:     {
// CHECK-NEXT:         int _r_d0 = _d_b.flags;
// CHECK-NEXT:         _d_b.flags = 0;
// CHECK-NEXT:     }
// CHECK-NEXT: }


// 18. Array subscript LHS
double f_arr_rem(double x, int y) {
  int arr[2] = {10, 20};
  arr[0] %= y;
  return x * arr[0];
}

// CHECK-LABEL: f_arr_rem_grad
// CHECK-NEXT:     int _d_y = 0;
// CHECK-NEXT:     int _d_arr[2] = {0};
// CHECK-NEXT:     int arr[2] = {10, 20};
// CHECK-NEXT:     (arr[0] %= y);
// CHECK-NEXT:     {
// CHECK-NEXT:         *_d_x += 1 * arr[0];
// CHECK-NEXT:         _d_arr[0] += x * 1;
// CHECK-NEXT:     }
// CHECK-NEXT:     _d_arr[0] = 0;
// CHECK-NEXT: }



// 19. Nested bit-field value consumption under parent *=.
// Must not bind T& to the bit-field; parent consumes the post-assign value.
double nested_bitfield_mul_assign(double x, int mask) {
  double result = x;
  Bits b;
  b.flags = 12;
  result *= (b.flags &= mask);
  return result;
}

// CHECK-LABEL: nested_bitfield_mul_assign_grad
// CHECK-NOT: int &{{.*}} = b.flags
// CHECK-NEXT:     int _d_mask = 0;
// CHECK-NEXT:     int _t1;
// CHECK-NEXT:     double _d_result = 0.;
// CHECK-NEXT:     double result = x;
// CHECK-NEXT:     Bits _d_b = {0};
// CHECK-NEXT:     Bits b;
// CHECK-NEXT:     b.flags = 12;
// CHECK-NEXT:     double _t0 = result;
// CHECK-NEXT:     result *= (_t1 = b.flags &= mask , b.flags);
// CHECK-NEXT:     _d_result += 1;
// CHECK-NEXT:     {
// CHECK-NEXT:         result = _t0;
// CHECK-NEXT:         double _r_d1 = _d_result;
// CHECK-NEXT:         _d_result = 0.;
// CHECK-NEXT:         _d_result += _r_d1 * _t1;
// CHECK-NEXT:         _d_b.flags += result * _r_d1;
// CHECK-NEXT:         _d_b.flags = 0;
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         int _r_d0 = _d_b.flags;
// CHECK-NEXT:         _d_b.flags = 0;
// CHECK-NEXT:     }
// CHECK-NEXT:     *_d_x += _d_result;
// CHECK-NEXT: }




// --- Astra review findings coverage (diff w.r.t. x) ---

// Finding 1 BLOCKER: unparenthesized discrete under *= in a loop.
// Must not pop-before-back; expected gradient 108 (same as nested_xor_loop).
double f_unparen_xor_mul_loop(double x, int count, int y) {
  double r = x;
  int n = 12;
  for (int i = 0; i < count; ++i)
    r *= n ^= y;
  return r;
}

// CHECK-LABEL: f_unparen_xor_mul_loop_grad
// CHECK-NEXT:     int _d_count = 0;
// CHECK-NEXT:     int _d_y = 0;
// CHECK-NEXT:     int _d_i = 0;
// CHECK-NEXT:     int i = 0;
// CHECK-NEXT:     clad::tape<double> _t1 = {};
// CHECK-NEXT:     clad::tape<int> _t2 = {};
// CHECK-NEXT:     clad::tape<int> _t3 = {};
// CHECK-NEXT:     double _d_r = 0.;
// CHECK-NEXT:     double r = x;
// CHECK-NEXT:     int _d_n = 0;
// CHECK-NEXT:     int n = 12;
// CHECK-NEXT:     unsigned {{int|long}} _t0 = 0;
// CHECK-NEXT:     for (i = 0; i < count; ++i) {
// CHECK-NEXT:         _t0++;
// CHECK-NEXT:         clad::push(_t1, r);
// CHECK-NEXT:         clad::push(_t2, n);
// CHECK-NEXT:         r *= (clad::push(_t3, n ^= y) , n);
// CHECK-NEXT:     }
// CHECK-NEXT:     _d_r += 1;
// CHECK-NEXT:     for (; _t0; _t0--) {
// CHECK-NEXT:         r = clad::pop(_t1);
// CHECK-NEXT:         double _r_d0 = _d_r;
// CHECK-NEXT:         _d_r = 0.;
// CHECK-NEXT:         _d_r += _r_d0 * clad::back(_t3);
// CHECK-NEXT:         _d_n += r * _r_d0;
// CHECK-NEXT:         n = clad::pop(_t2);
// CHECK-NEXT:         _d_n = 0;
// CHECK-NEXT:         clad::pop(_t3);
// CHECK-NEXT:     }
// CHECK-NEXT:     *_d_x += _d_r;
// CHECK-NEXT: }


// Finding 2: discrete inside comma expression.
double f_comma_xor(double x) {
  int n = 0;
  return x * ((n = 12), (n ^= 5));
}

// CHECK-LABEL: f_comma_xor_grad
// CHECK-NEXT:     int _d_n = 0;
// CHECK-NEXT:     int n = 0;
// CHECK-NEXT:     double _t0 = ((n = 12) , (n ^= 5));
// CHECK-NEXT:     {
// CHECK-NEXT:         *_d_x += 1 * _t0;
// CHECK-NEXT:         _d_n += x * 1;
// CHECK-NEXT:         _d_n = 0;
// CHECK-NEXT:         _d_n += 0;
// CHECK-NEXT:         _d_n = 0;
// CHECK-NEXT:     }
// CHECK-NEXT: }


// Finding 3: side-effectful bit-field base under *=.
// Harden: b[2] so a second reverse i++ would be OOB; assert index stays 1.
double f_bitfield_side_effect(double x) {
  Bits b[2];
  b[0].flags = 7;
  b[1].flags = 7;
  int i = 0;
  double r = x;
  r *= (b[i++].flags &= 5);
  // Forward must evaluate i++ once; reverse must reuse the stored index.
  if (i != 1)
    return -1; // signal failure without aborting lit
  return r;
}

// CHECK-LABEL: f_bitfield_side_effect_grad
// CHECK-NOT: int &{{.*}} = b[
// CHECK-NOT: _d_b[i++]
// CHECK-NEXT:     int _t2;
// CHECK-NEXT:     bool _cond0 = false;
// CHECK-NEXT:     Bits _d_b[2]{0};
// CHECK-NEXT:     Bits b[2];
// CHECK-NEXT:     double _d_r = 0.;
// CHECK-NEXT:     double r = x;
// CHECK-NEXT:     double _t0 = r;
// CHECK-NEXT:     int _t1 = 0;
// CHECK-NEXT:     auto _rev0 = [&] {
// CHECK-NEXT:         if (_cond0)
// CHECK-NEXT:             ;
// CHECK-NEXT:         {
// CHECK-NEXT:             r = _t0;
// CHECK-NEXT:             double _r_d2 = _d_r;
// CHECK-NEXT:             _d_r = 0.;
// CHECK-NEXT:             _d_r += _r_d2 * _t2;
// CHECK-NEXT:             _d_b[_t1].flags += r * _r_d2;
// CHECK-NEXT:             _d_b[_t1].flags = 0;
// CHECK-NEXT:         }
// CHECK-NEXT:         *_d_x += _d_r;
// CHECK-NEXT:         {
// CHECK-NEXT:             int _r_d1 = _d_b[1].flags;
// CHECK-NEXT:             _d_b[1].flags = 0;
// CHECK-NEXT:         }
// CHECK-NEXT:         {
// CHECK-NEXT:             int _r_d0 = _d_b[0].flags;
// CHECK-NEXT:             _d_b[0].flags = 0;
// CHECK-NEXT:         }
// CHECK-NEXT:     };
// CHECK-NEXT:     b[0].flags = 7;
// CHECK-NEXT:     b[1].flags = 7;
// CHECK-NEXT:     int _d_i = 0;
// CHECK-NEXT:     int i = 0;
// CHECK-NEXT:     _t1 = i++;
// CHECK-NEXT:     r *= (_t2 = b[_t1].flags &= 5 , b[_t1].flags);
// CHECK-NEXT:     {
// CHECK-NEXT:         _cond0 = i != 1;
// CHECK-NEXT:         if (_cond0) {
// CHECK-NEXT:             _rev0();
// CHECK-NEXT:             return;
// CHECK-NEXT:         }
// CHECK-NEXT:     }
// CHECK-NEXT:     _d_r += 1;
// CHECK-NEXT:     _rev0();
// CHECK-NEXT: }


// Finding 3b: side-effectful pointer base under bit-field discrete assign.
// Without stabilizing the complete designator, the snap comma re-runs p++ and
// the reverse value reads b[1].flags (==3) instead of the post-assign 5.
double f_bitfield_base_side_effect(double x) {
  Bits b[2] = {{7}, {3}};
  Bits* p = b;
  double r = x;
  r *= ((p++)[0].flags &= 5);
  return r;
}

// CHECK-LABEL: f_bitfield_base_side_effect_grad
// Stabilized base: one p++ into a stored pointer; no second p++ on value/adjoint.
// CHECK-NOT: ({{.*}}p++{{.*}}flags &= 5{{.*}},{{.*}}p++
// CHECK:     {{.*}} = p++;
// CHECK:     r *= ({{.*}} = {{.*}}[0].flags &= 5 , {{.*}}[0].flags);


// Finding 4: discrete in ternary arm under *=.
double f_ternary_xor(double x, int cond) {
  double r = x;
  int n = 12;
  int m = 3;
  r *= (cond > 0 ? n ^= 5 : m);
  return r;
}

// CHECK-LABEL: f_ternary_xor_grad
// CHECK-NEXT:     int _d_cond = 0;
// CHECK-NEXT:     int _t1;
// CHECK-NEXT:     int *_{{[a-zA-Z0-9]+}};
// CHECK-NEXT:     double _d_r = 0.;
// CHECK-NEXT:     double r = x;
// CHECK-NEXT:     int _d_n = 0;
// CHECK-NEXT:     int n = 12;
// CHECK-NEXT:     int _d_m = 0;
// CHECK-NEXT:     int m = 3;
// CHECK-NEXT:     double _t0 = r;
// CHECK-NEXT:     bool _cond0 = cond > 0;
// CHECK-NEXT:     if (_cond0)
// CHECK-NEXT:         _t{{[0-9]+}} = &(_t{{[0-9]+}} = n ^= 5 , n);
// CHECK-NEXT:     r *= (_cond0 ? *_t{{[0-9]+}} : m);
// CHECK-NEXT:     _d_r += 1;
// CHECK-NEXT:     {
// CHECK-NEXT:         r = _t0;
// CHECK-NEXT:         double _r_d0 = _d_r;
// CHECK-NEXT:         _d_r = 0.;
// CHECK-NEXT:         _d_r += _r_d0 * (_cond0 ? _t1 : m);
// CHECK-NEXT:         if (_cond0) {
// CHECK-NEXT:             _d_n += r * _r_d0;
// CHECK-NEXT:             _d_n = 0;
// CHECK-NEXT:         } else
// CHECK-NEXT:             _d_m += r * _r_d0;
// CHECK-NEXT:     }
// CHECK-NEXT:     *_d_x += _d_r;
// CHECK-NEXT: }


// Finding 5: discrete in for-loop increment (must not crash cast<Expr>).
double f_for_inc_xor(double x) {
  double r = 0;
  int n = 12;
  for (int i = 0; i < 2; ++i, n ^= 5)
    r += x * n;
  return r;
}

// CHECK-LABEL: f_for_inc_xor_grad
// CHECK-NOT: clad::push({{.*}}n ^= 5
// CHECK-NEXT:     int _d_i = 0;
// CHECK-NEXT:     int i = 0;
// CHECK-NEXT:     clad::tape<int> _t1 = {};
// CHECK-NEXT:     double _d_r = 0.;
// CHECK-NEXT:     double r = 0;
// CHECK-NEXT:     int _d_n = 0;
// CHECK-NEXT:     int n = 12;
// CHECK-NEXT:     unsigned {{int|long}} _t0 = 0;
// CHECK-NEXT:     for (i = 0; i < 2; clad::push(_t1, n) , (++i , (n ^= 5))) {
// CHECK-NEXT:         _t0++;
// CHECK-NEXT:         r += x * n;
// CHECK-NEXT:     }
// CHECK-NEXT:     _d_r += 1;
// CHECK-NEXT:     for (; _t0; _t0--) {
// CHECK-NEXT:         {
// CHECK-NEXT:             n = clad::pop(_t1);
// CHECK-NEXT:             _d_n = 0;
// CHECK-NEXT:             _d_i += 0;
// CHECK-NEXT:         }
// CHECK-NEXT:         *_d_x += _d_r * n;
// CHECK-NEXT:         _d_n += x * _d_r;
// CHECK-NEXT:     }
// CHECK-NEXT: }



// Finding 6 BLOCKER: chained discrete must keep LHS lvalue identity.
// (n ^= 5) &= 3 must modify n, not a snapshot temporary _t.
double f_discrete_lvalue_chain(double x) {
  int n = 0;
  (n ^= 5) &= 3;
  return x * n;
}

// CHECK-LABEL: f_discrete_lvalue_chain_grad
// CHECK-NOT: (_t{{[0-9]*}} &=
// CHECK-NEXT:     int _d_n = 0;
// CHECK-NEXT:     int n = 0;
// CHECK-NEXT:     (n ^= 5);
// CHECK-NEXT:     (n &= 3);
// CHECK-NEXT:     {
// CHECK-NEXT:         *_d_x += 1 * n;
// CHECK-NEXT:         _d_n += x * 1;
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         _d_n = 0;
// CHECK-NEXT:         _d_n = 0;
// CHECK-NEXT:     }
// CHECK-NEXT: }


double f_discrete_lvalue_chain_loop(double x, int count) {
  int n = 0;
  for (int i = 0; i < count; ++i)
    (n ^= 5) &= 3;
  return x * n;
}

// CHECK-LABEL: f_discrete_lvalue_chain_loop_grad
// CHECK-NEXT:     int _d_count = 0;
// CHECK-NEXT:     int _d_i = 0;
// CHECK-NEXT:     int i = 0;
// CHECK-NEXT:     int _d_n = 0;
// CHECK-NEXT:     int n = 0;
// CHECK-NEXT:     unsigned {{int|long}} _t0 = 0;
// CHECK-NEXT:     for (i = 0; i < count; ++i) {
// CHECK-NEXT:         _t0++;
// CHECK-NEXT:         (n ^= 5);
// CHECK-NEXT:         (n &= 3);
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         *_d_x += 1 * n;
// CHECK-NEXT:         _d_n += x * 1;
// CHECK-NEXT:     }
// CHECK-NEXT:     for (; _t0; _t0--) {
// CHECK-NEXT:         _d_n = 0;
// CHECK-NEXT:         _d_n = 0;
// CHECK-NEXT:     }
// CHECK-NEXT: }


// Finding 7: discarded standalone discrete in a loop must not always push.
double f_loop_discarded_discrete(double x, int n) {
  double res = x;
  int mask = 15;
  for (int i = 0; i < n; ++i) {
    mask %= 4; // discarded; reverse never consumes the post-assign value
    res += x;
  }
  return res;
}

// CHECK-LABEL: f_loop_discarded_discrete_grad
// CHECK-NOT: clad::push({{.*}}mask %=
// CHECK-NEXT:     int _d_n = 0;
// CHECK-NEXT:     int _d_i = 0;
// CHECK-NEXT:     int i = 0;
// CHECK-NEXT:     double _d_res = 0.;
// CHECK-NEXT:     double res = x;
// CHECK-NEXT:     int _d_mask = 0;
// CHECK-NEXT:     int mask = 15;
// CHECK-NEXT:     unsigned {{int|long}} _t0 = 0;
// CHECK-NEXT:     for (i = 0; i < n; ++i) {
// CHECK-NEXT:         _t0++;
// CHECK-NEXT:         (mask %= 4);
// CHECK-NEXT:         res += x;
// CHECK-NEXT:     }
// CHECK-NEXT:     _d_res += 1;
// CHECK-NEXT:     for (; _t0; _t0--) {
// CHECK-NEXT:         *_d_x += _d_res;
// CHECK-NEXT:         _d_mask = 0;
// CHECK-NEXT:     }
// CHECK-NEXT:     *_d_x += _d_res;
// CHECK-NEXT: }



// Finding 4b: ternary discrete arm inside a loop -- Pop must live in the
// arm's reverse If, not the outer reverse block (mis-pop when cond==0).
double f_ternary_xor_loop(double x, int count, int cond) {
  double r = x;
  int n = 12;
  int m = 3;
  for (int i = 0; i < count; ++i)
    r *= (cond > 0 ? (n ^= 5) : m);
  return r;
}

// CHECK-LABEL: f_ternary_xor_loop_grad
// Pop for discrete arm must be inside the reverse If arm (not outer mis-pop).
// CHECK:         clad::push(_cond0, cond > 0);
// CHECK:         if (clad::back(_cond0)) {
// CHECK-NEXT:         clad::push(_t{{[0-9]+}}, n);
// CHECK-NEXT:         clad::push(_t{{[0-9]+}}, &(clad::push(_t{{[0-9]+}}, n ^= 5) , n));
// CHECK:         r *= (clad::back(_cond0) ? *clad::back(_t{{[0-9]+}}) : m);
// CHECK:         for (; _t{{[0-9]+}}; _t{{[0-9]+}}--)
// CHECK:         if (clad::back(_cond0)) {
// CHECK:             clad::pop(_t{{[0-9]+}});
// CHECK:         } else
// CHECK:         clad::pop(_cond0);

// count=2, cond=0 => df/dx = 3*3 = 9
// count=2, cond=1 => df/dx = 9*12 = 108

// Finding 4c: parenthesized discarded discrete in a loop -- VisitParenExpr
// must not eagerly request rev (no discrete tape/push).
double f_loop_paren_discarded_discrete(double x, int n) {
  double res = x;
  int mask = 15;
  for (int i = 0; i < n; ++i) {
    (mask %= 4);
    res += x;
  }
  return res;
}

// CHECK-LABEL: f_loop_paren_discarded_discrete_grad
// CHECK-NOT: clad::push({{.*}}mask %=
// CHECK-NEXT:     int _d_n = 0;
// CHECK-NEXT:     int _d_i = 0;
// CHECK-NEXT:     int i = 0;
// CHECK-NEXT:     double _d_res = 0.;
// CHECK-NEXT:     double res = x;
// CHECK-NEXT:     int _d_mask = 0;
// CHECK-NEXT:     int mask = 15;
// CHECK-NEXT:     unsigned {{int|long}} _t0 = 0;
// CHECK-NEXT:     for (i = 0; i < n; ++i) {
// CHECK-NEXT:         _t0++;
// CHECK-NEXT:         (mask %= 4);
// CHECK-NEXT:         res += x;
// CHECK-NEXT:     }
// CHECK-NEXT:     _d_res += 1;
// CHECK-NEXT:     for (; _t0; _t0--) {
// CHECK-NEXT:         *_d_x += _d_res;
// CHECK-NEXT:         _d_mask = 0;
// CHECK-NEXT:     }
// CHECK-NEXT:     *_d_x += _d_res;
// CHECK-NEXT: }
// Finding 4d: discarded ternary discrete arm in a loop -- must not push.
double f_loop_discarded_ternary_discrete(double x, int n, int cond) {
  double res = x;
  int mask = 15;
  for (int i = 0; i < n; ++i) {
    (cond ? (mask ^= 5) : mask);
    res += x;
  }
  return res;
}

// CHECK-LABEL: f_loop_discarded_ternary_discrete_grad
// CHECK-NOT: clad::push({{.*}}mask ^=


// Finding 8: non-const-ref call arg with discrete assign -- DifferentiateCallArg
// must prepareForFwdClone before CloneNode(getExpr()) so StoreAndRestore sees
// the upgraded snapshot wrapper (order-independent / clone-safe).
// disable_tbr forces shouldBeRecorded so the CloneNode path is exercised.
int id_ref(int& n) { return n; }

double f_ref_call_discrete(double x) {
  int n = 12;
  return x * id_ref(n ^= 5);
}

// CHECK-LABEL: f_ref_call_discrete_grad
// Snap/assign runs once as a stmt; call/store use stable n.
// CHECK: {{_t[0-9]+ = n \^= 5;|clad::push\(.*n \^= 5}}
// CHECK-NOT: id_ref((n ^= 5)
// CHECK: id_ref(n);


// Finding 2: unary-plus parent wrapping a discrete compound assign under *=.
// Must snapshot n's post-assign value, not re-execute the assignment.
double f_unary_plus_discrete(double x) {
  double r = x;
  int n = 12;
  r *= +(n ^= 5);
  return r;
}

// CHECK-LABEL: f_unary_plus_discrete_grad
// CHECK: _t{{[0-9]+}} = n ^= 5
// CHECK-NOT: +(n ^= 5)
// CHECK: _d_r += _r_d{{[0-9]+}} * _t

// Finding 2b: unary-minus parent wrapping a discrete compound assign under *=.
double f_unary_minus_discrete(double x) {
  double r = x;
  int n = 12;
  r *= -(n ^= 5);
  return r;
}

// CHECK-LABEL: f_unary_minus_discrete_grad
// CHECK: _t{{[0-9]+}} = n ^= 5
// CHECK-NOT: -(n ^= 5)
// CHECK: _d_r += _r_d{{[0-9]+}} * -_t

// Finding 2c: C-style cast parent wrapping a discrete compound assign under *=.
double f_cast_discrete(double x) {
  double r = x;
  int n = 12;
  r *= (double)(n ^= 5);
  return r;
}

// CHECK-LABEL: f_cast_discrete_grad
// CHECK: _t{{[0-9]+}} = n ^= 5
// CHECK-NOT: (double)(n ^= 5)
// CHECK: _d_r += _r_d{{[0-9]+}} * (double)_t

// Finding 2d: unary-plus under *= in a loop with non-self-inverse op.
// Exposes the re-execution bug when <<= is not its own inverse.
double f_unary_plus_shl_loop(double x, int count) {
  double r = x;
  int n = 3;
  for (int i = 0; i < count; ++i)
    r *= +(n <<= 1);
  return r;
}

// CHECK-LABEL: f_unary_plus_shl_loop_grad
// Pop for snapshot must be inside the reverse loop body.
// CHECK: clad::push(_t{{[0-9]+}}, n <<= 1)
// CHECK: clad::back(_t{{[0-9]+}})
// CHECK-NOT: +(n <<= 1)


// Call-parent discrete: idd(n <<= 1) in a loop.
// The assignment must not re-execute in the pullback or reverse-forward call.
double idd(double v) { return v; }

double f_call_discrete_shl(double x) {
  int n = 3; double r = x;
  for (int i = 0; i < 2; ++i) r *= idd(n <<= 1);
  return r;
}

// CHECK-LABEL: f_call_discrete_shl_grad
// The forward loop separates the push from the call argument.
// CHECK: clad::push({{.*}}, n <<= 1)
// CHECK: idd(n)
// The reverse must not re-execute n <<= 1 inside a call.
// CHECK-NOT: idd((n <<= 1))
// CHECK-NOT: idd_pushforward((n <<= 1)

// int-returning identity variant.
int idi(int v) { return v; }

double f_call_discrete_shl_int(double x) {
  int n = 3; double r = x;
  for (int i = 0; i < 2; ++i) r *= idi(n <<= 1);
  return r;
}

// Functional-cast discrete: double(n ^= 5).
double f_funccast_discrete(double x) {
  int n = 12;
  double r = x * double(n ^= 5);
  return r;
}

// CHECK-LABEL: f_funccast_discrete_grad
// Snapshot must use the cast-wrapped assignment, not re-execute it.
// CHECK: _t{{[0-9]+}} = {{.*}}(n ^= 5)
// Reverse must not re-execute the assignment inside the cast.
// CHECK-NOT: double(n ^= 5)

// Named-cast discrete: static_cast<double>(n ^= 5).
double f_namedcast_discrete(double x) {
  int n = 12;
  double r = x * static_cast<double>(n ^= 5);
  return r;
}

// CHECK-LABEL: f_namedcast_discrete_grad
// Snapshot must use the cast-wrapped assignment, not re-execute it.
// CHECK: _t{{[0-9]+}} = {{.*}}(n ^= 5)
// Reverse must not re-execute the assignment inside the cast.
// CHECK-NOT: static_cast<double>(n ^= 5)

// Conditional-discrete regressions:
// 1. Unary-plus wrapping conditional with discrete arm in loop:
double f_cond_uplus_discrete(double x) {
  int n = 3, m = 3; double r = x;
  for (int i = 0; i < 2; ++i) r *= +(i == 0 ? n <<= 1 : m);
  return r;
}

// CHECK-LABEL: f_cond_uplus_discrete_grad
// CHECK: if (clad::back(_cond0))
// CHECK: clad::push(_t{{[0-9]+}}, &(clad::push(_t{{[0-9]+}}, n <<= 1) , n))
// CHECK: +(clad::back(_cond0) ? *clad::back(_t{{[0-9]+}}) : m)
// CHECK: _d_r += _r_d0 * (clad::back(_cond0) ? clad::back(_t{{[0-9]+}}) : m);

// 2. C-style cast wrapping conditional with discrete arm in loop:
double f_cond_cstyle_discrete(double x) {
  int n = 3, m = 3; double r = x;
  for (int i = 0; i < 2; ++i) r *= (double)(i == 0 ? n <<= 1 : m);
  return r;
}

// CHECK-LABEL: f_cond_cstyle_discrete_grad
// CHECK: if (clad::back(_cond0))
// CHECK: clad::push(_t{{[0-9]+}}, &(clad::push(_t{{[0-9]+}}, n <<= 1) , n))
// CHECK: (double)(clad::back(_cond0) ? *clad::back(_t{{[0-9]+}}) : m)
// CHECK: _d_r += _r_d0 * (double)(clad::back(_cond0) ? clad::back(_t{{[0-9]+}}) : m);

// 3. Call-parent wrapping conditional with discrete arm in loop:
double f_cond_call_discrete(double x) {
  int n = 3, m = 3; double r = x;
  for (int i = 0; i < 2; ++i) r *= idd(i == 0 ? n <<= 1 : m);
  return r;
}

// CHECK-LABEL: f_cond_call_discrete_grad
// CHECK: if (clad::back(_cond0))
// CHECK: clad::push(_t{{[0-9]+}}, &(clad::push(_t{{[0-9]+}}, n <<= 1) , n))
// CHECK: idd(clad::back(_cond0) ? *clad::back(_t{{[0-9]+}}) : m)
// The reverse pass must NOT re-execute n <<= 1 or push again.
// CHECK-NOT: idd(clad::back(_cond0) ? (clad::push
// CHECK-NOT: idd_pushforward(clad::back(_cond0) ? (clad::push

// 4. Branch containing unary-plus inside call: idd(i == 0 ? +(n <<= 1) : m)
double f_branch_plus(double x) {
  int n = 3, m = 3; double r = x;
  for (int i = 0; i < 2; ++i) r *= idd(i == 0 ? +(n <<= 1) : m);
  return r;
}

// CHECK-LABEL: f_branch_plus_grad
// CHECK: if (clad::back(_cond0))
// CHECK: clad::push(_t{{[0-9]+}}, +(clad::push(_t{{[0-9]+}}, n <<= 1) , n))
// CHECK: idd(clad::back(_cond0) ? clad::back(_t{{[0-9]+}}) : m)
// CHECK-NOT: idd(clad::back(_cond0) ? +(clad::push
// CHECK-NOT: idd_pushforward(clad::back(_cond0) ? +(clad::push

// 5. Branch containing C-style cast inside call: idd(i == 0 ? (double)(n <<= 1) : m)
double f_branch_cast(double x) {
  int n = 3, m = 3; double r = x;
  for (int i = 0; i < 2; ++i) r *= idd(i == 0 ? (double)(n <<= 1) : m);
  return r;
}

// CHECK-LABEL: f_branch_cast_grad
// CHECK: if (clad::back(_cond0))
// CHECK: clad::push(_t{{[0-9]+}}, (double)(clad::push(_t{{[0-9]+}}, n <<= 1) , n))
// CHECK: idd(clad::back(_cond0) ? clad::back(_t{{[0-9]+}}) : m)
// CHECK-NOT: idd(clad::back(_cond0) ? (double)(clad::push
// CHECK-NOT: idd_pushforward(clad::back(_cond0) ? (double)(clad::push

// 6. Branch containing comma operator inside call: idd(i == 0 ? ((n = 4), (n <<= 1)) : m)
double f_branch_comma(double x) {
  int n = 3, m = 3; double r = x;
  for (int i = 0; i < 2; ++i) r *= idd(i == 0 ? ((n = 4), (n <<= 1)) : m);
  return r;
}

// CHECK-LABEL: f_branch_comma_grad
// CHECK: if (clad::back(_cond0))
// CHECK: clad::push(_t{{[0-9]+}}, &((n = 4) , (clad::push(_t{{[0-9]+}}, n <<= 1) , n)))
// CHECK: idd(clad::back(_cond0) ? *clad::back(_t{{[0-9]+}}) : m)
// CHECK-NOT: idd(clad::back(_cond0) ? ((n = 4)
// CHECK-NOT: idd_pushforward(clad::back(_cond0) ? ((n = 4)

// 7. xvalue in conditional arm: stored by value (cannot take address of xvalue).
double f_xvalue_conditional(double x) {
  double n = x;
  double fallback = 3;
  return x > 0 ? static_cast<double&&>(n += 1) : fallback;
}

// CHECK-LABEL: f_xvalue_conditional_grad
// CHECK: if (_cond0)
// CHECK: _t{{[0-9]+}} = static_cast<double &&>(n += 1);

// 8. xvalue identity probe: mutation through rvalue reference.
double set_seven(double&& value) { value = 7; return value; }

double f_xvalue_identity(double x) {
  double a = 3, b = 5;
  set_seven(x > 0 ? static_cast<double&&>(a += 1)
                  : static_cast<double&&>(b += 1));
  return x * a;
}

// CHECK-LABEL: f_xvalue_identity_grad
// CHECK: set_seven(_cond0 ? static_cast<double &&>(a += 1) : static_cast<double &&>(b += 1));


int main() {
  // Test 1: rem (10 % 3 = 1 -> df/dx = 1)
  auto df_rem = clad::gradient(f_rem, "x");
  double dx_rem = 0;
  df_rem.execute(5.0, 3, &dx_rem);
  std::cout << "df_rem/dx = " << dx_rem << std::endl;
  // CHECK-EXEC: df_rem/dx = 1

  // Test 2: and (11 & 7 = 3 -> df/dx = 3)
  auto df_and = clad::gradient(f_and, "x");
  double dx_and = 0;
  df_and.execute(4.0, 11, &dx_and);
  std::cout << "df_and/dx = " << dx_and << std::endl;
  // CHECK-EXEC: df_and/dx = 3

  // Test 3: or (4 | 3 = 7 -> df/dx = 7)
  auto df_or = clad::gradient(f_or, "x");
  double dx_or = 0;
  df_or.execute(2.0, 4, &dx_or);
  std::cout << "df_or/dx = " << dx_or << std::endl;
  // CHECK-EXEC: df_or/dx = 7

  // Test 4: xor (12 ^ 5 = 9 -> df/dx = 9)
  auto df_xor = clad::gradient(f_xor, "x");
  double dx_xor = 0;
  df_xor.execute(3.0, 12, &dx_xor);
  std::cout << "df_xor/dx = " << dx_xor << std::endl;
  // CHECK-EXEC: df_xor/dx = 9

  // Test 5: shl (3 << 2 = 12 -> df/dx = 12)
  auto df_shl = clad::gradient(f_shl, "x");
  double dx_shl = 0;
  df_shl.execute(2.0, 3, &dx_shl);
  std::cout << "df_shl/dx = " << dx_shl << std::endl;
  // CHECK-EXEC: df_shl/dx = 12

  // Test 6: shr (10 >> 1 = 5 -> df/dx = 5)
  auto df_shr = clad::gradient(f_shr, "x");
  double dx_shr = 0;
  df_shr.execute(2.0, 10, &dx_shr);
  std::cout << "df_shr/dx = " << dx_shr << std::endl;
  // CHECK-EXEC: df_shr/dx = 5

  // Test 7: loop
  auto df_loop = clad::gradient(f_loop, "x");
  double dx_loop = 0;
  df_loop.execute(2.0, 3, &dx_loop);
  std::cout << "df_loop/dx = " << dx_loop << std::endl;
  // CHECK-EXEC: df_loop/dx = 5

  // Test 8: nested_rem (x * (n %= y) with x=5, n=10, y=3 -> n becomes 1 -> df/dx = 1)
  auto df_nested_rem = clad::gradient(f_nested_rem, "x");
  double dx_nested_rem = 0;
  df_nested_rem.execute(5.0, 3, &dx_nested_rem);
  std::cout << "df_nested_rem/dx = " << dx_nested_rem << std::endl;
  // CHECK-EXEC: df_nested_rem/dx = 1

  // Test 9: nested_and (x * (n &= y) with x=4, n=11, y=7 -> n becomes 3 -> df/dx = 3)
  auto df_nested_and = clad::gradient(f_nested_and, "x");
  double dx_nested_and = 0;
  df_nested_and.execute(4.0, 7, &dx_nested_and);
  std::cout << "df_nested_and/dx = " << dx_nested_and << std::endl;
  // CHECK-EXEC: df_nested_and/dx = 3

  // Test 10: nested_side_effect
  auto df_nested_side_effect = clad::gradient(f_nested_side_effect, "x");
  double dx_nested_side_effect = 0;
  df_nested_side_effect.execute(5.0, 2, &dx_nested_side_effect);
  std::cout << "df_nested_side_effect/dx = " << dx_nested_side_effect << std::endl;
  // CHECK-EXEC: df_nested_side_effect/dx = 1

  // Test 11: loop_side_effect
  auto df_loop_side_effect = clad::gradient(f_loop_side_effect, "x");
  double dx_loop_side_effect = 0;
  df_loop_side_effect.execute(2.0, 2, 2, &dx_loop_side_effect);
  std::cout << "df_loop_side_effect/dx = " << dx_loop_side_effect << std::endl;
  // CHECK-EXEC: df_loop_side_effect/dx = 1

  // Test 12: nested_xor_mul_assign (12 ^ 5 = 9 -> expected df/dx = 9)
  auto df_nested_xor = clad::gradient(nested_xor_mul_assign, "x");
  double dx_nested_xor = 0;
  df_nested_xor.execute(2.0, 5, &dx_nested_xor);
  std::cout << "nested_xor_mul_assign df/dx = " << dx_nested_xor << std::endl;
  // CHECK-EXEC: nested_xor_mul_assign df/dx = 9

  // Test 13: nested_shl_mul_assign (3 << 2 = 12 -> expected df/dx = 12)
  auto df_nested_shl = clad::gradient(nested_shl_mul_assign, "x");
  double dx_nested_shl = 0;
  df_nested_shl.execute(2.0, 2, &dx_nested_shl);
  std::cout << "nested_shl_mul_assign df/dx = " << dx_nested_shl << std::endl;
  // CHECK-EXEC: nested_shl_mul_assign df/dx = 12

  // Test 14: nested_shr_mul_assign (20 >> 1 = 10 -> expected df/dx = 10)
  auto df_nested_shr = clad::gradient(nested_shr_mul_assign, "x");
  double dx_nested_shr = 0;
  df_nested_shr.execute(2.0, 1, &dx_nested_shr);
  std::cout << "nested_shr_mul_assign df/dx = " << dx_nested_shr << std::endl;
  // CHECK-EXEC: nested_shr_mul_assign df/dx = 10

  // Test 15: nested_narrow_shift (200 <<= 1 -> unsigned char 144 -> expected df/dx = 144)
  auto df_nested_narrow = clad::gradient(nested_narrow_shift, "x");
  double dx_nested_narrow = 0;
  df_nested_narrow.execute(2.0, &dx_nested_narrow);
  std::cout << "nested_narrow_shift df/dx = " << dx_nested_narrow << std::endl;
  // CHECK-EXEC: nested_narrow_shift df/dx = 144

  // Test 16: nested_xor_loop (iter 1: 12^5=9, iter 2: 9^5=12 -> 9 * 12 = 108 -> expected df/dx = 108)
  auto df_nested_xor_loop = clad::gradient(nested_xor_loop, "x");
  double dx_nested_xor_loop = 0;
  df_nested_xor_loop.execute(2.0, 2, 5, &dx_nested_xor_loop);
  std::cout << "nested_xor_loop df/dx = " << dx_nested_xor_loop << std::endl;
  // CHECK-EXEC: nested_xor_loop df/dx = 108

  // Test 17: bit-field LHS (12 & 5 = 4 -> df/dx = 4)
  auto df_bitfield = clad::gradient(f_bitfield, "x");
  double dx_bitfield = 0;
  df_bitfield.execute(3.0, 5, &dx_bitfield);
  std::cout << "f_bitfield df/dx = " << dx_bitfield << std::endl;
  // CHECK-EXEC: f_bitfield df/dx = 4

  // Test 18: array subscript LHS (10 % 3 = 1 -> df/dx = 1)
  auto df_arr_rem = clad::gradient(f_arr_rem, "x");
  double dx_arr_rem = 0;
  df_arr_rem.execute(5.0, 3, &dx_arr_rem);
  std::cout << "f_arr_rem df/dx = " << dx_arr_rem << std::endl;
  // CHECK-EXEC: f_arr_rem df/dx = 1


  // Test 19: nested bit-field under *= (12 & 5 = 4 -> df/dx = 4)
  auto df_nested_bitfield = clad::gradient(nested_bitfield_mul_assign, "x");
  double dx_nested_bitfield = 0;
  df_nested_bitfield.execute(3.0, 5, &dx_nested_bitfield);
  std::cout << "nested_bitfield_mul_assign df/dx = " << dx_nested_bitfield << std::endl;
  // CHECK-EXEC: nested_bitfield_mul_assign df/dx = 4


  // Finding 1: unparenthesized r *= n ^= y in loop -> 108
  auto df_unparen = clad::gradient(f_unparen_xor_mul_loop, "x");
  double dx_unparen = 0;
  df_unparen.execute(2.0, 2, 5, &dx_unparen);
  std::cout << "f_unparen_xor_mul_loop df/dx = " << dx_unparen << std::endl;
  // CHECK-EXEC: f_unparen_xor_mul_loop df/dx = 108

  // Finding 2: comma (n=12, n^=5) -> 9
  auto df_comma = clad::gradient(f_comma_xor, "x");
  double dx_comma = 0;
  df_comma.execute(2.0, &dx_comma);
  std::cout << "f_comma_xor df/dx = " << dx_comma << std::endl;
  // CHECK-EXEC: f_comma_xor df/dx = 9

  // Finding 3: side-effectful bit-field -> 5
  auto df_bf_se = clad::gradient(f_bitfield_side_effect, "x");
  double dx_bf_se = 0;
  df_bf_se.execute(2.0, &dx_bf_se);
  std::cout << "f_bitfield_side_effect df/dx = " << dx_bf_se << std::endl;
  // CHECK-EXEC: f_bitfield_side_effect df/dx = 5

  // Finding 3b: side-effectful pointer base -> 5
  auto df_bf_base_se = clad::gradient(f_bitfield_base_side_effect, "x");
  double dx_bf_base_se = 0;
  df_bf_base_se.execute(2.0, &dx_bf_base_se);
  std::cout << "f_bitfield_base_side_effect df/dx = " << dx_bf_base_se << std::endl;
  // CHECK-EXEC: f_bitfield_base_side_effect df/dx = 5

  // Finding 4: ternary discrete arm (cond>0: 12^5=9)
  auto df_tern = clad::gradient(f_ternary_xor, "x");
  double dx_tern = 0;
  df_tern.execute(2.0, 1, &dx_tern);
  std::cout << "f_ternary_xor df/dx = " << dx_tern << std::endl;
  // CHECK-EXEC: f_ternary_xor df/dx = 9

  // Finding 5: for-increment discrete
  // i=0: n=12, r+=x*12; then n^=5 -> 9; i=1: r+=x*9; then n^=5
  // gradient = 12+9 = 21
  auto df_for_inc = clad::gradient(f_for_inc_xor, "x");
  double dx_for_inc = 0;
  df_for_inc.execute(2.0, &dx_for_inc);
  std::cout << "f_for_inc_xor df/dx = " << dx_for_inc << std::endl;
  // CHECK-EXEC: f_for_inc_xor df/dx = 21

  // Finding 6: lvalue chain (n^=5)&=3 -> n becomes 1 -> dx=1
  auto df_lvalue = clad::gradient(f_discrete_lvalue_chain, "x");
  double dx_lvalue = 0;
  df_lvalue.execute(2.0, &dx_lvalue);
  std::cout << "f_discrete_lvalue_chain df/dx = " << dx_lvalue << std::endl;
  // CHECK-EXEC: f_discrete_lvalue_chain df/dx = 1

  auto df_lvalue_loop = clad::gradient(f_discrete_lvalue_chain_loop, "x");
  double dx_lvalue_loop = 0;
  df_lvalue_loop.execute(2.0, 1, &dx_lvalue_loop);
  std::cout << "f_discrete_lvalue_chain_loop df/dx = " << dx_lvalue_loop << std::endl;
  // CHECK-EXEC: f_discrete_lvalue_chain_loop df/dx = 1

  // Finding 7: discarded loop discrete (gradient is n+1 for res+=x each iter + init)
  auto df_disc = clad::gradient(f_loop_discarded_discrete, "x");
  double dx_disc = 0;
  df_disc.execute(2.0, 3, &dx_disc);
  std::cout << "f_loop_discarded_discrete df/dx = " << dx_disc << std::endl;
  // CHECK-EXEC: f_loop_discarded_discrete df/dx = 4

  // Ternary+loop: cond=0 => 9; cond=1 => 108
  auto df_tern_loop = clad::gradient(f_ternary_xor_loop, "x");
  double dx_tern_loop0 = 0;
  df_tern_loop.execute(2.0, 2, 0, &dx_tern_loop0);
  std::cout << "f_ternary_xor_loop cond0 df/dx = " << dx_tern_loop0 << std::endl;
  // CHECK-EXEC: f_ternary_xor_loop cond0 df/dx = 9
  double dx_tern_loop1 = 0;
  df_tern_loop.execute(2.0, 2, 1, &dx_tern_loop1);
  std::cout << "f_ternary_xor_loop cond1 df/dx = " << dx_tern_loop1 << std::endl;
  // CHECK-EXEC: f_ternary_xor_loop cond1 df/dx = 108

  auto df_paren_disc = clad::gradient(f_loop_paren_discarded_discrete, "x");
  double dx_paren_disc = 0;
  df_paren_disc.execute(2.0, 3, &dx_paren_disc);
  std::cout << "f_loop_paren_discarded_discrete df/dx = " << dx_paren_disc << std::endl;
  // CHECK-EXEC: f_loop_paren_discarded_discrete df/dx = 4

  auto df_tern_disc = clad::gradient(f_loop_discarded_ternary_discrete, "x");
  double dx_tern_disc = 0;
  df_tern_disc.execute(2.0, 3, 1, &dx_tern_disc);
  std::cout << "f_loop_discarded_ternary_discrete df/dx = " << dx_tern_disc << std::endl;
  // CHECK-EXEC: f_loop_discarded_ternary_discrete df/dx = 4

  auto df_ref_call = clad::gradient<clad::opts::disable_tbr>(f_ref_call_discrete, "x");
  double dx_ref_call = 0;
  df_ref_call.execute(2.0, &dx_ref_call);
  std::cout << "f_ref_call_discrete df/dx = " << dx_ref_call << std::endl;
  // CHECK-EXEC: f_ref_call_discrete df/dx = 9

  // Finding 2: unary-plus parent (12 ^ 5 = 9, r = x * 9, df/dx = 9)
  auto df_uplus = clad::gradient(f_unary_plus_discrete, "x");
  double dx_uplus = 0;
  df_uplus.execute(2.0, &dx_uplus);
  std::cout << "f_unary_plus_discrete df/dx = " << dx_uplus << std::endl;
  // CHECK-EXEC: f_unary_plus_discrete df/dx = 9

  // Finding 2b: unary-minus parent (-(12 ^ 5) = -9, r = x * (-9), df/dx = -9)
  auto df_uminus = clad::gradient(f_unary_minus_discrete, "x");
  double dx_uminus = 0;
  df_uminus.execute(2.0, &dx_uminus);
  std::cout << "f_unary_minus_discrete df/dx = " << dx_uminus << std::endl;
  // CHECK-EXEC: f_unary_minus_discrete df/dx = -9

  // Finding 2c: C-style cast parent ((double)(12 ^ 5) = 9, df/dx = 9)
  auto df_cast = clad::gradient(f_cast_discrete, "x");
  double dx_cast = 0;
  df_cast.execute(2.0, &dx_cast);
  std::cout << "f_cast_discrete df/dx = " << dx_cast << std::endl;
  // CHECK-EXEC: f_cast_discrete df/dx = 9

  // Finding 2d: unary-plus <<= in loop (3<<1=6, 6<<1=12, r=x*6*12=72x, df/dx=72)
  auto df_shl_loop = clad::gradient(f_unary_plus_shl_loop, "x");
  double dx_shl_loop = 0;
  df_shl_loop.execute(1.0, 2, &dx_shl_loop);
  std::cout << "f_unary_plus_shl_loop df/dx = " << dx_shl_loop << std::endl;
  // CHECK-EXEC: f_unary_plus_shl_loop df/dx = 72

  // Call-parent discrete: idd(n <<= 1) in loop, f(x) = x * 6 * 12 = 72x
  auto df_call_shl = clad::gradient(f_call_discrete_shl, "x");
  double dx_call_shl = 0;
  df_call_shl.execute(1.0, &dx_call_shl);
  std::cout << "f_call_discrete_shl df/dx = " << dx_call_shl << std::endl;
  // CHECK-EXEC: f_call_discrete_shl df/dx = 72

  // int-returning identity variant
  auto df_call_shl_int = clad::gradient(f_call_discrete_shl_int, "x");
  double dx_call_shl_int = 0;
  df_call_shl_int.execute(1.0, &dx_call_shl_int);
  std::cout << "f_call_discrete_shl_int df/dx = " << dx_call_shl_int << std::endl;
  // CHECK-EXEC: f_call_discrete_shl_int df/dx = 72

  // Functional-cast discrete: int(12 ^ 5) = 9, f = x * 9
  auto df_funccast = clad::gradient(f_funccast_discrete, "x");
  double dx_funccast = 0;
  df_funccast.execute(3.0, &dx_funccast);
  std::cout << "f_funccast_discrete df/dx = " << dx_funccast << std::endl;
  // CHECK-EXEC: f_funccast_discrete df/dx = 9

  // Named-cast discrete: static_cast<double>(12 ^ 5) = 9, f = x * 9
  auto df_namedcast = clad::gradient(f_namedcast_discrete, "x");
  double dx_namedcast = 0;
  df_namedcast.execute(3.0, &dx_namedcast);
  std::cout << "f_namedcast_discrete df/dx = " << dx_namedcast << std::endl;
  // CHECK-EXEC: f_namedcast_discrete df/dx = 9

  // Conditional-discrete regressions:
  auto df_cond_uplus = clad::gradient(f_cond_uplus_discrete, "x");
  double dx_cond_uplus = 0;
  df_cond_uplus.execute(1.0, &dx_cond_uplus);
  std::cout << "f_cond_uplus_discrete df/dx = " << dx_cond_uplus << std::endl;
  // CHECK-EXEC: f_cond_uplus_discrete df/dx = 18

  auto df_cond_cstyle = clad::gradient(f_cond_cstyle_discrete, "x");
  double dx_cond_cstyle = 0;
  df_cond_cstyle.execute(1.0, &dx_cond_cstyle);
  std::cout << "f_cond_cstyle_discrete df/dx = " << dx_cond_cstyle << std::endl;
  // CHECK-EXEC: f_cond_cstyle_discrete df/dx = 18

  auto df_cond_call = clad::gradient(f_cond_call_discrete, "x");
  double dx_cond_call = 0;
  df_cond_call.execute(1.0, &dx_cond_call);
  std::cout << "f_cond_call_discrete df/dx = " << dx_cond_call << std::endl;
  // CHECK-EXEC: f_cond_call_discrete df/dx = 18

  auto df_bplus = clad::gradient(f_branch_plus, "x");
  double dx_bplus = 0;
  df_bplus.execute(1.0, &dx_bplus);
  std::cout << "f_branch_plus df/dx = " << dx_bplus << std::endl;
  // CHECK-EXEC: f_branch_plus df/dx = 18

  auto df_bcast = clad::gradient(f_branch_cast, "x");
  double dx_bcast = 0;
  df_bcast.execute(1.0, &dx_bcast);
  std::cout << "f_branch_cast df/dx = " << dx_bcast << std::endl;
  // CHECK-EXEC: f_branch_cast df/dx = 18

  auto df_bcomma = clad::gradient(f_branch_comma, "x");
  double dx_bcomma = 0;
  df_bcomma.execute(1.0, &dx_bcomma);
  std::cout << "f_branch_comma df/dx = " << dx_bcomma << std::endl;
  // CHECK-EXEC: f_branch_comma df/dx = 24

  auto df_xval = clad::gradient(f_xvalue_conditional);
  double dx_xval = 0;
  df_xval.execute(2.0, &dx_xval);
  std::cout << "f_xvalue_conditional df/dx = " << dx_xval << std::endl;
  // CHECK-EXEC: f_xvalue_conditional df/dx = 1

  auto df_xval_id = clad::gradient(f_xvalue_identity);
  double dx_xval_id = 0;
  df_xval_id.execute(2.0, &dx_xval_id);
  std::cout << "f_xvalue_identity df/dx = " << dx_xval_id << std::endl;
  // CHECK-EXEC: f_xvalue_identity df/dx = 7

  return 0;
}
