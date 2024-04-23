// RUN: %cladclang %s -I%S/../../include -oRecursive.out 2>&1 | FileCheck %s
// RUN: ./Recursive.out | FileCheck -check-prefix=CHECK-EXEC %s

//CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"

extern "C" int printf(const char* fmt, ...);

float f_dec(float arg) {
  if (arg == 0)
    return arg;
  else
    return f_dec(arg-1);
}

// CHECK: clad::ValueAndPushforward<float, float> f_dec_pushforward(float arg, float _d_arg);

// CHECK: float f_dec_darg0(float arg) {
// CHECK-NEXT:     float _d_arg = 1;
// CHECK-NEXT:     if (arg == 0)
// CHECK-NEXT:         return _d_arg;
// CHECK-NEXT:     else {
// CHECK-NEXT:         clad::ValueAndPushforward<float, float> _t0 = f_dec_pushforward(arg - 1, _d_arg - 0);
// CHECK-NEXT:         return _t0.pushforward;
// CHECK-NEXT:     }
// CHECK-NEXT: }

float f_dec_darg0(float arg);

float f_pow(float arg, int p) {
  if (p == 0)
    return 1;
  else
    return arg * f_pow(arg, p - 1);
}

// CHECK: clad::ValueAndPushforward<float, float> f_pow_pushforward(float arg, int p, float _d_arg);

// CHECK: float f_pow_darg0(float arg, int p) {
// CHECK-NEXT:     float _d_arg = 1;
// CHECK-NEXT:     if (p == 0)
// CHECK-NEXT:         return 0;
// CHECK-NEXT:     else {
// CHECK-NEXT:         clad::ValueAndPushforward<float, float> _t0 = f_pow_pushforward(arg, p - 1, _d_arg);
// CHECK-NEXT:         float &_t1 = _t0.value;
// CHECK-NEXT:         return _d_arg * _t1 + arg * _t0.pushforward;
// CHECK-NEXT:     }
// CHECK-NEXT: }

float f_pow_darg0(float arg, int p);

int main() {
  clad::differentiate(f_dec, 0);
  printf("Result is = %.2f\n", f_dec_darg0(2)); // CHECK-EXEC: Result is = 1.00
  clad::differentiate(f_pow, 0);
  printf("Result is = %.2f\n", f_pow_darg0(10, 2)); //CHECK-EXEC: Result is = 20.00

// CHECK: clad::ValueAndPushforward<float, float> f_dec_pushforward(float arg, float _d_arg) {
// CHECK-NEXT:     if (arg == 0)
// CHECK-NEXT:         return {arg, _d_arg};
// CHECK-NEXT:     else {
// CHECK-NEXT:         clad::ValueAndPushforward<float, float> _t0 = f_dec_pushforward(arg - 1, _d_arg - 0);
// CHECK-NEXT:         return {_t0.value, _t0.pushforward};
// CHECK-NEXT:     }
// CHECK-NEXT: }

// CHECK: clad::ValueAndPushforward<float, float> f_pow_pushforward(float arg, int p, float _d_arg) {
// CHECK-NEXT:     if (p == 0)
// CHECK-NEXT:         return {1, 0};
// CHECK-NEXT:     else {
// CHECK-NEXT:         clad::ValueAndPushforward<float, float> _t0 = f_pow_pushforward(arg, p - 1, _d_arg);
// CHECK-NEXT:         float &_t1 = _t0.value;
// CHECK-NEXT:         return {arg * _t1, _d_arg * _t1 + arg * _t0.pushforward};
// CHECK-NEXT:     }
// CHECK-NEXT: }
}
