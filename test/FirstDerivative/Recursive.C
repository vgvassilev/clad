// RUN: %cladclang %s -I%S/../../include -oRecursive.out 
// RUN: ./Recursive.out | FileCheck -check-prefix=CHECK-EXEC %s

//CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"

extern "C" int printf(const char* fmt, ...);

int f_dec(int arg) {
  if (arg == 0)
    return arg;
  else
    return f_dec(arg-1);
}

// CHECK: clad::ValueAndPushforward<int, int> f_dec_pushforward(int arg, int _d_arg) {
// CHECK-NEXT:     if (arg == 0)
// CHECK-NEXT:         return {arg, _d_arg};
// CHECK-NEXT:     else {
// CHECK-NEXT:         clad::ValueAndPushforward<int, int> _t0 = f_dec_pushforward(arg - 1, _d_arg - 0);
// CHECK-NEXT:         return {_t0.value, _t0.pushforward};
// CHECK-NEXT:     }
// CHECK-NEXT: }

// CHECK: int f_dec_darg0(int arg) {
// CHECK-NEXT:     int _d_arg = 1;
// CHECK-NEXT:     if (arg == 0)
// CHECK-NEXT:         return _d_arg;
// CHECK-NEXT:     else {
// CHECK-NEXT:         clad::ValueAndPushforward<int, int> _t0 = f_dec_pushforward(arg - 1, _d_arg - 0);
// CHECK-NEXT:         return _t0.pushforward;
// CHECK-NEXT:     }
// CHECK-NEXT: }

int f_dec_darg0(int arg);

int f_pow(int arg, int p) {
  if (p == 0)
    return 1;
  else
    return arg * f_pow(arg, p - 1);
}

// CHECK: clad::ValueAndPushforward<int, int> f_pow_pushforward(int arg, int p, int _d_arg, int _d_p) {
// CHECK-NEXT:     if (p == 0)
// CHECK-NEXT:         return {1, 0};
// CHECK-NEXT:     else {
// CHECK-NEXT:         clad::ValueAndPushforward<int, int> _t0 = f_pow_pushforward(arg, p - 1, _d_arg, _d_p - 0);
// CHECK-NEXT:         int &_t1 = _t0.value;
// CHECK-NEXT:         return {arg * _t1, _d_arg * _t1 + arg * _t0.pushforward};
// CHECK-NEXT:     }
// CHECK-NEXT: }

// CHECK: int f_pow_darg0(int arg, int p) {
// CHECK-NEXT:     int _d_arg = 1;
// CHECK-NEXT:     int _d_p = 0;
// CHECK-NEXT:     if (p == 0)
// CHECK-NEXT:         return 0;
// CHECK-NEXT:     else {
// CHECK-NEXT:         clad::ValueAndPushforward<int, int> _t0 = f_pow_pushforward(arg, p - 1, _d_arg, _d_p - 0);
// CHECK-NEXT:         int &_t1 = _t0.value;
// CHECK-NEXT:         return _d_arg * _t1 + arg * _t0.pushforward;
// CHECK-NEXT:     }
// CHECK-NEXT: }

int f_pow_darg0(int arg, int p);

int main() {
  clad::differentiate(f_dec, 0);
  printf("Result is = %d\n", f_dec_darg0(2)); // CHECK-EXEC: Result is = 1
  clad::differentiate(f_pow, 0);
  printf("Result is = %d\n", f_pow_darg0(10, 2)); //CHECK-EXEC: Result is = 20
}
