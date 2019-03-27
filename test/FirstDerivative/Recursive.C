// RUN: %cladclang %s -I%S/../../include -oBasicArithmeticAddSub.out 2>&1 | FileCheck %s
// RUN: ./BasicArithmeticAddSub.out | FileCheck -check-prefix=CHECK-EXEC %s

//CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"

extern "C" int printf(const char* fmt, ...);

int f_dec(int arg) {
  if (arg == 0)
    return arg;
  else
    return f_dec(arg-1);
}
// CHECK:   int f_dec_darg0(int arg) {
// CHECK-NEXT:       int _d_arg = 1;
// CHECK-NEXT:       if (arg == 0)
// CHECK-NEXT:           return _d_arg;
// CHECK-NEXT:       else
// CHECK-NEXT:           return f_dec_darg0(arg - 1) * (_d_arg - 0);
// CHECK-NEXT:   }
int f_dec_darg0(int arg);

int f_pow(int arg, int p) {
  if (p == 0)
    return 1;
  else
    return arg * f_pow(arg, p - 1);
}

// FIXME-CHECK: int f_pow_darg0(int arg, int p) {
// FIXME-CHECK-NEXT:   int _d_arg = 1;
// FIXME-CHECK-NEXT:   int _d_p = 0;
// FIXME-CHECK-NEXT:   if (p == 0)
// FIXME-CHECK-NEXT:     return 0;
// FIXME-CHECK-NEXT:   else {
// FIXME-CHECK-NEXT:     int _t0 = f_pow(arg, p - 1);
// FIXME-CHECK-NEXT:     return _d_arg * _t0 + arg * (f_pow_darg0(arg, p - 1) * (_d_arg + _d_p - 0));
// FIXME-CHECK-NEXT:   }
// FIXME-CHECK-NEXT: }

int f_pow_darg0(int arg, int p);
// == p * f_pow(arg, p - 1)

int main() {
  clad::differentiate(f_dec, 0);
  printf("Result is = %d\n", f_dec_darg0(2)); // CHECK-EXEC: Result is = 1
  // This test is currently broken. TODO: fix calls in the forward mode.
  //clad::differentiate(f_pow, 0);
  //printf("Result is = %d\n", f_pow_darg0(10, 2)); //cCHECK-EXEC: Result is = 20
}
