// RUN: %autodiff %s -I%S/../../include -fsyntax-only 2>&1 | FileCheck %s

#include "autodiff/Differentiator/Differentiator.h"

extern "C" int printf(const char* fmt, ...);

int f_1(float y) {
  int x = 1, z = 3;
  return x*y*z; // x * z
}

// CHECK: int f_1_derived_y(float y) {
// CHECK-NEXT: int x = 1, z = 3;
// CHECK-NEXT: return ((0 * y + x * 1) * z + x * y * 0);
// CHECK-NEXT: }

int f_2(int x, float y, int z) {
  return x*y*z; // y * z; x * z; x * y
}
// CHECK: int f_2_derived_x(int x, float y, int z) {
// CHECK-NEXT: return ((1 * y + x * 0) * z + x * y * 0);
// CHECK-NEXT: }

// CHECK: int f_2_derived_y(int x, float y, int z) {
// CHECK-NEXT: return ((0 * y + x * 1) * z + x * y * 0);
// CHECK-NEXT: }

// CHECK: int f_2_derived_z(int x, float y, int z) {
// CHECK-NEXT: return ((0 * y + x * 0) * z + x * y * 1);
// CHECK-NEXT: }

int f_3() {
  int x = 1, z = 3;
  float y = 2;
  return x*y*z; // should not be differentiated
}

int main () {
  int z,x = 4*5;
  diff(f_1, 1);
  
  diff(f_2, 1);
  diff(f_2, 2);
  diff(f_2, 3);
  
  // invalid 2nd argument
  diff(f_3, 1);
  // CHECK: plugin ad: Error: invalid argument index 1 among 0 argument(s)
  diff(f_2, -1);
  // note that here "-" is regarded as a prefix
  // CHECK: plugin ad: Error: expected positions of independent variables
  diff(f_2, 0);
  // CHECK: plugin ad: Error: invalid argument index 0 among 3 argument(s)
  diff(f_2, 4);
  // CHECK: plugin ad: Error: invalid argument index 4 among 3 argument(s)
  diff(f_2, 10);
  // CHECK: plugin ad: Error: invalid argument index 10 among 3 argument(s)
  
  diff(f_2, x);
  // CHECK: plugin ad: Error: expected positions of independent variables
  diff(f_2, f_2);
  // CHECK: plugin ad: Error: expected positions of independent variables
  
  int kjsdcniue;
  diff(f_2, kjsdcniue);
  // CHECK: plugin ad: Error: expected positions of independent variables

  // TODO: this causes seg fault:
  //  float one = 1.0;
  //  diff(f_2, float);
  
  return 0;
}
