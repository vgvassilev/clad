// RUN: %cladclang %s -I%S/../../include -oCallArguments.out -lm 2>&1 | FileCheck %s
// RUN: ./CallArguments.out | FileCheck -check-prefix=CHECK-EXEC %s
//XFAIL:*

#include "clad/Differentiator/Differentiator.h"

//CHECK-NOT: {{.*error|warning|note:.*}}

namespace custom_derivatives {
  float f_dx(float y) { return 2*y; }
}

float f(float y) {
  return y * y - 10;
}

float g(float x) {
  return f(x*x*x);
}

// CHECK: float g_dx(float x) {
// CHECK-NEXT: return f_dx(x * x * x)  * (((1.F * x + x * 1.F) * x + x * x * 1.F));
// CHECK-NEXT: }

float sqrt_func(float x, float y) {
  return sqrt(x * x + y * y) - y;
}

// CHECK: float sqrt_func_dx(float x, float y) {
// CHECK-NEXT: sqrt_dx(x * x + y * y) * ((1.F * x + x * 1.F) + ((0.F * y + y * 0.F))) - (0.F);
// CHECK-NEXT: }

float f_const_args_func_1(const float x, const float y) {
  return x * x + y * y - 1.F;
}

// CHECK: float f_const_args_func_1_dx(const float x, const float y) {
// CHECK-NEXT: (1.F * x + x * 1.F) + ((0.F * y + y * 0.F)) - (0.F)
// CHECK-NEXT: }

float f_const_args_func_2(float x, const float y) {
  return x * x + y * y - 1.F;
}

// CHECK: float f_const_args_func_2_dx(float x, const float y) {
// CHECK-NEXT: (1.F * x + x * 1.F) + ((0.F * y + y * 0.F)) - (0.F)
// CHECK-NEXT: }

float f_const_args_func_3(const float x, float y) {
  return x * x + y * y - 1.F;
}

// CHECK: float f_const_args_func_3_dx(const float x, float y) {
// CHECK-NEXT: (1.F * x + x * 1.F) + ((0.F * y + y * 0.F)) - (0.F)
// CHECK-NEXT: }

struct Vec { float x,y,z; };
float f_const_args_func_4(float x, float y, const Vec v) {
  return x * x + y * y - v.x;
}

// CHECK: float f_const_args_func_4_dx(float x, float y, const Vec v) {
// CHECK-NEXT: (1.F * x + x * 1.F) + ((0.F * y + y * 0.F)) - (0.F)
// CHECK-NEXT: }

float f_const_args_func_5(float x, float y, const Vec &v) {
  return x * x + y * y - v.x;
}

// CHECK: float f_const_args_func_5_dx(float x, float y, const Vec &v) {
// CHECK-NEXT: (1.F * x + x * 1.F) + ((0.F * y + y * 0.F)) - (0.F)
// CHECK-NEXT: }

float f_const_args_func_6(const float x, const float y, const Vec &v) {
  return x * x + y * y - v.x;
}

// CHECK: float f_const_args_func_6_dx(const float x, const float y, const Vec &v) {
// CHECK-NEXT: (1.F * x + x * 1.F) + ((0.F * y + y * 0.F)) - (0.F)
// CHECK-NEXT: }

float f_const_helper(const float x) { return x * x; }
//namespace custom_derivatives {
//  float f_const_helper_dx(float x) { return 2*x; }
//}

// CHECK: float f_const_helper_dx(const float x) {
// CHECK-NEXT: (1.F * x + x * 1.F)
// CHECK-NEXT: }

float f_const_args_func_7(const float x, const float y) {
  return f_const_helper(x) + f_const_helper(y) - y;
}

// CHECK: float f_const_args_func_7_dx(const float x, const float y) {
// CHECK-NEXT: f_const_helper_dx(x) + (f_const_helper_dx(y)) - (0.F)
// CHECK-NEXT: }

float f_const_args_func_8(const float x, float y) {
  return f_const_helper(x) + f_const_helper(y) - y;
}

// CHECK: float f_const_args_func_8_dx(const float x, float y) {
// CHECK-NEXT: f_const_helper_dx(x) + (f_const_helper_dx(y)) - (0.F)
// CHECK-NEXT: }

extern "C" int printf(const char* fmt, ...);
int main () {
  auto f = clad::differentiate(g, 0);
  printf("g_dx=%f\n", f.execute(1));
  //CHECK-EXEC: g_dx=6.000000

  clad::differentiate(sqrt_func, 0);

  auto f1 = clad::differentiate(f_const_args_func_1, 0);
  printf("f1_dx=%f\n", f1.execute(1.F,2.F));
  //CHECK-EXEC: f1_dx=2.000000
  auto f2 = clad::differentiate(f_const_args_func_2, 0);
  printf("f2_dx=%f\n", f2.execute(1.F,2.F));
  //CHECK-EXEC: f2_dx=2.000000
  auto f3 = clad::differentiate(f_const_args_func_3, 0);
  printf("f3_dx=%f\n", f3.execute(1.F,2.F));
  //CHECK-EXEC: f3_dx=2.000000
  auto f4 = clad::differentiate(f_const_args_func_4, 0);
  printf("f4_dx=%f\n", f4.execute(1.F,2.F,Vec()));
  //CHECK-EXEC: f4_dx=2.000000
  auto f5 = clad::differentiate(f_const_args_func_5, 0);
  printf("f5_dx=%f\n", f5.execute(1.F,2.F,Vec()));
  //CHECK-EXEC: f5_dx=2.000000
  auto f6 = clad::differentiate(f_const_args_func_6, 0);
  printf("f6_dx=%f\n", f6.execute(1.F,2.F,Vec()));
  //CHECK-EXEC: f6_dx=2.000000
  clad::differentiate(f_const_helper, 0);
  auto f7 = clad::differentiate(f_const_args_func_7, 0);
  printf("f7_dx=%f\n", f7.execute(1.F,2.F));
  //CHECK-EXEC: f7_dx=2.000000
  auto f8 = clad::differentiate(f_const_args_func_8, 0);
  const float f8x = 1.F;
  printf("f8_dx=%f\n", f8.execute(f8x,2.F));
  //CHECK-EXEC: f8_dx=2.000000

  return 0;
}
