// RUN: %cladclang %s -I%S/../../include -oActivity.out 2>&1 | %filecheck %s
// RUN: ./Activity.out | %filecheck_exec %s
// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -enable-aa %s -I%S/../../include -oActivity.out
// RUN: ./Activity.out | %filecheck_exec %s
//CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"

double f1(double x){
  double a = x*x;
  double b = 1;
  b = b*b;
  return a;
  
}

//CHECK: void f1_grad(double x, double *_d_x) {
//CHECK-NEXT:     double _d_a = 0;
//CHECK-NEXT:     double a = x * x;
//CHECK-NEXT:     double _d_b = 0;
//CHECK-NEXT:     double b = 1;
//CHECK-NEXT:     double _t0 = b;
//CHECK-NEXT:     b = b * b;
//CHECK-NEXT:     _d_a += 1;
//CHECK-NEXT:     {
//CHECK-NEXT:         *_d_x += _d_a * x;
//CHECK-NEXT:         *_d_x += x * _d_a;
//CHECK-NEXT:     }
//CHECK-NEXT: }

double f2(double x){
  double a = x*x;
  double b = 1;
  double g;
  if(a)
    b=x;
  else if(b)
    double d = b;
  else
    g = a;
  return a;
}

//CHECK: void f2_grad(double x, double *_d_x) {
//CHECK-NEXT:     bool _cond0;
//CHECK-NEXT:     double _t0;
//CHECK-NEXT:     bool _cond1;
//CHECK-NEXT:     double _d_d = 0;
//CHECK-NEXT:     double d = 0;
//CHECK-NEXT:     double _t1;
//CHECK-NEXT:     double _d_a = 0;
//CHECK-NEXT:     double a = x * x;
//CHECK-NEXT:     double _d_b = 0;
//CHECK-NEXT:     double b = 1;
//CHECK-NEXT:     double _d_g = 0;
//CHECK-NEXT:     double g;
//CHECK-NEXT:     {
//CHECK-NEXT:         _cond0 = a;
//CHECK-NEXT:         if (_cond0) {
//CHECK-NEXT:             _t0 = b;
//CHECK-NEXT:             b = x;
//CHECK-NEXT:         } else {
//CHECK-NEXT:             _cond1 = b;
//CHECK-NEXT:             if (_cond1)
//CHECK-NEXT:                 d = b;
//CHECK-NEXT:             else {
//CHECK-NEXT:                 _t1 = g;
//CHECK-NEXT:                 g = a;
//CHECK-NEXT:             }
//CHECK-NEXT:         }
//CHECK-NEXT:     }
//CHECK-NEXT:     _d_a += 1;
//CHECK-NEXT:     if (_cond0) {
//CHECK-NEXT:         b = _t0;
//CHECK-NEXT:         double _r_d0 = _d_b;
//CHECK-NEXT:         _d_b = 0;
//CHECK-NEXT:         *_d_x += _r_d0;
//CHECK-NEXT:     } else if (!_cond1) {
//CHECK-NEXT:         g = _t1;
//CHECK-NEXT:         double _r_d1 = _d_g;
//CHECK-NEXT:         _d_g = 0;
//CHECK-NEXT:         _d_a += _r_d1;
//CHECK-NEXT:     }
//CHECK-NEXT:     {
//CHECK-NEXT:         *_d_x += _d_a * x;
//CHECK-NEXT:         *_d_x += x * _d_a;
//CHECK-NEXT:     }
//CHECK-NEXT: }


#define TEST(F, x) { \
  result[0] = 0; \
  auto F##grad = clad::gradient<clad::opts::enable_aa>(F);\
  F##grad.execute(x, result);\
  printf("{%.2f}\n", result[0]); \
}

int main(){
    double result[3] = {};
    TEST(f1, 3);// CHECK-EXEC: {6.00}
    TEST(f2, 3);// CHECK-EXEC: {6.00}
}