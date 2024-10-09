// RUN: %cladclang %s -I%S/../../include -oActivity.out 2>&1 | %filecheck %s
// RUN: ./Activity.out | %filecheck_exec %s
// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -enable-va %s -I%S/../../include -oActivity.out
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
//CHECK-NEXT:     double _d_a = 0.;
//CHECK-NEXT:     double a = x * x;
//CHECK-NEXT:     double b = 1;
//CHECK-NEXT:     double _t0 = b;
//CHECK-NEXT:     b = b * b;
//CHECK-NEXT:     _d_a += 1;
//CHECK-NEXT:     b = _t0;
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
//CHECK-NEXT:     double d = 0.;
//CHECK-NEXT:     double _t1;
//CHECK-NEXT:     double _d_a = 0.;
//CHECK-NEXT:     double a = x * x;
//CHECK-NEXT:     double _d_b = 0.;
//CHECK-NEXT:     double b = 1;
//CHECK-NEXT:     double _d_g = 0.;
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
//CHECK-NEXT:         _d_b = 0.;
//CHECK-NEXT:         *_d_x += _r_d0;
//CHECK-NEXT:     } else if (!_cond1) {
//CHECK-NEXT:         g = _t1;
//CHECK-NEXT:         double _r_d1 = _d_g;
//CHECK-NEXT:         _d_g = 0.;
//CHECK-NEXT:         _d_a += _r_d1;
//CHECK-NEXT:     }
//CHECK-NEXT:     {
//CHECK-NEXT:         *_d_x += _d_a * x;
//CHECK-NEXT:         *_d_x += x * _d_a;
//CHECK-NEXT:     }
//CHECK-NEXT: }

double f3(double x){
  double x1, x2, x3, x4, x5 = 0;
  while(!x3){
    x5 = x4;
    x4 = x3;
    x3 = x2;
    x2 = x1;
    x1 = x;
  }
  return x5;
}

//CHECK: void f3_grad(double x, double *_d_x) {
//CHECK-NEXT:     clad::tape<double> _t1 = {};
//CHECK-NEXT:     clad::tape<double> _t2 = {};
//CHECK-NEXT:     clad::tape<double> _t3 = {};
//CHECK-NEXT:     clad::tape<double> _t4 = {};
//CHECK-NEXT:     clad::tape<double> _t5 = {};
//CHECK-NEXT:     double _d_x1 = 0., _d_x2 = 0., _d_x3 = 0., _d_x4 = 0., _d_x5 = 0.;
//CHECK-NEXT:     double x1, x2, x3, x4, x5 = 0;
//CHECK-NEXT:     unsigned {{int|long}} _t0 = {{0U|0UL}};
//CHECK-NEXT:     while (!x3) 
//CHECK-NEXT:      {
//CHECK-NEXT:         _t0++;
//CHECK-NEXT:         clad::push(_t1, x5);
//CHECK-NEXT:         x5 = x4;
//CHECK-NEXT:         clad::push(_t2, x4);
//CHECK-NEXT:         x4 = x3;
//CHECK-NEXT:         clad::push(_t3, x3);
//CHECK-NEXT:         x3 = x2;
//CHECK-NEXT:         clad::push(_t4, x2);
//CHECK-NEXT:         x2 = x1;
//CHECK-NEXT:         clad::push(_t5, x1);
//CHECK-NEXT:         x1 = x;
//CHECK-NEXT:     }
//CHECK-NEXT:     _d_x5 += 1;
//CHECK-NEXT:     while (_t0) 
//CHECK-NEXT:      {
//CHECK-NEXT:         {
//CHECK-NEXT:             {
//CHECK-NEXT:                 x1 = clad::pop(_t5);
//CHECK-NEXT:                 double _r_d4 = _d_x1;
//CHECK-NEXT:                 _d_x1 = 0.;
//CHECK-NEXT:                 *_d_x += _r_d4;
//CHECK-NEXT:             }
//CHECK-NEXT:             {
//CHECK-NEXT:                 x2 = clad::pop(_t4);
//CHECK-NEXT:                 double _r_d3 = _d_x2;
//CHECK-NEXT:                 _d_x2 = 0.;
//CHECK-NEXT:                 _d_x1 += _r_d3;
//CHECK-NEXT:             }
//CHECK-NEXT:             {
//CHECK-NEXT:                 x3 = clad::pop(_t3);
//CHECK-NEXT:                 double _r_d2 = _d_x3;
//CHECK-NEXT:                 _d_x3 = 0.;
//CHECK-NEXT:                 _d_x2 += _r_d2;
//CHECK-NEXT:             }
//CHECK-NEXT:             {
//CHECK-NEXT:                 x4 = clad::pop(_t2);
//CHECK-NEXT:                 double _r_d1 = _d_x4;
//CHECK-NEXT:                 _d_x4 = 0.;
//CHECK-NEXT:                 _d_x3 += _r_d1;
//CHECK-NEXT:             }
//CHECK-NEXT:             {
//CHECK-NEXT:                 x5 = clad::pop(_t1);
//CHECK-NEXT:                 double _r_d0 = _d_x5;
//CHECK-NEXT:                 _d_x5 = 0.;
//CHECK-NEXT:                 _d_x4 += _r_d0;
//CHECK-NEXT:             }
//CHECK-NEXT:         }
//CHECK-NEXT:         _t0--;
//CHECK-NEXT:     }
//CHECK-NEXT: }

double f4_1(double v, double u){
  double k = 2*u;
  double n = 2*v;
  return n*k;
}
double f4(double x){
  double c = f4_1(x, 1);
  return c;
}
// CHECK-NEXT: void f4_1_pullback(double v, double u, double _d_y, double *_d_v, double *_d_u);

// CHECK: void f4_grad(double x, double *_d_x) {
// CHECK-NEXT:     double _d_c = 0.;
// CHECK-NEXT:     double c = f4_1(x, 1);
// CHECK-NEXT:     _d_c += 1;
// CHECK-NEXT:     {
// CHECK-NEXT:         double _r0 = 0.;
// CHECK-NEXT:         double _r1 = 0.;
// CHECK-NEXT:         f4_1_pullback(x, 1, _d_c, &_r0, &_r1);
// CHECK-NEXT:         *_d_x += _r0;
// CHECK-NEXT:     }
// CHECK-NEXT: }

double f5(double x){
  double g = x ? 1 : 2;
  return g;
}
// CHECK: void f5_grad(double x, double *_d_x) {
// CHECK-NEXT:     double _cond0 = x;
// CHECK-NEXT:     double _d_g = 0.;
// CHECK-NEXT:     double g = _cond0 ? 1 : 2;
// CHECK-NEXT:     _d_g += 1;
// CHECK-NEXT: }

double f6(double x){
  double a = 0;
  if(0){
    a = x;
  }
  return a;
}

// CHECK: void f6_grad(double x, double *_d_x) {
// CHECK-NEXT:     double _t0;
// CHECK-NEXT:     double a = 0;
// CHECK-NEXT:     if (0) {
// CHECK-NEXT:         _t0 = a;
// CHECK-NEXT:         a = x;
// CHECK-NEXT:     }
// CHECK-NEXT:     if (0) {
// CHECK-NEXT:         a = _t0;
// CHECK-NEXT:     }
// CHECK-NEXT: }

double f7(double x){
  double &a = x;
  double* b = &a;
  double arr[3] = {1,2,3};
  double c = arr[0]*(*b)+arr[1]*a+arr[2]*x; 
  return a;
}

// CHECK: void f7_grad(double x, double *_d_x) {
// CHECK-NEXT:     double &_d_a = *_d_x;
// CHECK-NEXT:     double &a = x;
// CHECK-NEXT:     double *_d_b = &_d_a;
// CHECK-NEXT:     double *b = &a;
// CHECK-NEXT:     double _d_arr[3] = {0};
// CHECK-NEXT:     double arr[3] = {1, 2, 3};
// CHECK-NEXT:     double _d_c = 0.;
// CHECK-NEXT:     double c = arr[0] * *b + arr[1] * a + arr[2] * x;
// CHECK-NEXT:     _d_a += 1;
// CHECK-NEXT:     {
// CHECK-NEXT:         _d_arr[0] += _d_c * *b;
// CHECK-NEXT:         *_d_b += arr[0] * _d_c;
// CHECK-NEXT:         _d_arr[1] += _d_c * a;
// CHECK-NEXT:         _d_a += arr[1] * _d_c;
// CHECK-NEXT:         _d_arr[2] += _d_c * x;
// CHECK-NEXT:         *_d_x += arr[2] * _d_c;
// CHECK-NEXT:     }
// CHECK-NEXT: }

#define TEST(F, x) { \
  result[0] = 0; \
  auto F##grad = clad::gradient<clad::opts::enable_va>(F);\
  F##grad.execute(x, result);\
  printf("{%.2f}\n", result[0]); \
}

int main(){
    double result[3] = {};
    TEST(f1, 3);// CHECK-EXEC: {6.00}
    TEST(f2, 3);// CHECK-EXEC: {6.00}
    TEST(f3, 3);// CHECK-EXEC: {0.00}
    TEST(f4, 3);// CHECK-EXEC: {4.00}
    TEST(f5, 3);// CHECK-EXEC: {0.00}
    TEST(f6, 3);// CHECK-EXEC: {0.00}
    TEST(f7, 3);// CHECK-EXEC: {1.00}
}

// CHECK: void f4_1_pullback(double v, double u, double _d_y, double *_d_v, double *_d_u) {
// CHECK-NEXT:     double _d_k = 0.;
// CHECK-NEXT:     double k = 2 * u;
// CHECK-NEXT:     double _d_n = 0.;
// CHECK-NEXT:     double n = 2 * v;
// CHECK-NEXT:     {
// CHECK-NEXT:         _d_n += _d_y * k;
// CHECK-NEXT:         _d_k += n * _d_y;
// CHECK-NEXT:     }
// CHECK-NEXT:     *_d_v += 2 * _d_n;
// CHECK-NEXT:     *_d_u += 2 * _d_k;
// CHECK-NEXT: }
