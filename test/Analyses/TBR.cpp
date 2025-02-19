// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -disable-tbr %s -I%S/../../include -oTBR.out | %filecheck %s
// RUN: ./TBR.out | %filecheck_exec %s
// RUN: %cladclang %s -I%S/../../include -oTBR.out
// RUN: ./TBR.out | %filecheck_exec %s

#include "clad/Differentiator/Differentiator.h"

double f1(double x) {
  double t = 1;
  for (int i = 0; i < 3; i++)
    t *= x;
  return t;
} // == x^3

//CHECK: void f1_grad(double x, double *_d_x) {
//CHECK-NEXT:     int _d_i = 0;
//CHECK-NEXT:     int i = 0;
//CHECK-NEXT:     clad::tape<double> _t1 = {};
//CHECK-NEXT:     double _d_t = 0.;
//CHECK-NEXT:     double t = 1;
//CHECK-NEXT:     unsigned {{int|long}} _t0 = {{0U|0UL}};
//CHECK-NEXT:     for (i = 0; ; i++) {
//CHECK-NEXT:         {
//CHECK-NEXT:             if (!(i < 3))
//CHECK-NEXT:                 break;
//CHECK-NEXT:         }
//CHECK-NEXT:         _t0++;
//CHECK-NEXT:         clad::push(_t1, t);
//CHECK-NEXT:         t *= x;
//CHECK-NEXT:     }
//CHECK-NEXT:     _d_t += 1;
//CHECK-NEXT:     for (;; _t0--) {
//CHECK-NEXT:         {
//CHECK-NEXT:             if (!_t0)
//CHECK-NEXT:                 break;
//CHECK-NEXT:         }
//CHECK-NEXT:         t = clad::pop(_t1);
//CHECK-NEXT:         double _r_d0 = _d_t;
//CHECK-NEXT:         _d_t = 0.;
//CHECK-NEXT:         _d_t += _r_d0 * x;
//CHECK-NEXT:         *_d_x += t * _r_d0;
//CHECK-NEXT:     }
//CHECK-NEXT: }

double f2(double val) {
  double res = 0;
  for (int i=1; i<5; ++i) {
    if (i == 3)
      continue;
    res += i * val;
  }
  return res;
}

//CHECK: void f2_grad(double val, double *_d_val) {
//CHECK-NEXT:     int _d_i = 0;
//CHECK-NEXT:     int i = 0;
//CHECK-NEXT:     clad::tape<bool> _cond0 = {};
//CHECK-NEXT:     clad::tape<unsigned {{int|long}}> _t1 = {};
//CHECK-NEXT:     double _d_res = 0.;
//CHECK-NEXT:     double res = 0;
//CHECK-NEXT:     unsigned {{int|long}} _t0 = {{0U|0UL}};
//CHECK-NEXT:     for (i = 1; ; ++i) {
//CHECK-NEXT:         {
//CHECK-NEXT:             if (!(i < 5))
//CHECK-NEXT:                 break;
//CHECK-NEXT:         }
//CHECK-NEXT:         _t0++;
//CHECK-NEXT:         {
//CHECK-NEXT:             clad::push(_cond0, i == 3);
//CHECK-NEXT:             if (clad::back(_cond0)) {
//CHECK-NEXT:                 clad::push(_t1, {{1U|1UL}});
//CHECK-NEXT:                 continue;
//CHECK-NEXT:             }
//CHECK-NEXT:         }
//CHECK-NEXT:         res += i * val;
//CHECK-NEXT:         clad::push(_t1, {{2U|2UL}});
//CHECK-NEXT:     }
//CHECK-NEXT:     _d_res += 1;
//CHECK-NEXT:     for (;; _t0--) {
//CHECK-NEXT:         {
//CHECK-NEXT:             if (!_t0)
//CHECK-NEXT:                 break;
//CHECK-NEXT:         }
//CHECK-NEXT:         --i;
//CHECK-NEXT:         switch (clad::pop(_t1)) {
//CHECK-NEXT:           case {{2U|2UL}}:
//CHECK-NEXT:             ;
//CHECK-NEXT:             {
//CHECK-NEXT:                 double _r_d0 = _d_res;
//CHECK-NEXT:                 _d_i += _r_d0 * val;
//CHECK-NEXT:                 *_d_val += i * _r_d0;
//CHECK-NEXT:             }
//CHECK-NEXT:             {
//CHECK-NEXT:                 if (clad::back(_cond0))
//CHECK-NEXT:                   case {{1U|1UL}}:
//CHECK-NEXT:                     ;
//CHECK-NEXT:                 clad::pop(_cond0);
//CHECK-NEXT:             }
//CHECK-NEXT:         }
//CHECK-NEXT:     }
//CHECK-NEXT: }

double f3 (double x){
  double i = 1;
  double j = 0;
  double res = 0;
  res += i*x;
  if(j)
    j++;
  else if(i)
    res += i*x;
  else if(j)
    i++;
  return res;
}

//CHECK: void f3_grad(double x, double *_d_x) {
//CHECK-NEXT:     bool _cond0;
//CHECK-NEXT:     bool _cond1;
//CHECK-NEXT:     bool _cond2;
//CHECK-NEXT:     double _d_i = 0.;
//CHECK-NEXT:     double i = 1;
//CHECK-NEXT:     double _d_j = 0.;
//CHECK-NEXT:     double j = 0;
//CHECK-NEXT:     double _d_res = 0.;
//CHECK-NEXT:     double res = 0;
//CHECK-NEXT:     res += i * x;
//CHECK-NEXT:     {
//CHECK-NEXT:         _cond0 = j;
//CHECK-NEXT:         if (_cond0)
//CHECK-NEXT:             j++;
//CHECK-NEXT:         else {
//CHECK-NEXT:             _cond1 = i;
//CHECK-NEXT:             if (_cond1)
//CHECK-NEXT:                 res += i * x;
//CHECK-NEXT:             else {
//CHECK-NEXT:                 _cond2 = j;
//CHECK-NEXT:                 if (_cond2)
//CHECK-NEXT:                     i++;
//CHECK-NEXT:             }
//CHECK-NEXT:         }
//CHECK-NEXT:     }
//CHECK-NEXT:     _d_res += 1;
//CHECK-NEXT:     if (!_cond0)
//CHECK-NEXT:         if (_cond1) {
//CHECK-NEXT:             double _r_d1 = _d_res;
//CHECK-NEXT:             _d_i += _r_d1 * x;
//CHECK-NEXT:             *_d_x += i * _r_d1;
//CHECK-NEXT:         } else if (_cond2)
//CHECK-NEXT:             i--;
//CHECK-NEXT:     {
//CHECK-NEXT:         double _r_d0 = _d_res;
//CHECK-NEXT:         _d_i += _r_d0 * x;
//CHECK-NEXT:         *_d_x += i * _r_d0;
//CHECK-NEXT:     }
//CHECK-NEXT: }


#define TEST(F, x) { \
  result[0] = 0; \
  auto F##grad = clad::gradient<clad::opts::enable_tbr>(F);\
  F##grad.execute(x, result);\
  printf("{%.2f}\n", result[0]); \
}

int main() {
  double result[3] = {};
  TEST(f1, 3); // CHECK-EXEC: {27.00}
  TEST(f2, 3); // CHECK-EXEC: {7.00}
  TEST(f3, 3); // CHECK-EXEC: {2.00}
}
