// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -disable-tbr %s -I%S/../../include -oTBR.out | %filecheck %s
// RUN: ./TBR.out | %filecheck_exec %s
// RUN: %cladclang %s -I%S/../../include -oTBR.out
// RUN: ./TBR.out | %filecheck_exec %s
// XFAIL: valgrind

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
//CHECK-NEXT:     for (i = 0; i < 3; i++) {
//CHECK-NEXT:         _t0++;
//CHECK-NEXT:         clad::push(_t1, t);
//CHECK-NEXT:         t *= x;
//CHECK-NEXT:     }
//CHECK-NEXT:     _d_t += 1;
//CHECK-NEXT:     for (; _t0; _t0--) {
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
//CHECK-NEXT:     for (i = 1; i < 5; ++i) {
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
//CHECK-NEXT:     for (; _t0; _t0--) {
//CHECK-NEXT:         --i;
//CHECK-NEXT:         switch (clad::pop(_t1)) {
//CHECK-NEXT:           case {{2U|2UL}}:
//CHECK-NEXT:             ;
//CHECK-NEXT:             {
//CHECK-NEXT:                 _d_i += _d_res * val;
//CHECK-NEXT:                 *_d_val += i * _d_res;
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

double f3(double x){
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
//CHECK-NEXT:             _d_i += _d_res * x;
//CHECK-NEXT:             *_d_x += i * _d_res;
//CHECK-NEXT:         } else if (_cond2)
//CHECK-NEXT:             i--;
//CHECK-NEXT:     {
//CHECK-NEXT:         _d_i += _d_res * x;
//CHECK-NEXT:         *_d_x += i * _d_res;
//CHECK-NEXT:     }
//CHECK-NEXT: }

double f4(double x, double y) {
  double arr[4];
  double res = y;
  arr[0] = x;
  int i = 0;
  res *= arr[i];
  arr[0] = 0; // This test primarily checks if this arr[0] gets stored
  return res;
} // x * y

//CHECK: void f4_grad(double x, double y, double *_d_x, double *_d_y) {
//CHECK-NEXT:     double _d_arr[4] = {0};
//CHECK-NEXT:     double arr[4];
//CHECK-NEXT:     double _d_res = 0.;
//CHECK-NEXT:     double res = y;
//CHECK-NEXT:     arr[0] = x;
//CHECK-NEXT:     int _d_i = 0;
//CHECK-NEXT:     int i = 0;
//CHECK-NEXT:     double _t0 = res;
//CHECK-NEXT:     res *= arr[i];
//CHECK-NEXT:     double _t1 = arr[0];
//CHECK-NEXT:     arr[0] = 0;
//CHECK-NEXT:     _d_res += 1;
//CHECK-NEXT:     {
//CHECK-NEXT:         arr[0] = _t1;
//CHECK-NEXT:         double _r_d2 = _d_arr[0];
//CHECK-NEXT:         _d_arr[0] = 0.;
//CHECK-NEXT:     }
//CHECK-NEXT:     {
//CHECK-NEXT:         res = _t0;
//CHECK-NEXT:         double _r_d1 = _d_res;
//CHECK-NEXT:         _d_res = 0.;
//CHECK-NEXT:         _d_res += _r_d1 * arr[i];
//CHECK-NEXT:         _d_arr[i] += res * _r_d1;
//CHECK-NEXT:     }
//CHECK-NEXT:     {
//CHECK-NEXT:         double _r_d0 = _d_arr[0];
//CHECK-NEXT:         _d_arr[0] = 0.;
//CHECK-NEXT:         *_d_x += _r_d0;
//CHECK-NEXT:     }
//CHECK-NEXT:     *_d_y += _d_res;
//CHECK-NEXT: }

double f5(double x, double y) {
  double& ref = x;
  double z = y * ref;
  x = z; // `x` should be stored because `ref` has been used
  ref -= y; // `ref` should not be stored because `x` has already been stored
  return ref; // x * y - y
}

//CHECK: void f5_grad(double x, double y, double *_d_x, double *_d_y) {
//CHECK-NEXT:     double &_d_ref = *_d_x;
//CHECK-NEXT:     double &ref = x;
//CHECK-NEXT:     double _d_z = 0.;
//CHECK-NEXT:     double z = y * ref;
//CHECK-NEXT:     double _t0 = x;
//CHECK-NEXT:     x = z;
//CHECK-NEXT:     ref -= y;
//CHECK-NEXT:     _d_ref += 1;
//CHECK-NEXT:     {
//CHECK-NEXT:         double _r_d0 = _d_ref;
//CHECK-NEXT:         *_d_y += -_r_d0;
//CHECK-NEXT:     }
//CHECK-NEXT:     {
//CHECK-NEXT:         x = _t0;
//CHECK-NEXT:         _d_z += *_d_x;
//CHECK-NEXT:         *_d_x = 0.;
//CHECK-NEXT:     }
//CHECK-NEXT:     {
//CHECK-NEXT:         *_d_y += _d_z * ref;
//CHECK-NEXT:         _d_ref += y * _d_z;
//CHECK-NEXT:     }
//CHECK-NEXT: }

double f6(double x, double y) {
  double arr[4];
  arr[0] = y;
  x *= arr[0]; // arr[0] gets used
  int i = 0;
  arr[i] = 0; // tbr should make an asumption that arr[i] might be arr[0]
  return x; // x * y
}

//CHECK: void f6_grad(double x, double y, double *_d_x, double *_d_y) {
//CHECK-NEXT:     double _d_arr[4] = {0};
//CHECK-NEXT:     double arr[4];
//CHECK-NEXT:     arr[0] = y;
//CHECK-NEXT:     double _t0 = x;
//CHECK-NEXT:     x *= arr[0];
//CHECK-NEXT:     int _d_i = 0;
//CHECK-NEXT:     int i = 0;
//CHECK-NEXT:     double _t1 = arr[i];
//CHECK-NEXT:     arr[i] = 0;
//CHECK-NEXT:     *_d_x += 1;
//CHECK-NEXT:     {
//CHECK-NEXT:         arr[i] = _t1;
//CHECK-NEXT:         double _r_d2 = _d_arr[i];
//CHECK-NEXT:         _d_arr[i] = 0.;
//CHECK-NEXT:     }
//CHECK-NEXT:     {
//CHECK-NEXT:         x = _t0;
//CHECK-NEXT:         double _r_d1 = *_d_x;
//CHECK-NEXT:         *_d_x = 0.;
//CHECK-NEXT:         *_d_x += _r_d1 * arr[0];
//CHECK-NEXT:         _d_arr[0] += x * _r_d1;
//CHECK-NEXT:     }
//CHECK-NEXT:     {
//CHECK-NEXT:         double _r_d0 = _d_arr[0];
//CHECK-NEXT:         _d_arr[0] = 0.;
//CHECK-NEXT:         *_d_y += _r_d0;
//CHECK-NEXT:     }
//CHECK-NEXT: }


#define TEST(F, x) { \
  result[0] = 0; \
  auto F##grad = clad::gradient<clad::opts::enable_tbr>(F);\
  F##grad.execute(x, result);\
  printf("{%.2f}\n", result[0]); \
}

#define TEST2(F, x, y) { \
  result[0] = 0; result[1] = 0; \
  auto F##grad = clad::gradient<clad::opts::enable_tbr>(F);\
  F##grad.execute(x, y, &result[0], &result[1]);\
  printf("{%.2f, %.2f}\n", result[0], result[1]); \
}

int main() {
  double result[2] = {};
  TEST(f1, 3); // CHECK-EXEC: {27.00}
  TEST(f2, 3); // CHECK-EXEC: {7.00}
  TEST(f3, 3); // CHECK-EXEC: {2.00}
  TEST2(f4, 3, 4) // CHECK-EXEC: {4.00, 3.00}
  TEST2(f5, 8, 3) // CHECK-EXEC: {3.00, 7.00}
  TEST2(f6, 5, 2) // CHECK-EXEC: {2.00, 5.00}
}
