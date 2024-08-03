// RUN: %cladclang %s -I%S/../../include -std=c++17 -oSwitchInit.out 2>&1 -lstdc++ -lm | %filecheck %s
// RUN: ./SwitchInit.out | %filecheck_exec %s
//CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"

double fn1(double i, double j) {
  double res = 0;
  switch (int count = 1;count) {
    case 0: res += i * j; break;
    case 1: res += i * i; {
        case 2: res += j * j;
      }
    default: res += i * i * j * j;
  }
  return res;
}

// CHECK: void fn1_grad(double i, double j, double *_d_i, double *_d_j) {
// CHECK-NEXT:     int _d_count = 0;
// CHECK-NEXT:     int count = 0;
// CHECK-NEXT:     int _cond0;
// CHECK-NEXT:     double _t0;
// CHECK-NEXT:     clad::tape<unsigned {{int|long}}> _t1 = {};
// CHECK-NEXT:     double _t2;
// CHECK-NEXT:     double _t3;
// CHECK-NEXT:     double _t4;
// CHECK-NEXT:     double _d_res = 0;
// CHECK-NEXT:     double res = 0;
// CHECK-NEXT:     {
// CHECK-NEXT:         count = 1;
// CHECK-NEXT:         _cond0 = count;
// CHECK-NEXT:         switch (_cond0) {
// CHECK-NEXT:             {
// CHECK-NEXT:               case 0:
// CHECK-NEXT:                 res += i * j;
// CHECK-NEXT:                 _t0 = res;
// CHECK-NEXT:             }
// CHECK-NEXT:             {
// CHECK-NEXT:                 clad::push(_t1, {{1U|1UL}});
// CHECK-NEXT:                 break;
// CHECK-NEXT:             }
// CHECK-NEXT:             {
// CHECK-NEXT:               case 1:
// CHECK-NEXT:                 res += i * i;
// CHECK-NEXT:                 _t2 = res;
// CHECK-NEXT:             }
// CHECK-NEXT:             {
// CHECK-NEXT:                 {
// CHECK-NEXT:                   case 2:
// CHECK-NEXT:                     res += j * j;
// CHECK-NEXT:                     _t3 = res;
// CHECK-NEXT:                 }
// CHECK-NEXT:             }
// CHECK-NEXT:             {
// CHECK-NEXT:               default:
// CHECK-NEXT:                 res += i * i * j * j;
// CHECK-NEXT:                 _t4 = res;
// CHECK-NEXT:             }
// CHECK-NEXT:             clad::push(_t1, {{2U|2UL}});
// CHECK-NEXT:         }
// CHECK-NEXT:     }
// CHECK-NEXT:     _d_res += 1;
// CHECK-NEXT:     {
// CHECK-NEXT:         switch (clad::pop(_t1)) {
// CHECK-NEXT:           case {{2U|2UL}}:
// CHECK-NEXT:             ;
// CHECK-NEXT:             {
// CHECK-NEXT:                 {
// CHECK-NEXT:                     res = _t4;
// CHECK-NEXT:                     double _r_d3 = _d_res;
// CHECK-NEXT:                     *_d_i += _r_d3 * j * j * i;
// CHECK-NEXT:                     *_d_i += i * _r_d3 * j * j;
// CHECK-NEXT:                     *_d_j += i * i * _r_d3 * j;
// CHECK-NEXT:                     *_d_j += i * i * j * _r_d3;
// CHECK-NEXT:                 }
// CHECK-NEXT:                 if (_cond0 != 0 && _cond0 != 1 && _cond0 != 2)
// CHECK-NEXT:                     break;
// CHECK-NEXT:             }
// CHECK-NEXT:             {
// CHECK-NEXT:                 {
// CHECK-NEXT:                     {
// CHECK-NEXT:                         res = _t3;
// CHECK-NEXT:                         double _r_d2 = _d_res;
// CHECK-NEXT:                         *_d_j += _r_d2 * j;
// CHECK-NEXT:                         *_d_j += j * _r_d2;
// CHECK-NEXT:                     }
// CHECK-NEXT:                     if (2 == _cond0)
// CHECK-NEXT:                         break;
// CHECK-NEXT:                 }
// CHECK-NEXT:             }
// CHECK-NEXT:             {
// CHECK-NEXT:                 {
// CHECK-NEXT:                     res = _t2;
// CHECK-NEXT:                     double _r_d1 = _d_res;
// CHECK-NEXT:                     *_d_i += _r_d1 * i;
// CHECK-NEXT:                     *_d_i += i * _r_d1;
// CHECK-NEXT:                 }
// CHECK-NEXT:                 if (1 == _cond0)
// CHECK-NEXT:                     break;
// CHECK-NEXT:             }
// CHECK-NEXT:           case {{1U|1UL}}:
// CHECK-NEXT:             ;
// CHECK-NEXT:             {
// CHECK-NEXT:                 {
// CHECK-NEXT:                     res = _t0;
// CHECK-NEXT:                     double _r_d0 = _d_res;
// CHECK-NEXT:                     *_d_i += _r_d0 * j;
// CHECK-NEXT:                     *_d_j += i * _r_d0;
// CHECK-NEXT:                 }
// CHECK-NEXT:                 if (0 == _cond0)
// CHECK-NEXT:                     break;
// CHECK-NEXT:             }
// CHECK-NEXT:         }
// CHECK-NEXT:     }
// CHECK-NEXT: }

#define TEST_2(F, x, y)                                                        \
  {                                                                            \
    result[0] = result[1] = 0;                                                 \
    auto d_##F = clad::gradient(F);                                            \
    d_##F.execute(x, y, result, result + 1);                                   \
    printf("{%.2f, %.2f}\n", result[0], result[1]);                            \
  }

int main() {
  double result[2] = {};

  TEST_2(fn1, 3, 5); // CHECK-EXEC: {156.00, 100.00}
}
