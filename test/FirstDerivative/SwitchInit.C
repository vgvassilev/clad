// RUN: %cladclang %s -I%S/../../include -std=c++17 -oSwitchInit.out 2>&1 | %filecheck %s
// RUN: ./SwitchInit.out | %filecheck_exec %s
//CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"

double fn1(double i, double j, int choice) {
  double a = 0;
  switch (short effective_choice = choice + 1, another_choice = choice + 2;
          effective_choice) {
    case 1: a = i; break;
    case 2: a = i * i; break;
    case 3: a = i * i * i; break;
    default: a = j;
  }
  return a;
}

// CHECK: double fn1_darg0(double i, double j, int choice) {
// CHECK-NEXT:     double _d_i = 1;
// CHECK-NEXT:     double _d_j = 0;
// CHECK-NEXT:     int _d_choice = 0;
// CHECK-NEXT:     double _d_a = 0;
// CHECK-NEXT:     double a = 0;
// CHECK-NEXT:     {
// CHECK-NEXT:         short _d_effective_choice = _d_choice + 0, _d_another_choice = _d_choice + 0;
// CHECK-NEXT:         switch ({{(short effective_choice = choice \+ 1, another_choice = choice \+ 2; )?}}effective_choice) {
// CHECK-NEXT:           case 1:
// CHECK-NEXT:             {
// CHECK-NEXT:                 _d_a = _d_i;
// CHECK-NEXT:                 a = i;
// CHECK-NEXT:                 break;
// CHECK-NEXT:             }
// CHECK-NEXT:           case 2:
// CHECK-NEXT:             {
// CHECK-NEXT:                 _d_a = _d_i * i + i * _d_i;
// CHECK-NEXT:                 a = i * i;
// CHECK-NEXT:                 break;
// CHECK-NEXT:             }
// CHECK-NEXT:           case 3:
// CHECK-NEXT:             {
// CHECK-NEXT:                 double _t0 = i * i;
// CHECK-NEXT:                 _d_a = (_d_i * i + i * _d_i) * i + _t0 * _d_i;
// CHECK-NEXT:                 a = _t0 * i;
// CHECK-NEXT:                 break;
// CHECK-NEXT:             }
// CHECK-NEXT:           default:
// CHECK-NEXT:             {
// CHECK-NEXT:                 _d_a = _d_j;
// CHECK-NEXT:                 a = j;
// CHECK-NEXT:             }
// CHECK-NEXT:         }
// CHECK-NEXT:     }
// CHECK-NEXT:     return _d_a;
// CHECK-NEXT: }

#define INIT(fn, args)\
auto d_##fn = clad::differentiate(fn, args);

#define TEST_SWITCH_CASES(fn, start, end)\
for (int i=start; i<=end; ++i)\
  printf("%.2f ", d_##fn.execute(3, 5, i));\
printf("\n");

int main() {
  INIT(fn1, "i");
  TEST_SWITCH_CASES(fn1, 0, 3); // CHECK-EXEC: 1.00 6.00 27.00 0.00 
}