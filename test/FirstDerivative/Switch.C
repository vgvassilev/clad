// RUN: %cladclang %s -I%S/../../include -oSwitch.out 2>&1 | %filecheck %s
// RUN: ./Switch.out | %filecheck_exec %s
//CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"

double fn1 (double i, double j, int choice) {
  double a = 0, b = 0;
  double c = i + j;
  switch (choice) {
    case 1:
      a = i*i;
      b = a/(a*j);
      break;
    case 2:
      a = i*j;
      b = a/(a+i);
      break;
    case 3:
      a = j*j;
      b = a/(a*i);
      break;
    default:
      a = i*i;
      b = j*j;
      break;
  }
  return a+b+c; 
}

// CHECK: double fn1_darg0(double i, double j, int choice) {
// CHECK-NEXT:     double _d_i = 1;
// CHECK-NEXT:     double _d_j = 0;
// CHECK-NEXT:     int _d_choice = 0;
// CHECK-NEXT:     double _d_a = 0, _d_b = 0;
// CHECK-NEXT:     double a = 0, b = 0;
// CHECK-NEXT:     double _d_c = _d_i + _d_j;
// CHECK-NEXT:     double c = i + j;
// CHECK-NEXT:     {
// CHECK-NEXT:         switch (choice) {
// CHECK-NEXT:           case 1:
// CHECK-NEXT:             {
// CHECK-NEXT:                 _d_a = _d_i * i + i * _d_i;
// CHECK-NEXT:                 a = i * i;
// CHECK-NEXT:                 double _t0 = (a * j);
// CHECK-NEXT:                 _d_b = (_d_a * _t0 - a * (_d_a * j + a * _d_j)) / (_t0 * _t0);
// CHECK-NEXT:                 b = a / _t0;
// CHECK-NEXT:                 break;
// CHECK-NEXT:             }
// CHECK-NEXT:           case 2:
// CHECK-NEXT:             {
// CHECK-NEXT:                 _d_a = _d_i * j + i * _d_j;
// CHECK-NEXT:                 a = i * j;
// CHECK-NEXT:                 double _t1 = (a + i);
// CHECK-NEXT:                 _d_b = (_d_a * _t1 - a * (_d_a + _d_i)) / (_t1 * _t1);
// CHECK-NEXT:                 b = a / _t1;
// CHECK-NEXT:                 break;
// CHECK-NEXT:             }
// CHECK-NEXT:           case 3:
// CHECK-NEXT:             {
// CHECK-NEXT:                 _d_a = _d_j * j + j * _d_j;
// CHECK-NEXT:                 a = j * j;
// CHECK-NEXT:                 double _t2 = (a * i);
// CHECK-NEXT:                 _d_b = (_d_a * _t2 - a * (_d_a * i + a * _d_i)) / (_t2 * _t2);
// CHECK-NEXT:                 b = a / _t2;
// CHECK-NEXT:                 break;
// CHECK-NEXT:             }
// CHECK-NEXT:           default:
// CHECK-NEXT:             {
// CHECK-NEXT:                 _d_a = _d_i * i + i * _d_i;
// CHECK-NEXT:                 a = i * i;
// CHECK-NEXT:                 _d_b = _d_j * j + j * _d_j;
// CHECK-NEXT:                 b = j * j;
// CHECK-NEXT:                 break;
// CHECK-NEXT:             }
// CHECK-NEXT:         }
// CHECK-NEXT:     }
// CHECK-NEXT:     return _d_a + _d_b + _d_c;
// CHECK-NEXT: }

double fn2 (double i, double j, int choice) {
  double a = 0, b = 0;
  double c = i + j;
  switch (int diff_choice = choice + 1) {
    case 1:
      a = i;
      b = a/(a*j);
      break;
    default:
      a = i*i;
      b = j*j;
      break;
    case 2:
      a = j;
      b = a/(a+i);
      break;
    case 3:
      a = 1;
      b = (i/j)*i;
      break;
  }
  return a+b+c; 
}

// CHECK: double fn2_darg0(double i, double j, int choice) {
// CHECK-NEXT:     double _d_i = 1;
// CHECK-NEXT:     double _d_j = 0;
// CHECK-NEXT:     int _d_choice = 0;
// CHECK-NEXT:     double _d_a = 0, _d_b = 0;
// CHECK-NEXT:     double a = 0, b = 0;
// CHECK-NEXT:     double _d_c = _d_i + _d_j;
// CHECK-NEXT:     double c = i + j;
// CHECK-NEXT:     {
// CHECK-NEXT:         int _d_diff_choice = _d_choice + 0;
// CHECK-NEXT:         switch (int diff_choice = choice + 1) {
// CHECK-NEXT:           case 1:
// CHECK-NEXT:             {
// CHECK-NEXT:                 _d_a = _d_i;
// CHECK-NEXT:                 a = i;
// CHECK-NEXT:                 double _t0 = (a * j);
// CHECK-NEXT:                 _d_b = (_d_a * _t0 - a * (_d_a * j + a * _d_j)) / (_t0 * _t0);
// CHECK-NEXT:                 b = a / _t0;
// CHECK-NEXT:                 break;
// CHECK-NEXT:             }
// CHECK-NEXT:           default:
// CHECK-NEXT:             {
// CHECK-NEXT:                 _d_a = _d_i * i + i * _d_i;
// CHECK-NEXT:                 a = i * i;
// CHECK-NEXT:                 _d_b = _d_j * j + j * _d_j;
// CHECK-NEXT:                 b = j * j;
// CHECK-NEXT:                 break;
// CHECK-NEXT:             }
// CHECK-NEXT:           case 2:
// CHECK-NEXT:             {
// CHECK-NEXT:                 _d_a = _d_j;
// CHECK-NEXT:                 a = j;
// CHECK-NEXT:                 double _t1 = (a + i);
// CHECK-NEXT:                 _d_b = (_d_a * _t1 - a * (_d_a + _d_i)) / (_t1 * _t1);
// CHECK-NEXT:                 b = a / _t1;
// CHECK-NEXT:                 break;
// CHECK-NEXT:             }
// CHECK-NEXT:           case 3:
// CHECK-NEXT:             {
// CHECK-NEXT:                 _d_a = 0;
// CHECK-NEXT:                 a = 1;
// CHECK-NEXT:                 double _t2 = (i / j);
// CHECK-NEXT:                 _d_b = ((_d_i * j - i * _d_j) / (j * j)) * i + _t2 * _d_i;
// CHECK-NEXT:                 b = _t2 * i;
// CHECK-NEXT:                 break;
// CHECK-NEXT:             }
// CHECK-NEXT:         }
// CHECK-NEXT:     }
// CHECK-NEXT:     return _d_a + _d_b + _d_c;
// CHECK-NEXT: }

double fn3 (double i, double j, int choice) {
  double a = 0;
  switch (choice)
    case 0:
    case 1:
    default:
    a = (i*i + j/i)*i;

  return a;
}

// CHECK: double fn3_darg0(double i, double j, int choice) {
// CHECK-NEXT:     double _d_i = 1;
// CHECK-NEXT:     double _d_j = 0;
// CHECK-NEXT:     int _d_choice = 0;
// CHECK-NEXT:     double _d_a = 0;
// CHECK-NEXT:     double a = 0;
// CHECK-NEXT:     {
// CHECK-NEXT:         switch (choice) {
// CHECK-NEXT:           case 0:
// CHECK-NEXT:             {
// CHECK-NEXT:             }
// CHECK-NEXT:           case 1:
// CHECK-NEXT:             {
// CHECK-NEXT:             }
// CHECK-NEXT:           default:
// CHECK-NEXT:             {
// CHECK-NEXT:                 double _t0 = (i * i + j / i);
// CHECK-NEXT:                 _d_a = (_d_i * i + i * _d_i + (_d_j * i - j * _d_i) / (i * i)) * i + _t0 * _d_i;
// CHECK-NEXT:                 a = _t0 * i;
// CHECK-NEXT:             }
// CHECK-NEXT:         }
// CHECK-NEXT:     }
// CHECK-NEXT:     return _d_a;
// CHECK-NEXT: }

double fn4(double i, double j, int choice) {
  double a = 0, b = 0;

  switch (choice) {
    case 0: {
      a = i + j;
      switch (0) {
        case 0: b = i * i; break;
        case 1: b = i * j; break;
        default: b = j * j;
      }
    } break;
    case 1: {
      a = i * i;
      b = i * i / j;
      break;
    }
    default: {
      a = i * j;
      switch (1) {
        case 0: b = i * i; break;
        case 1: b = i * j; break;
        default: b = j * j;
      }
    }
  }

  return a + b;
}

// CHECK: double fn4_darg0(double i, double j, int choice) {
// CHECK-NEXT:     double _d_i = 1;
// CHECK-NEXT:     double _d_j = 0;
// CHECK-NEXT:     int _d_choice = 0;
// CHECK-NEXT:     double _d_a = 0, _d_b = 0;
// CHECK-NEXT:     double a = 0, b = 0;
// CHECK-NEXT:     {
// CHECK-NEXT:         switch (choice) {
// CHECK-NEXT:           case 0:
// CHECK-NEXT:             {
// CHECK-NEXT:                 {
// CHECK-NEXT:                     _d_a = _d_i + _d_j;
// CHECK-NEXT:                     a = i + j;
// CHECK-NEXT:                     {
// CHECK-NEXT:                         switch (0) {
// CHECK-NEXT:                           case 0:
// CHECK-NEXT:                             {
// CHECK-NEXT:                                 _d_b = _d_i * i + i * _d_i;
// CHECK-NEXT:                                 b = i * i;
// CHECK-NEXT:                                 break;
// CHECK-NEXT:                             }
// CHECK-NEXT:                           case 1:
// CHECK-NEXT:                             {
// CHECK-NEXT:                                 _d_b = _d_i * j + i * _d_j;
// CHECK-NEXT:                                 b = i * j;
// CHECK-NEXT:                                 break;
// CHECK-NEXT:                             }
// CHECK-NEXT:                           default:
// CHECK-NEXT:                             {
// CHECK-NEXT:                                 _d_b = _d_j * j + j * _d_j;
// CHECK-NEXT:                                 b = j * j;
// CHECK-NEXT:                             }
// CHECK-NEXT:                         }
// CHECK-NEXT:                     }
// CHECK-NEXT:                 }
// CHECK-NEXT:                 break;
// CHECK-NEXT:             }
// CHECK-NEXT:           case 1:
// CHECK-NEXT:             {
// CHECK-NEXT:                 {
// CHECK-NEXT:                     _d_a = _d_i * i + i * _d_i;
// CHECK-NEXT:                     a = i * i;
// CHECK-NEXT:                     double _t0 = i * i;
// CHECK-NEXT:                     _d_b = ((_d_i * i + i * _d_i) * j - _t0 * _d_j) / (j * j);
// CHECK-NEXT:                     b = _t0 / j;
// CHECK-NEXT:                     break;
// CHECK-NEXT:                 }
// CHECK-NEXT:             }
// CHECK-NEXT:           default:
// CHECK-NEXT:             {
// CHECK-NEXT:                 {
// CHECK-NEXT:                     _d_a = _d_i * j + i * _d_j;
// CHECK-NEXT:                     a = i * j;
// CHECK-NEXT:                     {
// CHECK-NEXT:                         switch (1) {
// CHECK-NEXT:                           case 0:
// CHECK-NEXT:                             {
// CHECK-NEXT:                                 _d_b = _d_i * i + i * _d_i;
// CHECK-NEXT:                                 b = i * i;
// CHECK-NEXT:                                 break;
// CHECK-NEXT:                             }
// CHECK-NEXT:                           case 1:
// CHECK-NEXT:                             {
// CHECK-NEXT:                                 _d_b = _d_i * j + i * _d_j;
// CHECK-NEXT:                                 b = i * j;
// CHECK-NEXT:                                 break;
// CHECK-NEXT:                             }
// CHECK-NEXT:                           default:
// CHECK-NEXT:                             {
// CHECK-NEXT:                                 _d_b = _d_j * j + j * _d_j;
// CHECK-NEXT:                                 b = j * j;
// CHECK-NEXT:                             }
// CHECK-NEXT:                         }
// CHECK-NEXT:                     }
// CHECK-NEXT:                 }
// CHECK-NEXT:             }
// CHECK-NEXT:         }
// CHECK-NEXT:     }
// CHECK-NEXT:     return _d_a + _d_b;
// CHECK-NEXT: }

#define INIT(fn, args)\
auto d_##fn = clad::differentiate(fn, args);

#define TEST_SWITCH_CASES(fn, start, end)\
for (int i=start; i<=end; ++i)\
  printf("%.2f ", d_##fn.execute(3, 5, i));\
printf("\n");

int main() {
  INIT(fn1, "i");
  INIT(fn2, "i");
  INIT(fn3, "i");
  INIT(fn4, "i");

  TEST_SWITCH_CASES(fn1, 0, 4);   // CHECK-EXEC: 7.00 7.00 6.00 0.89 7.00
  TEST_SWITCH_CASES(fn2, 0, 4);   // CHECK-EXEC: 2.00 0.92 2.20 7.00 7.00
  TEST_SWITCH_CASES(fn3, 0, 2);   // CHECK-EXEC: 27.00 27.00 27.00
  TEST_SWITCH_CASES(fn4, 0, 3);   // CHECK-EXEC: 7.00 7.20 10.00 10.00
}