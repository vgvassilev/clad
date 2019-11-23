// RUN: %cladclang %s -lm -I%S/../../include -oLoops.out 2>&1 | FileCheck %s
// RUN: ./Loops.out | FileCheck -check-prefix=CHECK-EXEC %s
//CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"
#include <cmath>

double f1(double x, int y) {
    double r = 1.0;
    for (int i = 0; i < y; i = i + 1)
        r = r * x;
    return r;
} // = pow(x, y);

double f1_darg0(double x, int y);
// CHECK: double f1_darg0(double x, int y) {
// CHECK-NEXT:   double _d_x = 1;
// CHECK-NEXT:   int _d_y = 0;
// CHECK-NEXT:   double _d_r = 0.;
// CHECK-NEXT:   double r = 1.;
// CHECK-NEXT:   {
// CHECK-NEXT:     int _d_i = 0;
// CHECK-NEXT:     for (int i = 0; i < y; (_d_i = _d_i + 0) , (i = i + 1)) {
// CHECK-NEXT:       _d_r = _d_r * x + r * _d_x;
// CHECK-NEXT:       r = r * x;
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   return _d_r;
// CHECK-NEXT: }
// _d_r(i) = _d_r(i - 1) * x + r(i - 1) 
// r(i) = r(i - 1) * x = pow(x, i)
// ->
// _d_r(i) = _d_r(i - 1) * x + pow(x, i - 1) = i * pow(x, i - 1)
// _d_r(y) = y * pow(x, y - 1);

double f1_inc(double x, int y) {
    double r = 1.0;
    for (int i = 0; i < y; i++)
        r *= x;
    return r;
} // = pow(x, y);

double f1_inc_darg0(double x, int y);
//CHECK:   double f1_inc_darg0(double x, int y) {
//CHECK-NEXT:       double _d_x = 1;
//CHECK-NEXT:       int _d_y = 0;
//CHECK-NEXT:       double _d_r = 0.;
//CHECK-NEXT:       double r = 1.;
//CHECK-NEXT:       {
//CHECK-NEXT:           int _d_i = 0;
//CHECK-NEXT:           for (int i = 0; i < y; i++) {
//CHECK-NEXT:               _d_r = _d_r * x + r * _d_x;
//CHECK-NEXT:               r *= x;
//CHECK-NEXT:           }
//CHECK-NEXT:       }
//CHECK-NEXT:       return _d_r;
//CHECK-NEXT:   }


double f2(double x, int y) {
    for (int i = 0; i < y; i = i + 1)
        x = x * x;
    return x;
} // = pow(x, pow(2, y));

double f2_darg0(double x, int y);
// CHECK: double f2_darg0(double x, int y) {
// CHECK-NEXT:   double _d_x = 1;
// CHECK-NEXT:   int _d_y = 0;
// CHECK-NEXT:   {
// CHECK-NEXT:     int _d_i = 0;
// CHECK-NEXT:     for (int i = 0; i < y; (_d_i = _d_i + 0) , (i = i + 1)) {
// CHECK-NEXT:       _d_x = _d_x * x + x * _d_x;
// CHECK-NEXT:       x = x * x;
// CHECK-NEXT:     } 
// CHECK-NEXT:  }
// CHECK-NEXT:  return _d_x;
// CHECK-NEXT: }
// _d_x(i) = 2 * _d_x(i - 1) * x(i - 1) 
// x(i) = x(i - 1) * x(i - 1) = pow(x, pow(2, i))
// ->
// _d_x(i) = 2 * _d_x(i - 1) * pow(x, pow(2, i - 1)) = pow(2, i) * pow(x, sum(j = 0...i-1, pow(2, j))) 
// = pow(2, i) * pow(x, pow(2, i) - 1)
// _d_x(y) = pow(2, i) * pow(x, pow(2, y) - 1)

double f2_inc(double x, int y) {
    for (int i = 0; i < y; i++)
        x *= x;
    return x;
} // = pow(x, pow(2, y));

double f2_inc_darg0(double x, int y);
//CHECK:   double f2_inc_darg0(double x, int y) {
//CHECK-NEXT:       double _d_x = 1;
//CHECK-NEXT:       int _d_y = 0;
//CHECK-NEXT:       {
//CHECK-NEXT:           int _d_i = 0;
//CHECK-NEXT:           for (int i = 0; i < y; i++) {
//CHECK-NEXT:               _d_x = _d_x * x + x * _d_x;
//CHECK-NEXT:               x *= x;
//CHECK-NEXT:           }
//CHECK-NEXT:       }
//CHECK-NEXT:       return _d_x;
//CHECK-NEXT:   }


double f3(double x, int y) {
    double r = 1.0;
    for (int i = 0; i < y; r = r * x)
        i = i + 1;
    return r;
} // = pow(x, y);

double f3_darg0(double x, int y);
// CHECK: double f3_darg0(double x, int y) {
// CHECK-NEXT:   double _d_x = 1;
// CHECK-NEXT:   int _d_y = 0;
// CHECK-NEXT:   double _d_r = 0.;
// CHECK-NEXT:   double r = 1.;
// CHECK-NEXT:   {
// CHECK-NEXT:     int _d_i = 0;
// CHECK-NEXT:     for (int i = 0; i < y; (_d_r = _d_r * x + r * _d_x) , (r = r * x)) {
// CHECK-NEXT:       _d_i = _d_i + 0;
// CHECK-NEXT:       i = i + 1;
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   return _d_r;
// CHECK-NEXT: }
// = y * pow(x, y-1)

double f3_inc(double x, int y) {
    double r = 1.0;
    for (int i = 0; i < y; r *= x)
        i++;
    return r;
} // = pow(x, y);

double f3_inc_darg0(double x, int y);
//CHECK:   double f3_inc_darg0(double x, int y) {
//CHECK-NEXT:       double _d_x = 1;
//CHECK-NEXT:       int _d_y = 0;
//CHECK-NEXT:       double _d_r = 0.;
//CHECK-NEXT:       double r = 1.;
//CHECK-NEXT:       {
//CHECK-NEXT:           int _d_i = 0;
//CHECK-NEXT:           for (int i = 0; i < y; (_d_r = _d_r * x + r * _d_x) , (r *= x))
//CHECK-NEXT:               i++;
//CHECK-NEXT:       }
//CHECK-NEXT:       return _d_r;
//CHECK-NEXT:   }


double f4(double x, int y) {
    double r = 1;
    int i;
    for (i = 0; i < y; r = r * std::sin(x))
        i = i + 1;
    return r;
} // = pow(sin(x), y)

double f4_darg0(double x, int y);
// CHECK: double f4_darg0(double x, int y) {
// CHECK-NEXT:   double _d_x = 1;
// CHECK-NEXT:   int _d_y = 0;
// CHECK-NEXT:   double _d_r = 0;
// CHECK-NEXT:   double r = 1;
// CHECK-NEXT:   int _d_i;
// CHECK-NEXT:   int i;
// CHECK-NEXT:   {
// CHECK-NEXT:     _d_i = 0;
// CHECK-NEXT:     for (i = 0; i < y; [&]             {
// CHECK-NEXT:       double _t1 = std::sin(x);
// CHECK-NEXT:      _d_r = _d_r * _t1 + r * (custom_derivatives::sin_darg0(x) * _d_x);
// CHECK-NEXT:      r = r * _t1;
// CHECK:        }
// CHECK:        ()) {
// CHECK-NEXT:          _d_i = _d_i + 0;
// CHECK-NEXT:          i = i + 1;
// CHECK-NEXT:        }
// CHECK-NEXT:     }
// CHECK-NEXT: return _d_r;
// CHECK-NEXT: }
// = y * cos(x) * pow(sin(x), y-1) = y/2.0 * sin(2*x) * pow(sin(x), y-2)

double f4_inc(double x, int y) {
    double r = 1;
    int i;
    for (i = 0; i < y; r *= std::sin(x))
        i++;
    return r;
} // = pow(sin(x), y)

double f4_inc_darg0(double x, int y);
//CHECK:   double f4_inc_darg0(double x, int y) {
//CHECK-NEXT:       double _d_x = 1;
//CHECK-NEXT:       int _d_y = 0;
//CHECK-NEXT:       double _d_r = 0;
//CHECK-NEXT:       double r = 1;
//CHECK-NEXT:       int _d_i;
//CHECK-NEXT:       int i;
//CHECK-NEXT:       {
//CHECK-NEXT:           _d_i = 0;
//CHECK-NEXT:           for (i = 0; i < y; [&]             {
//CHECK-NEXT:                   double _t1 = std::sin(x);
//CHECK-NEXT:                   _d_r = _d_r * _t1 + r * (custom_derivatives::sin_darg0(x) * _d_x);
//CHECK-NEXT:                   r *= _t1;
//CHECK:        }
//CHECK:        ())
//CHECK-NEXT:           i++;
//CHECK-NEXT:       }
//CHECK-NEXT:       return _d_r;
//CHECK-NEXT:   }

int main() {
  clad::differentiate(f1, 0);
  printf("Result is = %.2f\n", f1_darg0(10, 2)); // CHECK-EXEC: Result is = 20.00

  clad::differentiate(f1_inc, 0);
  printf("Result is = %.2f\n", f1_inc_darg0(10, 2)); // CHECK-EXEC: Result is = 20.00

  clad::differentiate(f2, 0);
  printf("Result is = %.2f\n", f2_darg0(10, 2)); // CHECK-EXEC: Result is = 4000.00

  clad::differentiate(f2_inc, 0);
  printf("Result is = %.2f\n", f2_inc_darg0(10, 2)); // CHECK-EXEC: Result is = 4000.00

  clad::differentiate(f3, 0);
  printf("Result is = %.2f\n", f3_darg0(10, 2)); // CHECK-EXEC: Result is = 20.00

  clad::differentiate(f3_inc, 0);
  printf("Result is = %.2f\n", f3_inc_darg0(10, 2)); // CHECK-EXEC: Result is = 20.00

  clad::differentiate(f4, 0);
  printf("Result is = %.2f\n", f4_darg0(M_PI/4, 8)); // CHECK-EXEC: Result is = 0.50

  clad::differentiate(f4_inc, 0);
  printf("Result is = %.2f\n", f4_inc_darg0(M_PI/4, 8)); // CHECK-EXEC: Result is = 0.50
  
}


