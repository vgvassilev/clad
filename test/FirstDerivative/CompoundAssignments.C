// RUN: %cladclang %s -I%S/../../include -oCompoundAssignments.out 2>&1 | %filecheck %s
// RUN: ./CompoundAssignments.out
//CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"
#include <cmath>

double f1(double x, double y) {
  x += y;
  x += x;
  x -= y;
  x -= x;
  return x; // == 0
}

double f1_darg0(double x, double y);
//CHECK:   double f1_darg0(double x, double y) {
//CHECK-NEXT:       double _d_x = 1;
//CHECK-NEXT:       double _d_y = 0;
//CHECK-NEXT:       _d_x += _d_y;
//CHECK-NEXT:       x += y;
//CHECK-NEXT:       _d_x += _d_x;
//CHECK-NEXT:       x += x;
//CHECK-NEXT:       _d_x -= _d_y;
//CHECK-NEXT:       x -= y;
//CHECK-NEXT:       _d_x -= _d_x;
//CHECK-NEXT:       x -= x;
//CHECK-NEXT:       return _d_x;
//CHECK-NEXT:   }

double f2(double x, double y) {
  x *= y;
  x *= x;
  x /= y;
  x /= x;
  return x; // == 1
}

double f2_darg0(double x, double y);
//CHECK:   double f2_darg0(double x, double y) {
//CHECK-NEXT:       double _d_x = 1;
//CHECK-NEXT:       double _d_y = 0;
//CHECK-NEXT:       _d_x = _d_x * y + x * _d_y;
//CHECK-NEXT:       x *= y;
//CHECK-NEXT:       _d_x = _d_x * x + x * _d_x;
//CHECK-NEXT:       x *= x;
//CHECK-NEXT:       _d_x = (_d_x * y - x * _d_y) / (y * y);
//CHECK-NEXT:       x /= y;
//CHECK-NEXT:       _d_x = (_d_x * x - x * _d_x) / (x * x);
//CHECK-NEXT:       x /= x;
//CHECK-NEXT:       return _d_x;
//CHECK-NEXT:   }


double f3(double x) {
  x++;
  x--;
  ++x;
  --x;
  return x; // == x_in
}

double f3_darg0(double x);
//CHECK:   double f3_darg0(double x) {
//CHECK-NEXT:       double _d_x = 1;
//CHECK-NEXT:       x++;
//CHECK-NEXT:       x--;
//CHECK-NEXT:       ++x;
//CHECK-NEXT:       --x;
//CHECK-NEXT:       return _d_x;
//CHECK-NEXT:   }

double f4(double x, double y) {
    x = (y += x);
    return x; // == x_in + y_in
}

double f4_darg0(double x, double y);
//CHECK:   double f4_darg0(double x, double y) {
//CHECK-NEXT:       double _d_x = 1;
//CHECK-NEXT:       double _d_y = 0;
//CHECK-NEXT:       _d_x = (_d_y += _d_x);
//CHECK-NEXT:       x = (y += x);
//CHECK-NEXT:       return _d_x;
//CHECK-NEXT:   }

double f5(double x, double y) {
    x /= std::pow(2.0, y);
    x *= std::pow(2.0, y);
    return x; // == x_in
}

double f5_darg0(double x, double y);
//FIXME-CHECK:   double f5_darg0(double x, double y) {
//FIXME-CHECK-NEXT:       double _d_x = 1;
//FIXME-CHECK-NEXT:       double _d_y = 0;
//FIXME-CHECK-NEXT:       double _t0 = std::pow(2., y);
//FIXME-CHECK-NEXT:       _d_x = (_d_x * _t0 - x * (custom_derivatives::pow_darg0(2., y) * (0. + _d_y))) / (_t0 * _t0);
//FIXME-CHECK-NEXT:       x /= _t0;
//FIXME-CHECK-NEXT:       double _t1 = std::pow(2., y);
//FIXME-CHECK-NEXT:       _d_x = _d_x * _t1 + x * (custom_derivatives::pow_darg0(2., y) * (0. + _d_y));
//FIXME-CHECK-NEXT:       x *= _t1;
//FIXME-CHECK-NEXT:       return _d_x;
//FIXME-CHECK-NEXT:   }


int main() {
  clad::differentiate(f1, 0);
  printf("Result is = %.2f\n", f1_darg0(100, 100)); // CHECK-EXEC: Result is = 0.00

  clad::differentiate(f2, 0);
  printf("Result is = %.2f\n", f2_darg0(100, 100)); // CHECK-EXEC: Result is = 0.00

  clad::differentiate(f3, 0);
  printf("Result is = %.2f\n", f3_darg0(100)); // CHECK-EXEC: Result is = 1.00

  clad::differentiate(f4, 0);
  printf("Result is = %.2f\n", f4_darg0(100, 100)); // CHECK-EXEC: Result is = 1.00

  // This test is currently broken.
  //clad::differentiate(f5, 0);
  //printf("Result is = %.2f\n", f5_darg0(100, 10)); // CHECK-EXEC: Result is = 1.00
}


