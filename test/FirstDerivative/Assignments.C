// RUN: %cladclang %s -I%S/../../include -oAssignments.out 2>&1 | %filecheck %s
// RUN: ./Assignments.out
//CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"
#include <cmath>

double f1(double x, double y) {
  x = y;
  return y;
}

// CHECK: double f1_darg0(double x, double y) {
// CHECK-NEXT:   double _d_x = 1;
// CHECK-NEXT:   double _d_y = 0;
// CHECK-NEXT:   _d_x = _d_y;
// CHECK-NEXT:   x = y;
// CHECK-NEXT:   return _d_y;
// CHECK-NEXT: }

double f2(double x, double y) {
  if (x < y)
    x = y;
  return x;
}

// CHECK: double f2_darg0(double x, double y) {
// CHECK-NEXT:   double _d_x = 1;
// CHECK-NEXT:   double _d_y = 0;
// CHECK-NEXT:   if (x < y) {
// CHECK-NEXT:     _d_x = _d_y;
// CHECK-NEXT:     x = y;
// CHECK-NEXT:   }
// CHECK-NEXT:   return _d_x;
// CHECK-NEXT: }

// CHECK: double f2_darg1(double x, double y) {
// CHECK-NEXT:   double _d_x = 0;
// CHECK-NEXT:   double _d_y = 1;
// CHECK-NEXT:   if (x < y) {
// CHECK-NEXT:     _d_x = _d_y;
// CHECK-NEXT:     x = y;
// CHECK-NEXT:   }
// CHECK-NEXT:   return _d_x;
// CHECK-NEXT: }

double f3(double x, double y) {
  x = x;
  x = x * x;
  y = x * x;
  x = y;
  return y;
}

// CHECK: double f3_darg0(double x, double y) {
// CHECK-NEXT:   double _d_x = 1;
// CHECK-NEXT:   double _d_y = 0;
// CHECK-NEXT:   _d_x = _d_x;
// CHECK-NEXT:   x = x;
// CHECK-NEXT:   _d_x = _d_x * x + x * _d_x;
// CHECK-NEXT:   x = x * x;
// CHECK-NEXT:   _d_y = _d_x * x + x * _d_x;
// CHECK-NEXT:   y = x * x;
// CHECK-NEXT:   _d_x = _d_y;
// CHECK-NEXT:   x = y;
// CHECK-NEXT:   return _d_y;
// CHECK-NEXT: }

// CHECK: double f3_darg1(double x, double y) {
// CHECK-NEXT:   double _d_x = 0;
// CHECK-NEXT:   double _d_y = 1;
// CHECK-NEXT:   _d_x = _d_x;
// CHECK-NEXT:   x = x;
// CHECK-NEXT:   _d_x = _d_x * x + x * _d_x;
// CHECK-NEXT:   x = x * x;
// CHECK-NEXT:   _d_y = _d_x * x + x * _d_x;
// CHECK-NEXT:   y = x * x;
// CHECK-NEXT:   _d_x = _d_y;
// CHECK-NEXT:   x = y;
// CHECK-NEXT:   return _d_y;
// CHECK-NEXT: }

double f4(double x, double y) {
   y = x;
   x = 0;
   return y;
}

// CHECK: double f4_darg0(double x, double y) {
// CHECK-NEXT:   double _d_x = 1;
// CHECK-NEXT:   double _d_y = 0;
// CHECK-NEXT:   _d_y = _d_x;
// CHECK-NEXT:   y = x;
// CHECK-NEXT:   _d_x = 0;
// CHECK-NEXT:   x = 0;
// CHECK-NEXT:   return _d_y;
// CHECK-NEXT: }

int main() {
  clad::differentiate(f1, 0);
  clad::differentiate(f2, 0);
  clad::differentiate(f2, 1);
  clad::differentiate(f3, 0);
  clad::differentiate(f3, 1);
  clad::differentiate <clad::order::first> (f4, 0); // testing order template parameter
}


