// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -disable-tbr %s -I%S/../../include -oUseful.out 2>&1 | %filecheck %s
// RUN: ./Useful.out | %filecheck_exec %s
// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -enable-ua -Xclang -plugin-arg-clad -Xclang -disable-tbr %s -I%S/../../include -oUseful.out
// RUN: ./Useful.out | %filecheck_exec %s
//CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"
#include "../TestUtils.h"

double f1(double x){
    double b = 1;
    return x;
}

// CHECK: double f1_darg0(double x) {
// CHECK-NEXT:    double _d_x = 1;
// CHECK-NEXT:    double b = 1;
// CHECK-NEXT:    return _d_x;
// CHECK-NEXT:}

double f2(double x){
  double a = 0;
  a = 1;
  return x;
}

// CHECK: double f2_darg0(double x) {
// CHECK-NEXT:    double _d_x = 1;
// CHECK-NEXT:    double a = 0;
// CHECK-NEXT:    a = 1;
// CHECK-NEXT:    return _d_x;
// CHECK-NEXT:}

double f3(double x){
    double x1 = 0, x2 = 1, x3 = 1, x4 = 1, x5 = 1;
    while(x5){
      x5 = x4;
      x4 = x3;
      x3 = x2;
      x2 = x1;
    }
    return x5;
}

// CHECK: double f3_darg0(double x) {
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     double _d_x1 = 0, _d_x2 = 0, _d_x3 = 0, _d_x4 = 0, _d_x5 = 0;
// CHECK-NEXT:     double x1 = 0, x2 = 1, x3 = 1, x4 = 1, x5 = 1;
// CHECK-NEXT:     while (x5)
// CHECK-NEXT:         {
// CHECK-NEXT:             _d_x5 = _d_x4;
// CHECK-NEXT:             x5 = x4;
// CHECK-NEXT:             _d_x4 = _d_x3;
// CHECK-NEXT:             x4 = x3;
// CHECK-NEXT:             _d_x3 = _d_x2;
// CHECK-NEXT:             x3 = x2;
// CHECK-NEXT:             _d_x2 = _d_x1;
// CHECK-NEXT:             x2 = x1;
// CHECK-NEXT:         }
// CHECK-NEXT:     return _d_x5;

double f4(double x){
  double a = 0;
  if(0){
    a = x;
  }
  return a;
}

// CHECK: double f4_darg0(double x) {
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     double _d_a = 0;
// CHECK-NEXT:     double a = 0;
// CHECK-NEXT:     if (0) {
// CHECK-NEXT:         _d_a = _d_x;
// CHECK-NEXT:         a = x;
// CHECK-NEXT:     }
// CHECK-NEXT:     return _d_a;

double f5_1(double x){
  double b = 1;
  double c = x;
  return c;
}

double f5(double x){
  double a = f5_1(x);
  return a;
}

// CHECK: clad::ValueAndPushforward<double, double> f5_1_pushforward(double x, double _d_x) {
// CHECK-NEXT:     double b = 1;
// CHECK-NEXT:     double _d_c = _d_x;
// CHECK-NEXT:     double c = x;
// CHECK-NEXT:     return {c, _d_c};
// CHECK-NEXT: }
// CHECK-NEXT: double f5_darg0(double x) {
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     clad::ValueAndPushforward<double, double> _t0 = f5_1_pushforward(x, _d_x);
// CHECK-NEXT:     double _d_a = _t0.pushforward;
// CHECK-NEXT:     double a = _t0.value;
// CHECK-NEXT:     return _d_a;

double f6(double x){
  double j = 0;
  for(int i = 0; i<5; i++){
    j += x + x;
  }
  return j;
}

// CHECK: double f6_darg0(double x) {
// CHECK-NEXT:    double _d_x = 1;
// CHECK-NEXT:    double _d_j = 0;
// CHECK-NEXT:    double j = 0;
// CHECK-NEXT:    for (int i = 0; i < 5; i++) {
// CHECK-NEXT:        _d_j += _d_x + _d_x;
// CHECK-NEXT:        j += x + x;
// CHECK-NEXT:    }
// CHECK-NEXT:    return _d_j;
// CHECK-NEXT:}

int main(){
    INIT_DIFFERENTIATE_UA(f1, "x");
    INIT_DIFFERENTIATE_UA(f2, "x");
    INIT_DIFFERENTIATE_UA(f3, "x");
    INIT_DIFFERENTIATE_UA(f4, "x");
    INIT_DIFFERENTIATE_UA(f5, "x");
    INIT_DIFFERENTIATE_UA(f6, "x");

    TEST_DIFFERENTIATE(f1, 3); // CHECK-EXEC: {1.00}
    TEST_DIFFERENTIATE(f2, 3); // CHECK-EXEC: {1.00}
    TEST_DIFFERENTIATE(f3, 3); // CHECK-EXEC: {0.00}
    TEST_DIFFERENTIATE(f4, 3); // CHECK-EXEC: {0.00}
    TEST_DIFFERENTIATE(f5, 3); // CHECK-EXEC: {1.00}
    TEST_DIFFERENTIATE(f6, 3); // CHECK-EXEC: {10.00}
    return 0;
}