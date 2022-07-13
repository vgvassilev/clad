// RUN: %cladclang %s -I%S/../../include -oFunctionsInNamespaces.out 
// RUN: ./FunctionsInNamespaces.out | FileCheck -check-prefix=CHECK-EXEC %s

// CHECK-NOT: {{.*error|warning|note:.*}}
#include "clad/Differentiator/Differentiator.h"

#include "../TestUtils.h"

namespace function_namespace10 {
  int func1(int x) {
    return x*x*x + x*x;
  }

  int func2(int x) {
    return x*x + x;
  }

  namespace function_namespace11 {
    int func3(int x, int y) {
      return x*x*x + y*y;
    }

    int func4(int x, int y) {
      return x*x + y;
    }
  }
}

namespace function_namespace2 {
  float func1(float x) {
    return x*x*x + x*x;
  }

  float func2(float x) {
    return x*x + x;
  }

  int func3(int x, int y) {
    return function_namespace10::function_namespace11::func4(x, y);
  }
}

int test_1(int x, int y) {
  return function_namespace2::func3(x, y);
}

// CHECK: clad::ValueAndPushforward<int, int> func4_pushforward(int x, int y, int _d_x, int _d_y) {
// CHECK-NEXT:     return {x * x + y, _d_x * x + x * _d_x + _d_y};
// CHECK-NEXT: }

// CHECK: clad::ValueAndPushforward<int, int> func3_pushforward(int x, int y, int _d_x, int _d_y) {
// CHECK-NEXT:     clad::ValueAndPushforward<int, int> _t0 = func4_pushforward(x, y, _d_x, _d_y);
// CHECK-NEXT:     return {_t0.value, _t0.pushforward};
// CHECK-NEXT: }

// CHECK: int test_1_darg1(int x, int y) {
// CHECK-NEXT:     int _d_x = 0;
// CHECK-NEXT:     int _d_y = 1;
// CHECK-NEXT:     clad::ValueAndPushforward<int, int> _t0 = func3_pushforward(x, y, _d_x, _d_y);
// CHECK-NEXT:     return _t0.pushforward;
// CHECK-NEXT: }

namespace A {
namespace B {
namespace C {
  double someFn_1(double& i, double j, double k) {
    i = j;
    return 1;
  }

  // CHECK: clad::ValueAndPushforward<double, double> someFn_1_pushforward(double &i, double j, double k, double &_d_i, double _d_j, double _d_k) {
  // CHECK-NEXT:     _d_i = _d_j;
  // CHECK-NEXT:     i = j;
  // CHECK-NEXT:     return {1, 0};
  // CHECK-NEXT: }

  double someFn_1(double& i, double j) {
    someFn_1(i, j, j);
    return 2;
  }

  // CHECK: clad::ValueAndPushforward<double, double> someFn_1_pushforward(double &i, double j, double &_d_i, double _d_j) {
  // CHECK-NEXT:     clad::ValueAndPushforward<double, double> _t0 = someFn_1_pushforward(i, j, j, _d_i, _d_j, _d_j);
  // CHECK-NEXT:     return {2, 0};
  // CHECK-NEXT: }

  double someFn(double& i, double& j) {
    someFn_1(i, j);
    return 3;
  }

  // CHECK: clad::ValueAndPushforward<double, double> someFn_pushforward(double &i, double &j, double &_d_i, double &_d_j) {
  // CHECK-NEXT:     clad::ValueAndPushforward<double, double> _t0 = someFn_1_pushforward(i, j, _d_i, _d_j);
  // CHECK-NEXT:     return {3, 0};
  // CHECK-NEXT: }
} // namespace C
} // namespace B
} // namespace A

double fn1(double i, double j) {
  A::B::C::someFn(i, j);
  return i + j;
}

// CHECK: double fn1_darg1(double i, double j) {
// CHECK-NEXT:     double _d_i = 0;
// CHECK-NEXT:     double _d_j = 1;
// CHECK-NEXT:     clad::ValueAndPushforward<double, double> _t0 = someFn_pushforward(i, j, _d_i, _d_j);
// CHECK-NEXT:     return _d_i + _d_j;
// CHECK-NEXT: }

int main () {
  INIT_DIFFERENTIATE(test_1, 1);
  INIT_DIFFERENTIATE(fn1, "j");

  TEST_DIFFERENTIATE(test_1, 3, 5);   // CHECK-EXEC: {1}
  TEST_DIFFERENTIATE(fn1, 3, 5);      // CHECK-EXEC: {2.00}
  return 0;
}