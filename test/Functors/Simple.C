// RUN: %cladclang %s -I%S/../../include -Xclang -verify 2>&1 | FileCheck %s

//CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"

class AFunctor {
public:
  int operator()(int x) { return x * 2;}
};

class AFunctorWithState {
private:
  int sum;
public:
  AFunctorWithState() : sum(0) { }
  int operator()(int x) { return sum += x;}
};

class Matcher {
  int target;
public:
  Matcher(int m) : target(m) {}
  int operator()(int x) { return x == target;}
};

class SimpleExpression {
  float x, y;
public:
  SimpleExpression(float x, float y) : x(x), y(y) {}
  float operator()(float x, float y) { return x * x + y * y;}
};

// CHECK: float operator_call_dx(float x, float y) {
// CHECK-NEXT: (1.F * x + x * 1.F) + ((0.F * y + y * 0.F));
// CHECK-NEXT: }

// CHECK: float operator_call_dy(float x, float y) {
// CHECK-NEXT: (0.F * x + x * 0.F) + ((1.F * y + y * 1.F));
// CHECK-NEXT: }

float f(float x) {
  return x;
}

int main() {// expected-no-diagnostics
  AFunctor doubler;
  int x = doubler(5);
  AFunctorWithState summer;
  int sum = summer(2);

  Matcher Is5(5);

  SimpleExpression expr(3.5, 4.5);
  auto f1_dx = clad::differentiate(&SimpleExpression::operator(), 0);
  // printf("Result is = %f\n", f1_dx.execute(3.5F, 4.5F)); // CHECK-EXEC: Result is = 0

  auto f1_dy = clad::differentiate(&SimpleExpression::operator(), 1);
  // printf("Result is = %f\n", f1_dy.execute(3.5, 4.5)); // CHECK-EXEC: Result is = 0

  return 0;
}
