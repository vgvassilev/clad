// RUN: %cladclang %s -I%S/../../include -oSimpleFunctor.out 2>&1 | %filecheck %s
// RUN: ./SimpleFunctor.out | %filecheck_exec %s

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
  Matcher(int m=0) : target(m) {}
  int operator()(int x) { return x == target;}
};

class SimpleExpression {
  float x, y;
public:
  SimpleExpression() : x(), y() {}
  SimpleExpression(float x, float y) : x(x), y(y) {}
  float operator()(float x, float y) { return x * x + y * y;}
  float operator_call_darg0(float x, float y);
};

// CHECK: float operator_call_darg0(float x, float y) {
// CHECK-NEXT: float _d_x = 1;
// CHECK-NEXT: float _d_y = 0;
// CHECK-NEXT: SimpleExpression _d_this_obj;
// CHECK-NEXT: SimpleExpression *_d_this = &_d_this_obj;
// CHECK-NEXT: float _d_x0 = 0;
// CHECK-NEXT: float _d_y0 = 0;
// CHECK-NEXT: return _d_x * x + x * _d_x + _d_y * y + y * _d_y;
// CHECK-NEXT: }

// CHECK: float operator_call_darg1(float x, float y) {
// CHECK-NEXT: float _d_x = 0;
// CHECK-NEXT: float _d_y = 1;
// CHECK-NEXT: SimpleExpression _d_this_obj;
// CHECK-NEXT: SimpleExpression *_d_this = &_d_this_obj;
// CHECK-NEXT: float _d_x0 = 0;
// CHECK-NEXT: float _d_y0 = 0;
// CHECK-NEXT: return _d_x * x + x * _d_x + _d_y * y + y * _d_y;
// CHECK-NEXT: }

float f(float x) {
  return x;
}

namespace clad {
namespace custom_derivatives {
  template <typename F>
  void use_functor_pushforward(double x, F& f, double d_x, F& d_f) {
      f.operator_call_pushforward(x, &d_f, d_x);
  }
}
}
template <typename F>
void use_functor(double x, F& f) {
    f(x);
}

struct Foo {
    double &y;
    Foo(double &y): y(y) {} 

    double operator()(double x) {
        y = 2*x;

        return x;
    }
};

double fn0(double x) {
    Foo func = Foo({x});
    use_functor(x, func);
    return x;
}

// CHECK: clad::ValueAndPushforward<double, double> operator_call_pushforward(double x, Foo *_d_this, double _d_x);
// CHECK: double fn0_darg0(double x) {
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:    Foo _d_func = Foo({_d_x});
// CHECK-NEXT:    Foo func = Foo({x});
// CHECK-NEXT:    clad::custom_derivatives::use_functor_pushforward(x, func, _d_x, _d_func);
// CHECK-NEXT:    return _d_x;
// CHECK-NEXT:}
// CHECK: clad::ValueAndPushforward<double, double> operator_call_pushforward(double x, Foo *_d_this, double _d_x) {
// CHECK-NEXT:    _d_this->y = 0 * x + 2 * _d_x;
// CHECK-NEXT:    this->y = 2 * x;
// CHECK-NEXT:    return {x, _d_x};
// CHECK-NEXT:}

int main() {
  AFunctor doubler;
  int x = doubler(5);
  AFunctorWithState summer;
  int sum = summer(2);

  Matcher Is5(5);

  SimpleExpression expr(3.5, 4.5);
  auto f1_darg0 = clad::differentiate(&SimpleExpression::operator(), 0);
  printf("Result is = %f\n", expr.operator_call_darg0(3.5, 4.5)); // CHECK-EXEC: Result is = 7

  auto f1_darg1 = clad::differentiate(&SimpleExpression::operator(), 1);
  printf("Result is = %f\n", f1_darg1.execute(expr, 3.5, 4.5)); // CHECK-EXEC: Result is = 9

  auto dfn0 = clad::differentiate(fn0, "x");
  printf("RES: %f\n", dfn0.execute(3.0)); // CHECK-EXEC: RES: 2

  return 0;
}
