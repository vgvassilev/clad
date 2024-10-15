// RUN: %cladclang %s -I%S/../../include -oReferenceArguments.out 2>&1 | %filecheck %s
// RUN: ./ReferenceArguments.out | %filecheck_exec %s

#include "clad/Differentiator/Differentiator.h"

namespace clad {
namespace custom_derivatives {
  template <typename F>
  void use_functor_pushforward(double &x, F& f, double &d_x, F& d_f) {
      f.operator_call_pushforward(x, &d_f, d_x);
  }
}
}
template <typename F>
void use_functor(double &x, F& f) {
    f(x);
}

struct Foo {
    double operator()(double& x) {
        x = 2*x*x;
        return x;
    }
};

double fn0(double x, Foo& func) {
    use_functor(x, func);
    return x;
}

double fn1(double& i, double& j) {
  double res = i * i * j;
  return res;
}

// CHECK: double fn1_darg0(double &i, double &j) {
// CHECK-NEXT:     double _d_i = 1;
// CHECK-NEXT:     double _d_j = 0;
// CHECK-NEXT:     double _t0 = i * i;
// CHECK-NEXT:     double _d_res = (_d_i * i + i * _d_i) * j + _t0 * _d_j;
// CHECK-NEXT:     double res = _t0 * j;
// CHECK-NEXT:     return _d_res;
// CHECK-NEXT: }

#define INIT(fn, ...) auto d_##fn = clad::differentiate(fn, __VA_ARGS__);

#define TEST(fn, ...)                                                          \
  printf("{%.2f}\n", d_##fn.execute(__VA_ARGS__))

int main() {
    INIT(fn0, "x");
    INIT(fn1, "i");
    
    double i = 3, j = 5;
    TEST(fn1, i, j);    // CHECK-EXEC: {30.00}
    Foo fff;
    TEST(fn0, i, fff);    // CHECK-EXEC: {12.00}
}
