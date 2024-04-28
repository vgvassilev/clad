// RUN: %cladnumdiffclang %s -I%S/../../include -oPureCentralDiffCalls.out
// -Xclang -verify 2>&1 RUN: ./PureCentralDiffCalls.out | %filecheck_exec %s

// CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"

#include <functional>

extern "C" int printf(const char* fmt, ...);

// Scalar args
double func(double x, double y, double z) { return x * y + x * (y + z); }

// All pointer args
double func1(double* x, double* y) {
  return x[0] * y[0] + x[1] * y[1] + x[2] * y[2];
}

// Mix args...
double func2(double* x, double y, int z) {
  double sum = 0;
  for (int i = 0; i < z; i++) {
    x[i] += y;
    sum += x[i];
  }
  return sum;
}

struct myStruct {
  double data;
  bool effect;
  myStruct(double x, bool eff) {
    data = x;
    effect = eff;
  }
};

double operator+(myStruct a, double b) { return a.data + b; }

myStruct operator+(myStruct a, myStruct b) {
  myStruct out(0, false);
  out.data = a.data + b.data;
  out.effect = a.effect || b.effect;
  return out;
}

myStruct
updateIndexParamValue(myStruct arg, std::size_t idx, std::size_t currIdx,
                      int multiplier, numerical_diff::precision& h_val,
                      std::size_t n = 0, std::size_t i = 0) {
  if (idx == currIdx) {
    h_val = (h_val == 0) ? numerical_diff::get_h(arg.data) : h_val;
    if (arg.effect)
      return myStruct(arg.data + h_val * multiplier, arg.effect);
  }
  return arg;
}

double func3(myStruct a, myStruct b) { return (a + b + a).data; }

int main() { // expected-no-diagnostics
  double x = 10, y = 2, z = 10, dy = 0, dz = 0, z1 = 3;
  double x1[3] = {1, 1, 1}, y1[3] = {2, 3, 4}, dx[3] = {0, 0, 0};
  myStruct a(10, true), b(11, false);

  // Forward mode, derivative wrt one arg
  // df/dx
  double func_res = numerical_diff::forward_central_difference(func, y, 1, false, x, y,
                                                               z);
  printf("Result is = %f\n", func_res); // CHECK-EXEC: Result is = 20.000000
  // df/dx[0]
  double func1_res = numerical_diff::forward_central_difference(func1, x1, 0, 3,
                                                                0, false, x1, y1);
  printf("Result is = %f\n", func1_res); // CHECK-EXEC: Result is = 2.000000

  // Gradients, derivative wrt all args
  clad::tape<clad::array_ref<double>> grad = {};
  grad.emplace_back(dx, 3);
  grad.emplace_back(&dy);
  grad.emplace_back(&dz);
  numerical_diff::central_difference(func2, grad, false, x1, y, z1);
  printf("Result is = %f\n", dx[0]); // CHECK-EXEC: Result is = 1.000000
  printf("Result is = %f\n", dx[1]); // CHECK-EXEC: Result is = 1.000000
  printf("Result is = %f\n", dx[2]); // CHECK-EXEC: Result is = 1.000000
  printf("Result is = %f\n", dy);    // CHECK-EXEC: Result is = 3.000000

  // Functors...
  double functor_res = numerical_diff::
      forward_central_difference(std::multiplies<double>(), y, 1, false, x, y);
  printf("Result is = %f\n", functor_res); // CHECK-EXEC: Result is = 10.000000

  // Overloaded operators...
  double operator_res = numerical_diff::forward_central_difference<
      double (*)(myStruct, double)>(operator+, a, 0, false, a, y);
  printf("Result is = %f\n", operator_res); // CHECK-EXEC: Result is = 1.000000

  // functions with user defined data types...
  double userDefined_res = numerical_diff::forward_central_difference(func3, b,
                                                                      1, false, a, b);
  printf("Result is = %f\n",
         userDefined_res); // CHECK-EXEC: Result is = 0.000000
}
