// RUN: %cladclang %s -I%S/../../include -oVariables.out 2>&1 | %filecheck %s
// RUN: ./Variables.out
//CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"
#include <cmath>
#include <string>
#include <iostream>

double f_x(double x) {
  double t0 = x;
  double t1 = t0;
  return t1;
}

// CHECK: double f_x_darg0(double x) {
// CHECK-NEXT:    double _d_x = 1;
// CHECK-NEXT:    double _d_t0 = _d_x;
// CHECK-NEXT:    double t0 = x;
// CHECK-NEXT:    double _d_t1 = _d_t0;
// CHECK-NEXT:    double t1 = t0;
// CHECK-NEXT:    return _d_t1;
// CHECK-NEXT: }

double f_ops1(double x) {
  double t0 = x;
  double t1 = 2 * x;
  double t2 = 0;
  double t3 = t1 * 2 + t2;
  return t3;
}

// CHECK: double f_ops1_darg0(double x) {
// CHECK-NEXT:    double _d_x = 1;
// CHECK-NEXT:    double _d_t0 = _d_x;
// CHECK-NEXT:    double t0 = x;
// CHECK-NEXT:    double _d_t1 = 0 * x + 2 * _d_x;
// CHECK-NEXT:    double t1 = 2 * x;
// CHECK-NEXT:    double _d_t2 = 0;
// CHECK-NEXT:    double t2 = 0;
// CHECK-NEXT:    double _d_t3 = _d_t1 * 2 + t1 * 0 + _d_t2;
// CHECK-NEXT:    double t3 = t1 * 2 + t2;
// CHECK-NEXT:    return _d_t3;
// CHECK-NEXT: }

double f_ops2(double x) {
  double t0 = x;
  double t1 = 2 * x;
  double t2 = 5;
  double t3 = t1 * x + t2;
  return t3;
}

// CHECK: double f_ops2_darg0(double x) {
// CHECK-NEXT:    double _d_x = 1;
// CHECK-NEXT:    double _d_t0 = _d_x;
// CHECK-NEXT:    double t0 = x;
// CHECK-NEXT:    double _d_t1 = 0 * x + 2 * _d_x;
// CHECK-NEXT:    double t1 = 2 * x;
// CHECK-NEXT:    double _d_t2 = 0;
// CHECK-NEXT:    double t2 = 5;
// CHECK-NEXT:    double _d_t3 = _d_t1 * x + t1 * _d_x + _d_t2;
// CHECK-NEXT:    double t3 = t1 * x + t2;
// CHECK-NEXT:    return _d_t3;
// CHECK-NEXT: }

double f_sin(double x, double y) {
  double xsin = std::sin(x);
  double ysin = std::sin(y);
  auto xt = xsin * xsin;
  auto yt = ysin * ysin;
  return xt + yt;
}

// CHECK: double f_sin_darg0(double x, double y) {
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     double _d_y = 0;
// CHECK-NEXT:     ValueAndPushforward<double, double> _t0 = clad::custom_derivatives{{(::std)?}}::sin_pushforward(x, _d_x);
// CHECK-NEXT:     double _d_xsin = _t0.pushforward;
// CHECK-NEXT:     double xsin = _t0.value;
// CHECK-NEXT:     ValueAndPushforward<double, double> _t1 = clad::custom_derivatives{{(::std)?}}::sin_pushforward(y, _d_y);
// CHECK-NEXT:     double _d_ysin = _t1.pushforward;
// CHECK-NEXT:     double ysin = _t1.value;
// CHECK-NEXT:     double _d_xt = _d_xsin * xsin + xsin * _d_xsin;
// CHECK-NEXT:     double xt = xsin * xsin;
// CHECK-NEXT:     double _d_yt = _d_ysin * ysin + ysin * _d_ysin;
// CHECK-NEXT:     double yt = ysin * ysin;
// CHECK-NEXT:     return _d_xt + _d_yt;
// CHECK-NEXT: }

double f_string(double x) {
  const char *s = "string literal";
  return x;
}

// CHECK: double f_string_darg0(double x) {
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     const char *_d_s = "";
// CHECK-NEXT:     const char *s = "string literal";
// CHECK-NEXT:     return _d_x;
// CHECK-NEXT: }

namespace clad {
namespace custom_derivatives {
clad::ValueAndPushforward<double, double> string_test_pushforward(double x, const char s[], double _d_x, const char *_d_s) {
    return {1, 0};
}
}}
double string_test(double x, const char s[]) {
    return 1;
}
double f_string_call(double x) {
  return string_test(x, "string literal");
}

// CHECK: double f_string_call_darg0(double x) {
// CHECK-NEXT:         double _d_x = 1;
// CHECK-NEXT:         clad::ValueAndPushforward<double, double> _t0 = clad::custom_derivatives::string_test_pushforward(x, "string literal", _d_x, "");
// CHECK-NEXT:         return _t0.pushforward;
// CHECK-NEXT:     }

double f_stdstring(double x) {
  std::string s = "string literal";
  return x;
}

// CHECK: double f_stdstring_darg0(double x) {
// CHECK-NEXT:         double _d_x = 1;
// CHECK-NEXT:         std::string _d_s = {{[{]"", std::allocator<char>\(\)[}]|""}};
// CHECK-NEXT:         std::string s = {{[{]"string literal", std::allocator<char>\(\)[}]|"string literal"}};
// CHECK-NEXT:         return _d_x;
// CHECK-NEXT:     }

namespace clad {
namespace custom_derivatives {
clad::ValueAndPushforward<double, double> stdstring_test_pushforward(double x, const ::std::string& s, double _d_x, const ::std::string& _d_s) {
    return {x, 1};
}
}}
double stdstring_test(double x, const std::string& s) {
    return x;
}
double f_stdstring_call(double x) {
  return stdstring_test(x, "string literal");
}

// CHECK: double f_stdstring_call_darg0(double x) {
// CHECK-NEXT:         double _d_x = 1;
// CHECK-NEXT:         clad::ValueAndPushforward<double, double> _t0 = clad::custom_derivatives::stdstring_test_pushforward(x, {{[{]"string literal", std::allocator<char>\(\)[}]|"string literal"}}, _d_x, {{[{]"", std::allocator<char>\(\)[}]|""}});
// CHECK-NEXT:         return _t0.pushforward;
// CHECK-NEXT:     }

int main() {
  clad::differentiate(f_x, 0);
  clad::differentiate(f_ops1, 0);
  clad::differentiate(f_ops2, 0);
  clad::differentiate(f_sin, 0);
  clad::differentiate(f_string, 0);
  clad::differentiate(f_string_call, 0);
  auto df_stdstring = clad::differentiate(f_stdstring, 0);
  std::cout << df_stdstring.execute(3.0) << '\n'; // CHECK-EXEC: 1
  auto df_stdstring_call = clad::differentiate(f_stdstring_call, 0);
  std::cout << df_stdstring_call.execute(3.0) << '\n'; // CHECK-EXEC: 1
}


