// RUN: %cladclang %s -I%S/../../include -oBasicArithmeticAddSub.out 2>&1 | %filecheck %s
// RUN: ./BasicArithmeticAddSub.out | %filecheck_exec %s

//CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"

#include <complex>

extern "C" int printf(const char* fmt, ...);

int a_1(int x) {
  int y = 4;
  return y + y; // == 0
}
// CHECK: int a_1_darg0(int x) {
// CHECK-NEXT: int _d_x = 1;
// CHECK-NEXT: int _d_y = 0;
// CHECK-NEXT: int y = 4;
// CHECK-NEXT: return _d_y + _d_y;
// CHECK-NEXT: }

int a_2(int x) {
  return 1 + 1; // == 0
}
// CHECK: int a_2_darg0(int x) {
// CHECK-NEXT: int _d_x = 1;
// CHECK-NEXT: return 0 + 0;
// CHECK-NEXT: }

int a_3(int x) {
  return x + x; // == 2
}
// CHECK: int a_3_darg0(int x) {
// CHECK-NEXT: int _d_x = 1;
// CHECK-NEXT: return _d_x + _d_x;
// CHECK-NEXT: }

int a_4(int x) {
  int y = 4;
  return x + y + x + 3 + x; // == 3x
}
// CHECK: int a_4_darg0(int x) {
// CHECK-NEXT: int _d_x = 1;
// CHECK-NEXT: int _d_y = 0;
// CHECK-NEXT: int y = 4;
// CHECK-NEXT: return _d_x + _d_y + _d_x + 0 + _d_x;
// CHECK-NEXT: }

int s_1(int x) {
  int y = 4;
  return y - y; // == 0
}
// CHECK: int s_1_darg0(int x) {
// CHECK-NEXT: int _d_x = 1;
// CHECK-NEXT: int _d_y = 0;
// CHECK-NEXT: int y = 4;
// CHECK-NEXT: return _d_y - _d_y;
// CHECK-NEXT: }

int s_2(int x) {
  return 1 - 1; // == 0
}
// CHECK: int s_2_darg0(int x) {
// CHECK-NEXT: int _d_x = 1;
// CHECK-NEXT: return 0 - 0;
// CHECK-NEXT: }

int s_3(int x) {
  return x - x; // == 0
}
// CHECK: int s_3_darg0(int x) {
// CHECK-NEXT: int _d_x = 1;
// CHECK-NEXT: return _d_x - _d_x;
// CHECK-NEXT: }

int s_4(int x) {
  int y = 4;
  return x - y - x - 3 - x; // == -1
}
// CHECK: int s_4_darg0(int x) {
// CHECK-NEXT: int _d_x = 1;
// CHECK-NEXT: int _d_y = 0;
// CHECK-NEXT: int y = 4;
// CHECK-NEXT: return _d_x - _d_y - _d_x - 0 - _d_x;
// CHECK-NEXT: }

int as_1(int x) {
  int y = 4;
  return x + x - x + y - y + 3 - 3; // == 1
}
// CHECK: int as_1_darg0(int x) {
// CHECK-NEXT: int _d_x = 1;
// CHECK-NEXT: int _d_y = 0;
// CHECK-NEXT: int y = 4;
// CHECK-NEXT: return _d_x + _d_x - _d_x + _d_y - _d_y + 0 - 0;
// CHECK-NEXT: }

float IntegerLiteralToFloatLiteral(float x, float y) {
  return x * x - y;
}
// CHECK: float IntegerLiteralToFloatLiteral_darg0(float x, float y) {
// CHECK-NEXT: float _d_x = 1;
// CHECK-NEXT: float _d_y = 0;
// CHECK-NEXT: return _d_x * x + x * _d_x - _d_y;
// CHECK-NEXT: }

std::complex<double> add_real_complex(double x, std::complex<double> y) {
  return std::complex<double>(x + y.real(), y.imag());
} 

// CHECK: constexpr clad::ValueAndPushforward<double, double> real_pushforward(const std::complex<double> *_d_this) const __attribute__((abi_tag("cxx11")));
// CHECK-NEXT: constexpr clad::ValueAndPushforward<double, double> imag_pushforward(const std::complex<double> *_d_this) const __attribute__((abi_tag("cxx11")));
// CHECK-NEXT:std::complex<double> add_real_complex_darg0(double x, std::complex<double> y) {
// CHECK-NEXT:    double _d_x = 1;
// CHECK-NEXT:    std::complex<double> _d_y;
// CHECK-NEXT:    clad::ValueAndPushforward<double, double> _t0 = y.real_pushforward(&_d_y);
// CHECK-NEXT:    clad::ValueAndPushforward<double, double> _t1 = y.imag_pushforward(&_d_y);
// CHECK-NEXT:    return std::complex<double>(_d_x + _t0.pushforward, _t1.pushforward);
// CHECK-NEXT:}
// CHECK-NEXT:constexpr clad::ValueAndPushforward<double, double> real_pushforward(const std::complex<double> *_d_this) const __attribute__((abi_tag("cxx11"))) {
// CHECK-NEXT:    return {__real this->_M_value, __real _d_this->_M_value};
// CHECK-NEXT:}
// CHECK-NEXT:constexpr clad::ValueAndPushforward<double, double> imag_pushforward(const std::complex<double> *_d_this) const __attribute__((abi_tag("cxx11"))) {
// CHECK-NEXT:    return {__imag this->_M_value, __imag _d_this->_M_value};
// CHECK-NEXT:}

int a_1_darg0(int x);
int a_2_darg0(int x);
int a_3_darg0(int x);
int a_4_darg0(int x);
int s_1_darg0(int x);
int s_2_darg0(int x);
int s_3_darg0(int x);
int s_4_darg0(int x);
int as_1_darg0(int x);
float IntegerLiteralToFloatLiteral_darg0(float x, float y);
std::complex<double> add_real_complex_darg0(double x, std::complex<double> y);

int main () { // expected-no-diagnostics
  int x = 4;
  clad::differentiate(a_1, 0);
  printf("Result is = %d\n", a_1_darg0(1)); // CHECK-EXEC: Result is = 0

  clad::differentiate(a_2, 0);
  printf("Result is = %d\n", a_2_darg0(1)); // CHECK-EXEC: Result is = 0

  clad::differentiate(a_3, 0);
  printf("Result is = %d\n", a_3_darg0(1)); // CHECK-EXEC: Result is = 2

  clad::differentiate(a_4, 0);
  printf("Result is = %d\n", a_4_darg0(1)); // CHECK-EXEC: Result is = 3

  clad::differentiate(s_1, 0);
  printf("Result is = %d\n", s_1_darg0(1)); // CHECK-EXEC: Result is = 0

  clad::differentiate(s_2, 0);
  printf("Result is = %d\n", s_2_darg0(1)); // CHECK-EXEC: Result is = 0

  clad::differentiate(s_3, 0);
  printf("Result is = %d\n", s_3_darg0(1)); // CHECK-EXEC: Result is = 0

  clad::differentiate(s_4, 0);
  printf("Result is = %d\n", s_4_darg0(1)); // CHECK-EXEC: Result is = -1

  clad::differentiate(as_1, 0);
  printf("Result is = %d\n", as_1_darg0(1)); // CHECK-EXEC: Result is = 1

  clad::differentiate(IntegerLiteralToFloatLiteral, 0);
  printf("Result is = %f\n", IntegerLiteralToFloatLiteral_darg0(5., 0.)); // CHECK-EXEC: Result is = 10

  clad::differentiate(add_real_complex, "x");
  printf("Result is = %f\n", add_real_complex_darg0(5., std::complex<double>(1., 2.)).real()); // CHECK-EXEC: Result is = 1.000000

  return 0;
}
