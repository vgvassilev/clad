// RUN: %cladclang %s -I%S/../../include -oScalarWrtUserDefined.out 2>&1 -lstdc++ -lm | FileCheck %s
// RUN: ./ScalarWrtUserDefined.out | FileCheck -check-prefix=CHECK-EXEC %s
// CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"

class ComplexNumber {
public:
  double real = 0, im = 0;
  ComplexNumber(double p_real = 0, double p_im = 0) : real(p_real), im(p_im) {}
};

class __clad_double_wrt_ComplexNumber;
class __clad_ComplexNumber_wrt_ComplexNumber;

double fn(ComplexNumber c, double i) {
  ComplexNumber d;
  double res;
  // d.real = c.real;
  // res += c.real;
  // res -= i;
  // res *= i;
  // res /= c.im;
  res = c.real + c.im;
  return res;
}

int main() {
  ComplexNumber c(3, 5);
  auto d_fn = clad::differentiate(fn, "c");

  auto res = static_cast<__clad_double_wrt_ComplexNumber*>(d_fn.execute(c, 7));
  printf("%.2f %.2f\n", res->real, res->im);  // CHECK-EXEC: 1.00 1.00
  return 0;
}