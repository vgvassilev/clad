// RUN: %cladclang %s -I%S/../../include -lstdc++ -lm -oForwardSimple.out 2>&1 | FileCheck %s
// RUN: ./ForwardSimple.out | FileCheck -check-prefix=CHECK-EXEC %s
// CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"

class ComplexNumber {
public:
  ComplexNumber(double p_real = 0, double p_im = 0) : real(p_real), im(p_im) {}
  double real, im;
};

clad::BuildTangentType<double, ComplexNumber> buildDoubleWrtComplexNumber;
clad::BuildTangentType<ComplexNumber, ComplexNumber>
    buildComplexNumberWrtComplexNumber;

double fn(ComplexNumber c) {
  double res = c.real + c.im;
  res -= c.real + c.real;
  res = res * c.im;
  res = res / c.im;
  return res;
}

int main() {
  auto d_fn = clad::differentiate(fn, "c");
  ComplexNumber c(3, 5);
  typename clad::TangentTypeInfo<double, ComplexNumber>::type* d_c;
  d_c = static_cast<
      typename clad::TangentTypeInfo<double, ComplexNumber>::type*>(
      d_fn.execute(c));
  printf("%.2f\n", d_c->real);  // CHECK-EXEC: -1.00
  printf("%.2f\n", d_c->im);    // CHECK-EXEC: 1.00
}