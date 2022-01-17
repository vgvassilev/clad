// RUN: %cladclang %s -I%S/../../include -lstdc++ -lm -oSimple.out 2>&1
// RUN: ./Simple.out | FileCheck -check-prefix=CHECK-EXEC %s
// CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"

class ComplexNumber {
  double real, im;
};

clad::BuildTangentType<double, ComplexNumber> buildDoubleWrtComplexNumber;

int main() {
  printf("%d\n", sizeof(clad::DerivativeOf<double, ComplexNumber>) >= sizeof(double) * 2); // CHECK-EXEC: 1
  printf("%d\n", sizeof(typename clad::TangentTypeInfo<double, ComplexNumber>::type) >= sizeof(double) * 2); // CHECK-EXEC: 1
}