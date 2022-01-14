// RUN: %cladclang %s -I%S/../../include -oSimple.out 2>&1 | FileCheck %s
// RUN: ./Simple.out | FileCheck -check-prefix=CHECK-EXEC %s
// CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"

template <typename YType, typename XType> struct DerivedTypeInfo {
  using type = void;
};

class ComplexNumber {
  double real, im;
};

class __clad_double_wrt_ComplexNumber;

int main() {
  printf("%d\n", sizeof(__clad_double_wrt_ComplexNumber) >= sizeof(double) * 2); // CHECK-EXEC: 1

  printf("%d\n",
         sizeof(typename DerivedTypeInfo<double, ComplexNumber>::type) >=
             sizeof(double) * 2); ; // CHECK-EXEC: 1
}