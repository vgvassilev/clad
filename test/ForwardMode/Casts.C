// RUN: %cladclang %s -I%S/../../include -oCasts.out 
// RUN: ./Casts.out | FileCheck -check-prefix=CHECK-EXEC %s
// CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"

#include "../TestUtils.h"

long double fn1(double i, double j) {
  long double res =
      static_cast<long double>(7 * i) + static_cast<long double>(i * j);
  return res;
}

int main() {
  INIT_DIFFERENTIATE(fn1, "i");

  TEST_DIFFERENTIATE(fn1, 3, 5);  // CHECK-EXEC: {12.00}
}