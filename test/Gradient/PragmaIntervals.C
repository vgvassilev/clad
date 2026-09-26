// RUN: %cladclang %s -I%S/../../include -o %t
// RUN: %t | %filecheck_exec %s

#include "clad/Differentiator/Differentiator.h"

extern "C" int printf(const char*, ...);

double square(double x) { return x * x; }

int main() {
#pragma clad ON
  auto before = clad::gradient(square);
#pragma clad OFF
  auto disabled = clad::gradient(square);
#pragma clad ON
  auto after = clad::gradient(square);

  printf("%d %d %d\n", before.getFunctionPtr() != nullptr,
         disabled.getFunctionPtr() != nullptr,
         after.getFunctionPtr() != nullptr);
  // CHECK-EXEC: 1 0 1

  double before_dx = 0, after_dx = 0;
  before.execute(3, &before_dx);
  after.execute(3, &after_dx);
  printf("%.1f %.1f\n", before_dx, after_dx);
  // CHECK-EXEC-NEXT: 6.0 6.0
}
