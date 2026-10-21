// RUN: %cladclang %s -std=c++17 -I%S/../../include -o %t
// RUN: %t | %filecheck_exec %s
// RUN: %cladclang %s -std=c++20 -I%S/../../include -o %t20
// RUN: %t20 | %filecheck_exec %s
// RUN: %cladclang %s -std=c++17 -I%S/../../include -Xclang -plugin-arg-clad -Xclang -enable-va -o %t.va
// RUN: %t.va | %filecheck_exec %s

#include "clad/Differentiator/Differentiator.h"
#include "clad/Differentiator/STLBuiltins.h"

#include <optional>

extern "C" int printf(const char*, ...);

struct NonDefault {
  explicit NonDefault(int) {}
};

template <typename T>
double configured(double x, std::optional<T> = std::nullopt) {
  return x * x;
}

double tagged(double x, std::nullopt_t) { return x * x; }

double nullopt_argument(double x) { return tagged(x, std::nullopt); }

double default_optional(double x) {
  return configured<int>(x) + configured<double>(x);
}

double explicit_optional(double x) {
  return configured<int>(x, std::nullopt) +
         configured<NonDefault>(x, std::nullopt);
}

int main() {
  double dx = 0.;
  clad::gradient(nullopt_argument).execute(3., &dx);
  printf("tag %.1f\n", dx); // CHECK-EXEC: tag 6.0

  dx = 0.;
  clad::gradient(default_optional).execute(3., &dx);
  printf("default %.1f\n", dx); // CHECK-EXEC-NEXT: default 12.0

  dx = 0.;
  clad::gradient(explicit_optional).execute(3., &dx);
  printf("explicit %.1f\n", dx); // CHECK-EXEC-NEXT: explicit 12.0
}
