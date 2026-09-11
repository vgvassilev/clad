// RUN: %cladclang %s -I%S/../../include -Xclang -verify -o %t
// RUN: %t | %filecheck_exec %s
// RUN: %cladclang %s -I%S/../../include -Xclang -verify -Xclang -plugin-arg-clad -Xclang -disable-tbr -Xclang -plugin-arg-clad -Xclang -disable-va -o %t.no-analysis
// RUN: %t.no-analysis | %filecheck_exec %s
// RUN: %cladclang %s -I%S/../../include -Xclang -verify -Xclang -plugin-arg-clad -Xclang -enable-va -o %t.va
// RUN: %t.va | %filecheck_exec %s
// UNSUPPORTED: clang-11, clang-12, clang-13, clang-14, clang-15, clang-16
// expected-no-diagnostics

#include "clad/Differentiator/Differentiator.h"
#include "../TestUtils.h"

double parenthesized_capture(double x) {
  auto inner = ([x] { return x * x; });
  x *= 3;
  return inner();
}

double braced_capture(double x) {
  auto inner{[x] { return x * x; }};
  x *= 3;
  return inner();
}

double parenthesized_reference(double x) {
  auto inner = (([&x] { return x * x; }));
  double first = inner();
  x *= 3;
  return first + inner();
}

double braced_reference(double x) {
  auto inner{([&value = x] { return value * value; })};
  double first = inner();
  x *= 3;
  return first + inner();
}

double nested_mutating_reference(double x) {
  auto outer = ([&x] {
    auto inner{([&x] { x *= 2; return x * x; })};
    double first = inner();
    return first + inner();
  });
  double first = outer();
  return first + outer();
}

double nested_mutable_capture(double x) {
  auto outer{[&x] {
    auto inner = (([value = x]() mutable {
      value *= 2;
      return value * value;
    }));
    double first = inner();
    x *= 2;
    return first + inner();
  }};
  double first = outer();
  return first + outer();
}

int main() {
  INIT_GRADIENT(parenthesized_capture);
  INIT_GRADIENT(braced_capture);
  INIT_GRADIENT(parenthesized_reference);
  INIT_GRADIENT(braced_reference);
  INIT_GRADIENT(nested_mutating_reference);
  INIT_GRADIENT(nested_mutable_capture);

  double dx = 0;
  TEST_GRADIENT(parenthesized_capture, 1, 2, &dx); // CHECK-EXEC: {4.00}
  TEST_GRADIENT(parenthesized_capture, 1, -2, &dx); // CHECK-EXEC-NEXT: {-4.00}
  TEST_GRADIENT(braced_capture, 1, 2, &dx); // CHECK-EXEC-NEXT: {4.00}
  TEST_GRADIENT(braced_capture, 1, -2, &dx); // CHECK-EXEC-NEXT: {-4.00}
  TEST_GRADIENT(parenthesized_reference, 1, 2, &dx); // CHECK-EXEC-NEXT: {40.00}
  TEST_GRADIENT(braced_reference, 1, -2, &dx); // CHECK-EXEC-NEXT: {-40.00}
  TEST_GRADIENT(nested_mutating_reference, 1, 2, &dx); // CHECK-EXEC-NEXT: {1360.00}
  TEST_GRADIENT(nested_mutable_capture, 1, 2, &dx); // CHECK-EXEC-NEXT: {400.00}
}
