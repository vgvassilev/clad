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

double switch_condition(double x) {
  auto inner = [x] {
    switch (x > 0 ? 0 : 1) {
    case 0: return x * x;
    default: return 3 * x * x;
    }
  };
  double value = inner();
  return value * value;
}

double switch_initializer(double x) {
  auto inner = [x] {
    switch (double value = x * x; x > 0 ? 0 : 1) {
    case 0: return value;
    default: return 3 * value;
    }
  };
  double value = inner();
  return value * value;
}

double nested_switch(double x) {
  auto inner = [x] {
    double result = 0;
    switch (int choice = x > 0 ? 0 : 1) {
    case 0:
      switch (int choice = x > 3 ? 0 : 1) {
      case 0: result = x * x; break;
      default: result = 2 * x * x; break;
      }
      break;
    default: result = 3 * x * x; break;
    }
    return result;
  };
  double value = inner();
  return value * value;
}

double switch_labels(double x) {
  auto inner = [x] {
    double result = 0;
    switch (x > 0 ? 1 : 2) {
    default: result = 3 * x * x; break;
    case 0:
    case 1: result = x * x; break;
    }
    return result;
  };
  double value = inner();
  return value * value;
}

double mutating_switch(double x) {
  auto inner = [&x](int choice) {
    switch (choice) {
    case 0: {
      x *= 2;
      break;
    }
    default: {
      x *= 3;
      break;
    }
    }
    return x * x;
  };
  double first = inner(0);
  return first + inner(1);
}

int main() {
  INIT_GRADIENT(switch_condition);
  INIT_GRADIENT(switch_initializer);
  INIT_GRADIENT(nested_switch);
  INIT_GRADIENT(switch_labels);
  INIT_GRADIENT(mutating_switch);

  double dx = 0;
  TEST_GRADIENT(switch_condition, 1, 2, &dx); // CHECK-EXEC: {32.00}
  TEST_GRADIENT(switch_condition, 1, -2, &dx); // CHECK-EXEC-NEXT: {-288.00}
  TEST_GRADIENT(switch_initializer, 1, 2, &dx); // CHECK-EXEC-NEXT: {32.00}
  TEST_GRADIENT(switch_initializer, 1, -2, &dx); // CHECK-EXEC-NEXT: {-288.00}
  TEST_GRADIENT(nested_switch, 1, 2, &dx); // CHECK-EXEC-NEXT: {128.00}
  TEST_GRADIENT(nested_switch, 1, 4, &dx); // CHECK-EXEC-NEXT: {256.00}
  TEST_GRADIENT(nested_switch, 1, -2, &dx); // CHECK-EXEC-NEXT: {-288.00}
  TEST_GRADIENT(switch_labels, 1, 2, &dx); // CHECK-EXEC-NEXT: {32.00}
  TEST_GRADIENT(switch_labels, 1, -2, &dx); // CHECK-EXEC-NEXT: {-288.00}
  TEST_GRADIENT(mutating_switch, 1, 2, &dx); // CHECK-EXEC-NEXT: {160.00}
}
