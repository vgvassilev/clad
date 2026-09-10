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

double matrix_capture(double x) {
  const double values[2][3] = {{x, x * x, 2 * x}, {3 * x, 4 * x, 5 * x}};
  auto inner = [values] { return values[0][1] * values[1][2]; };
  return inner();
}

double tensor_capture(double x) {
  double values[2][1][3] = {{{x, 2 * x, 3 * x}}, {{4 * x, 5 * x, x * x}}};
  auto inner = [values] { return values[0][0][1] * values[1][0][2]; };
  values[0][0][1] = 0;
  return inner();
}

double mutable_matrix_capture(double x) {
  double values[2][2] = {{x, x * x}, {2 * x, 3 * x}};
  auto inner = [values]() mutable {
    values[0][0] *= 2;
    return values[0][0] * values[1][1];
  };
  double first = inner();
  return first + inner();
}

double loop_matrix_capture(double x) {
  double result = 0;
  for (int i = 0; i < 3; ++i) {
    double values[1][2] = {{x, x * x}};
    auto inner = [values]() mutable {
      values[0][0] *= 2;
      return values[0][0] * values[0][1];
    };
    result += inner() + inner();
    x += 1;
  }
  return result;
}

double nested_matrix_capture(double x) {
  auto outer = [&x] {
    double values[2][1] = {{x}, {x}};
    auto inner = [values]() mutable {
      values[0][0] *= 2;
      return values[0][0] * values[1][0];
    };
    double first = inner();
    x *= 2;
    return first + inner();
  };
  double first = outer();
  return first + outer();
}

int main() {
  INIT_GRADIENT(matrix_capture);
  INIT_GRADIENT(tensor_capture);
  INIT_GRADIENT(mutable_matrix_capture);
  INIT_GRADIENT(loop_matrix_capture);
  INIT_GRADIENT(nested_matrix_capture);

  double dx = 0;
  TEST_GRADIENT(matrix_capture, 1, 2, &dx); // CHECK-EXEC: {60.00}
  TEST_GRADIENT(matrix_capture, 1, -2, &dx); // CHECK-EXEC-NEXT: {60.00}
  TEST_GRADIENT(tensor_capture, 1, 2, &dx); // CHECK-EXEC-NEXT: {24.00}
  TEST_GRADIENT(mutable_matrix_capture, 1, 2, &dx); // CHECK-EXEC-NEXT: {72.00}
  TEST_GRADIENT(mutable_matrix_capture, 1, -2, &dx); // CHECK-EXEC-NEXT: {-72.00}
  TEST_GRADIENT(loop_matrix_capture, 1, 2, &dx); // CHECK-EXEC-NEXT: {522.00}
  TEST_GRADIENT(nested_matrix_capture, 1, 2, &dx); // CHECK-EXEC-NEXT: {120.00}
}
