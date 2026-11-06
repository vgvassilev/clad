// RUN: %cladclang %s -I%S/../../include -Xclang -verify -o %t 2>&1 | %filecheck %s
// RUN: %t | %filecheck_exec %s
// RUN: %cladclang %s -I%S/../../include -Xclang -plugin-arg-clad -Xclang -fenable-analysis=activity -o %t.va
// RUN: %t.va | %filecheck_exec %s
// expected-no-diagnostics

#include "clad/Differentiator/Differentiator.h"

extern "C" int printf(const char*, ...);

// Reverse mode uses a pushforward for a scalar function of one argument.
double square(double weight) { return weight * weight; }

double scalar_loss(double data, double weight) {
  return data * square(weight);
}

// A reference return needs a reverse-forward helper to preserve its adjoint.
double& select(double* values, int index = 0) { return values[index]; }

double reference_loss(double data, double weight) {
  return data * select(&weight);
}

int main() {
  // Both variants must coexist, whichever option is requested first.
  auto scalar = clad::gradient(scalar_loss, "weight");
  auto scalar_assumed = clad::gradient<clad::opts::assume_unselected_inactive>(
      scalar_loss, "weight");
  double dw = 0;
  scalar.execute(3, 2, &dw);
  printf("scalar: %.0f\n", dw);
  // CHECK-EXEC: scalar: 12
  dw = 0;
  scalar_assumed.execute(3, 2, &dw);
  printf("scalar assumed: %.0f\n", dw);
  // CHECK-EXEC: scalar assumed: 12

  auto reference_assumed =
      clad::gradient<clad::opts::assume_unselected_inactive>(reference_loss,
                                                           "weight");
  auto reference = clad::gradient(reference_loss, "weight");
  dw = 0;
  reference_assumed.execute(3, 2, &dw);
  printf("reference assumed: %.0f\n", dw);
  // CHECK-EXEC: reference assumed: 3
  dw = 0;
  reference.execute(3, 2, &dw);
  printf("reference: %.0f\n", dw);
  // CHECK-EXEC: reference: 3
}

// CHECK-DAG: clad::ValueAndPushforward<double, double> square_pushforward(double weight, double _d_weight)
// CHECK-DAG: clad::ValueAndPushforward<double, double> square_pushforward_assume_inactive(double weight, double _d_weight)
// CHECK-DAG: clad::ValueAndAdjoint<double &, double &> select_reverse_forw_assume_inactive(
// CHECK-DAG: clad::ValueAndAdjoint<double &, double &> select_reverse_forw(
