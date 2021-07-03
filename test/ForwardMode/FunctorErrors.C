// RUN: %cladclang %s -I%S/../../include -fsyntax-only -Xclang -verify 2>&1
// CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"

struct ExperimentNoCallOperator {
};

struct ExperimentMultipleCallOperators {
  double operator()(double i, double j) { // expected-note {{candidate function}}
    return i*j;
  }
  double operator()(double i) { // expected-note {{candidate function}}
    return i*i;
  }
  double operator()(double i) const { // expected-note {{candidate function}}
    return i*i*i;
  }
};

int main() {
  ExperimentNoCallOperator E_NoCallOperator;
  auto d_NoCallOperator = clad::differentiate(E_NoCallOperator, "i"); // expected-error {{'ExperimentNoCallOperator' has no defined operator()}}
  d_NoCallOperator.execute(1, 3);
  ExperimentMultipleCallOperators E_MultipleCallOperators;
  auto d_MultipleCallOperators = clad::differentiate(E_MultipleCallOperators, "i"); // expected-error {{'ExperimentMultipleCallOperators' has multiple definitions of operator().Multiple definitions of call operators are not supported.}}
}