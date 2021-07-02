// RUN: %cladclang %s -I%S/../../include -fsyntax-only -Xclang -verify 2>&1
// CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"

struct ExperimentNoCallOperator {
};

struct ExperimentMultipleCallOperators {
  double operator()(double i, double j) {
    return i*j;
  }
  double operator()(double i) {
    return i*i;
  }
  double operator()(double i) const {
    return i*i*i;
  }
};

int main() {
  ExperimentNoCallOperator E_NoCallOperator;
  auto d_NoCallOperator = clad::differentiate(E_NoCallOperator, "i"); // expected-error {{No call operator is defined for the class ExperimentNoCallOperator}}
  d_NoCallOperator.execute(1, 3);
  ExperimentMultipleCallOperators E_MultipleCallOperators;
  auto d_MultipleCallOperators = clad::differentiate(E_MultipleCallOperators, "i"); // expected-error {{Class ExperimentMultipleCallOperators defines multiple overloads of call operator. Multiple overloads of call operators are not supported.}}
  d_MultipleCallOperators.execute(3, 5);
}