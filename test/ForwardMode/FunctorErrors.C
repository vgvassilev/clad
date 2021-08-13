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

class ExperimentPrivateCallOperator {
  double operator()(double i, double j) { // expected-note {{implicitly declared private here}}
    return i*i;
  }
};

struct ExperimentProtectedCallOperator {
  protected:
    double operator()(double i, double j) { // expected-note {{declared protected here}}
      return i*i;
    }
};

struct Widget {
  double i, j;
  const char* char_arr[10];
  char** p2p_char;
  Widget(double p_i, double p_j) : i(p_i), j(p_j) {}
  double operator()() {
    char **p;
    p = p2p_char;
    char_arr[0] = char_arr[1];  // expected-warning {{derivative of an assignment attempts to assign to unassignable expr, assignment ignored}}
    return i*i + j;
  }
};

int main() {
  ExperimentNoCallOperator E_NoCallOperator;
  auto d_NoCallOperator = clad::differentiate(E_NoCallOperator, "i"); // expected-error {{'ExperimentNoCallOperator' has no defined operator()}}
  d_NoCallOperator.execute(1, 3);
  ExperimentMultipleCallOperators E_MultipleCallOperators;
  auto d_MultipleCallOperators = clad::differentiate(E_MultipleCallOperators, "i"); // expected-error {{'ExperimentMultipleCallOperators' has multiple definitions of operator(). Multiple definitions of call operators are not supported.}}
  ExperimentPrivateCallOperator E_PrivateCallOperator;
  auto d_PrivateCallOperator = clad::differentiate(E_PrivateCallOperator, "i"); // expected-error {{'ExperimentPrivateCallOperator' contains private call operator. Differentiation of private/protected call operator is not supported.}}
  ExperimentProtectedCallOperator E_ProtectedCallOperator;
  auto d_ProtectedCallOperator = clad::differentiate(E_ProtectedCallOperator, "i"); // expected-error {{'ExperimentProtectedCallOperator' contains protected call operator. Differentiation of private/protected call operator is not supported.}}
  Widget W(3, 5);
  auto d_W = clad::differentiate(W, 0);
  auto d_W_Again = clad::differentiate(W, 10);  // expected-error {{Invalid member variable index '10' of '4' member variable(s)}}
}