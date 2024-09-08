// RUN: %cladclang %s -I%S/../../include -oHessian.out 2>&1 | %filecheck %s
// RUN: ./Hessian.out | %filecheck_exec %s
// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -enable-tbr %s -I%S/../../include -oHessian.out
// RUN: ./Hessian.out | %filecheck_exec %s

//CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"
#include <cmath>

using Double_t = double;

namespace TMath {
  Double_t Abs(Double_t x) { return ::std::abs(x); }
  Double_t Exp(Double_t x) { return ::std::exp(x); }
  Double_t Sin(Double_t x) { return ::std::sin(x); }
}

Double_t TFormula_example(Double_t* x, Double_t* p) {
  return x[0]*(p[0] + p[1] + p[2]) + TMath::Exp(-p[0]) + TMath::Abs(p[1]);
}

int main() {
  Double_t x[] = { 3 };
  Double_t p[] = { -std::log(2), -1, 3 };
  Double_t matrix[9] = { 0 };

  auto hessian = clad::hessian(TFormula_example, "p[0:2]");
  hessian.execute(x, p, matrix);

  printf("Result is = {%.2f, %.2f, %.2f, %.2f,"
         " %.2f, %.2f, %.2f, %.2f, %.2f}\n",
         matrix[0], matrix[1], matrix[2],
         matrix[3], matrix[4], matrix[5],
         matrix[6], matrix[7], matrix[8]); // CHECK-EXEC: Result is = {2.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00}
}