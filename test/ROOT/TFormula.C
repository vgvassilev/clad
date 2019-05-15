// RUN: %cladclang %s -lm -I%S/../../include -oTFormula.out 2>&1 | FileCheck %s
// RUN: ./TFormula.out | FileCheck -check-prefix=CHECK-EXEC %s

//CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"
#include <cmath>

using Double_t = double;

namespace TMath {
  Double_t Abs(Double_t x) { return ::std::abs(x); }
  Double_t Exp(Double_t x) { return ::std::exp(x); }
  Double_t Sin(Double_t x) { return ::std::sin(x); }
}

namespace custom_derivatives {
  Double_t Abs_darg0(Double_t x) { return (x < 0) ? -1 : 1; }
  Double_t Exp_darg0(Double_t x) { return ::std::exp(x); }
  Double_t Sin_darg0(Double_t x) { return ::std::cos(x); }
}

Double_t TFormula_example(Double_t* x, Double_t* p) {
  return x[0]*(p[0] + p[1] + p[2]) + TMath::Exp(-p[0]) + TMath::Abs(p[1]);
}
// _grad = { x[0] + (-1) * Exp_darg0(-p[0]), x[0] + Abs_darg0(p[1]), x[0] }

void TFormula_example_grad(Double_t *x, Double_t *p, Double_t *_result);
//CHECK:   void TFormula_example_grad(Double_t *x, Double_t *p, Double_t *_result) {
//CHECK-NEXT:       double _t0;
//CHECK-NEXT:       Double_t _t1;
//CHECK-NEXT:       Double_t _t2;
//CHECK-NEXT:       Double_t _t3;
//CHECK-NEXT:       _t1 = x[0];
//CHECK-NEXT:       _t0 = (p[0] + p[1] + p[2]);
//CHECK-NEXT:       _t2 = -p[0];
//CHECK-NEXT:       _t3 = p[1];
//CHECK-NEXT:       double TFormula_example_return = _t1 * _t0 + TMath::Exp(_t2) + TMath::Abs(_t3);
//CHECK-NEXT:       goto _label0;
//CHECK-NEXT:     _label0:
//CHECK-NEXT:       {
//CHECK-NEXT:           double _r0 = 1 * _t0;
//CHECK-NEXT:           double _r1 = _t1 * 1;
//CHECK-NEXT:           _result[0] += _r1;
//CHECK-NEXT:           _result[1] += _r1;
//CHECK-NEXT:           _result[2] += _r1;
//CHECK-NEXT:           Double_t _r2 = 1 * custom_derivatives::Exp_darg0(_t2);
//CHECK-NEXT:           _result[0] += -_r2;
//CHECK-NEXT:           Double_t _r3 = 1 * custom_derivatives::Abs_darg0(_t3);
//CHECK-NEXT:           _result[1] += _r3;
//CHECK-NEXT:       }
//CHECK-NEXT:   }
      
int main() {
  auto gradient = clad::gradient(TFormula_example);
  Double_t x[] = { 3 };
  Double_t p[] = { -std::log(2), -1, 3 };
  Double_t result[3] = { 0 };
  gradient.execute(x, p, result);
  printf("Result is = {%.2f, %.2f, %.2f}\n", result[0], result[1], result[2]); // CHECK-EXEC: Result is = {1.00, 2.00, 3.00}
}
