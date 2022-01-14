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
  Double_t Abs_darg0_darg0(Double_t x) { return 0; }
  Double_t Exp_darg0_darg0(Double_t x) { return ::std::exp(x); }
  Double_t Sin_darg0_darg0(Double_t x) { return -1 * ::std::sin(x); }
}

Double_t TFormula_example(Double_t* x, Double_t* p) {
  return x[0]*(p[0] + p[1] + p[2]) + TMath::Exp(-p[0]) + TMath::Abs(p[1]);
}
// _grad = { x[0] + (-1) * Exp_darg0(-p[0]), x[0] + Abs_darg0(p[1]), x[0] }

void TFormula_example_grad_1(Double_t* x, Double_t* p, Double_t* _d_p);
//CHECK:   void TFormula_example_grad_1(Double_t *x, Double_t *p, clad::array_ref<Double_t> _d_p) {
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
//CHECK-NEXT:           _d_p[0] += _r1;
//CHECK-NEXT:           _d_p[1] += _r1;
//CHECK-NEXT:           _d_p[2] += _r1;
//CHECK-NEXT:           Double_t _r2 = 1 * custom_derivatives::Exp_darg0(_t2);
//CHECK-NEXT:           _d_p[0] += -_r2;
//CHECK-NEXT:           Double_t _r3 = 1 * custom_derivatives::Abs_darg0(_t3);
//CHECK-NEXT:           _d_p[1] += _r3;
//CHECK-NEXT:       }
//CHECK-NEXT:   }

//CHECK:   Double_t TFormula_example_darg1_0(Double_t *x, Double_t *p) {
//CHECK-NEXT:       double _t0 = (p[0] + p[1] + p[2]);
//CHECK-NEXT:       return 0 * _t0 + x[0] * (1 + 0 + 0) + custom_derivatives::Exp_darg0(-p[0]) * -1 + custom_derivatives::Abs_darg0(p[1]) * 0;
//CHECK-NEXT:   }

//CHECK:   void TFormula_example_darg1_0_grad_1(Double_t *x, Double_t *p, clad::array_ref<Double_t> _d_p) {
//CHECK-NEXT:       double _d__t0 = 0;
//CHECK-NEXT:       double _t0;
//CHECK-NEXT:       double _t1;
//CHECK-NEXT:       Double_t _t2;
//CHECK-NEXT:       Double_t _t3;
//CHECK-NEXT:       Double_t _t4;
//CHECK-NEXT:       double _t00 = (p[0] + p[1] + p[2]);
//CHECK-NEXT:       _t0 = _t00;
//CHECK-NEXT:       _t2 = x[0];
//CHECK-NEXT:       _t1 = (1 + 0 + 0);
//CHECK-NEXT:       _t3 = -p[0];
//CHECK-NEXT:       _t4 = p[1];
//CHECK-NEXT:       double TFormula_example_darg1_0_return = 0 * _t0 + _t2 * _t1 + custom_derivatives::Exp_darg0(_t3) * -1 + custom_derivatives::Abs_darg0(_t4) * 0;
//CHECK-NEXT:       goto _label0;
//CHECK-NEXT:     _label0:
//CHECK-NEXT:       {
//CHECK-NEXT:           double _r0 = 1 * _t0;
//CHECK-NEXT:           double _r1 = 0 * 1;
//CHECK-NEXT:           _d__t0 += _r1;
//CHECK-NEXT:           double _r2 = 1 * _t1;
//CHECK-NEXT:           double _r3 = _t2 * 1;
//CHECK-NEXT:           double _r4 = 1 * -1;
//CHECK-NEXT:           Double_t _r5 = _r4 * custom_derivatives::Exp_darg0_darg0(_t3);
//CHECK-NEXT:           _d_p[0] += -_r5;
//CHECK-NEXT:           double _r6 = 1 * 0;
//CHECK-NEXT:           Double_t _r7 = _r6 * custom_derivatives::Abs_darg0_darg0(_t4);
//CHECK-NEXT:           _d_p[1] += _r7;
//CHECK-NEXT:       }
//CHECK-NEXT:       {
//CHECK-NEXT:           _d_p[0] += _d__t0;
//CHECK-NEXT:           _d_p[1] += _d__t0;
//CHECK-NEXT:           _d_p[2] += _d__t0;
//CHECK-NEXT:       }
//CHECK-NEXT:   }

//CHECK:   Double_t TFormula_example_darg1_1(Double_t *x, Double_t *p) {
//CHECK-NEXT:       double _t0 = (p[0] + p[1] + p[2]);
//CHECK-NEXT:       return 0 * _t0 + x[0] * (0 + 1 + 0) + custom_derivatives::Exp_darg0(-p[0]) * -0 + custom_derivatives::Abs_darg0(p[1]) * 1;
//CHECK-NEXT:   }

//CHECK:   void TFormula_example_darg1_1_grad_1(Double_t *x, Double_t *p, clad::array_ref<Double_t> _d_p) {
//CHECK-NEXT:       double _d__t0 = 0;
//CHECK-NEXT:       double _t0;
//CHECK-NEXT:       double _t1;
//CHECK-NEXT:       Double_t _t2;
//CHECK-NEXT:       Double_t _t3;
//CHECK-NEXT:       Double_t _t4;
//CHECK-NEXT:       double _t00 = (p[0] + p[1] + p[2]);
//CHECK-NEXT:       _t0 = _t00;
//CHECK-NEXT:       _t2 = x[0];
//CHECK-NEXT:       _t1 = (0 + 1 + 0);
//CHECK-NEXT:       _t3 = -p[0];
//CHECK-NEXT:       _t4 = p[1];
//CHECK-NEXT:       double TFormula_example_darg1_1_return = 0 * _t0 + _t2 * _t1 + custom_derivatives::Exp_darg0(_t3) * -0 + custom_derivatives::Abs_darg0(_t4) * 1;
//CHECK-NEXT:       goto _label0;
//CHECK-NEXT:     _label0:
//CHECK-NEXT:       {
//CHECK-NEXT:           double _r0 = 1 * _t0;
//CHECK-NEXT:           double _r1 = 0 * 1;
//CHECK-NEXT:           _d__t0 += _r1;
//CHECK-NEXT:           double _r2 = 1 * _t1;
//CHECK-NEXT:           double _r3 = _t2 * 1;
//CHECK-NEXT:           double _r4 = 1 * -0;
//CHECK-NEXT:           Double_t _r5 = _r4 * custom_derivatives::Exp_darg0_darg0(_t3);
//CHECK-NEXT:           _d_p[0] += -_r5;
//CHECK-NEXT:           double _r6 = 1 * 1;
//CHECK-NEXT:           Double_t _r7 = _r6 * custom_derivatives::Abs_darg0_darg0(_t4);
//CHECK-NEXT:           _d_p[1] += _r7;
//CHECK-NEXT:       }
//CHECK-NEXT:       {
//CHECK-NEXT:           _d_p[0] += _d__t0;
//CHECK-NEXT:           _d_p[1] += _d__t0;
//CHECK-NEXT:           _d_p[2] += _d__t0;
//CHECK-NEXT:       }
//CHECK-NEXT:   }

//CHECK:   Double_t TFormula_example_darg1_2(Double_t *x, Double_t *p) {
//CHECK-NEXT:       double _t0 = (p[0] + p[1] + p[2]);
//CHECK-NEXT:       return 0 * _t0 + x[0] * (0 + 0 + 1) + custom_derivatives::Exp_darg0(-p[0]) * -0 + custom_derivatives::Abs_darg0(p[1]) * 0;
//CHECK-NEXT:   }

//CHECK:   void TFormula_example_darg1_2_grad_1(Double_t *x, Double_t *p, clad::array_ref<Double_t> _d_p) {
//CHECK-NEXT:       double _d__t0 = 0;
//CHECK-NEXT:       double _t0;
//CHECK-NEXT:       double _t1;
//CHECK-NEXT:       Double_t _t2;
//CHECK-NEXT:       Double_t _t3;
//CHECK-NEXT:       Double_t _t4;
//CHECK-NEXT:       double _t00 = (p[0] + p[1] + p[2]);
//CHECK-NEXT:       _t0 = _t00;
//CHECK-NEXT:       _t2 = x[0];
//CHECK-NEXT:       _t1 = (0 + 0 + 1);
//CHECK-NEXT:       _t3 = -p[0];
//CHECK-NEXT:       _t4 = p[1];
//CHECK-NEXT:       double TFormula_example_darg1_2_return = 0 * _t0 + _t2 * _t1 + custom_derivatives::Exp_darg0(_t3) * -0 + custom_derivatives::Abs_darg0(_t4) * 0;
//CHECK-NEXT:       goto _label0;
//CHECK-NEXT:     _label0:
//CHECK-NEXT:       {
//CHECK-NEXT:           double _r0 = 1 * _t0;
//CHECK-NEXT:           double _r1 = 0 * 1;
//CHECK-NEXT:           _d__t0 += _r1;
//CHECK-NEXT:           double _r2 = 1 * _t1;
//CHECK-NEXT:           double _r3 = _t2 * 1;
//CHECK-NEXT:           double _r4 = 1 * -0;
//CHECK-NEXT:           Double_t _r5 = _r4 * custom_derivatives::Exp_darg0_darg0(_t3);
//CHECK-NEXT:           _d_p[0] += -_r5;
//CHECK-NEXT:           double _r6 = 1 * 0;
//CHECK-NEXT:           Double_t _r7 = _r6 * custom_derivatives::Abs_darg0_darg0(_t4);
//CHECK-NEXT:           _d_p[1] += _r7;
//CHECK-NEXT:       }
//CHECK-NEXT:       {
//CHECK-NEXT:           _d_p[0] += _d__t0;
//CHECK-NEXT:           _d_p[1] += _d__t0;
//CHECK-NEXT:           _d_p[2] += _d__t0;
//CHECK-NEXT:       }
//CHECK-NEXT:   }

//CHECK:   void TFormula_example_hessian_1(Double_t *x, Double_t *p, clad::array_ref<Double_t> hessianMatrix) {
//CHECK-NEXT:       TFormula_example_darg1_0_grad_1(x, p, hessianMatrix.slice(0UL, 3UL));
//CHECK-NEXT:       TFormula_example_darg1_1_grad_1(x, p, hessianMatrix.slice(3UL, 3UL));
//CHECK-NEXT:       TFormula_example_darg1_2_grad_1(x, p, hessianMatrix.slice(6UL, 3UL));
//CHECK-NEXT:   }

// forward mode differentiation w.r.t. p[0]:
//CHECK:   Double_t TFormula_example_darg1_0(Double_t *x, Double_t *p) {
//CHECK-NEXT:       double _t0 = (p[0] + p[1] + p[2]);
//CHECK-NEXT:       return 0 * _t0 + x[0] * (1 + 0 + 0) + custom_derivatives::Exp_darg0(-p[0]) * -1 + custom_derivatives::Abs_darg0(p[1]) * 0;
//CHECK-NEXT:   }

// forward mode differentiation w.r.t. p[1]:
//CHECK:   Double_t TFormula_example_darg1_1(Double_t *x, Double_t *p) {
//CHECK-NEXT:       double _t0 = (p[0] + p[1] + p[2]);
//CHECK-NEXT:       return 0 * _t0 + x[0] * (0 + 1 + 0) + custom_derivatives::Exp_darg0(-p[0]) * -0 + custom_derivatives::Abs_darg0(p[1]) * 1;
//CHECK-NEXT:   }

// forward mode differentiation w.r.t. p[2]:
//CHECK:   Double_t TFormula_example_darg1_2(Double_t *x, Double_t *p) {
//CHECK-NEXT:       double _t0 = (p[0] + p[1] + p[2]);
//CHECK-NEXT:       return 0 * _t0 + x[0] * (0 + 0 + 1) + custom_derivatives::Exp_darg0(-p[0]) * -0 + custom_derivatives::Abs_darg0(p[1]) * 0;
//CHECK-NEXT:   }

int main() {
  Double_t x[] = { 3 };
  Double_t p[] = { -std::log(2), -1, 3 };
  Double_t result[3] = { 0 };
  clad::array_ref<Double_t> result_ref(result, 3);

  auto gradient = clad::gradient(TFormula_example, "p");
  gradient.execute(x, p, result_ref);
  printf("Result is = {%.2f, %.2f, %.2f}\n", result[0], result[1], result[2]); // CHECK-EXEC: Result is = {1.00, 2.00, 3.00}

  Double_t matrix[9] = { 0 };
  clad::array_ref<Double_t> matrix_ref(matrix, 9);

  auto hessian = clad::hessian(TFormula_example, "p[0:2]");
  hessian.execute(x, p, matrix_ref);

  printf("Result is = {%.2f, %.2f, %.2f, %.2f,"
         " %.2f, %.2f, %.2f, %.2f, %.2f}\n",
         matrix[0], matrix[1], matrix[2],
         matrix[3], matrix[4], matrix[5],
         matrix[6], matrix[7], matrix[8]); // CHECK-EXEC: Result is = {2.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00}

  auto differentiation0 = clad::differentiate(TFormula_example, "p[0]");
  printf("Result is = {%.2f}\n", differentiation0.execute(x, p)); // CHECK-EXEC: Result is = {1.00}

  auto differentiation1 = clad::differentiate(TFormula_example, "p[1]");
  printf("Result is = {%.2f}\n", differentiation1.execute(x, p)); // CHECK-EXEC: Result is = {2.00}

  auto differentiation2 = clad::differentiate(TFormula_example, "p[2]");
  printf("Result is = {%.2f}\n", differentiation2.execute(x, p)); // CHECK-EXEC: Result is = {3.00}
}
