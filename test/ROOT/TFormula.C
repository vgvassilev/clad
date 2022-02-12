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

namespace clad {
namespace custom_derivatives {
namespace TMath {
Double_t Abs_pushforward(Double_t x, Double_t d_x) {
  return std::abs_pushforward(x, d_x);
}
Double_t Exp_pushforward(Double_t x, Double_t d_x) {
  return std::exp_pushforward(x, d_x);
}
Double_t Sin_pushforward(Double_t x, Double_t d_x) {
  return std::cos_pushforward(x, d_x);
}
Double_t Abs_pushforward_pushforward(Double_t x, Double_t d_x, Double_t d_x0,
                                     Double_t d_d_x) {
  return 0;
}
Double_t Exp_pushforward_pushforward(Double_t x, Double_t d_x, Double_t d_x0,
                                     Double_t d_d_x) {
  return std::exp_pushforward_pushforward(x, d_x, d_x0, d_d_x);
}
Double_t Sin_pushforward_pushforward(Double_t x, Double_t d_x, Double_t d_x0,
                                     Double_t d_d_x) {
  return -1 * std::sin_pushforward_pushforward(x, d_x, d_x0, d_d_x);
}
} // namespace TMath
} // namespace custom_derivatives
} // namespace clad

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
//CHECK-NEXT:           Double_t _r2 = 1 * clad::custom_derivatives::TMath::Exp_pushforward(_t2, 1.);
//CHECK-NEXT:           _d_p[0] += -_r2;
//CHECK-NEXT:           Double_t _r3 = 1 * clad::custom_derivatives::TMath::Abs_pushforward(_t3, 1.);
//CHECK-NEXT:           _d_p[1] += _r3;
//CHECK-NEXT:       }
//CHECK-NEXT:   }

//CHECK:   Double_t TFormula_example_darg1_0(Double_t *x, Double_t *p) {
//CHECK-NEXT:       double _t0 = (p[0] + p[1] + p[2]);
//CHECK-NEXT:       return 0 * _t0 + x[0] * (1 + 0 + 0) + clad::custom_derivatives::TMath::Exp_pushforward(-p[0], -1) + clad::custom_derivatives::TMath::Abs_pushforward(p[1], 0);
//CHECK-NEXT:   }

// CHECK: void exp_pushforward_pullback(double x, double d_x, double _d_y, clad::array_ref<double> _d_x, clad::array_ref<double> _d_d_x) {
// CHECK-NEXT:     double _t0;
// CHECK-NEXT:     double _t1;
// CHECK-NEXT:     double _t2;
// CHECK-NEXT:     _t1 = x;
// CHECK-NEXT:     _t2 = ::std::exp(_t1);
// CHECK-NEXT:     _t0 = d_x;
// CHECK-NEXT:     double exp_pushforward_return = _t2 * _t0;
// CHECK-NEXT:     goto _label0;
// CHECK-NEXT:   _label0:
// CHECK-NEXT:     {
// CHECK-NEXT:         double _r0 = _d_y * _t0;
// CHECK-NEXT:         double _r1 = _r0 * clad::custom_derivatives::exp_pushforward(_t1, 1.);
// CHECK-NEXT:         * _d_x += _r1;
// CHECK-NEXT:         double _r2 = _t2 * _d_y;
// CHECK-NEXT:         * _d_d_x += _r2;
// CHECK-NEXT:     }
// CHECK-NEXT: }

// CHECK: void Exp_pushforward_pullback(Double_t x, Double_t d_x, Double_t _d_y, clad::array_ref<Double_t> _d_x, clad::array_ref<Double_t> _d_d_x) {
// CHECK-NEXT:     Double_t _t0;
// CHECK-NEXT:     Double_t _t1;
// CHECK-NEXT:     _t0 = x;
// CHECK-NEXT:     _t1 = d_x;
// CHECK-NEXT:     double Exp_pushforward_return = std::exp_pushforward(_t0, _t1);
// CHECK-NEXT:     goto _label0;
// CHECK-NEXT:   _label0:
// CHECK-NEXT:     {
// CHECK-NEXT:         double _grad0 = 0.;
// CHECK-NEXT:         double _grad1 = 0.;
// CHECK-NEXT:         exp_pushforward_pullback(_t0, _t1, _d_y, &_grad0, &_grad1);
// CHECK-NEXT:         double _r0 = _grad0;
// CHECK-NEXT:         * _d_x += _r0;
// CHECK-NEXT:         double _r1 = _grad1;
// CHECK-NEXT:         * _d_d_x += _r1;
// CHECK-NEXT:     }
// CHECK-NEXT: }

// CHECK: void abs_pushforward_pullback(double x, double d_x, double _d_y, clad::array_ref<double> _d_x, clad::array_ref<double> _d_d_x) {
// CHECK-NEXT:     bool _cond0;
// CHECK-NEXT:     _cond0 = x >= 0;
// CHECK-NEXT:     if (_cond0) {
// CHECK-NEXT:         double abs_pushforward_return = d_x;
// CHECK-NEXT:         goto _label0;
// CHECK-NEXT:     } else {
// CHECK-NEXT:         double abs_pushforward_return0 = -d_x;
// CHECK-NEXT:         goto _label1;
// CHECK-NEXT:     }
// CHECK-NEXT:     if (_cond0)
// CHECK-NEXT:       _label0:
// CHECK-NEXT:         * _d_d_x += _d_y;
// CHECK-NEXT:     else
// CHECK-NEXT:       _label1:
// CHECK-NEXT:         * _d_d_x += -_d_y;
// CHECK-NEXT: }

// CHECK: void Abs_pushforward_pullback(Double_t x, Double_t d_x, Double_t _d_y, clad::array_ref<Double_t> _d_x, clad::array_ref<Double_t> _d_d_x) {
// CHECK-NEXT:     Double_t _t0;
// CHECK-NEXT:     Double_t _t1;
// CHECK-NEXT:     _t0 = x;
// CHECK-NEXT:     _t1 = d_x;
// CHECK-NEXT:     double Abs_pushforward_return = std::abs_pushforward(_t0, _t1);
// CHECK-NEXT:     goto _label0;
// CHECK-NEXT:   _label0:
// CHECK-NEXT:     {
// CHECK-NEXT:         double _grad0 = 0.;
// CHECK-NEXT:         double _grad1 = 0.;
// CHECK-NEXT:         abs_pushforward_pullback(_t0, _t1, _d_y, &_grad0, &_grad1);
// CHECK-NEXT:         double _r0 = _grad0;
// CHECK-NEXT:         * _d_x += _r0;
// CHECK-NEXT:         double _r1 = _grad1;
// CHECK-NEXT:         * _d_d_x += _r1;
// CHECK-NEXT:     }
// CHECK-NEXT: }

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
//CHECK-NEXT:       double TFormula_example_darg1_0_return = 0 * _t0 + _t2 * _t1 + clad::custom_derivatives::TMath::Exp_pushforward(_t3, -1) + clad::custom_derivatives::TMath::Abs_pushforward(_t4, 0);
//CHECK-NEXT:       goto _label0;
//CHECK-NEXT:     _label0:
//CHECK-NEXT:       {
//CHECK-NEXT:           double _r0 = 1 * _t0;
//CHECK-NEXT:           double _r1 = 0 * 1;
//CHECK-NEXT:           _d__t0 += _r1;
//CHECK-NEXT:           double _r2 = 1 * _t1;
//CHECK-NEXT:           double _r3 = _t2 * 1;
// CHECK-NEXT:          Double_t _grad0 = 0.;
// CHECK-NEXT:          Double_t _grad1 = 0.;
// CHECK-NEXT:          Exp_pushforward_pullback(_t3, -1, 1, &_grad0, &_grad1);
// CHECK-NEXT:          Double_t _r4 = _grad0;
// CHECK-NEXT:          _d_p[0] += -_r4;
// CHECK-NEXT:          Double_t _r5 = _grad1;
// CHECK-NEXT:          Double_t _grad2 = 0.;
// CHECK-NEXT:          Double_t _grad3 = 0.;
// CHECK-NEXT:          Abs_pushforward_pullback(_t4, 0, 1, &_grad2, &_grad3);
// CHECK-NEXT:          Double_t _r6 = _grad2;
// CHECK-NEXT:          _d_p[1] += _r6;
// CHECK-NEXT:          Double_t _r7 = _grad3;
// CHECK-NEXT:      }
//CHECK-NEXT:       {
//CHECK-NEXT:           _d_p[0] += _d__t0;
//CHECK-NEXT:           _d_p[1] += _d__t0;
//CHECK-NEXT:           _d_p[2] += _d__t0;
//CHECK-NEXT:       }
//CHECK-NEXT:   }

//CHECK:   Double_t TFormula_example_darg1_1(Double_t *x, Double_t *p) {
//CHECK-NEXT:       double _t0 = (p[0] + p[1] + p[2]);
//CHECK-NEXT:       return 0 * _t0 + x[0] * (0 + 1 + 0) + clad::custom_derivatives::TMath::Exp_pushforward(-p[0], -0) + clad::custom_derivatives::TMath::Abs_pushforward(p[1], 1);
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
//CHECK-NEXT:       double TFormula_example_darg1_1_return = 0 * _t0 + _t2 * _t1 + clad::custom_derivatives::TMath::Exp_pushforward(_t3, -0) + clad::custom_derivatives::TMath::Abs_pushforward(_t4, 1);
//CHECK-NEXT:       goto _label0;
//CHECK-NEXT:     _label0:
//CHECK-NEXT:       {
//CHECK-NEXT:           double _r0 = 1 * _t0;
//CHECK-NEXT:           double _r1 = 0 * 1;
//CHECK-NEXT:           _d__t0 += _r1;
//CHECK-NEXT:           double _r2 = 1 * _t1;
//CHECK-NEXT:           double _r3 = _t2 * 1;
// CHECK-NEXT:          Double_t _grad0 = 0.;
// CHECK-NEXT:          Double_t _grad1 = 0.;
// CHECK-NEXT:          Exp_pushforward_pullback(_t3, -0, 1, &_grad0, &_grad1);
// CHECK-NEXT:          Double_t _r4 = _grad0;
// CHECK-NEXT:          _d_p[0] += -_r4;
// CHECK-NEXT:          Double_t _r5 = _grad1;
// CHECK-NEXT:          Double_t _grad2 = 0.;
// CHECK-NEXT:          Double_t _grad3 = 0.;
// CHECK-NEXT:          Abs_pushforward_pullback(_t4, 1, 1, &_grad2, &_grad3);
// CHECK-NEXT:          Double_t _r6 = _grad2;
// CHECK-NEXT:          _d_p[1] += _r6;
// CHECK-NEXT:          Double_t _r7 = _grad3;
// CHECK-NEXT:      }
//CHECK-NEXT:       {
//CHECK-NEXT:           _d_p[0] += _d__t0;
//CHECK-NEXT:           _d_p[1] += _d__t0;
//CHECK-NEXT:           _d_p[2] += _d__t0;
//CHECK-NEXT:       }
//CHECK-NEXT:   }

//CHECK:   Double_t TFormula_example_darg1_2(Double_t *x, Double_t *p) {
//CHECK-NEXT:       double _t0 = (p[0] + p[1] + p[2]);
//CHECK-NEXT:       return 0 * _t0 + x[0] * (0 + 0 + 1) + clad::custom_derivatives::TMath::Exp_pushforward(-p[0], -0) + clad::custom_derivatives::TMath::Abs_pushforward(p[1], 0);
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
//CHECK-NEXT:       double TFormula_example_darg1_2_return = 0 * _t0 + _t2 * _t1 + clad::custom_derivatives::TMath::Exp_pushforward(_t3, -0) + clad::custom_derivatives::TMath::Abs_pushforward(_t4, 0);
//CHECK-NEXT:       goto _label0;
//CHECK-NEXT:     _label0:
//CHECK-NEXT:       {
//CHECK-NEXT:           double _r0 = 1 * _t0;
//CHECK-NEXT:           double _r1 = 0 * 1;
//CHECK-NEXT:           _d__t0 += _r1;
//CHECK-NEXT:           double _r2 = 1 * _t1;
//CHECK-NEXT:           double _r3 = _t2 * 1;
// CHECK-NEXT:          Double_t _grad0 = 0.;
// CHECK-NEXT:          Double_t _grad1 = 0.;
// CHECK-NEXT:          Exp_pushforward_pullback(_t3, -0, 1, &_grad0, &_grad1);
// CHECK-NEXT:          Double_t _r4 = _grad0;
// CHECK-NEXT:          _d_p[0] += -_r4;
// CHECK-NEXT:          Double_t _r5 = _grad1;
// CHECK-NEXT:          Double_t _grad2 = 0.;
// CHECK-NEXT:          Double_t _grad3 = 0.;
// CHECK-NEXT:          Abs_pushforward_pullback(_t4, 0, 1, &_grad2, &_grad3);
// CHECK-NEXT:          Double_t _r6 = _grad2;
// CHECK-NEXT:          _d_p[1] += _r6;
// CHECK-NEXT:          Double_t _r7 = _grad3;
// CHECK-NEXT:      }
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
