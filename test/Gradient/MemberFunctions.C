// RUN: %cladclang %s -fno-exceptions -I%S/../../include -oMemberFunctions.out 2>&1 | FileCheck %s
// RUN: ./MemberFunctions.out | FileCheck -check-prefix=CHECK-EXEC %s
// RUN: %cladclang -std=c++14 %s -fno-exceptions -I%S/../../include -oMemberFunctions-cpp14.out 2>&1 | FileCheck %s
// RUN: ./MemberFunctions-cpp14.out | FileCheck -check-prefix=CHECK-EXEC %s
// RUN: %cladclang -std=c++17 %s -fno-exceptions -I%S/../../include -oMemberFunctions-cpp17.out 2>&1 | FileCheck %s
// RUN: ./MemberFunctions-cpp17.out | FileCheck -check-prefix=CHECK-EXEC %s

//CHECK-NOT: {{.*error|warning|note:.*}}
#include "clad/Differentiator/Differentiator.h"

class SimpleFunctions {
public:
  SimpleFunctions(double p_x = 0, double p_y = 0) : x(p_x), y(p_y) {}
  double x, y;
  double mem_fn(double i, double j) { return (x + y) * i + i * j; }

  // CHECK: void mem_fn_grad(double i, double j, clad::array_ref<SimpleFunctions> _d_this, clad::array_ref<double> _d_i, clad::array_ref<double> _d_j) {
  // CHECK-NEXT:     double _t0;
  // CHECK-NEXT:     double _t1;
  // CHECK-NEXT:     double _t2;
  // CHECK-NEXT:     double _t3;
  // CHECK-NEXT:     _t1 = (this->x + this->y);
  // CHECK-NEXT:     _t0 = i;
  // CHECK-NEXT:     _t3 = i;
  // CHECK-NEXT:     _t2 = j;
  // CHECK-NEXT:     {
  // CHECK-NEXT:         double _r0 = 1 * _t0;
  // CHECK-NEXT:         (* _d_this).x += _r0;
  // CHECK-NEXT:         (* _d_this).y += _r0;
  // CHECK-NEXT:         double _r1 = _t1 * 1;
  // CHECK-NEXT:         * _d_i += _r1;
  // CHECK-NEXT:         double _r2 = 1 * _t2;
  // CHECK-NEXT:         * _d_i += _r2;
  // CHECK-NEXT:         double _r3 = _t3 * 1;
  // CHECK-NEXT:         * _d_j += _r3;
  // CHECK-NEXT:     }
  // CHECK-NEXT: }

  double const_mem_fn(double i, double j) const { return (x + y) * i + i * j; }

  // CHECK: void const_mem_fn_grad(double i, double j, clad::array_ref<SimpleFunctions> _d_this, clad::array_ref<double> _d_i, clad::array_ref<double> _d_j) const {
  // CHECK-NEXT:     double _t0;
  // CHECK-NEXT:     double _t1;
  // CHECK-NEXT:     double _t2;
  // CHECK-NEXT:     double _t3;
  // CHECK-NEXT:     _t1 = (this->x + this->y);
  // CHECK-NEXT:     _t0 = i;
  // CHECK-NEXT:     _t3 = i;
  // CHECK-NEXT:     _t2 = j;
  // CHECK-NEXT:     {
  // CHECK-NEXT:         double _r0 = 1 * _t0;
  // CHECK-NEXT:         (* _d_this).x += _r0;
  // CHECK-NEXT:         (* _d_this).y += _r0;
  // CHECK-NEXT:         double _r1 = _t1 * 1;
  // CHECK-NEXT:         * _d_i += _r1;
  // CHECK-NEXT:         double _r2 = 1 * _t2;
  // CHECK-NEXT:         * _d_i += _r2;
  // CHECK-NEXT:         double _r3 = _t3 * 1;
  // CHECK-NEXT:         * _d_j += _r3;
  // CHECK-NEXT:     }
  // CHECK-NEXT: }

  double volatile_mem_fn(double i, double j) volatile {
    return (x+y)*i + i*j;
  }

  // CHECK: void volatile_mem_fn_grad(double i, double j, clad::array_ref<volatile SimpleFunctions> _d_this, clad::array_ref<double> _d_i, clad::array_ref<double> _d_j) volatile {
  // CHECK-NEXT:     double _t0;
  // CHECK-NEXT:     double _t1;
  // CHECK-NEXT:     double _t2;
  // CHECK-NEXT:     double _t3;
  // CHECK-NEXT:     _t1 = (this->x + this->y);
  // CHECK-NEXT:     _t0 = i;
  // CHECK-NEXT:     _t3 = i;
  // CHECK-NEXT:     _t2 = j;
  // CHECK-NEXT:     {
  // CHECK-NEXT:         double _r0 = 1 * _t0;
  // CHECK-NEXT:         (* _d_this).x += _r0;
  // CHECK-NEXT:         (* _d_this).y += _r0;
  // CHECK-NEXT:         double _r1 = _t1 * 1;
  // CHECK-NEXT:         * _d_i += _r1;
  // CHECK-NEXT:         double _r2 = 1 * _t2;
  // CHECK-NEXT:         * _d_i += _r2;
  // CHECK-NEXT:         double _r3 = _t3 * 1;
  // CHECK-NEXT:         * _d_j += _r3;
  // CHECK-NEXT:     }
  // CHECK-NEXT: }

  double const_volatile_mem_fn(double i, double j) const volatile {
    return (x+y)*i + i*j;
  }

  // CHECK: void const_volatile_mem_fn_grad(double i, double j, clad::array_ref<volatile SimpleFunctions> _d_this, clad::array_ref<double> _d_i, clad::array_ref<double> _d_j) const volatile {
  // CHECK-NEXT:     double _t0;
  // CHECK-NEXT:     double _t1;
  // CHECK-NEXT:     double _t2;
  // CHECK-NEXT:     double _t3;
  // CHECK-NEXT:     _t1 = (this->x + this->y);
  // CHECK-NEXT:     _t0 = i;
  // CHECK-NEXT:     _t3 = i;
  // CHECK-NEXT:     _t2 = j;
  // CHECK-NEXT:     {
  // CHECK-NEXT:         double _r0 = 1 * _t0;
  // CHECK-NEXT:         (* _d_this).x += _r0;
  // CHECK-NEXT:         (* _d_this).y += _r0;
  // CHECK-NEXT:         double _r1 = _t1 * 1;
  // CHECK-NEXT:         * _d_i += _r1;
  // CHECK-NEXT:         double _r2 = 1 * _t2;
  // CHECK-NEXT:         * _d_i += _r2;
  // CHECK-NEXT:         double _r3 = _t3 * 1;
  // CHECK-NEXT:         * _d_j += _r3;
  // CHECK-NEXT:     }
  // CHECK-NEXT: }

  double lval_ref_mem_fn(double i, double j) & { return (x + y) * i + i * j; }

  // CHECK: void lval_ref_mem_fn_grad(double i, double j, clad::array_ref<SimpleFunctions> _d_this, clad::array_ref<double> _d_i, clad::array_ref<double> _d_j) & {
  // CHECK-NEXT:     double _t0;
  // CHECK-NEXT:     double _t1;
  // CHECK-NEXT:     double _t2;
  // CHECK-NEXT:     double _t3;
  // CHECK-NEXT:     _t1 = (this->x + this->y);
  // CHECK-NEXT:     _t0 = i;
  // CHECK-NEXT:     _t3 = i;
  // CHECK-NEXT:     _t2 = j;
  // CHECK-NEXT:     {
  // CHECK-NEXT:         double _r0 = 1 * _t0;
  // CHECK-NEXT:         (* _d_this).x += _r0;
  // CHECK-NEXT:         (* _d_this).y += _r0;
  // CHECK-NEXT:         double _r1 = _t1 * 1;
  // CHECK-NEXT:         * _d_i += _r1;
  // CHECK-NEXT:         double _r2 = 1 * _t2;
  // CHECK-NEXT:         * _d_i += _r2;
  // CHECK-NEXT:         double _r3 = _t3 * 1;
  // CHECK-NEXT:         * _d_j += _r3;
  // CHECK-NEXT:     }
  // CHECK-NEXT: }

  double const_lval_ref_mem_fn(double i, double j) const & {
    return (x+y)*i + i*j;
  }

  // CHECK: void const_lval_ref_mem_fn_grad(double i, double j, clad::array_ref<SimpleFunctions> _d_this, clad::array_ref<double> _d_i, clad::array_ref<double> _d_j) const & {
  // CHECK-NEXT:     double _t0;
  // CHECK-NEXT:     double _t1;
  // CHECK-NEXT:     double _t2;
  // CHECK-NEXT:     double _t3;
  // CHECK-NEXT:     _t1 = (this->x + this->y);
  // CHECK-NEXT:     _t0 = i;
  // CHECK-NEXT:     _t3 = i;
  // CHECK-NEXT:     _t2 = j;
  // CHECK-NEXT:     {
  // CHECK-NEXT:         double _r0 = 1 * _t0;
  // CHECK-NEXT:         (* _d_this).x += _r0;
  // CHECK-NEXT:         (* _d_this).y += _r0;
  // CHECK-NEXT:         double _r1 = _t1 * 1;
  // CHECK-NEXT:         * _d_i += _r1;
  // CHECK-NEXT:         double _r2 = 1 * _t2;
  // CHECK-NEXT:         * _d_i += _r2;
  // CHECK-NEXT:         double _r3 = _t3 * 1;
  // CHECK-NEXT:         * _d_j += _r3;
  // CHECK-NEXT:     }
  // CHECK-NEXT: }

  double volatile_lval_ref_mem_fn(double i, double j) volatile & {
    return (x+y)*i + i*j;
  }

  // CHECK: void volatile_lval_ref_mem_fn_grad(double i, double j, clad::array_ref<volatile SimpleFunctions> _d_this, clad::array_ref<double> _d_i, clad::array_ref<double> _d_j) volatile & {
  // CHECK-NEXT:     double _t0;
  // CHECK-NEXT:     double _t1;
  // CHECK-NEXT:     double _t2;
  // CHECK-NEXT:     double _t3;
  // CHECK-NEXT:     _t1 = (this->x + this->y);
  // CHECK-NEXT:     _t0 = i;
  // CHECK-NEXT:     _t3 = i;
  // CHECK-NEXT:     _t2 = j;
  // CHECK-NEXT:     {
  // CHECK-NEXT:         double _r0 = 1 * _t0;
  // CHECK-NEXT:         (* _d_this).x += _r0;
  // CHECK-NEXT:         (* _d_this).y += _r0;
  // CHECK-NEXT:         double _r1 = _t1 * 1;
  // CHECK-NEXT:         * _d_i += _r1;
  // CHECK-NEXT:         double _r2 = 1 * _t2;
  // CHECK-NEXT:         * _d_i += _r2;
  // CHECK-NEXT:         double _r3 = _t3 * 1;
  // CHECK-NEXT:         * _d_j += _r3;
  // CHECK-NEXT:     }
  // CHECK-NEXT: }

  double const_volatile_lval_ref_mem_fn(double i, double j) const volatile & {
    return (x+y)*i + i*j;
  }

  // CHECK: void const_volatile_lval_ref_mem_fn_grad(double i, double j, clad::array_ref<volatile SimpleFunctions> _d_this, clad::array_ref<double> _d_i, clad::array_ref<double> _d_j) const volatile & {
  // CHECK-NEXT:     double _t0;
  // CHECK-NEXT:     double _t1;
  // CHECK-NEXT:     double _t2;
  // CHECK-NEXT:     double _t3;
  // CHECK-NEXT:     _t1 = (this->x + this->y);
  // CHECK-NEXT:     _t0 = i;
  // CHECK-NEXT:     _t3 = i;
  // CHECK-NEXT:     _t2 = j;
  // CHECK-NEXT:     {
  // CHECK-NEXT:         double _r0 = 1 * _t0;
  // CHECK-NEXT:         (* _d_this).x += _r0;
  // CHECK-NEXT:         (* _d_this).y += _r0;
  // CHECK-NEXT:         double _r1 = _t1 * 1;
  // CHECK-NEXT:         * _d_i += _r1;
  // CHECK-NEXT:         double _r2 = 1 * _t2;
  // CHECK-NEXT:         * _d_i += _r2;
  // CHECK-NEXT:         double _r3 = _t3 * 1;
  // CHECK-NEXT:         * _d_j += _r3;
  // CHECK-NEXT:     }
  // CHECK-NEXT: }

  double rval_ref_mem_fn(double i, double j) && { return (x + y) * i + i * j; }

  // CHECK: void rval_ref_mem_fn_grad(double i, double j, clad::array_ref<SimpleFunctions> _d_this, clad::array_ref<double> _d_i, clad::array_ref<double> _d_j) && {
  // CHECK-NEXT:     double _t0;
  // CHECK-NEXT:     double _t1;
  // CHECK-NEXT:     double _t2;
  // CHECK-NEXT:     double _t3;
  // CHECK-NEXT:     _t1 = (this->x + this->y);
  // CHECK-NEXT:     _t0 = i;
  // CHECK-NEXT:     _t3 = i;
  // CHECK-NEXT:     _t2 = j;
  // CHECK-NEXT:     {
  // CHECK-NEXT:         double _r0 = 1 * _t0;
  // CHECK-NEXT:         (* _d_this).x += _r0;
  // CHECK-NEXT:         (* _d_this).y += _r0;
  // CHECK-NEXT:         double _r1 = _t1 * 1;
  // CHECK-NEXT:         * _d_i += _r1;
  // CHECK-NEXT:         double _r2 = 1 * _t2;
  // CHECK-NEXT:         * _d_i += _r2;
  // CHECK-NEXT:         double _r3 = _t3 * 1;
  // CHECK-NEXT:         * _d_j += _r3;
  // CHECK-NEXT:     }
  // CHECK-NEXT: }

  double const_rval_ref_mem_fn(double i, double j) const && {
    return (x+y)*i + i*j;
  }

  // CHECK: void const_rval_ref_mem_fn_grad(double i, double j, clad::array_ref<SimpleFunctions> _d_this, clad::array_ref<double> _d_i, clad::array_ref<double> _d_j) const && {
  // CHECK-NEXT:     double _t0;
  // CHECK-NEXT:     double _t1;
  // CHECK-NEXT:     double _t2;
  // CHECK-NEXT:     double _t3;
  // CHECK-NEXT:     _t1 = (this->x + this->y);
  // CHECK-NEXT:     _t0 = i;
  // CHECK-NEXT:     _t3 = i;
  // CHECK-NEXT:     _t2 = j;
  // CHECK-NEXT:     {
  // CHECK-NEXT:         double _r0 = 1 * _t0;
  // CHECK-NEXT:         (* _d_this).x += _r0;
  // CHECK-NEXT:         (* _d_this).y += _r0;
  // CHECK-NEXT:         double _r1 = _t1 * 1;
  // CHECK-NEXT:         * _d_i += _r1;
  // CHECK-NEXT:         double _r2 = 1 * _t2;
  // CHECK-NEXT:         * _d_i += _r2;
  // CHECK-NEXT:         double _r3 = _t3 * 1;
  // CHECK-NEXT:         * _d_j += _r3;
  // CHECK-NEXT:     }
  // CHECK-NEXT: }

  double volatile_rval_ref_mem_fn(double i, double j) volatile && {
    return (x+y)*i + i*j;
  }

  // CHECK: void volatile_rval_ref_mem_fn_grad(double i, double j, clad::array_ref<volatile SimpleFunctions> _d_this, clad::array_ref<double> _d_i, clad::array_ref<double> _d_j) volatile && {
  // CHECK-NEXT:     double _t0;
  // CHECK-NEXT:     double _t1;
  // CHECK-NEXT:     double _t2;
  // CHECK-NEXT:     double _t3;
  // CHECK-NEXT:     _t1 = (this->x + this->y);
  // CHECK-NEXT:     _t0 = i;
  // CHECK-NEXT:     _t3 = i;
  // CHECK-NEXT:     _t2 = j;
  // CHECK-NEXT:     {
  // CHECK-NEXT:         double _r0 = 1 * _t0;
  // CHECK-NEXT:         (* _d_this).x += _r0;
  // CHECK-NEXT:         (* _d_this).y += _r0;
  // CHECK-NEXT:         double _r1 = _t1 * 1;
  // CHECK-NEXT:         * _d_i += _r1;
  // CHECK-NEXT:         double _r2 = 1 * _t2;
  // CHECK-NEXT:         * _d_i += _r2;
  // CHECK-NEXT:         double _r3 = _t3 * 1;
  // CHECK-NEXT:         * _d_j += _r3;
  // CHECK-NEXT:     }
  // CHECK-NEXT: }

  double const_volatile_rval_ref_mem_fn(double i, double j) const volatile && {
    return (x+y)*i + i*j;
  }

  // CHECK: void const_volatile_rval_ref_mem_fn_grad(double i, double j, clad::array_ref<volatile SimpleFunctions> _d_this, clad::array_ref<double> _d_i, clad::array_ref<double> _d_j) const volatile && {
  // CHECK-NEXT:     double _t0;
  // CHECK-NEXT:     double _t1;
  // CHECK-NEXT:     double _t2;
  // CHECK-NEXT:     double _t3;
  // CHECK-NEXT:     _t1 = (this->x + this->y);
  // CHECK-NEXT:     _t0 = i;
  // CHECK-NEXT:     _t3 = i;
  // CHECK-NEXT:     _t2 = j;
  // CHECK-NEXT:     {
  // CHECK-NEXT:         double _r0 = 1 * _t0;
  // CHECK-NEXT:         (* _d_this).x += _r0;
  // CHECK-NEXT:         (* _d_this).y += _r0;
  // CHECK-NEXT:         double _r1 = _t1 * 1;
  // CHECK-NEXT:         * _d_i += _r1;
  // CHECK-NEXT:         double _r2 = 1 * _t2;
  // CHECK-NEXT:         * _d_i += _r2;
  // CHECK-NEXT:         double _r3 = _t3 * 1;
  // CHECK-NEXT:         * _d_j += _r3;
  // CHECK-NEXT:     }
  // CHECK-NEXT: }

  double noexcept_mem_fn(double i, double j) noexcept {
    return (x+y)*i + i*j;
  }

  // CHECK: void noexcept_mem_fn_grad(double i, double j, clad::array_ref<SimpleFunctions> _d_this, clad::array_ref<double> _d_i, clad::array_ref<double> _d_j) noexcept {
  // CHECK-NEXT:     double _t0;
  // CHECK-NEXT:     double _t1;
  // CHECK-NEXT:     double _t2;
  // CHECK-NEXT:     double _t3;
  // CHECK-NEXT:     _t1 = (this->x + this->y);
  // CHECK-NEXT:     _t0 = i;
  // CHECK-NEXT:     _t3 = i;
  // CHECK-NEXT:     _t2 = j;
  // CHECK-NEXT:     {
  // CHECK-NEXT:         double _r0 = 1 * _t0;
  // CHECK-NEXT:         (* _d_this).x += _r0;
  // CHECK-NEXT:         (* _d_this).y += _r0;
  // CHECK-NEXT:         double _r1 = _t1 * 1;
  // CHECK-NEXT:         * _d_i += _r1;
  // CHECK-NEXT:         double _r2 = 1 * _t2;
  // CHECK-NEXT:         * _d_i += _r2;
  // CHECK-NEXT:         double _r3 = _t3 * 1;
  // CHECK-NEXT:         * _d_j += _r3;
  // CHECK-NEXT:     }
  // CHECK-NEXT: }

  double const_noexcept_mem_fn(double i, double j) const noexcept {
    return (x+y)*i + i*j;
  }

  // CHECK: void const_noexcept_mem_fn_grad(double i, double j, clad::array_ref<SimpleFunctions> _d_this, clad::array_ref<double> _d_i, clad::array_ref<double> _d_j) const noexcept {
  // CHECK-NEXT:     double _t0;
  // CHECK-NEXT:     double _t1;
  // CHECK-NEXT:     double _t2;
  // CHECK-NEXT:     double _t3;
  // CHECK-NEXT:     _t1 = (this->x + this->y);
  // CHECK-NEXT:     _t0 = i;
  // CHECK-NEXT:     _t3 = i;
  // CHECK-NEXT:     _t2 = j;
  // CHECK-NEXT:     {
  // CHECK-NEXT:         double _r0 = 1 * _t0;
  // CHECK-NEXT:         (* _d_this).x += _r0;
  // CHECK-NEXT:         (* _d_this).y += _r0;
  // CHECK-NEXT:         double _r1 = _t1 * 1;
  // CHECK-NEXT:         * _d_i += _r1;
  // CHECK-NEXT:         double _r2 = 1 * _t2;
  // CHECK-NEXT:         * _d_i += _r2;
  // CHECK-NEXT:         double _r3 = _t3 * 1;
  // CHECK-NEXT:         * _d_j += _r3;
  // CHECK-NEXT:     }
  // CHECK-NEXT: }

  double volatile_noexcept_mem_fn(double i, double j) volatile noexcept {
    return (x+y)*i + i*j;
  }

  // CHECK: void volatile_noexcept_mem_fn_grad(double i, double j, clad::array_ref<volatile SimpleFunctions> _d_this, clad::array_ref<double> _d_i, clad::array_ref<double> _d_j) volatile noexcept {
  // CHECK-NEXT:     double _t0;
  // CHECK-NEXT:     double _t1;
  // CHECK-NEXT:     double _t2;
  // CHECK-NEXT:     double _t3;
  // CHECK-NEXT:     _t1 = (this->x + this->y);
  // CHECK-NEXT:     _t0 = i;
  // CHECK-NEXT:     _t3 = i;
  // CHECK-NEXT:     _t2 = j;
  // CHECK-NEXT:     {
  // CHECK-NEXT:         double _r0 = 1 * _t0;
  // CHECK-NEXT:         (* _d_this).x += _r0;
  // CHECK-NEXT:         (* _d_this).y += _r0;
  // CHECK-NEXT:         double _r1 = _t1 * 1;
  // CHECK-NEXT:         * _d_i += _r1;
  // CHECK-NEXT:         double _r2 = 1 * _t2;
  // CHECK-NEXT:         * _d_i += _r2;
  // CHECK-NEXT:         double _r3 = _t3 * 1;
  // CHECK-NEXT:         * _d_j += _r3;
  // CHECK-NEXT:     }
  // CHECK-NEXT: }

  double const_volatile_noexcept_mem_fn(double i, double j) const volatile noexcept {
    return (x+y)*i + i*j;
  }

  // CHECK: void const_volatile_noexcept_mem_fn_grad(double i, double j, clad::array_ref<volatile SimpleFunctions> _d_this, clad::array_ref<double> _d_i, clad::array_ref<double> _d_j) const volatile noexcept {
  // CHECK-NEXT:     double _t0;
  // CHECK-NEXT:     double _t1;
  // CHECK-NEXT:     double _t2;
  // CHECK-NEXT:     double _t3;
  // CHECK-NEXT:     _t1 = (this->x + this->y);
  // CHECK-NEXT:     _t0 = i;
  // CHECK-NEXT:     _t3 = i;
  // CHECK-NEXT:     _t2 = j;
  // CHECK-NEXT:     {
  // CHECK-NEXT:         double _r0 = 1 * _t0;
  // CHECK-NEXT:         (* _d_this).x += _r0;
  // CHECK-NEXT:         (* _d_this).y += _r0;
  // CHECK-NEXT:         double _r1 = _t1 * 1;
  // CHECK-NEXT:         * _d_i += _r1;
  // CHECK-NEXT:         double _r2 = 1 * _t2;
  // CHECK-NEXT:         * _d_i += _r2;
  // CHECK-NEXT:         double _r3 = _t3 * 1;
  // CHECK-NEXT:         * _d_j += _r3;
  // CHECK-NEXT:     }
  // CHECK-NEXT: }

  double lval_ref_noexcept_mem_fn(double i, double j) & noexcept {
    return (x+y)*i + i*j;
  }

  // CHECK: void lval_ref_noexcept_mem_fn_grad(double i, double j, clad::array_ref<SimpleFunctions> _d_this, clad::array_ref<double> _d_i, clad::array_ref<double> _d_j) & noexcept {
  // CHECK-NEXT:     double _t0;
  // CHECK-NEXT:     double _t1;
  // CHECK-NEXT:     double _t2;
  // CHECK-NEXT:     double _t3;
  // CHECK-NEXT:     _t1 = (this->x + this->y);
  // CHECK-NEXT:     _t0 = i;
  // CHECK-NEXT:     _t3 = i;
  // CHECK-NEXT:     _t2 = j;
  // CHECK-NEXT:     {
  // CHECK-NEXT:         double _r0 = 1 * _t0;
  // CHECK-NEXT:         (* _d_this).x += _r0;
  // CHECK-NEXT:         (* _d_this).y += _r0;
  // CHECK-NEXT:         double _r1 = _t1 * 1;
  // CHECK-NEXT:         * _d_i += _r1;
  // CHECK-NEXT:         double _r2 = 1 * _t2;
  // CHECK-NEXT:         * _d_i += _r2;
  // CHECK-NEXT:         double _r3 = _t3 * 1;
  // CHECK-NEXT:         * _d_j += _r3;
  // CHECK-NEXT:     }
  // CHECK-NEXT: }

  double const_lval_ref_noexcept_mem_fn(double i, double j) const & noexcept {
    return (x+y)*i + i*j;
  }

  // CHECK: void const_lval_ref_noexcept_mem_fn_grad(double i, double j, clad::array_ref<SimpleFunctions> _d_this, clad::array_ref<double> _d_i, clad::array_ref<double> _d_j) const & noexcept {
  // CHECK-NEXT:     double _t0;
  // CHECK-NEXT:     double _t1;
  // CHECK-NEXT:     double _t2;
  // CHECK-NEXT:     double _t3;
  // CHECK-NEXT:     _t1 = (this->x + this->y);
  // CHECK-NEXT:     _t0 = i;
  // CHECK-NEXT:     _t3 = i;
  // CHECK-NEXT:     _t2 = j;
  // CHECK-NEXT:     {
  // CHECK-NEXT:         double _r0 = 1 * _t0;
  // CHECK-NEXT:         (* _d_this).x += _r0;
  // CHECK-NEXT:         (* _d_this).y += _r0;
  // CHECK-NEXT:         double _r1 = _t1 * 1;
  // CHECK-NEXT:         * _d_i += _r1;
  // CHECK-NEXT:         double _r2 = 1 * _t2;
  // CHECK-NEXT:         * _d_i += _r2;
  // CHECK-NEXT:         double _r3 = _t3 * 1;
  // CHECK-NEXT:         * _d_j += _r3;
  // CHECK-NEXT:     }
  // CHECK-NEXT: }

  double volatile_lval_ref_noexcept_mem_fn(double i, double j) volatile & noexcept {
    return (x+y)*i + i*j;
  }

  // CHECK: void volatile_lval_ref_noexcept_mem_fn_grad(double i, double j, clad::array_ref<volatile SimpleFunctions> _d_this, clad::array_ref<double> _d_i, clad::array_ref<double> _d_j) volatile & noexcept {
  // CHECK-NEXT:     double _t0;
  // CHECK-NEXT:     double _t1;
  // CHECK-NEXT:     double _t2;
  // CHECK-NEXT:     double _t3;
  // CHECK-NEXT:     _t1 = (this->x + this->y);
  // CHECK-NEXT:     _t0 = i;
  // CHECK-NEXT:     _t3 = i;
  // CHECK-NEXT:     _t2 = j;
  // CHECK-NEXT:     {
  // CHECK-NEXT:         double _r0 = 1 * _t0;
  // CHECK-NEXT:         (* _d_this).x += _r0;
  // CHECK-NEXT:         (* _d_this).y += _r0;
  // CHECK-NEXT:         double _r1 = _t1 * 1;
  // CHECK-NEXT:         * _d_i += _r1;
  // CHECK-NEXT:         double _r2 = 1 * _t2;
  // CHECK-NEXT:         * _d_i += _r2;
  // CHECK-NEXT:         double _r3 = _t3 * 1;
  // CHECK-NEXT:         * _d_j += _r3;
  // CHECK-NEXT:     }
  // CHECK-NEXT: }

  double const_volatile_lval_ref_noexcept_mem_fn(double i, double j) const volatile & noexcept {
    return (x+y)*i + i*j;
  }

  // CHECK: void const_volatile_lval_ref_noexcept_mem_fn_grad(double i, double j, clad::array_ref<volatile SimpleFunctions> _d_this, clad::array_ref<double> _d_i, clad::array_ref<double> _d_j) const volatile & noexcept {
  // CHECK-NEXT:     double _t0;
  // CHECK-NEXT:     double _t1;
  // CHECK-NEXT:     double _t2;
  // CHECK-NEXT:     double _t3;
  // CHECK-NEXT:     _t1 = (this->x + this->y);
  // CHECK-NEXT:     _t0 = i;
  // CHECK-NEXT:     _t3 = i;
  // CHECK-NEXT:     _t2 = j;
  // CHECK-NEXT:     {
  // CHECK-NEXT:         double _r0 = 1 * _t0;
  // CHECK-NEXT:         (* _d_this).x += _r0;
  // CHECK-NEXT:         (* _d_this).y += _r0;
  // CHECK-NEXT:         double _r1 = _t1 * 1;
  // CHECK-NEXT:         * _d_i += _r1;
  // CHECK-NEXT:         double _r2 = 1 * _t2;
  // CHECK-NEXT:         * _d_i += _r2;
  // CHECK-NEXT:         double _r3 = _t3 * 1;
  // CHECK-NEXT:         * _d_j += _r3;
  // CHECK-NEXT:     }
  // CHECK-NEXT: }

  double rval_ref_noexcept_mem_fn(double i, double j) && noexcept {
    return (x+y)*i + i*j;
  }

  // CHECK: void rval_ref_noexcept_mem_fn_grad(double i, double j, clad::array_ref<SimpleFunctions> _d_this, clad::array_ref<double> _d_i, clad::array_ref<double> _d_j) && noexcept {
  // CHECK-NEXT:     double _t0;
  // CHECK-NEXT:     double _t1;
  // CHECK-NEXT:     double _t2;
  // CHECK-NEXT:     double _t3;
  // CHECK-NEXT:     _t1 = (this->x + this->y);
  // CHECK-NEXT:     _t0 = i;
  // CHECK-NEXT:     _t3 = i;
  // CHECK-NEXT:     _t2 = j;
  // CHECK-NEXT:     {
  // CHECK-NEXT:         double _r0 = 1 * _t0;
  // CHECK-NEXT:         (* _d_this).x += _r0;
  // CHECK-NEXT:         (* _d_this).y += _r0;
  // CHECK-NEXT:         double _r1 = _t1 * 1;
  // CHECK-NEXT:         * _d_i += _r1;
  // CHECK-NEXT:         double _r2 = 1 * _t2;
  // CHECK-NEXT:         * _d_i += _r2;
  // CHECK-NEXT:         double _r3 = _t3 * 1;
  // CHECK-NEXT:         * _d_j += _r3;
  // CHECK-NEXT:     }
  // CHECK-NEXT: }

  double const_rval_ref_noexcept_mem_fn(double i, double j) const && noexcept {
    return (x+y)*i + i*j;
  }

  // CHECK: void const_rval_ref_noexcept_mem_fn_grad(double i, double j, clad::array_ref<SimpleFunctions> _d_this, clad::array_ref<double> _d_i, clad::array_ref<double> _d_j) const && noexcept {
  // CHECK-NEXT:     double _t0;
  // CHECK-NEXT:     double _t1;
  // CHECK-NEXT:     double _t2;
  // CHECK-NEXT:     double _t3;
  // CHECK-NEXT:     _t1 = (this->x + this->y);
  // CHECK-NEXT:     _t0 = i;
  // CHECK-NEXT:     _t3 = i;
  // CHECK-NEXT:     _t2 = j;
  // CHECK-NEXT:     {
  // CHECK-NEXT:         double _r0 = 1 * _t0;
  // CHECK-NEXT:         (* _d_this).x += _r0;
  // CHECK-NEXT:         (* _d_this).y += _r0;
  // CHECK-NEXT:         double _r1 = _t1 * 1;
  // CHECK-NEXT:         * _d_i += _r1;
  // CHECK-NEXT:         double _r2 = 1 * _t2;
  // CHECK-NEXT:         * _d_i += _r2;
  // CHECK-NEXT:         double _r3 = _t3 * 1;
  // CHECK-NEXT:         * _d_j += _r3;
  // CHECK-NEXT:     }
  // CHECK-NEXT: }

  double volatile_rval_ref_noexcept_mem_fn(double i, double j) volatile && noexcept {
    return (x+y)*i + i*j;
  }

  // CHECK: void volatile_rval_ref_noexcept_mem_fn_grad(double i, double j, clad::array_ref<volatile SimpleFunctions> _d_this, clad::array_ref<double> _d_i, clad::array_ref<double> _d_j) volatile && noexcept {
  // CHECK-NEXT:     double _t0;
  // CHECK-NEXT:     double _t1;
  // CHECK-NEXT:     double _t2;
  // CHECK-NEXT:     double _t3;
  // CHECK-NEXT:     _t1 = (this->x + this->y);
  // CHECK-NEXT:     _t0 = i;
  // CHECK-NEXT:     _t3 = i;
  // CHECK-NEXT:     _t2 = j;
  // CHECK-NEXT:     {
  // CHECK-NEXT:         double _r0 = 1 * _t0;
  // CHECK-NEXT:         (* _d_this).x += _r0;
  // CHECK-NEXT:         (* _d_this).y += _r0;
  // CHECK-NEXT:         double _r1 = _t1 * 1;
  // CHECK-NEXT:         * _d_i += _r1;
  // CHECK-NEXT:         double _r2 = 1 * _t2;
  // CHECK-NEXT:         * _d_i += _r2;
  // CHECK-NEXT:         double _r3 = _t3 * 1;
  // CHECK-NEXT:         * _d_j += _r3;
  // CHECK-NEXT:     }
  // CHECK-NEXT: }

  double const_volatile_rval_ref_noexcept_mem_fn(double i, double j) const volatile && noexcept {
    return (x+y)*i + i*j;
  }

  // CHECK: void const_volatile_rval_ref_noexcept_mem_fn_grad(double i, double j, clad::array_ref<volatile SimpleFunctions> _d_this, clad::array_ref<double> _d_i, clad::array_ref<double> _d_j) const volatile && noexcept {
  // CHECK-NEXT:     double _t0;
  // CHECK-NEXT:     double _t1;
  // CHECK-NEXT:     double _t2;
  // CHECK-NEXT:     double _t3;
  // CHECK-NEXT:     _t1 = (this->x + this->y);
  // CHECK-NEXT:     _t0 = i;
  // CHECK-NEXT:     _t3 = i;
  // CHECK-NEXT:     _t2 = j;
  // CHECK-NEXT:     {
  // CHECK-NEXT:         double _r0 = 1 * _t0;
  // CHECK-NEXT:         (* _d_this).x += _r0;
  // CHECK-NEXT:         (* _d_this).y += _r0;
  // CHECK-NEXT:         double _r1 = _t1 * 1;
  // CHECK-NEXT:         * _d_i += _r1;
  // CHECK-NEXT:         double _r2 = 1 * _t2;
  // CHECK-NEXT:         * _d_i += _r2;
  // CHECK-NEXT:         double _r3 = _t3 * 1;
  // CHECK-NEXT:         * _d_j += _r3;
  // CHECK-NEXT:     }
  // CHECK-NEXT: }

  double partial_mem_fn(double i, double j) { return (x + y) * i + i * j; }

  // CHECK: void partial_mem_fn_grad_0(double i, double j, clad::array_ref<SimpleFunctions> _d_this, clad::array_ref<double> _d_i) {
  // CHECK-NEXT:     double _d_j = 0;
  // CHECK-NEXT:     double _t0;
  // CHECK-NEXT:     double _t1;
  // CHECK-NEXT:     double _t2;
  // CHECK-NEXT:     double _t3;
  // CHECK-NEXT:     _t1 = (this->x + this->y);
  // CHECK-NEXT:     _t0 = i;
  // CHECK-NEXT:     _t3 = i;
  // CHECK-NEXT:     _t2 = j;
  // CHECK-NEXT:     {
  // CHECK-NEXT:         double _r0 = 1 * _t0;
  // CHECK-NEXT:         (* _d_this).x += _r0;
  // CHECK-NEXT:         (* _d_this).y += _r0;
  // CHECK-NEXT:         double _r1 = _t1 * 1;
  // CHECK-NEXT:         * _d_i += _r1;
  // CHECK-NEXT:         double _r2 = 1 * _t2;
  // CHECK-NEXT:         * _d_i += _r2;
  // CHECK-NEXT:         double _r3 = _t3 * 1;
  // CHECK-NEXT:         _d_j += _r3;
  // CHECK-NEXT:     }
  // CHECK-NEXT: }

  void mem_fn_grad(double i, double j, clad::array_ref<double> _d_i, clad::array_ref<double> _d_j);
  void const_mem_fn_grad(double i, double j, clad::array_ref<double> _d_i, clad::array_ref<double> _d_j);
  void volatile_mem_fn_grad(double i, double j, clad::array_ref<double> _d_i, clad::array_ref<double> _d_j);
  void const_volatile_mem_fn_grad(double i, double j, clad::array_ref<double> _d_i, clad::array_ref<double> _d_j);
  void lval_ref_mem_fn_grad(double i, double j, clad::array_ref<double> _d_i, clad::array_ref<double> _d_j);
  void const_lval_ref_mem_fn_grad(double i, double j, clad::array_ref<double> _d_i, clad::array_ref<double> _d_j);
  void volatile_lval_ref_mem_fn_grad(double i, double j, clad::array_ref<double> _d_i, clad::array_ref<double> _d_j);
  void const_volatile_lval_ref_mem_fn_grad(double i, double j, clad::array_ref<double> _d_i, clad::array_ref<double> _d_j);
  void rval_ref_mem_fn_grad(double i, double j, clad::array_ref<double> _d_i, clad::array_ref<double> _d_j);
  void const_rval_ref_mem_fn_grad(double i, double j, clad::array_ref<double> _d_i, clad::array_ref<double> _d_j);
  void volatile_rval_ref_mem_fn_grad(double i, double j, clad::array_ref<double> _d_i, clad::array_ref<double> _d_j);
  void const_volatile_rval_ref_mem_fn_grad(double i, double j, clad::array_ref<double> _d_i, clad::array_ref<double> _d_j);
  void noexcept_mem_fn_grad(double i, double j, clad::array_ref<double> _d_i, clad::array_ref<double> _d_j);
  void const_noexcept_mem_fn_grad(double i, double j, clad::array_ref<double> _d_i, clad::array_ref<double> _d_j);
  void volatile_noexcept_mem_fn_grad(double i, double j, clad::array_ref<double> _d_i, clad::array_ref<double> _d_j);
  void const_volatile_noexcept_mem_fn_grad(double i, double j, clad::array_ref<double> _d_i, clad::array_ref<double> _d_j);
  void lval_ref_noexcept_mem_fn_grad(double i, double j, clad::array_ref<double> _d_i, clad::array_ref<double> _d_j);
  void const_lval_ref_noexcept_mem_fn_grad(double i, double j, clad::array_ref<double> _d_i, clad::array_ref<double> _d_j);
  void volatile_lval_ref_noexcept_mem_fn_grad(double i, double j, clad::array_ref<double> _d_i, clad::array_ref<double> _d_j);
  void const_volatile_lval_ref_noexcept_mem_fn_grad(double i, double j, clad::array_ref<double> _d_i, clad::array_ref<double> _d_j);
  void rval_ref_noexcept_mem_fn_grad(double i, double j, clad::array_ref<double> _d_i, clad::array_ref<double> _d_j);
  void const_rval_ref_noexcept_mem_fn_grad(double i, double j, clad::array_ref<double> _d_i, clad::array_ref<double> _d_j);
  void volatile_rval_ref_noexcept_mem_fn_grad(double i, double j, clad::array_ref<double> _d_i, clad::array_ref<double> _d_j);
  void const_volatile_rval_ref_noexcept_mem_fn_grad(double i, double j, clad::array_ref<double> _d_i, clad::array_ref<double> _d_j);
  void partial_mem_fn_grad(double i, double j, clad::array_ref<double> _d_i);
};

double fn(double i,double j) {
  return i*i*j;
}

// CHECK: void fn_grad(double i, double j, clad::array_ref<double> _d_i, clad::array_ref<double> _d_j) {
// CHECK-NEXT:     double _t0;
// CHECK-NEXT:     double _t1;
// CHECK-NEXT:     double _t2;
// CHECK-NEXT:     double _t3;
// CHECK-NEXT:     _t2 = i;
// CHECK-NEXT:     _t1 = i;
// CHECK-NEXT:     _t3 = _t2 * _t1;
// CHECK-NEXT:     _t0 = j;
// CHECK-NEXT:     {
// CHECK-NEXT:         double _r0 = 1 * _t0;
// CHECK-NEXT:         double _r1 = _r0 * _t1;
// CHECK-NEXT:         * _d_i += _r1;
// CHECK-NEXT:         double _r2 = _t2 * _r0;
// CHECK-NEXT:         * _d_i += _r2;
// CHECK-NEXT:         double _r3 = _t3 * 1;
// CHECK-NEXT:         * _d_j += _r3;
// CHECK-NEXT:     }
// CHECK-NEXT: }

int main() {
  auto d_mem_fn = clad::gradient(&SimpleFunctions::mem_fn);
  auto d_const_mem_fn = clad::gradient(&SimpleFunctions::const_mem_fn);
  auto d_volatile_mem_fn = clad::gradient(&SimpleFunctions::volatile_mem_fn);
  auto d_const_volatile_mem_fn = clad::gradient(&SimpleFunctions::const_volatile_mem_fn);
  auto d_lval_ref_mem_fn = clad::gradient(&SimpleFunctions::lval_ref_mem_fn);
  auto d_const_lval_ref_mem_fn = clad::gradient(&SimpleFunctions::const_lval_ref_mem_fn);
  auto d_volatile_lval_ref_mem_fn = clad::gradient(&SimpleFunctions::volatile_lval_ref_mem_fn);
  auto d_const_volatile_lval_ref_mem_fn = clad::gradient(&SimpleFunctions::const_volatile_lval_ref_mem_fn);
  auto d_rval_ref_mem_fn = clad::gradient(&SimpleFunctions::rval_ref_mem_fn);
  auto d_const_rval_ref_mem_fn = clad::gradient(&SimpleFunctions::const_rval_ref_mem_fn);
  auto d_volatile_rval_ref_mem_fn = clad::gradient(&SimpleFunctions::volatile_rval_ref_mem_fn);
  auto d_const_volatile_rval_ref_mem_fn = clad::gradient(&SimpleFunctions::const_volatile_rval_ref_mem_fn);
  auto d_noexcept_mem_fn = clad::gradient(&SimpleFunctions::noexcept_mem_fn);
  auto d_const_noexcept_mem_fn = clad::gradient(&SimpleFunctions::const_noexcept_mem_fn);
  auto d_volatile_noexcept_mem_fn = clad::gradient(&SimpleFunctions::volatile_noexcept_mem_fn);
  auto d_const_volatile_noexcept_mem_fn = clad::gradient(&SimpleFunctions::const_volatile_noexcept_mem_fn);
  auto d_lval_ref_noexcept_mem_fn = clad::gradient(&SimpleFunctions::lval_ref_noexcept_mem_fn);
  auto d_const_lval_ref_noexcept_mem_fn = clad::gradient(&SimpleFunctions::const_lval_ref_noexcept_mem_fn);
  auto d_volatile_lval_ref_noexcept_mem_fn = clad::gradient(&SimpleFunctions::volatile_lval_ref_noexcept_mem_fn);
  auto d_const_volatile_lval_ref_noexcept_mem_fn = clad::gradient(&SimpleFunctions::const_volatile_lval_ref_noexcept_mem_fn);
  auto d_rval_ref_noexcept_mem_fn = clad::gradient(&SimpleFunctions::rval_ref_noexcept_mem_fn);
  auto d_const_rval_ref_noexcept_mem_fn = clad::gradient(&SimpleFunctions::const_rval_ref_noexcept_mem_fn);
  auto d_volatile_rval_ref_noexcept_mem_fn = clad::gradient(&SimpleFunctions::volatile_rval_ref_noexcept_mem_fn);
  auto d_const_volatile_rval_ref_noexcept_mem_fn = clad::gradient(&SimpleFunctions::const_volatile_rval_ref_noexcept_mem_fn);
  auto d_partial_mem_fn = clad::gradient(&SimpleFunctions::partial_mem_fn, "i");

  auto d_fn = clad::gradient(fn);
  double result[2] = {};
  d_fn.execute(4, 5, &result[0], &result[1]);
  for(unsigned i=0;i<2;++i) {
    printf("%.2f ",result[i]);  //CHECK-EXEC: 40.00 16.00
  }

  auto d_const_volatile_lval_ref_mem_fn_i = clad::gradient(&SimpleFunctions::const_volatile_lval_ref_mem_fn, "i");

  // CHECK:   void const_volatile_lval_ref_mem_fn_grad_0(double i, double j, clad::array_ref<volatile SimpleFunctions> _d_this, clad::array_ref<double> _d_i) const volatile & {
  // CHECK-NEXT:       double _d_j = 0;
  // CHECK-NEXT:       double _t0;
  // CHECK-NEXT:       double _t1;
  // CHECK-NEXT:       double _t2;
  // CHECK-NEXT:       double _t3;
  // CHECK-NEXT:       _t1 = (this->x + this->y);
  // CHECK-NEXT:       _t0 = i;
  // CHECK-NEXT:       _t3 = i;
  // CHECK-NEXT:       _t2 = j;
  // CHECK-NEXT:       {
  // CHECK-NEXT:           double _r0 = 1 * _t0;
  // CHECK-NEXT:           (* _d_this).x += _r0;
  // CHECK-NEXT:           (* _d_this).y += _r0;
  // CHECK-NEXT:           double _r1 = _t1 * 1;
  // CHECK-NEXT:           * _d_i += _r1;
  // CHECK-NEXT:           double _r2 = 1 * _t2;
  // CHECK-NEXT:           * _d_i += _r2;
  // CHECK-NEXT:           double _r3 = _t3 * 1;
  // CHECK-NEXT:           _d_j += _r3;
  // CHECK-NEXT:       }
  // CHECK-NEXT:   }

  auto d_const_volatile_rval_ref_mem_fn_j = clad::gradient(&SimpleFunctions::const_volatile_rval_ref_mem_fn, "j");

  // CHECK:   void const_volatile_rval_ref_mem_fn_grad_1(double i, double j, clad::array_ref<volatile SimpleFunctions> _d_this, clad::array_ref<double> _d_j) const volatile && {
  // CHECK-NEXT:       double _d_i = 0;
  // CHECK-NEXT:       double _t0;
  // CHECK-NEXT:       double _t1;
  // CHECK-NEXT:       double _t2;
  // CHECK-NEXT:       double _t3;
  // CHECK-NEXT:       _t1 = (this->x + this->y);
  // CHECK-NEXT:       _t0 = i;
  // CHECK-NEXT:       _t3 = i;
  // CHECK-NEXT:       _t2 = j;
  // CHECK-NEXT:       {
  // CHECK-NEXT:           double _r0 = 1 * _t0;
  // CHECK-NEXT:           (* _d_this).x += _r0;
  // CHECK-NEXT:           (* _d_this).y += _r0;
  // CHECK-NEXT:           double _r1 = _t1 * 1;
  // CHECK-NEXT:           _d_i += _r1;
  // CHECK-NEXT:           double _r2 = 1 * _t2;
  // CHECK-NEXT:           _d_i += _r2;
  // CHECK-NEXT:           double _r3 = _t3 * 1;
  // CHECK-NEXT:           * _d_j += _r3;
  // CHECK-NEXT:       }
  // CHECK-NEXT:   }
}
