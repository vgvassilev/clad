// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -disable-tbr %s -fno-exceptions -I%S/../../include -oMemberFunctions.out 2>&1 | %filecheck %s
// RUN: ./MemberFunctions.out | %filecheck_exec %s
// RUN: %cladclang %s -fno-exceptions -I%S/../../include -oMemberFunctions.out
// RUN: ./MemberFunctions.out | %filecheck_exec %s

// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -disable-tbr -std=c++14 %s -fno-exceptions -I%S/../../include -oMemberFunctions-cpp14.out 2>&1 | %filecheck %s
// RUN: ./MemberFunctions-cpp14.out | %filecheck_exec %s
// RUN: %cladclang -std=c++14 %s -fno-exceptions -I%S/../../include -oMemberFunctions-cpp14.out
// RUN: ./MemberFunctions-cpp14.out | %filecheck_exec %s

// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -disable-tbr -std=c++17 %s -fno-exceptions -I%S/../../include -oMemberFunctions-cpp17.out 2>&1 | %filecheck %s
// RUN: ./MemberFunctions-cpp17.out | %filecheck_exec %s
// RUN: %cladclang -std=c++17 %s -fno-exceptions -I%S/../../include -oMemberFunctions-cpp17.out
// RUN: ./MemberFunctions-cpp17.out | %filecheck_exec %s

#include "clad/Differentiator/Differentiator.h"

class SimpleFunctions {
public:
  SimpleFunctions(double p_x = 0, double p_y = 0) : x(p_x), y(p_y) {}
  double x, y;
  double mem_fn(double i, double j) { return (x + y) * i + i * j; }

  // CHECK: void SimpleFunctions::mem_fn_grad(double i, double j, SimpleFunctions *_d_this, double *_d_i, double *_d_j) {
  // CHECK-NEXT:     {
  // CHECK-NEXT:         _d_this->x += 1 * i;
  // CHECK-NEXT:         _d_this->y += 1 * i;
  // CHECK-NEXT:         *_d_i += (this->x + this->y) * 1;
  // CHECK-NEXT:         *_d_i += 1 * j;
  // CHECK-NEXT:         *_d_j += i * 1;
  // CHECK-NEXT:     }
  // CHECK-NEXT: }

  double const_mem_fn(double i, double j) const { return (x + y) * i + i * j; }

  // CHECK: void SimpleFunctions::const_mem_fn_grad(double i, double j, SimpleFunctions *_d_this, double *_d_i, double *_d_j) const {
  // CHECK-NEXT:     {
  // CHECK-NEXT:         _d_this->x += 1 * i;
  // CHECK-NEXT:         _d_this->y += 1 * i;
  // CHECK-NEXT:         *_d_i += (this->x + this->y) * 1;
  // CHECK-NEXT:         *_d_i += 1 * j;
  // CHECK-NEXT:         *_d_j += i * 1;
  // CHECK-NEXT:     }
  // CHECK-NEXT: }

  double volatile_mem_fn(double i, double j) volatile {
    return (x+y)*i + i*j;
  }

  // CHECK: void SimpleFunctions::volatile_mem_fn_grad(double i, double j, volatile SimpleFunctions *_d_this, double *_d_i, double *_d_j) volatile {
  // CHECK-NEXT:     double _t0 = (this->x + this->y);
  // CHECK-NEXT:     {
  // CHECK-NEXT:         _d_this->x += 1 * i;
  // CHECK-NEXT:         _d_this->y += 1 * i;
  // CHECK-NEXT:         *_d_i += _t0 * 1;
  // CHECK-NEXT:         *_d_i += 1 * j;
  // CHECK-NEXT:         *_d_j += i * 1;
  // CHECK-NEXT:     }
  // CHECK-NEXT: }

  double const_volatile_mem_fn(double i, double j) const volatile {
    return (x+y)*i + i*j;
  }

  // CHECK: void SimpleFunctions::const_volatile_mem_fn_grad(double i, double j, volatile SimpleFunctions *_d_this, double *_d_i, double *_d_j) const volatile {
  // CHECK-NEXT:     double _t0 = (this->x + this->y);
  // CHECK-NEXT:     {
  // CHECK-NEXT:         _d_this->x += 1 * i;
  // CHECK-NEXT:         _d_this->y += 1 * i;
  // CHECK-NEXT:         *_d_i += _t0 * 1;
  // CHECK-NEXT:         *_d_i += 1 * j;
  // CHECK-NEXT:         *_d_j += i * 1;
  // CHECK-NEXT:     }
  // CHECK-NEXT: }

  double lval_ref_mem_fn(double i, double j) & { return (x + y) * i + i * j; }

  // CHECK: void SimpleFunctions::lval_ref_mem_fn_grad(double i, double j, SimpleFunctions *_d_this, double *_d_i, double *_d_j) & {
  // CHECK-NEXT:     {
  // CHECK-NEXT:         _d_this->x += 1 * i;
  // CHECK-NEXT:         _d_this->y += 1 * i;
  // CHECK-NEXT:         *_d_i += (this->x + this->y) * 1;
  // CHECK-NEXT:         *_d_i += 1 * j;
  // CHECK-NEXT:         *_d_j += i * 1;
  // CHECK-NEXT:     }
  // CHECK-NEXT: }

  double const_lval_ref_mem_fn(double i, double j) const & {
    return (x+y)*i + i*j;
  }

  // CHECK: void SimpleFunctions::const_lval_ref_mem_fn_grad(double i, double j, SimpleFunctions *_d_this, double *_d_i, double *_d_j) const & {
  // CHECK-NEXT:     {
  // CHECK-NEXT:         _d_this->x += 1 * i;
  // CHECK-NEXT:         _d_this->y += 1 * i;
  // CHECK-NEXT:         *_d_i += (this->x + this->y) * 1;
  // CHECK-NEXT:         *_d_i += 1 * j;
  // CHECK-NEXT:         *_d_j += i * 1;
  // CHECK-NEXT:     }
  // CHECK-NEXT: }

  double volatile_lval_ref_mem_fn(double i, double j) volatile & {
    return (x+y)*i + i*j;
  }

  // CHECK: void SimpleFunctions::volatile_lval_ref_mem_fn_grad(double i, double j, volatile SimpleFunctions *_d_this, double *_d_i, double *_d_j) volatile & {
  // CHECK-NEXT:     double _t0 = (this->x + this->y);
  // CHECK-NEXT:     {
  // CHECK-NEXT:         _d_this->x += 1 * i;
  // CHECK-NEXT:         _d_this->y += 1 * i;
  // CHECK-NEXT:         *_d_i += _t0 * 1;
  // CHECK-NEXT:         *_d_i += 1 * j;
  // CHECK-NEXT:         *_d_j += i * 1;
  // CHECK-NEXT:     }
  // CHECK-NEXT: }

  double const_volatile_lval_ref_mem_fn(double i, double j) const volatile & {
    return (x+y)*i + i*j;
  }

  // CHECK: void SimpleFunctions::const_volatile_lval_ref_mem_fn_grad(double i, double j, volatile SimpleFunctions *_d_this, double *_d_i, double *_d_j) const volatile & {
  // CHECK-NEXT:     double _t0 = (this->x + this->y);
  // CHECK-NEXT:     {
  // CHECK-NEXT:         _d_this->x += 1 * i;
  // CHECK-NEXT:         _d_this->y += 1 * i;
  // CHECK-NEXT:         *_d_i += _t0 * 1;
  // CHECK-NEXT:         *_d_i += 1 * j;
  // CHECK-NEXT:         *_d_j += i * 1;
  // CHECK-NEXT:     }
  // CHECK-NEXT: }

  double rval_ref_mem_fn(double i, double j) && { return (x + y) * i + i * j; }

  // CHECK: void SimpleFunctions::rval_ref_mem_fn_grad(double i, double j, SimpleFunctions *_d_this, double *_d_i, double *_d_j) && {
  // CHECK-NEXT:     {
  // CHECK-NEXT:         _d_this->x += 1 * i;
  // CHECK-NEXT:         _d_this->y += 1 * i;
  // CHECK-NEXT:         *_d_i += (this->x + this->y) * 1;
  // CHECK-NEXT:         *_d_i += 1 * j;
  // CHECK-NEXT:         *_d_j += i * 1;
  // CHECK-NEXT:     }
  // CHECK-NEXT: }

  double const_rval_ref_mem_fn(double i, double j) const && {
    return (x+y)*i + i*j;
  }

  // CHECK: void SimpleFunctions::const_rval_ref_mem_fn_grad(double i, double j, SimpleFunctions *_d_this, double *_d_i, double *_d_j) const && {
  // CHECK-NEXT:     {
  // CHECK-NEXT:         _d_this->x += 1 * i;
  // CHECK-NEXT:         _d_this->y += 1 * i;
  // CHECK-NEXT:         *_d_i += (this->x + this->y) * 1;
  // CHECK-NEXT:         *_d_i += 1 * j;
  // CHECK-NEXT:         *_d_j += i * 1;
  // CHECK-NEXT:     }
  // CHECK-NEXT: }

  double volatile_rval_ref_mem_fn(double i, double j) volatile && {
    return (x+y)*i + i*j;
  }

  // CHECK: void SimpleFunctions::volatile_rval_ref_mem_fn_grad(double i, double j, volatile SimpleFunctions *_d_this, double *_d_i, double *_d_j) volatile && {
  // CHECK-NEXT:     double _t0 = (this->x + this->y);
  // CHECK-NEXT:     {
  // CHECK-NEXT:         _d_this->x += 1 * i;
  // CHECK-NEXT:         _d_this->y += 1 * i;
  // CHECK-NEXT:         *_d_i += _t0 * 1;
  // CHECK-NEXT:         *_d_i += 1 * j;
  // CHECK-NEXT:         *_d_j += i * 1;
  // CHECK-NEXT:     }
  // CHECK-NEXT: }

  double const_volatile_rval_ref_mem_fn(double i, double j) const volatile && {
    return (x+y)*i + i*j;
  }

  // CHECK: void SimpleFunctions::const_volatile_rval_ref_mem_fn_grad(double i, double j, volatile SimpleFunctions *_d_this, double *_d_i, double *_d_j) const volatile && {
  // CHECK-NEXT:     double _t0 = (this->x + this->y);
  // CHECK-NEXT:     {
  // CHECK-NEXT:         _d_this->x += 1 * i;
  // CHECK-NEXT:         _d_this->y += 1 * i;
  // CHECK-NEXT:         *_d_i += _t0 * 1;
  // CHECK-NEXT:         *_d_i += 1 * j;
  // CHECK-NEXT:         *_d_j += i * 1;
  // CHECK-NEXT:     }
  // CHECK-NEXT: }

  double noexcept_mem_fn(double i, double j) noexcept {
    return (x+y)*i + i*j;
  }

  // CHECK: void SimpleFunctions::noexcept_mem_fn_grad(double i, double j, SimpleFunctions *_d_this, double *_d_i, double *_d_j) noexcept {
  // CHECK-NEXT:     {
  // CHECK-NEXT:         _d_this->x += 1 * i;
  // CHECK-NEXT:         _d_this->y += 1 * i;
  // CHECK-NEXT:         *_d_i += (this->x + this->y) * 1;
  // CHECK-NEXT:         *_d_i += 1 * j;
  // CHECK-NEXT:         *_d_j += i * 1;
  // CHECK-NEXT:     }
  // CHECK-NEXT: }

  double const_noexcept_mem_fn(double i, double j) const noexcept {
    return (x+y)*i + i*j;
  }

  // CHECK: void SimpleFunctions::const_noexcept_mem_fn_grad(double i, double j, SimpleFunctions *_d_this, double *_d_i, double *_d_j) const noexcept {
  // CHECK-NEXT:     {
  // CHECK-NEXT:         _d_this->x += 1 * i;
  // CHECK-NEXT:         _d_this->y += 1 * i;
  // CHECK-NEXT:         *_d_i += (this->x + this->y) * 1;
  // CHECK-NEXT:         *_d_i += 1 * j;
  // CHECK-NEXT:         *_d_j += i * 1;
  // CHECK-NEXT:     }
  // CHECK-NEXT: }

  double volatile_noexcept_mem_fn(double i, double j) volatile noexcept {
    return (x+y)*i + i*j;
  }

  // CHECK: void SimpleFunctions::volatile_noexcept_mem_fn_grad(double i, double j, volatile SimpleFunctions *_d_this, double *_d_i, double *_d_j) volatile noexcept {
  // CHECK-NEXT:     double _t0 = (this->x + this->y);
  // CHECK-NEXT:     {
  // CHECK-NEXT:         _d_this->x += 1 * i;
  // CHECK-NEXT:         _d_this->y += 1 * i;
  // CHECK-NEXT:         *_d_i += _t0 * 1;
  // CHECK-NEXT:         *_d_i += 1 * j;
  // CHECK-NEXT:         *_d_j += i * 1;
  // CHECK-NEXT:     }
  // CHECK-NEXT: }

  double const_volatile_noexcept_mem_fn(double i, double j) const volatile noexcept {
    return (x+y)*i + i*j;
  }

  // CHECK: void SimpleFunctions::const_volatile_noexcept_mem_fn_grad(double i, double j, volatile SimpleFunctions *_d_this, double *_d_i, double *_d_j) const volatile noexcept {
  // CHECK-NEXT:     double _t0 = (this->x + this->y);
  // CHECK-NEXT:     {
  // CHECK-NEXT:         _d_this->x += 1 * i;
  // CHECK-NEXT:         _d_this->y += 1 * i;
  // CHECK-NEXT:         *_d_i += _t0 * 1;
  // CHECK-NEXT:         *_d_i += 1 * j;
  // CHECK-NEXT:         *_d_j += i * 1;
  // CHECK-NEXT:     }
  // CHECK-NEXT: }

  double lval_ref_noexcept_mem_fn(double i, double j) & noexcept {
    return (x+y)*i + i*j;
  }

  // CHECK: void SimpleFunctions::lval_ref_noexcept_mem_fn_grad(double i, double j, SimpleFunctions *_d_this, double *_d_i, double *_d_j) & noexcept {
  // CHECK-NEXT:     {
  // CHECK-NEXT:         _d_this->x += 1 * i;
  // CHECK-NEXT:         _d_this->y += 1 * i;
  // CHECK-NEXT:         *_d_i += (this->x + this->y) * 1;
  // CHECK-NEXT:         *_d_i += 1 * j;
  // CHECK-NEXT:         *_d_j += i * 1;
  // CHECK-NEXT:     }
  // CHECK-NEXT: }

  double const_lval_ref_noexcept_mem_fn(double i, double j) const & noexcept {
    return (x+y)*i + i*j;
  }

  // CHECK: void SimpleFunctions::const_lval_ref_noexcept_mem_fn_grad(double i, double j, SimpleFunctions *_d_this, double *_d_i, double *_d_j) const & noexcept {
  // CHECK-NEXT:     {
  // CHECK-NEXT:         _d_this->x += 1 * i;
  // CHECK-NEXT:         _d_this->y += 1 * i;
  // CHECK-NEXT:         *_d_i += (this->x + this->y) * 1;
  // CHECK-NEXT:         *_d_i += 1 * j;
  // CHECK-NEXT:         *_d_j += i * 1;
  // CHECK-NEXT:     }
  // CHECK-NEXT: }

  double volatile_lval_ref_noexcept_mem_fn(double i, double j) volatile & noexcept {
    return (x+y)*i + i*j;
  }

  // CHECK: void SimpleFunctions::volatile_lval_ref_noexcept_mem_fn_grad(double i, double j, volatile SimpleFunctions *_d_this, double *_d_i, double *_d_j) volatile & noexcept {
  // CHECK-NEXT:     double _t0 = (this->x + this->y);
  // CHECK-NEXT:     {
  // CHECK-NEXT:         _d_this->x += 1 * i;
  // CHECK-NEXT:         _d_this->y += 1 * i;
  // CHECK-NEXT:         *_d_i += _t0 * 1;
  // CHECK-NEXT:         *_d_i += 1 * j;
  // CHECK-NEXT:         *_d_j += i * 1;
  // CHECK-NEXT:     }
  // CHECK-NEXT: }

  double const_volatile_lval_ref_noexcept_mem_fn(double i, double j) const volatile & noexcept {
    return (x+y)*i + i*j;
  }

  // CHECK: void SimpleFunctions::const_volatile_lval_ref_noexcept_mem_fn_grad(double i, double j, volatile SimpleFunctions *_d_this, double *_d_i, double *_d_j) const volatile & noexcept {
  // CHECK-NEXT:     double _t0 = (this->x + this->y);
  // CHECK-NEXT:     {
  // CHECK-NEXT:         _d_this->x += 1 * i;
  // CHECK-NEXT:         _d_this->y += 1 * i;
  // CHECK-NEXT:         *_d_i += _t0 * 1;
  // CHECK-NEXT:         *_d_i += 1 * j;
  // CHECK-NEXT:         *_d_j += i * 1;
  // CHECK-NEXT:     }
  // CHECK-NEXT: }

  double rval_ref_noexcept_mem_fn(double i, double j) && noexcept {
    return (x+y)*i + i*j;
  }

  // CHECK: void SimpleFunctions::rval_ref_noexcept_mem_fn_grad(double i, double j, SimpleFunctions *_d_this, double *_d_i, double *_d_j) && noexcept {
  // CHECK-NEXT:     {
  // CHECK-NEXT:         _d_this->x += 1 * i;
  // CHECK-NEXT:         _d_this->y += 1 * i;
  // CHECK-NEXT:         *_d_i += (this->x + this->y) * 1;
  // CHECK-NEXT:         *_d_i += 1 * j;
  // CHECK-NEXT:         *_d_j += i * 1;
  // CHECK-NEXT:     }
  // CHECK-NEXT: }

  double const_rval_ref_noexcept_mem_fn(double i, double j) const && noexcept {
    return (x+y)*i + i*j;
  }

  // CHECK: void SimpleFunctions::const_rval_ref_noexcept_mem_fn_grad(double i, double j, SimpleFunctions *_d_this, double *_d_i, double *_d_j) const && noexcept {
  // CHECK-NEXT:     {
  // CHECK-NEXT:         _d_this->x += 1 * i;
  // CHECK-NEXT:         _d_this->y += 1 * i;
  // CHECK-NEXT:         *_d_i += (this->x + this->y) * 1;
  // CHECK-NEXT:         *_d_i += 1 * j;
  // CHECK-NEXT:         *_d_j += i * 1;
  // CHECK-NEXT:     }
  // CHECK-NEXT: }

  double volatile_rval_ref_noexcept_mem_fn(double i, double j) volatile && noexcept {
    return (x+y)*i + i*j;
  }

  // CHECK: void SimpleFunctions::volatile_rval_ref_noexcept_mem_fn_grad(double i, double j, volatile SimpleFunctions *_d_this, double *_d_i, double *_d_j) volatile && noexcept {
  // CHECK-NEXT:     double _t0 = (this->x + this->y);
  // CHECK-NEXT:     {
  // CHECK-NEXT:         _d_this->x += 1 * i;
  // CHECK-NEXT:         _d_this->y += 1 * i;
  // CHECK-NEXT:         *_d_i += _t0 * 1;
  // CHECK-NEXT:         *_d_i += 1 * j;
  // CHECK-NEXT:         *_d_j += i * 1;
  // CHECK-NEXT:     }
  // CHECK-NEXT: }

  double const_volatile_rval_ref_noexcept_mem_fn(double i, double j) const volatile && noexcept {
    return (x+y)*i + i*j;
  }

  // CHECK: void SimpleFunctions::const_volatile_rval_ref_noexcept_mem_fn_grad(double i, double j, volatile SimpleFunctions *_d_this, double *_d_i, double *_d_j) const volatile && noexcept {
  // CHECK-NEXT:     double _t0 = (this->x + this->y);
  // CHECK-NEXT:     {
  // CHECK-NEXT:         _d_this->x += 1 * i;
  // CHECK-NEXT:         _d_this->y += 1 * i;
  // CHECK-NEXT:         *_d_i += _t0 * 1;
  // CHECK-NEXT:         *_d_i += 1 * j;
  // CHECK-NEXT:         *_d_j += i * 1;
  // CHECK-NEXT:     }
  // CHECK-NEXT: }

  double partial_mem_fn(double i, double j) { return (x + y) * i + i * j; }

  // CHECK: void partial_mem_fn_grad_0(double i, double j, SimpleFunctions *_d_this, double *_d_i) {
  // CHECK-NEXT:     double _d_j = 0.;
  // CHECK-NEXT:     {
  // CHECK-NEXT:         _d_this->x += 1 * i;
  // CHECK-NEXT:         _d_this->y += 1 * i;
  // CHECK-NEXT:         *_d_i += (this->x + this->y) * 1;
  // CHECK-NEXT:         *_d_i += 1 * j;
  // CHECK-NEXT:         _d_j += i * 1;
  // CHECK-NEXT:     }
  // CHECK-NEXT: }


  static double static_mem_fn(double u, double v) { return u + v; }
  
  // CHECK: static void static_mem_fn_grad(double u, double v, double *_d_u, double *_d_v) {
  // CHECK-NEXT:     {
  // CHECK-NEXT:         *_d_u += 1;
  // CHECK-NEXT:         *_d_v += 1;
  // CHECK-NEXT:     }
  // CHECK-NEXT: }

  double& ref_mem_fn(double i) {
    x = +i;
    x = -i;
    return x;
  }
  SimpleFunctions& operator+=(double value) {
    x += value;
    return *this;
  }
  SimpleFunctions& operator++() {
    x += 1.0;
    return *this;
  }

  void mem_fn_grad(double i, double j, double *_d_i, double *_d_j);
  void const_mem_fn_grad(double i, double j, double *_d_i, double *_d_j);
  void volatile_mem_fn_grad(double i, double j, double *_d_i, double *_d_j);
  void const_volatile_mem_fn_grad(double i, double j, double *_d_i, double *_d_j);
  void lval_ref_mem_fn_grad(double i, double j, double *_d_i, double *_d_j);
  void const_lval_ref_mem_fn_grad(double i, double j, double *_d_i, double *_d_j);
  void volatile_lval_ref_mem_fn_grad(double i, double j, double *_d_i, double *_d_j);
  void const_volatile_lval_ref_mem_fn_grad(double i, double j, double *_d_i, double *_d_j);
  void rval_ref_mem_fn_grad(double i, double j, double *_d_i, double *_d_j);
  void const_rval_ref_mem_fn_grad(double i, double j, double *_d_i, double *_d_j);
  void volatile_rval_ref_mem_fn_grad(double i, double j, double *_d_i, double *_d_j);
  void const_volatile_rval_ref_mem_fn_grad(double i, double j, double *_d_i, double *_d_j);
  void noexcept_mem_fn_grad(double i, double j, double *_d_i, double *_d_j);
  void const_noexcept_mem_fn_grad(double i, double j, double *_d_i, double *_d_j);
  void volatile_noexcept_mem_fn_grad(double i, double j, double *_d_i, double *_d_j);
  void const_volatile_noexcept_mem_fn_grad(double i, double j, double *_d_i, double *_d_j);
  void lval_ref_noexcept_mem_fn_grad(double i, double j, double *_d_i, double *_d_j);
  void const_lval_ref_noexcept_mem_fn_grad(double i, double j, double *_d_i, double *_d_j);
  void volatile_lval_ref_noexcept_mem_fn_grad(double i, double j, double *_d_i, double *_d_j);
  void const_volatile_lval_ref_noexcept_mem_fn_grad(double i, double j, double *_d_i, double *_d_j);
  void rval_ref_noexcept_mem_fn_grad(double i, double j, double *_d_i, double *_d_j);
  void const_rval_ref_noexcept_mem_fn_grad(double i, double j, double *_d_i, double *_d_j);
  void volatile_rval_ref_noexcept_mem_fn_grad(double i, double j, double *_d_i, double *_d_j);
  void const_volatile_rval_ref_noexcept_mem_fn_grad(double i, double j, double *_d_i, double *_d_j);
  void partial_mem_fn_grad(double i, double j, double *_d_i);
};

double fn(double i,double j) {
  return i*i*j;
}

// CHECK: void fn_grad(double i, double j, double *_d_i, double *_d_j) {
// CHECK-NEXT:     {
// CHECK-NEXT:         *_d_i += 1 * j * i;
// CHECK-NEXT:         *_d_i += i * 1 * j;
// CHECK-NEXT:         *_d_j += i * i * 1;
// CHECK-NEXT:     }
// CHECK-NEXT: }

double fn2(SimpleFunctions& sf, double i) {
  return sf.ref_mem_fn(i);
}

// CHECK: void ref_mem_fn_pullback(double i, double _d_y, SimpleFunctions *_d_this, double *_d_i) {
// CHECK-NEXT:     double _t0 = this->x;
// CHECK-NEXT:     this->x = +i;
// CHECK-NEXT:     double _t1 = this->x;
// CHECK-NEXT:     this->x = -i;
// CHECK-NEXT:     _d_this->x += _d_y;
// CHECK-NEXT:     {
// CHECK-NEXT:         this->x = _t1;
// CHECK-NEXT:         double _r_d1 = _d_this->x;
// CHECK-NEXT:         _d_this->x = 0.;
// CHECK-NEXT:         *_d_i += -_r_d1;
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         this->x = _t0;
// CHECK-NEXT:         double _r_d0 = _d_this->x;
// CHECK-NEXT:         _d_this->x = 0.;
// CHECK-NEXT:         *_d_i += _r_d0;
// CHECK-NEXT:     }
// CHECK-NEXT: }

// CHECK: clad::ValueAndAdjoint<double &, double &> ref_mem_fn_forw(double i, SimpleFunctions *_d_this, double _d_i);

// CHECK: void fn2_grad(SimpleFunctions &sf, double i, SimpleFunctions *_d_sf, double *_d_i) {
// CHECK-NEXT:     SimpleFunctions _t0 = sf;
// CHECK-NEXT:     clad::ValueAndAdjoint<double &, double &> _t1 = sf.ref_mem_fn_forw(i, &(*_d_sf), 0.);
// CHECK-NEXT:     {
// CHECK-NEXT:         double _r0 = 0.;
// CHECK-NEXT:         sf = _t0;
// CHECK-NEXT:         sf.ref_mem_fn_pullback(i, 1, &(*_d_sf), &_r0);
// CHECK-NEXT:         *_d_i += _r0;
// CHECK-NEXT:     }
// CHECK-NEXT: }

double fn3(double x, double y, double i, double j) {
  SimpleFunctions sf(x, y);
  return sf.mem_fn(i, j);
}

double fn5(SimpleFunctions& v, double value) {
  v += value;
  return v.x;
}

// CHECK: void operator_plus_equal_pullback(double value, SimpleFunctions _d_y, SimpleFunctions *_d_this, double *_d_value) {
// CHECK-NEXT:     double _t0 = this->x;
// CHECK-NEXT:     this->x += value;
// CHECK-NEXT:     {
// CHECK-NEXT:         this->x = _t0;
// CHECK-NEXT:         double _r_d0 = _d_this->x;
// CHECK-NEXT:         *_d_value += _r_d0;
// CHECK-NEXT:     }
// CHECK-NEXT: }

// CHECK: clad::ValueAndAdjoint<SimpleFunctions &, SimpleFunctions &> operator_plus_equal_forw(double value, SimpleFunctions *_d_this, double _d_value);

// CHECK: void fn5_grad(SimpleFunctions &v, double value, SimpleFunctions *_d_v, double *_d_value) {
// CHECK-NEXT:     SimpleFunctions _t0 = v;
// CHECK-NEXT:     clad::ValueAndAdjoint<SimpleFunctions &, SimpleFunctions &> _t1 = v.operator_plus_equal_forw(value, &(*_d_v), 0.);
// CHECK-NEXT:     (*_d_v).x += 1;
// CHECK-NEXT:     {
// CHECK-NEXT:         double _r0 = 0.;
// CHECK-NEXT:         v = _t0;
// CHECK-NEXT:         v.operator_plus_equal_pullback(value, {}, &(*_d_v), &_r0);
// CHECK-NEXT:         *_d_value += _r0;
// CHECK-NEXT:     }
// CHECK-NEXT: }

double fn4(SimpleFunctions& v) {
  ++v;
  return v.x;
}

// CHECK: void operator_plus_plus_pullback(SimpleFunctions _d_y, SimpleFunctions *_d_this) {
// CHECK-NEXT:     double _t0 = this->x;
// CHECK-NEXT:     this->x += 1.;
// CHECK-NEXT:     {
// CHECK-NEXT:         this->x = _t0;
// CHECK-NEXT:         double _r_d0 = _d_this->x;
// CHECK-NEXT:     }
// CHECK-NEXT: }

// CHECK: clad::ValueAndAdjoint<SimpleFunctions &, SimpleFunctions &> operator_plus_plus_forw(SimpleFunctions *_d_this);

// CHECK: void fn4_grad(SimpleFunctions &v, SimpleFunctions *_d_v) {
// CHECK-NEXT:     SimpleFunctions _t0 = v;
// CHECK-NEXT:     clad::ValueAndAdjoint<SimpleFunctions &, SimpleFunctions &> _t1 = v.operator_plus_plus_forw(&(*_d_v));
// CHECK-NEXT:     (*_d_v).x += 1;
// CHECK-NEXT:     {
// CHECK-NEXT:         v = _t0;
// CHECK-NEXT:         v.operator_plus_plus_pullback({}, &(*_d_v));
// CHECK-NEXT:     }
// CHECK-NEXT: }

class SafeTestClass {
    public:
    SafeTestClass() {};
    SafeTestClass(double &x) {
    }
    SafeTestClass(double x, double* y) {
        *y = x;
    }
};

namespace clad {
namespace custom_derivatives {
namespace class_functions {
    clad::ValueAndAdjoint<SafeTestClass, SafeTestClass>
    constructor_reverse_forw(clad::ConstructorReverseForwTag<SafeTestClass>, double x, double* y, double d_x, double* d_y) {
        return {SafeTestClass(x, y), SafeTestClass(d_x, d_y)};
    }
    clad::ValueAndAdjoint<SafeTestClass, SafeTestClass>
    constructor_reverse_forw(clad::ConstructorReverseForwTag<SafeTestClass>, double &x, double &d_x) {
        return {SafeTestClass(x), SafeTestClass(d_x)};
    }
    clad::ValueAndAdjoint<SafeTestClass, SafeTestClass>
    constructor_reverse_forw(clad::ConstructorReverseForwTag<SafeTestClass>) {
        return {SafeTestClass(), SafeTestClass()};
    }
    void constructor_pullback(double x, double* y, SafeTestClass *d_this, double* d_x, double* d_y) {
        *d_x += *d_y;
        *d_y = 0;
    }
}}}

double fn6(double u, double v) {
    double &w = u;
    SafeTestClass s1;
    SafeTestClass s2(u, &v);
    SafeTestClass s3(w);
    return v;
}

// CHECK: static void constructor_pullback(double &x, SafeTestClass *_d_this, double *_d_x) {
// CHECK-NEXT: }

// CHECK: void fn6_grad(double u, double v, double *_d_u, double *_d_v) {
// CHECK-NEXT:      double &_d_w = *_d_u;
// CHECK-NEXT:      double &w = u;
// CHECK-NEXT:      clad::ValueAndAdjoint<SafeTestClass, SafeTestClass> _t0 = {{.*}}constructor_reverse_forw(clad::ConstructorReverseForwTag<SafeTestClass>());
// CHECK-NEXT:      SafeTestClass s1(_t0.value);
// CHECK-NEXT:      SafeTestClass _d_s1 = _t0.adjoint;
// CHECK-NEXT:      clad::ValueAndAdjoint<SafeTestClass, SafeTestClass> _t1 = {{.*}}constructor_reverse_forw(clad::ConstructorReverseForwTag<SafeTestClass>(), u, &v, 0., &*_d_v);
// CHECK-NEXT:      SafeTestClass s2(_t1.value);
// CHECK-NEXT:      SafeTestClass _d_s2 = _t1.adjoint;
// CHECK-NEXT:      clad::ValueAndAdjoint<SafeTestClass, SafeTestClass> _t2 = {{.*}}constructor_reverse_forw(clad::ConstructorReverseForwTag<SafeTestClass>(), w, _d_w);
// CHECK-NEXT:      SafeTestClass s3(_t2.value);
// CHECK-NEXT:      SafeTestClass _d_s3 = _t2.adjoint;
// CHECK-NEXT:      *_d_v += 1;
// CHECK-NEXT:      SafeTestClass::constructor_pullback(w, &_d_s3, &_d_w);
// CHECK-NEXT:      {
// CHECK-NEXT:          double _r0 = 0.;  
// CHECK-NEXT:          {{.*}}constructor_pullback(u, &v, &_d_s2, &_r0, &*_d_v);
// CHECK-NEXT:          *_d_u += _r0;
// CHECK-NEXT:      }
// CHECK-NEXT:  }

double fn7(double u, double v) {
  return SimpleFunctions::static_mem_fn(u, v);
}

struct S {
  double val;
  bool cond;
  bool Cond(const double& t) const {
    return t > 0 || cond;
  }
  double getVal() const{
    return val;
  }
  
  S operator-(const double& x) const {
    return {val - x, cond};
  }
};

double fn8(double x, double y) {
  S s = {x, false};
  if (s.Cond(y))
    return 0;
  return y;
}

// CHECK:  void fn8_grad(double x, double y, double *_d_x, double *_d_y) {
// CHECK-NEXT:      S _d_s = {0., false};
// CHECK-NEXT:      S s = {x, false};
// CHECK-NEXT:      if (s.Cond(y))
// CHECK-NEXT:          goto _label0;
// CHECK-NEXT:      *_d_y += 1;
// CHECK-NEXT:      if (s.Cond(y))
// CHECK-NEXT:        _label0:
// CHECK-NEXT:          ;
// CHECK-NEXT:      *_d_x += _d_s.val;
// CHECK-NEXT:  }

double fn9(double x, double y) {
  S* s = new S{x, false};
  return s->getVal();
}

// CHECK:  void fn9_grad(double x, double y, double *_d_x, double *_d_y) {
// CHECK-NEXT:      S *_d_s = new S();
// CHECK-NEXT:      S *s = new S({x, false});
// CHECK-NEXT:      s->getVal_pullback(1, _d_s);
// CHECK-NEXT:      *_d_x += *_d_s.val;
// CHECK-NEXT:  }

// CHECK:  void operator_minus_pullback(const double &x, S _d_y, S *_d_this, double *_d_x) const {
// CHECK-NEXT:      {
// CHECK-NEXT:          _d_this->val += _d_y.val;
// CHECK-NEXT:          *_d_x += -_d_y.val;
// CHECK-NEXT:          _d_this->cond += _d_y.cond;
// CHECK-NEXT:      }
// CHECK-NEXT:  }

double fn10(double x, double y) {
  S s = {x, false};
  return ((s - 4*x) - y).getVal();
}

// CHECK:  void fn10_grad(double x, double y, double *_d_x, double *_d_y) {
// CHECK-NEXT:      S _d_s = {0., false};
// CHECK-NEXT:      S s = {x, false};
// CHECK-NEXT:      {
// CHECK-NEXT:          S _r0 = {0., false};
// CHECK-NEXT:          ((s - 4 * x) - y).getVal_pullback(1, &_r0);
// CHECK-NEXT:          S _r1 = {0., false};
// CHECK-NEXT:          (s - 4 * x).operator_minus_pullback(y, _r0, &_r1, &*_d_y);
// CHECK-NEXT:          double _r2 = 0.;
// CHECK-NEXT:          s.operator_minus_pullback(4 * x, _r1, &_d_s, &_r2);
// CHECK-NEXT:          *_d_x += 4 * _r2;
// CHECK-NEXT:      }
// CHECK-NEXT:      *_d_x += _d_s.val;
// CHECK-NEXT:  }

class A {
public:
  void increment() { data++; }
  void setData(double u) { data = u; }
  double data = 0;
};

// CHECK:  void setData_pullback(double u, A *_d_this, double *_d_u) {
// CHECK-NEXT:      double _t0 = this->data;
// CHECK-NEXT:      this->data = u;
// CHECK-NEXT:      {
// CHECK-NEXT:          this->data = _t0;
// CHECK-NEXT:          double _r_d0 = _d_this->data;
// CHECK-NEXT:          _d_this->data = 0.;
// CHECK-NEXT:          *_d_u += _r_d0;
// CHECK-NEXT:      }
// CHECK-NEXT:  }

// CHECK:  void increment_pullback(A *_d_this) {
// CHECK-NEXT:      this->data++;
// CHECK-NEXT:      this->data--;
// CHECK-NEXT:  }

double fn11(double u, double v) {
  double res = 0;
  A a;
  a.setData(u);
  res += a.data * v;
  a.increment();
  return res;
}

// CHECK:  void fn11_grad(double u, double v, double *_d_u, double *_d_v) {
// CHECK-NEXT:      double _d_res = 0.;
// CHECK-NEXT:      double res = 0;
// CHECK-NEXT:      A _d_a = {0.};
// CHECK-NEXT:      A a;
// CHECK-NEXT:      A _t0 = a;
// CHECK-NEXT:      a.setData(u);
// CHECK-NEXT:      double _t1 = res;
// CHECK-NEXT:      res += a.data * v;
// CHECK-NEXT:      A _t2 = a;
// CHECK-NEXT:      a.increment();
// CHECK-NEXT:      _d_res += 1;
// CHECK-NEXT:      {
// CHECK-NEXT:          a = _t2;
// CHECK-NEXT:          a.increment_pullback(&_d_a);
// CHECK-NEXT:      }
// CHECK-NEXT:      {
// CHECK-NEXT:          res = _t1;
// CHECK-NEXT:          double _r_d0 = _d_res;
// CHECK-NEXT:          _d_a.data += _r_d0 * v;
// CHECK-NEXT:          *_d_v += a.data * _r_d0;
// CHECK-NEXT:      }
// CHECK-NEXT:      {
// CHECK-NEXT:          double _r0 = 0.;
// CHECK-NEXT:          a = _t0;
// CHECK-NEXT:          a.setData_pullback(u, &_d_a, &_r0);
// CHECK-NEXT:          *_d_u += _r0;
// CHECK-NEXT:      }
// CHECK-NEXT:  }

struct B {
  float m = 0;
  void scale(const float* in, float* out) const {
    out[0] = in[0] * m;
  }
};

float fn12(const B b, const float* in) {
  float res = 0;
  b.scale(in, &res);
  return res + 3;
}

// CHECK:  void fn12_grad_0(const B b, const float *in, B *_d_b) {
// CHECK-NEXT:      float _d_res = 0.F;
// CHECK-NEXT:      float res = 0;
// CHECK-NEXT:      b.scale(in, &res);
// CHECK-NEXT:      _d_res += 1;
// CHECK-NEXT:      b.scale_pullback(in, &res, &(*_d_b), &_d_res);
// CHECK-NEXT:  }

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
  auto d_static_mem_fn = clad::gradient(&SimpleFunctions::static_mem_fn);

  auto d_fn = clad::gradient(fn);
  double result[2] = {};
  d_fn.execute(4, 5, &result[0], &result[1]);
  for(unsigned i=0;i<2;++i) {
    printf("%.2f ",result[i]);  //CHECK-EXEC: 40.00 16.00
  }

  SimpleFunctions sf1(2, 3), sf2(3, 4), sf3(4, 5);
  SimpleFunctions d_sf;

  auto d_fn2 = clad::gradient(fn2);
  d_fn2.execute(sf1, 2, &d_sf, &result[0]);
  printf("%.2f", result[0]); //CHECK-EXEC: 39.00

  auto d_fn5 = clad::gradient(fn5);
  d_fn5.execute(sf2, 3, &d_sf, &result[0]);
  printf("%.2f", result[0]); //CHECK-EXEC: 40.00

  auto d_fn4 = clad::gradient(fn4);
  d_fn4.execute(sf3, &d_sf);
  printf("%.2f", d_sf.x); //CHECK-EXEC: 2.00

  double dx = 0, dy = 0;
  auto d_fn6 = clad::gradient(fn6);
  d_fn6.execute(3, 5, &dx, &dy);
  printf("%.2f", dx); //CHECK-EXEC: 1.00
  printf("%.2f", dy); //CHECK-EXEC: 0.00
  
  double du = 0, dv = 0;
  auto d_fn7 = clad::gradient(fn7);
  d_fn7.execute(3, 5, &du, &dv);
  printf("%.2f", du); //CHECK-EXEC: 1.00
  printf("%.2f", dv); //CHECK-EXEC: 1.00

  dx = 0, dy = 0;
  auto d_fn8 = clad::gradient(fn8);
  d_fn8.execute(3, -5, &dx, &dy);
  printf("%.2f", dx); //CHECK-EXEC: 0.00
  printf("%.2f", dy); //CHECK-EXEC: 1.00

  dx = 0, dy = 0;
  auto d_fn9 = clad::gradient(fn9);
  d_fn9.execute(3, -5, &dx, &dy);
  printf("%.2f", dx); //CHECK-EXEC: 1.00
  printf("%.2f", dy); //CHECK-EXEC: 0.00

  dx = 0, dy = 0;
  auto d_fn10 = clad::gradient(fn10);
  d_fn10.execute(3, -5, &dx, &dy);
  printf("%.2f", dx); //CHECK-EXEC: -3.00
  printf("%.2f", dy); //CHECK-EXEC: -1.00

  dx = 0, dy = 0;
  auto d_fn11 = clad::gradient(fn11);
  d_fn11.execute(3, 5, &dx, &dy);
  printf("%.2f", dx); //CHECK-EXEC: 5.00
  printf("%.2f", dy); //CHECK-EXEC: 3.00
  
  B b{3}, d_b{0};
  float in = 2.0f;
  auto d_fn12 = clad::gradient(fn12, "0");
  d_fn12.execute(b, &in, &d_b);
  printf("%.2f", d_b.m); //CHECK-EXEC: 2.00
  
  auto d_const_volatile_lval_ref_mem_fn_i = clad::gradient(&SimpleFunctions::const_volatile_lval_ref_mem_fn, "i");

  // CHECK:   void const_volatile_lval_ref_mem_fn_grad_0(double i, double j, volatile SimpleFunctions *_d_this, double *_d_i) const volatile & {
  // CHECK-NEXT:       double _d_j = 0.;
  // CHECK-NEXT:       double _t0 = (this->x + this->y);
  // CHECK-NEXT:       {
  // CHECK-NEXT:           _d_this->x += 1 * i;
  // CHECK-NEXT:           _d_this->y += 1 * i;
  // CHECK-NEXT:           *_d_i += _t0 * 1;
  // CHECK-NEXT:           *_d_i += 1 * j;
  // CHECK-NEXT:           _d_j += i * 1;
  // CHECK-NEXT:       }
  // CHECK-NEXT:   }

  auto d_const_volatile_rval_ref_mem_fn_j = clad::gradient(&SimpleFunctions::const_volatile_rval_ref_mem_fn, "j");

  // CHECK:   void const_volatile_rval_ref_mem_fn_grad_1(double i, double j, volatile SimpleFunctions *_d_this, double *_d_j) const volatile && {
  // CHECK-NEXT:       double _d_i = 0.;
  // CHECK-NEXT:       double _t0 = (this->x + this->y);
  // CHECK-NEXT:       {
  // CHECK-NEXT:           _d_this->x += 1 * i;
  // CHECK-NEXT:           _d_this->y += 1 * i;
  // CHECK-NEXT:           _d_i += _t0 * 1;
  // CHECK-NEXT:           _d_i += 1 * j;
  // CHECK-NEXT:           *_d_j += i * 1;
  // CHECK-NEXT:       }
  // CHECK-NEXT:   }

  auto d_fn3 = clad::gradient(fn3, "i,j");
  result[0] = result[1] = 0;
  d_fn3.execute(2, 3, 4, 5, &result[0], &result[1]);
  printf("%.2f %.2f", result[0], result[1]); // CHECK-EXEC: 10.00 4.00

// CHECK: static void constructor_pullback(double p_x, double p_y, SimpleFunctions *_d_this, double *_d_p_x, double *_d_p_y) {
// CHECK-NEXT:     {
// CHECK-NEXT:         *_d_p_y += _d_this->y;
// CHECK-NEXT:         _d_this->y = 0.;
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         *_d_p_x += _d_this->x;
// CHECK-NEXT:         _d_this->x = 0.;
// CHECK-NEXT:     }
// CHECK-NEXT: }

// CHECK: void fn3_grad_2_3(double x, double y, double i, double j, double *_d_i, double *_d_j) {
// CHECK-NEXT:     double _d_x = 0.;
// CHECK-NEXT:     double _d_y = 0.;
// CHECK-NEXT:     SimpleFunctions sf(x, y);
// CHECK-NEXT:     SimpleFunctions _d_sf(sf);
// CHECK-NEXT:     clad::zero_init(_d_sf);
// CHECK-NEXT:     SimpleFunctions _t0 = sf;
// CHECK-NEXT:     {
// CHECK-NEXT:         double _r2 = 0.;
// CHECK-NEXT:         double _r3 = 0.;
// CHECK-NEXT:         sf = _t0;
// CHECK-NEXT:         sf.mem_fn_pullback(i, j, 1, &_d_sf, &_r2, &_r3);
// CHECK-NEXT:         *_d_i += _r2;
// CHECK-NEXT:         *_d_j += _r3;
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         double _r0 = 0.;
// CHECK-NEXT:         double _r1 = 0.;
// CHECK-NEXT:         SimpleFunctions::constructor_pullback(x, y, &_d_sf, &_r0, &_r1);
// CHECK-NEXT:         _d_x += _r0;
// CHECK-NEXT:         _d_y += _r1;
// CHECK-NEXT:     }
// CHECK-NEXT:     }

// CHECK: clad::ValueAndAdjoint<double &, double &> ref_mem_fn_forw(double i, SimpleFunctions *_d_this, double _d_i) {
// CHECK-NEXT:     this->x = +i;
// CHECK-NEXT:     this->x = -i;
// CHECK-NEXT:     return {this->x, _d_this->x};
// CHECK-NEXT: }

// CHECK: clad::ValueAndAdjoint<SimpleFunctions &, SimpleFunctions &> operator_plus_equal_forw(double value, SimpleFunctions *_d_this, double _d_value) {
// CHECK-NEXT:     this->x += value;
// CHECK-NEXT:     return {*this, *_d_this};
// CHECK-NEXT: }

// CHECK: clad::ValueAndAdjoint<SimpleFunctions &, SimpleFunctions &> operator_plus_plus_forw(SimpleFunctions *_d_this) {
// CHECK-NEXT:     this->x += 1.;
// CHECK-NEXT:     return {*this, *_d_this};
// CHECK-NEXT: }
}

// CHECK:  void scale_pullback(const float *in, float *out, B *_d_this, float *_d_out) const {
// CHECK-NEXT:      float _t0 = out[0];
// CHECK-NEXT:      out[0] = in[0] * this->m;
// CHECK-NEXT:      {
// CHECK-NEXT:          out[0] = _t0;
// CHECK-NEXT:          float _r_d0 = _d_out[0];
// CHECK-NEXT:          _d_out[0] = 0.F;
// CHECK-NEXT:          _d_this->m += in[0] * _r_d0;
// CHECK-NEXT:      }
// CHECK-NEXT:  }
