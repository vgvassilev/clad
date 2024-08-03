// RUN: %cladclang %s -fno-exceptions -I%S/../../include -oMemberFunctions.out 2>&1 | %filecheck %s
// RUN: ./MemberFunctions.out | %filecheck_exec %s
// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -enable-tbr %s -fno-exceptions -I%S/../../include -oMemberFunctions.out
// RUN: ./MemberFunctions.out | %filecheck_exec %s

// RUN: %cladclang -std=c++14 %s -fno-exceptions -I%S/../../include -oMemberFunctions-cpp14.out 2>&1 | %filecheck %s
// RUN: ./MemberFunctions-cpp14.out | %filecheck_exec %s
// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -enable-tbr -std=c++14 %s -fno-exceptions -I%S/../../include -oMemberFunctions-cpp14.out
// RUN: ./MemberFunctions-cpp14.out | %filecheck_exec %s

// RUN: %cladclang -std=c++17 %s -fno-exceptions -I%S/../../include -oMemberFunctions-cpp17.out 2>&1 | %filecheck %s
// RUN: ./MemberFunctions-cpp17.out | %filecheck_exec %s
// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -enable-tbr -std=c++17 %s -fno-exceptions -I%S/../../include -oMemberFunctions-cpp17.out
// RUN: ./MemberFunctions-cpp17.out | %filecheck_exec %s

//CHECK-NOT: {{.*error|warning|note:.*}}
#include "clad/Differentiator/Differentiator.h"

class SimpleFunctions {
public:
  SimpleFunctions(double p_x = 0, double p_y = 0) : x(p_x), y(p_y) {}
  double x, y;
  double mem_fn(double i, double j) { return (x + y) * i + i * j; }

  // CHECK: void mem_fn_grad(double i, double j, SimpleFunctions *_d_this, double *_d_i, double *_d_j) {
  // CHECK-NEXT:     {
  // CHECK-NEXT:         (*_d_this).x += 1 * i;
  // CHECK-NEXT:         (*_d_this).y += 1 * i;
  // CHECK-NEXT:         *_d_i += (this->x + this->y) * 1;
  // CHECK-NEXT:         *_d_i += 1 * j;
  // CHECK-NEXT:         *_d_j += i * 1;
  // CHECK-NEXT:     }
  // CHECK-NEXT: }

  double const_mem_fn(double i, double j) const { return (x + y) * i + i * j; }

  // CHECK: void const_mem_fn_grad(double i, double j, SimpleFunctions *_d_this, double *_d_i, double *_d_j) const {
  // CHECK-NEXT:     {
  // CHECK-NEXT:         (*_d_this).x += 1 * i;
  // CHECK-NEXT:         (*_d_this).y += 1 * i;
  // CHECK-NEXT:         *_d_i += (this->x + this->y) * 1;
  // CHECK-NEXT:         *_d_i += 1 * j;
  // CHECK-NEXT:         *_d_j += i * 1;
  // CHECK-NEXT:     }
  // CHECK-NEXT: }

  double volatile_mem_fn(double i, double j) volatile {
    return (x+y)*i + i*j;
  }

  // CHECK: void volatile_mem_fn_grad(double i, double j, volatile SimpleFunctions *_d_this, double *_d_i, double *_d_j) volatile {
  // CHECK-NEXT:     double _t0 = (this->x + this->y);
  // CHECK-NEXT:     {
  // CHECK-NEXT:         (*_d_this).x += 1 * i;
  // CHECK-NEXT:         (*_d_this).y += 1 * i;
  // CHECK-NEXT:         *_d_i += _t0 * 1;
  // CHECK-NEXT:         *_d_i += 1 * j;
  // CHECK-NEXT:         *_d_j += i * 1;
  // CHECK-NEXT:     }
  // CHECK-NEXT: }

  double const_volatile_mem_fn(double i, double j) const volatile {
    return (x+y)*i + i*j;
  }

  // CHECK: void const_volatile_mem_fn_grad(double i, double j, volatile SimpleFunctions *_d_this, double *_d_i, double *_d_j) const volatile {
  // CHECK-NEXT:     double _t0 = (this->x + this->y);
  // CHECK-NEXT:     {
  // CHECK-NEXT:         (*_d_this).x += 1 * i;
  // CHECK-NEXT:         (*_d_this).y += 1 * i;
  // CHECK-NEXT:         *_d_i += _t0 * 1;
  // CHECK-NEXT:         *_d_i += 1 * j;
  // CHECK-NEXT:         *_d_j += i * 1;
  // CHECK-NEXT:     }
  // CHECK-NEXT: }

  double lval_ref_mem_fn(double i, double j) & { return (x + y) * i + i * j; }

  // CHECK: void lval_ref_mem_fn_grad(double i, double j, SimpleFunctions *_d_this, double *_d_i, double *_d_j) & {
  // CHECK-NEXT:     {
  // CHECK-NEXT:         (*_d_this).x += 1 * i;
  // CHECK-NEXT:         (*_d_this).y += 1 * i;
  // CHECK-NEXT:         *_d_i += (this->x + this->y) * 1;
  // CHECK-NEXT:         *_d_i += 1 * j;
  // CHECK-NEXT:         *_d_j += i * 1;
  // CHECK-NEXT:     }
  // CHECK-NEXT: }

  double const_lval_ref_mem_fn(double i, double j) const & {
    return (x+y)*i + i*j;
  }

  // CHECK: void const_lval_ref_mem_fn_grad(double i, double j, SimpleFunctions *_d_this, double *_d_i, double *_d_j) const & {
  // CHECK-NEXT:     {
  // CHECK-NEXT:         (*_d_this).x += 1 * i;
  // CHECK-NEXT:         (*_d_this).y += 1 * i;
  // CHECK-NEXT:         *_d_i += (this->x + this->y) * 1;
  // CHECK-NEXT:         *_d_i += 1 * j;
  // CHECK-NEXT:         *_d_j += i * 1;
  // CHECK-NEXT:     }
  // CHECK-NEXT: }

  double volatile_lval_ref_mem_fn(double i, double j) volatile & {
    return (x+y)*i + i*j;
  }

  // CHECK: void volatile_lval_ref_mem_fn_grad(double i, double j, volatile SimpleFunctions *_d_this, double *_d_i, double *_d_j) volatile & {
  // CHECK-NEXT:     double _t0 = (this->x + this->y);
  // CHECK-NEXT:     {
  // CHECK-NEXT:         (*_d_this).x += 1 * i;
  // CHECK-NEXT:         (*_d_this).y += 1 * i;
  // CHECK-NEXT:         *_d_i += _t0 * 1;
  // CHECK-NEXT:         *_d_i += 1 * j;
  // CHECK-NEXT:         *_d_j += i * 1;
  // CHECK-NEXT:     }
  // CHECK-NEXT: }

  double const_volatile_lval_ref_mem_fn(double i, double j) const volatile & {
    return (x+y)*i + i*j;
  }

  // CHECK: void const_volatile_lval_ref_mem_fn_grad(double i, double j, volatile SimpleFunctions *_d_this, double *_d_i, double *_d_j) const volatile & {
  // CHECK-NEXT:     double _t0 = (this->x + this->y);
  // CHECK-NEXT:     {
  // CHECK-NEXT:         (*_d_this).x += 1 * i;
  // CHECK-NEXT:         (*_d_this).y += 1 * i;
  // CHECK-NEXT:         *_d_i += _t0 * 1;
  // CHECK-NEXT:         *_d_i += 1 * j;
  // CHECK-NEXT:         *_d_j += i * 1;
  // CHECK-NEXT:     }
  // CHECK-NEXT: }

  double rval_ref_mem_fn(double i, double j) && { return (x + y) * i + i * j; }

  // CHECK: void rval_ref_mem_fn_grad(double i, double j, SimpleFunctions *_d_this, double *_d_i, double *_d_j) && {
  // CHECK-NEXT:     {
  // CHECK-NEXT:         (*_d_this).x += 1 * i;
  // CHECK-NEXT:         (*_d_this).y += 1 * i;
  // CHECK-NEXT:         *_d_i += (this->x + this->y) * 1;
  // CHECK-NEXT:         *_d_i += 1 * j;
  // CHECK-NEXT:         *_d_j += i * 1;
  // CHECK-NEXT:     }
  // CHECK-NEXT: }

  double const_rval_ref_mem_fn(double i, double j) const && {
    return (x+y)*i + i*j;
  }

  // CHECK: void const_rval_ref_mem_fn_grad(double i, double j, SimpleFunctions *_d_this, double *_d_i, double *_d_j) const && {
  // CHECK-NEXT:     {
  // CHECK-NEXT:         (*_d_this).x += 1 * i;
  // CHECK-NEXT:         (*_d_this).y += 1 * i;
  // CHECK-NEXT:         *_d_i += (this->x + this->y) * 1;
  // CHECK-NEXT:         *_d_i += 1 * j;
  // CHECK-NEXT:         *_d_j += i * 1;
  // CHECK-NEXT:     }
  // CHECK-NEXT: }

  double volatile_rval_ref_mem_fn(double i, double j) volatile && {
    return (x+y)*i + i*j;
  }

  // CHECK: void volatile_rval_ref_mem_fn_grad(double i, double j, volatile SimpleFunctions *_d_this, double *_d_i, double *_d_j) volatile && {
  // CHECK-NEXT:     double _t0 = (this->x + this->y);
  // CHECK-NEXT:     {
  // CHECK-NEXT:         (*_d_this).x += 1 * i;
  // CHECK-NEXT:         (*_d_this).y += 1 * i;
  // CHECK-NEXT:         *_d_i += _t0 * 1;
  // CHECK-NEXT:         *_d_i += 1 * j;
  // CHECK-NEXT:         *_d_j += i * 1;
  // CHECK-NEXT:     }
  // CHECK-NEXT: }

  double const_volatile_rval_ref_mem_fn(double i, double j) const volatile && {
    return (x+y)*i + i*j;
  }

  // CHECK: void const_volatile_rval_ref_mem_fn_grad(double i, double j, volatile SimpleFunctions *_d_this, double *_d_i, double *_d_j) const volatile && {
  // CHECK-NEXT:     double _t0 = (this->x + this->y);
  // CHECK-NEXT:     {
  // CHECK-NEXT:         (*_d_this).x += 1 * i;
  // CHECK-NEXT:         (*_d_this).y += 1 * i;
  // CHECK-NEXT:         *_d_i += _t0 * 1;
  // CHECK-NEXT:         *_d_i += 1 * j;
  // CHECK-NEXT:         *_d_j += i * 1;
  // CHECK-NEXT:     }
  // CHECK-NEXT: }

  double noexcept_mem_fn(double i, double j) noexcept {
    return (x+y)*i + i*j;
  }

  // CHECK: void noexcept_mem_fn_grad(double i, double j, SimpleFunctions *_d_this, double *_d_i, double *_d_j) noexcept {
  // CHECK-NEXT:     {
  // CHECK-NEXT:         (*_d_this).x += 1 * i;
  // CHECK-NEXT:         (*_d_this).y += 1 * i;
  // CHECK-NEXT:         *_d_i += (this->x + this->y) * 1;
  // CHECK-NEXT:         *_d_i += 1 * j;
  // CHECK-NEXT:         *_d_j += i * 1;
  // CHECK-NEXT:     }
  // CHECK-NEXT: }

  double const_noexcept_mem_fn(double i, double j) const noexcept {
    return (x+y)*i + i*j;
  }

  // CHECK: void const_noexcept_mem_fn_grad(double i, double j, SimpleFunctions *_d_this, double *_d_i, double *_d_j) const noexcept {
  // CHECK-NEXT:     {
  // CHECK-NEXT:         (*_d_this).x += 1 * i;
  // CHECK-NEXT:         (*_d_this).y += 1 * i;
  // CHECK-NEXT:         *_d_i += (this->x + this->y) * 1;
  // CHECK-NEXT:         *_d_i += 1 * j;
  // CHECK-NEXT:         *_d_j += i * 1;
  // CHECK-NEXT:     }
  // CHECK-NEXT: }

  double volatile_noexcept_mem_fn(double i, double j) volatile noexcept {
    return (x+y)*i + i*j;
  }

  // CHECK: void volatile_noexcept_mem_fn_grad(double i, double j, volatile SimpleFunctions *_d_this, double *_d_i, double *_d_j) volatile noexcept {
  // CHECK-NEXT:     double _t0 = (this->x + this->y);
  // CHECK-NEXT:     {
  // CHECK-NEXT:         (*_d_this).x += 1 * i;
  // CHECK-NEXT:         (*_d_this).y += 1 * i;
  // CHECK-NEXT:         *_d_i += _t0 * 1;
  // CHECK-NEXT:         *_d_i += 1 * j;
  // CHECK-NEXT:         *_d_j += i * 1;
  // CHECK-NEXT:     }
  // CHECK-NEXT: }

  double const_volatile_noexcept_mem_fn(double i, double j) const volatile noexcept {
    return (x+y)*i + i*j;
  }

  // CHECK: void const_volatile_noexcept_mem_fn_grad(double i, double j, volatile SimpleFunctions *_d_this, double *_d_i, double *_d_j) const volatile noexcept {
  // CHECK-NEXT:     double _t0 = (this->x + this->y);
  // CHECK-NEXT:     {
  // CHECK-NEXT:         (*_d_this).x += 1 * i;
  // CHECK-NEXT:         (*_d_this).y += 1 * i;
  // CHECK-NEXT:         *_d_i += _t0 * 1;
  // CHECK-NEXT:         *_d_i += 1 * j;
  // CHECK-NEXT:         *_d_j += i * 1;
  // CHECK-NEXT:     }
  // CHECK-NEXT: }

  double lval_ref_noexcept_mem_fn(double i, double j) & noexcept {
    return (x+y)*i + i*j;
  }

  // CHECK: void lval_ref_noexcept_mem_fn_grad(double i, double j, SimpleFunctions *_d_this, double *_d_i, double *_d_j) & noexcept {
  // CHECK-NEXT:     {
  // CHECK-NEXT:         (*_d_this).x += 1 * i;
  // CHECK-NEXT:         (*_d_this).y += 1 * i;
  // CHECK-NEXT:         *_d_i += (this->x + this->y) * 1;
  // CHECK-NEXT:         *_d_i += 1 * j;
  // CHECK-NEXT:         *_d_j += i * 1;
  // CHECK-NEXT:     }
  // CHECK-NEXT: }

  double const_lval_ref_noexcept_mem_fn(double i, double j) const & noexcept {
    return (x+y)*i + i*j;
  }

  // CHECK: void const_lval_ref_noexcept_mem_fn_grad(double i, double j, SimpleFunctions *_d_this, double *_d_i, double *_d_j) const & noexcept {
  // CHECK-NEXT:     {
  // CHECK-NEXT:         (*_d_this).x += 1 * i;
  // CHECK-NEXT:         (*_d_this).y += 1 * i;
  // CHECK-NEXT:         *_d_i += (this->x + this->y) * 1;
  // CHECK-NEXT:         *_d_i += 1 * j;
  // CHECK-NEXT:         *_d_j += i * 1;
  // CHECK-NEXT:     }
  // CHECK-NEXT: }

  double volatile_lval_ref_noexcept_mem_fn(double i, double j) volatile & noexcept {
    return (x+y)*i + i*j;
  }

  // CHECK: void volatile_lval_ref_noexcept_mem_fn_grad(double i, double j, volatile SimpleFunctions *_d_this, double *_d_i, double *_d_j) volatile & noexcept {
  // CHECK-NEXT:     double _t0 = (this->x + this->y);
  // CHECK-NEXT:     {
  // CHECK-NEXT:         (*_d_this).x += 1 * i;
  // CHECK-NEXT:         (*_d_this).y += 1 * i;
  // CHECK-NEXT:         *_d_i += _t0 * 1;
  // CHECK-NEXT:         *_d_i += 1 * j;
  // CHECK-NEXT:         *_d_j += i * 1;
  // CHECK-NEXT:     }
  // CHECK-NEXT: }

  double const_volatile_lval_ref_noexcept_mem_fn(double i, double j) const volatile & noexcept {
    return (x+y)*i + i*j;
  }

  // CHECK: void const_volatile_lval_ref_noexcept_mem_fn_grad(double i, double j, volatile SimpleFunctions *_d_this, double *_d_i, double *_d_j) const volatile & noexcept {
  // CHECK-NEXT:     double _t0 = (this->x + this->y);
  // CHECK-NEXT:     {
  // CHECK-NEXT:         (*_d_this).x += 1 * i;
  // CHECK-NEXT:         (*_d_this).y += 1 * i;
  // CHECK-NEXT:         *_d_i += _t0 * 1;
  // CHECK-NEXT:         *_d_i += 1 * j;
  // CHECK-NEXT:         *_d_j += i * 1;
  // CHECK-NEXT:     }
  // CHECK-NEXT: }

  double rval_ref_noexcept_mem_fn(double i, double j) && noexcept {
    return (x+y)*i + i*j;
  }

  // CHECK: void rval_ref_noexcept_mem_fn_grad(double i, double j, SimpleFunctions *_d_this, double *_d_i, double *_d_j) && noexcept {
  // CHECK-NEXT:     {
  // CHECK-NEXT:         (*_d_this).x += 1 * i;
  // CHECK-NEXT:         (*_d_this).y += 1 * i;
  // CHECK-NEXT:         *_d_i += (this->x + this->y) * 1;
  // CHECK-NEXT:         *_d_i += 1 * j;
  // CHECK-NEXT:         *_d_j += i * 1;
  // CHECK-NEXT:     }
  // CHECK-NEXT: }

  double const_rval_ref_noexcept_mem_fn(double i, double j) const && noexcept {
    return (x+y)*i + i*j;
  }

  // CHECK: void const_rval_ref_noexcept_mem_fn_grad(double i, double j, SimpleFunctions *_d_this, double *_d_i, double *_d_j) const && noexcept {
  // CHECK-NEXT:     {
  // CHECK-NEXT:         (*_d_this).x += 1 * i;
  // CHECK-NEXT:         (*_d_this).y += 1 * i;
  // CHECK-NEXT:         *_d_i += (this->x + this->y) * 1;
  // CHECK-NEXT:         *_d_i += 1 * j;
  // CHECK-NEXT:         *_d_j += i * 1;
  // CHECK-NEXT:     }
  // CHECK-NEXT: }

  double volatile_rval_ref_noexcept_mem_fn(double i, double j) volatile && noexcept {
    return (x+y)*i + i*j;
  }

  // CHECK: void volatile_rval_ref_noexcept_mem_fn_grad(double i, double j, volatile SimpleFunctions *_d_this, double *_d_i, double *_d_j) volatile && noexcept {
  // CHECK-NEXT:     double _t0 = (this->x + this->y);
  // CHECK-NEXT:     {
  // CHECK-NEXT:         (*_d_this).x += 1 * i;
  // CHECK-NEXT:         (*_d_this).y += 1 * i;
  // CHECK-NEXT:         *_d_i += _t0 * 1;
  // CHECK-NEXT:         *_d_i += 1 * j;
  // CHECK-NEXT:         *_d_j += i * 1;
  // CHECK-NEXT:     }
  // CHECK-NEXT: }

  double const_volatile_rval_ref_noexcept_mem_fn(double i, double j) const volatile && noexcept {
    return (x+y)*i + i*j;
  }

  // CHECK: void const_volatile_rval_ref_noexcept_mem_fn_grad(double i, double j, volatile SimpleFunctions *_d_this, double *_d_i, double *_d_j) const volatile && noexcept {
  // CHECK-NEXT:     double _t0 = (this->x + this->y);
  // CHECK-NEXT:     {
  // CHECK-NEXT:         (*_d_this).x += 1 * i;
  // CHECK-NEXT:         (*_d_this).y += 1 * i;
  // CHECK-NEXT:         *_d_i += _t0 * 1;
  // CHECK-NEXT:         *_d_i += 1 * j;
  // CHECK-NEXT:         *_d_j += i * 1;
  // CHECK-NEXT:     }
  // CHECK-NEXT: }

  double partial_mem_fn(double i, double j) { return (x + y) * i + i * j; }

  // CHECK: void partial_mem_fn_grad_0(double i, double j, SimpleFunctions *_d_this, double *_d_i) {
  // CHECK-NEXT:     double _d_j = 0;
  // CHECK-NEXT:     {
  // CHECK-NEXT:         (*_d_this).x += 1 * i;
  // CHECK-NEXT:         (*_d_this).y += 1 * i;
  // CHECK-NEXT:         *_d_i += (this->x + this->y) * 1;
  // CHECK-NEXT:         *_d_i += 1 * j;
  // CHECK-NEXT:         _d_j += i * 1;
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

// CHECK: void ref_mem_fn_pullback(double i, double _d_y, SimpleFunctions *_d_this, double *_d_i);

// CHECK: clad::ValueAndAdjoint<double &, double &> ref_mem_fn_forw(double i, SimpleFunctions *_d_this, double *_d_i);

// CHECK: void fn2_grad(SimpleFunctions &sf, double i, SimpleFunctions *_d_sf, double *_d_i) {
// CHECK-NEXT:     SimpleFunctions _t0 = sf;
// CHECK-NEXT:     clad::ValueAndAdjoint<double &, double &> _t1 = _t0.ref_mem_fn_forw(i, &(*_d_sf), nullptr);
// CHECK-NEXT:     {
// CHECK-NEXT:         double _r0 = 0;
// CHECK-NEXT:         _t0.ref_mem_fn_pullback(i, 1, &(*_d_sf), &_r0);
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

// CHECK: void operator_plus_equal_pullback(double value, SimpleFunctions _d_y, SimpleFunctions *_d_this, double *_d_value);

// CHECK: clad::ValueAndAdjoint<SimpleFunctions &, SimpleFunctions &> operator_plus_equal_forw(double value, SimpleFunctions *_d_this, SimpleFunctions *_d_value);

// CHECK: void fn5_grad(SimpleFunctions &v, double value, SimpleFunctions *_d_v, double *_d_value) {
// CHECK-NEXT:     SimpleFunctions _t0 = v;
// CHECK-NEXT:     clad::ValueAndAdjoint<SimpleFunctions &, SimpleFunctions &> _t1 = _t0.operator_plus_equal_forw(value, &(*_d_v), nullptr);
// CHECK-NEXT:     (*_d_v).x += 1;
// CHECK-NEXT:     {
// CHECK-NEXT:         double _r0 = 0;
// CHECK-NEXT:         _t0.operator_plus_equal_pullback(value, {}, &(*_d_v), &_r0);
// CHECK-NEXT:         *_d_value += _r0;
// CHECK-NEXT:     }
// CHECK-NEXT: }

double fn4(SimpleFunctions& v) {
  ++v;
  return v.x;
}

// CHECK: void operator_plus_plus_pullback(SimpleFunctions _d_y, SimpleFunctions *_d_this);

// CHECK: clad::ValueAndAdjoint<SimpleFunctions &, SimpleFunctions &> operator_plus_plus_forw(SimpleFunctions *_d_this);

// CHECK: void fn4_grad(SimpleFunctions &v, SimpleFunctions *_d_v) {
// CHECK-NEXT:     SimpleFunctions _t0 = v;
// CHECK-NEXT:     clad::ValueAndAdjoint<SimpleFunctions &, SimpleFunctions &> _t1 = _t0.operator_plus_plus_forw(&(*_d_v));
// CHECK-NEXT:     (*_d_v).x += 1;
// CHECK-NEXT:     _t0.operator_plus_plus_pullback({}, &(*_d_v));
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

  auto d_const_volatile_lval_ref_mem_fn_i = clad::gradient(&SimpleFunctions::const_volatile_lval_ref_mem_fn, "i");

  // CHECK:   void const_volatile_lval_ref_mem_fn_grad_0(double i, double j, volatile SimpleFunctions *_d_this, double *_d_i) const volatile & {
  // CHECK-NEXT:       double _d_j = 0;
  // CHECK-NEXT:       double _t0 = (this->x + this->y);
  // CHECK-NEXT:       {
  // CHECK-NEXT:           (*_d_this).x += 1 * i;
  // CHECK-NEXT:           (*_d_this).y += 1 * i;
  // CHECK-NEXT:           *_d_i += _t0 * 1;
  // CHECK-NEXT:           *_d_i += 1 * j;
  // CHECK-NEXT:           _d_j += i * 1;
  // CHECK-NEXT:       }
  // CHECK-NEXT:   }

  auto d_const_volatile_rval_ref_mem_fn_j = clad::gradient(&SimpleFunctions::const_volatile_rval_ref_mem_fn, "j");

  // CHECK:   void const_volatile_rval_ref_mem_fn_grad_1(double i, double j, volatile SimpleFunctions *_d_this, double *_d_j) const volatile && {
  // CHECK-NEXT:       double _d_i = 0;
  // CHECK-NEXT:       double _t0 = (this->x + this->y);
  // CHECK-NEXT:       {
  // CHECK-NEXT:           (*_d_this).x += 1 * i;
  // CHECK-NEXT:           (*_d_this).y += 1 * i;
  // CHECK-NEXT:           _d_i += _t0 * 1;
  // CHECK-NEXT:           _d_i += 1 * j;
  // CHECK-NEXT:           *_d_j += i * 1;
  // CHECK-NEXT:       }
  // CHECK-NEXT:   }

  auto d_fn3 = clad::gradient(fn3, "i,j");
  result[0] = result[1] = 0;
  d_fn3.execute(2, 3, 4, 5, &result[0], &result[1]);
  printf("%.2f %.2f", result[0], result[1]); // CHECK-EXEC: 10.00 4.00

// CHECK: void fn3_grad_2_3(double x, double y, double i, double j, double *_d_i, double *_d_j) {
// CHECK-NEXT:     double _d_x = 0;
// CHECK-NEXT:     double _d_y = 0;
// CHECK-NEXT:     SimpleFunctions _d_sf({});
// CHECK-NEXT:     SimpleFunctions sf(x, y);
// CHECK-NEXT:     SimpleFunctions _t0 = sf;
// CHECK-NEXT:     {
// CHECK-NEXT:         double _r0 = 0;
// CHECK-NEXT:         double _r1 = 0;
// CHECK-NEXT:         _t0.mem_fn_pullback(i, j, 1, &_d_sf, &_r0, &_r1);
// CHECK-NEXT:         *_d_i += _r0;
// CHECK-NEXT:         *_d_j += _r1;
// CHECK-NEXT:     }
// CHECK-NEXT: }

// CHECK: void ref_mem_fn_pullback(double i, double _d_y, SimpleFunctions *_d_this, double *_d_i) {
// CHECK-NEXT:     double _t0 = this->x;
// CHECK-NEXT:     this->x = +i;
// CHECK-NEXT:     double _t1 = this->x;
// CHECK-NEXT:     this->x = -i;
// CHECK-NEXT:     (*_d_this).x += _d_y;
// CHECK-NEXT:     {
// CHECK-NEXT:         this->x = _t1;
// CHECK-NEXT:         double _r_d1 = (*_d_this).x;
// CHECK-NEXT:         (*_d_this).x = 0;
// CHECK-NEXT:         *_d_i += -_r_d1;
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         this->x = _t0;
// CHECK-NEXT:         double _r_d0 = (*_d_this).x;
// CHECK-NEXT:         (*_d_this).x = 0;
// CHECK-NEXT:         *_d_i += _r_d0;
// CHECK-NEXT:     }
// CHECK-NEXT: }

// CHECK: clad::ValueAndAdjoint<double &, double &> ref_mem_fn_forw(double i, SimpleFunctions *_d_this, double *_d_i) {
// CHECK-NEXT:     double _t0 = this->x;
// CHECK-NEXT:     this->x = +i;
// CHECK-NEXT:     double _t1 = this->x;
// CHECK-NEXT:     this->x = -i;
// CHECK-NEXT:     return {this->x, (*_d_this).x};
// CHECK-NEXT: }

// CHECK: void operator_plus_equal_pullback(double value, SimpleFunctions _d_y, SimpleFunctions *_d_this, double *_d_value) {
// CHECK-NEXT:     double _t0 = this->x;
// CHECK-NEXT:     this->x += value;
// CHECK-NEXT:     {
// CHECK-NEXT:         this->x = _t0;
// CHECK-NEXT:         double _r_d0 = (*_d_this).x;
// CHECK-NEXT:         *_d_value += _r_d0;
// CHECK-NEXT:     }
// CHECK-NEXT: }

// CHECK: clad::ValueAndAdjoint<SimpleFunctions &, SimpleFunctions &> operator_plus_equal_forw(double value, SimpleFunctions *_d_this, SimpleFunctions *_d_value) {
// CHECK-NEXT:     double _t0 = this->x;
// CHECK-NEXT:     this->x += value;
// CHECK-NEXT:     return {*this, (*_d_this)};
// CHECK-NEXT: }

// CHECK: void operator_plus_plus_pullback(SimpleFunctions _d_y, SimpleFunctions *_d_this) {
// CHECK-NEXT:     double _t0 = this->x;
// CHECK-NEXT:     this->x += 1.;
// CHECK-NEXT:     {
// CHECK-NEXT:         this->x = _t0;
// CHECK-NEXT:         double _r_d0 = (*_d_this).x;
// CHECK-NEXT:     }
// CHECK-NEXT: }

// CHECK: clad::ValueAndAdjoint<SimpleFunctions &, SimpleFunctions &> operator_plus_plus_forw(SimpleFunctions *_d_this) {
// CHECK-NEXT:     double _t0 = this->x;
// CHECK-NEXT:     this->x += 1.;
// CHECK-NEXT:     return {*this, (*_d_this)};
// CHECK-NEXT: }
}
