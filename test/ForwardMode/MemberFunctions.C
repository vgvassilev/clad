// RUN: %cladclang %s -I%S/../../include -oMemberFunctions.out 2>&1 | %filecheck %s
// RUN: ./MemberFunctions.out | %filecheck_exec %s
// RUN: %cladclang -std=c++14 %s -I%S/../../include -oMemberFunctions-cpp14.out 2>&1 | %filecheck %s
// RUN: ./MemberFunctions-cpp14.out | %filecheck_exec %s
// RUN: %cladclang -std=c++17 %s -I%S/../../include -oMemberFunctions-cpp17.out 2>&1 | %filecheck %s
// RUN: ./MemberFunctions-cpp17.out | %filecheck_exec %s
// CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"

extern "C" int printf(const char *fmt, ...);
class SimpleFunctions {
public:
  SimpleFunctions() noexcept : x(0), y(0) {}
  SimpleFunctions(double p_x, double p_y) noexcept : x(p_x), y(p_y) {}
  double x, y;
  double mem_fn(double i, double j)  { 
    return (x+y)*i + i*j*j; 
  } 

  // CHECK: double mem_fn_darg0(double i, double j) {
  // CHECK-NEXT:     double _d_i = 1;
  // CHECK-NEXT:     double _d_j = 0;
  // CHECK-NEXT:     SimpleFunctions _d_this_obj;
  // CHECK-NEXT:     SimpleFunctions *_d_this = &_d_this_obj;
  // CHECK-NEXT:     double _t0 = (this->x + this->y);
  // CHECK-NEXT:     double _t1 = i * j;
  // CHECK-NEXT:     return (_d_this->x + _d_this->y) * i + _t0 * _d_i + (_d_i * j + i * _d_j) * j + _t1 * _d_j;
  // CHECK-NEXT: }

  void mem_fn_with_void_return() {
	  return;
  }

  // CHECK: void mem_fn_with_void_return_pushforward(SimpleFunctions *_d_this);

  double mem_fn_with_void_function_call(double i, double j) {
    mem_fn_with_void_return();
    return i*j;
  } 

  // CHECK:  double mem_fn_with_void_function_call_darg0(double i, double j) {
  // CHECK-NEXT:     double _d_i = 1;  
  // CHECK-NEXT:     double _d_j = 0; 
  // CHECK-NEXT:     SimpleFunctions _d_this_obj;
  // CHECK-NEXT:     SimpleFunctions *_d_this = &_d_this_obj; 
  // CHECK-NEXT:     this->mem_fn_with_void_return_pushforward(_d_this);  
  // CHECK-NEXT:     return _d_i * j + i * _d_j;  
  // CHECK-NEXT:}

  double mem_fn_with_var_arg_list(double i, double j, ...)  { 
    return (x+y)*i + i*j*j; 
  } 

  // CHECK: double mem_fn_with_var_arg_list_darg0(double i, double j, ...) {
  // CHECK-NEXT:     double _d_i = 1;
  // CHECK-NEXT:     double _d_j = 0;
  // CHECK-NEXT:     SimpleFunctions _d_this_obj;
  // CHECK-NEXT:     SimpleFunctions *_d_this = &_d_this_obj;
  // CHECK-NEXT:     double _t0 = (this->x + this->y);
  // CHECK-NEXT:     double _t1 = i * j;
  // CHECK-NEXT:     return (_d_this->x + _d_this->y) * i + _t0 * _d_i + (_d_i * j + i * _d_j) * j + _t1 * _d_j;
  // CHECK-NEXT: }

  double const_mem_fn(double i, double j) const { 
    return (x+y)*i + i*j*j; 
  } 

  // CHECK: double const_mem_fn_darg0(double i, double j) const {
  // CHECK-NEXT:     double _d_i = 1;
  // CHECK-NEXT:     double _d_j = 0;
  // CHECK-NEXT:     const SimpleFunctions _d_this_obj;
  // CHECK-NEXT:     const SimpleFunctions *_d_this = &_d_this_obj;
  // CHECK-NEXT:     double _t0 = (this->x + this->y);
  // CHECK-NEXT:     double _t1 = i * j;
  // CHECK-NEXT:     return (_d_this->x + _d_this->y) * i + _t0 * _d_i + (_d_i * j + i * _d_j) * j + _t1 * _d_j;
  // CHECK-NEXT: }

  double const_mem_fn_with_var_arg_list(double i, double j, ...) const { 
    return (x+y)*i + i*j*j; 
  } 

  // CHECK: double const_mem_fn_with_var_arg_list_darg0(double i, double j, ...) const {
  // CHECK-NEXT:     double _d_i = 1;
  // CHECK-NEXT:     double _d_j = 0;
  // CHECK-NEXT:     const SimpleFunctions _d_this_obj;
  // CHECK-NEXT:     const SimpleFunctions *_d_this = &_d_this_obj;
  // CHECK-NEXT:     double _t0 = (this->x + this->y);
  // CHECK-NEXT:     double _t1 = i * j;
  // CHECK-NEXT:     return (_d_this->x + _d_this->y) * i + _t0 * _d_i + (_d_i * j + i * _d_j) * j + _t1 * _d_j;
  // CHECK-NEXT: }

  double volatile_mem_fn(double i, double j) volatile { 
    return (x+y)*i + i*j*j; 
  } 

  // CHECK: double volatile_mem_fn_darg0(double i, double j) volatile {
  // CHECK-NEXT:     double _d_i = 1;
  // CHECK-NEXT:     double _d_j = 0;
  // CHECK-NEXT:     volatile SimpleFunctions _d_this_obj;
  // CHECK-NEXT:     volatile SimpleFunctions *_d_this = &_d_this_obj;
  // CHECK-NEXT:     double _t0 = (this->x + this->y);
  // CHECK-NEXT:     double _t1 = i * j;
  // CHECK-NEXT:     return (_d_this->x + _d_this->y) * i + _t0 * _d_i + (_d_i * j + i * _d_j) * j + _t1 * _d_j;
  // CHECK-NEXT: }

  double volatile_mem_fn_with_var_arg_list(double i, double j, ...) volatile { 
    return (x+y)*i + i*j*j; 
  } 

  // CHECK: double volatile_mem_fn_with_var_arg_list_darg0(double i, double j, ...) volatile {
  // CHECK-NEXT:     double _d_i = 1;
  // CHECK-NEXT:     double _d_j = 0;
  // CHECK-NEXT:     volatile SimpleFunctions _d_this_obj;
  // CHECK-NEXT:     volatile SimpleFunctions *_d_this = &_d_this_obj;
  // CHECK-NEXT:     double _t0 = (this->x + this->y);
  // CHECK-NEXT:     double _t1 = i * j;
  // CHECK-NEXT:     return (_d_this->x + _d_this->y) * i + _t0 * _d_i + (_d_i * j + i * _d_j) * j + _t1 * _d_j;
  // CHECK-NEXT: }

  double const_volatile_mem_fn(double i, double j) const volatile { 
    return (x+y)*i + i*j*j; 
  } 

  // CHECK: double const_volatile_mem_fn_darg0(double i, double j) const volatile {
  // CHECK-NEXT:     double _d_i = 1;
  // CHECK-NEXT:     double _d_j = 0;
  // CHECK-NEXT:     const volatile SimpleFunctions _d_this_obj;
  // CHECK-NEXT:     const volatile SimpleFunctions *_d_this = &_d_this_obj;
  // CHECK-NEXT:     double _t0 = (this->x + this->y);
  // CHECK-NEXT:     double _t1 = i * j;
  // CHECK-NEXT:     return (_d_this->x + _d_this->y) * i + _t0 * _d_i + (_d_i * j + i * _d_j) * j + _t1 * _d_j;
  // CHECK-NEXT: }

  double const_volatile_mem_fn_with_var_arg_list(double i, double j, ...) const volatile { 
    return (x+y)*i + i*j*j; 
  } 

  // CHECK: double const_volatile_mem_fn_with_var_arg_list_darg0(double i, double j, ...) const volatile {
  // CHECK-NEXT:     double _d_i = 1;
  // CHECK-NEXT:     double _d_j = 0;
  // CHECK-NEXT:     const volatile SimpleFunctions _d_this_obj;
  // CHECK-NEXT:     const volatile SimpleFunctions *_d_this = &_d_this_obj;
  // CHECK-NEXT:     double _t0 = (this->x + this->y);
  // CHECK-NEXT:     double _t1 = i * j;
  // CHECK-NEXT:     return (_d_this->x + _d_this->y) * i + _t0 * _d_i + (_d_i * j + i * _d_j) * j + _t1 * _d_j;
  // CHECK-NEXT: }

  double lval_ref_mem_fn(double i, double j) & { 
    return (x+y)*i + i*j*j; 
  } 

  // CHECK: double lval_ref_mem_fn_darg0(double i, double j) & {
  // CHECK-NEXT:     double _d_i = 1;
  // CHECK-NEXT:     double _d_j = 0;
  // CHECK-NEXT:     SimpleFunctions _d_this_obj;
  // CHECK-NEXT:     SimpleFunctions *_d_this = &_d_this_obj;
  // CHECK-NEXT:     double _t0 = (this->x + this->y);
  // CHECK-NEXT:     double _t1 = i * j;
  // CHECK-NEXT:     return (_d_this->x + _d_this->y) * i + _t0 * _d_i + (_d_i * j + i * _d_j) * j + _t1 * _d_j;
  // CHECK-NEXT: }

  double lval_ref_mem_fn_with_var_arg_list(double i, double j, ...) & { 
    return (x+y)*i + i*j*j; 
  } 

  // CHECK: double lval_ref_mem_fn_with_var_arg_list_darg0(double i, double j, ...) & {
  // CHECK-NEXT:     double _d_i = 1;
  // CHECK-NEXT:     double _d_j = 0;
  // CHECK-NEXT:     SimpleFunctions _d_this_obj;
  // CHECK-NEXT:     SimpleFunctions *_d_this = &_d_this_obj;
  // CHECK-NEXT:     double _t0 = (this->x + this->y);
  // CHECK-NEXT:     double _t1 = i * j;
  // CHECK-NEXT:     return (_d_this->x + _d_this->y) * i + _t0 * _d_i + (_d_i * j + i * _d_j) * j + _t1 * _d_j;
  // CHECK-NEXT: }

  double const_lval_ref_mem_fn(double i, double j) const & { 
    return (x+y)*i + i*j*j; 
  } 

  // CHECK: double const_lval_ref_mem_fn_darg0(double i, double j) const & {
  // CHECK-NEXT:     double _d_i = 1;
  // CHECK-NEXT:     double _d_j = 0;
  // CHECK-NEXT:     const SimpleFunctions _d_this_obj;
  // CHECK-NEXT:     const SimpleFunctions *_d_this = &_d_this_obj;
  // CHECK-NEXT:     double _t0 = (this->x + this->y);
  // CHECK-NEXT:     double _t1 = i * j;
  // CHECK-NEXT:     return (_d_this->x + _d_this->y) * i + _t0 * _d_i + (_d_i * j + i * _d_j) * j + _t1 * _d_j;
  // CHECK-NEXT: }

  double const_lval_ref_mem_fn_with_var_arg_list(double i, double j, ...) const & { 
    return (x+y)*i + i*j*j; 
  } 

  // CHECK: double const_lval_ref_mem_fn_with_var_arg_list_darg0(double i, double j, ...) const & {
  // CHECK-NEXT:     double _d_i = 1;
  // CHECK-NEXT:     double _d_j = 0;
  // CHECK-NEXT:     const SimpleFunctions _d_this_obj;
  // CHECK-NEXT:     const SimpleFunctions *_d_this = &_d_this_obj;
  // CHECK-NEXT:     double _t0 = (this->x + this->y);
  // CHECK-NEXT:     double _t1 = i * j;
  // CHECK-NEXT:     return (_d_this->x + _d_this->y) * i + _t0 * _d_i + (_d_i * j + i * _d_j) * j + _t1 * _d_j;
  // CHECK-NEXT: }

  double volatile_lval_ref_mem_fn(double i, double j) volatile & { 
    return (x+y)*i + i*j*j; 
  } 

  // CHECK: double volatile_lval_ref_mem_fn_darg0(double i, double j) volatile & {
  // CHECK-NEXT:     double _d_i = 1;
  // CHECK-NEXT:     double _d_j = 0;
  // CHECK-NEXT:     volatile SimpleFunctions _d_this_obj;
  // CHECK-NEXT:     volatile SimpleFunctions *_d_this = &_d_this_obj;
  // CHECK-NEXT:     double _t0 = (this->x + this->y);
  // CHECK-NEXT:     double _t1 = i * j;
  // CHECK-NEXT:     return (_d_this->x + _d_this->y) * i + _t0 * _d_i + (_d_i * j + i * _d_j) * j + _t1 * _d_j;
  // CHECK-NEXT: }

  double volatile_lval_ref_mem_fn_with_var_arg_list(double i, double j, ...) volatile & { 
    return (x+y)*i + i*j*j; 
  } 

  // CHECK: double volatile_lval_ref_mem_fn_with_var_arg_list_darg0(double i, double j, ...) volatile & {
  // CHECK-NEXT:     double _d_i = 1;
  // CHECK-NEXT:     double _d_j = 0;
  // CHECK-NEXT:     volatile SimpleFunctions _d_this_obj;
  // CHECK-NEXT:     volatile SimpleFunctions *_d_this = &_d_this_obj;
  // CHECK-NEXT:     double _t0 = (this->x + this->y);
  // CHECK-NEXT:     double _t1 = i * j;
  // CHECK-NEXT:     return (_d_this->x + _d_this->y) * i + _t0 * _d_i + (_d_i * j + i * _d_j) * j + _t1 * _d_j;
  // CHECK-NEXT: }

  double const_volatile_lval_ref_mem_fn(double i, double j) const volatile & { 
    return (x+y)*i + i*j*j; 
  } 

  // CHECK: double const_volatile_lval_ref_mem_fn_darg0(double i, double j) const volatile & {
  // CHECK-NEXT:     double _d_i = 1;
  // CHECK-NEXT:     double _d_j = 0;
  // CHECK-NEXT:     const volatile SimpleFunctions _d_this_obj;
  // CHECK-NEXT:     const volatile SimpleFunctions *_d_this = &_d_this_obj;
  // CHECK-NEXT:     double _t0 = (this->x + this->y);
  // CHECK-NEXT:     double _t1 = i * j;
  // CHECK-NEXT:     return (_d_this->x + _d_this->y) * i + _t0 * _d_i + (_d_i * j + i * _d_j) * j + _t1 * _d_j;
  // CHECK-NEXT: }

  double const_volatile_lval_ref_mem_fn_with_var_arg_list(double i, double j, ...) const volatile & { 
    return (x+y)*i + i*j*j; 
  } 

  // CHECK: double const_volatile_lval_ref_mem_fn_with_var_arg_list_darg0(double i, double j, ...) const volatile & {
  // CHECK-NEXT:     double _d_i = 1;
  // CHECK-NEXT:     double _d_j = 0;
  // CHECK-NEXT:     const volatile SimpleFunctions _d_this_obj;
  // CHECK-NEXT:     const volatile SimpleFunctions *_d_this = &_d_this_obj;
  // CHECK-NEXT:     double _t0 = (this->x + this->y);
  // CHECK-NEXT:     double _t1 = i * j;
  // CHECK-NEXT:     return (_d_this->x + _d_this->y) * i + _t0 * _d_i + (_d_i * j + i * _d_j) * j + _t1 * _d_j;
  // CHECK-NEXT: }

  double rval_ref_mem_fn(double i, double j) && { 
    return (x+y)*i + i*j*j; 
  } 

  // CHECK: double rval_ref_mem_fn_darg0(double i, double j) && {
  // CHECK-NEXT:     double _d_i = 1;
  // CHECK-NEXT:     double _d_j = 0;
  // CHECK-NEXT:     SimpleFunctions _d_this_obj;
  // CHECK-NEXT:     SimpleFunctions *_d_this = &_d_this_obj;
  // CHECK-NEXT:     double _t0 = (this->x + this->y);
  // CHECK-NEXT:     double _t1 = i * j;
  // CHECK-NEXT:     return (_d_this->x + _d_this->y) * i + _t0 * _d_i + (_d_i * j + i * _d_j) * j + _t1 * _d_j;
  // CHECK-NEXT: }

  double rval_ref_mem_fn_with_var_arg_list(double i, double j, ...) && { 
    return (x+y)*i + i*j*j; 
  } 

  // CHECK: double rval_ref_mem_fn_with_var_arg_list_darg0(double i, double j, ...) && {
  // CHECK-NEXT:     double _d_i = 1;
  // CHECK-NEXT:     double _d_j = 0;
  // CHECK-NEXT:     SimpleFunctions _d_this_obj;
  // CHECK-NEXT:     SimpleFunctions *_d_this = &_d_this_obj;
  // CHECK-NEXT:     double _t0 = (this->x + this->y);
  // CHECK-NEXT:     double _t1 = i * j;
  // CHECK-NEXT:     return (_d_this->x + _d_this->y) * i + _t0 * _d_i + (_d_i * j + i * _d_j) * j + _t1 * _d_j;
  // CHECK-NEXT: }

  double const_rval_ref_mem_fn(double i, double j) const && { 
    return (x+y)*i + i*j*j; 
  } 

  // CHECK: double const_rval_ref_mem_fn_darg0(double i, double j) const && {
  // CHECK-NEXT:     double _d_i = 1;
  // CHECK-NEXT:     double _d_j = 0;
  // CHECK-NEXT:     const SimpleFunctions _d_this_obj;
  // CHECK-NEXT:     const SimpleFunctions *_d_this = &_d_this_obj;
  // CHECK-NEXT:     double _t0 = (this->x + this->y);
  // CHECK-NEXT:     double _t1 = i * j;
  // CHECK-NEXT:     return (_d_this->x + _d_this->y) * i + _t0 * _d_i + (_d_i * j + i * _d_j) * j + _t1 * _d_j;
  // CHECK-NEXT: }

  double const_rval_ref_mem_fn_with_var_arg_list(double i, double j, ...) const && { 
    return (x+y)*i + i*j*j; 
  } 

  // CHECK: double const_rval_ref_mem_fn_with_var_arg_list_darg0(double i, double j, ...) const && {
  // CHECK-NEXT:     double _d_i = 1;
  // CHECK-NEXT:     double _d_j = 0;
  // CHECK-NEXT:     const SimpleFunctions _d_this_obj;
  // CHECK-NEXT:     const SimpleFunctions *_d_this = &_d_this_obj;
  // CHECK-NEXT:     double _t0 = (this->x + this->y);
  // CHECK-NEXT:     double _t1 = i * j;
  // CHECK-NEXT:     return (_d_this->x + _d_this->y) * i + _t0 * _d_i + (_d_i * j + i * _d_j) * j + _t1 * _d_j;
  // CHECK-NEXT: }

  double volatile_rval_ref_mem_fn(double i, double j) volatile && { 
    return (x+y)*i + i*j*j; 
  } 

  // CHECK: double volatile_rval_ref_mem_fn_darg0(double i, double j) volatile && {
  // CHECK-NEXT:     double _d_i = 1;
  // CHECK-NEXT:     double _d_j = 0;
  // CHECK-NEXT:     volatile SimpleFunctions _d_this_obj;
  // CHECK-NEXT:     volatile SimpleFunctions *_d_this = &_d_this_obj;
  // CHECK-NEXT:     double _t0 = (this->x + this->y);
  // CHECK-NEXT:     double _t1 = i * j;
  // CHECK-NEXT:     return (_d_this->x + _d_this->y) * i + _t0 * _d_i + (_d_i * j + i * _d_j) * j + _t1 * _d_j;
  // CHECK-NEXT: }

  double volatile_rval_ref_mem_fn_with_var_arg_list(double i, double j, ...) volatile && { 
    return (x+y)*i + i*j*j; 
  } 

  // CHECK: double volatile_rval_ref_mem_fn_with_var_arg_list_darg0(double i, double j, ...) volatile && {
  // CHECK-NEXT:     double _d_i = 1;
  // CHECK-NEXT:     double _d_j = 0;
  // CHECK-NEXT:     volatile SimpleFunctions _d_this_obj;
  // CHECK-NEXT:     volatile SimpleFunctions *_d_this = &_d_this_obj;
  // CHECK-NEXT:     double _t0 = (this->x + this->y);
  // CHECK-NEXT:     double _t1 = i * j;
  // CHECK-NEXT:     return (_d_this->x + _d_this->y) * i + _t0 * _d_i + (_d_i * j + i * _d_j) * j + _t1 * _d_j;
  // CHECK-NEXT: }

  double const_volatile_rval_ref_mem_fn(double i, double j) const volatile && { 
    return (x+y)*i + i*j*j; 
  } 

  // CHECK: double const_volatile_rval_ref_mem_fn_darg0(double i, double j) const volatile && {
  // CHECK-NEXT:     double _d_i = 1;
  // CHECK-NEXT:     double _d_j = 0;
  // CHECK-NEXT:     const volatile SimpleFunctions _d_this_obj;
  // CHECK-NEXT:     const volatile SimpleFunctions *_d_this = &_d_this_obj;
  // CHECK-NEXT:     double _t0 = (this->x + this->y);
  // CHECK-NEXT:     double _t1 = i * j;
  // CHECK-NEXT:     return (_d_this->x + _d_this->y) * i + _t0 * _d_i + (_d_i * j + i * _d_j) * j + _t1 * _d_j;
  // CHECK-NEXT: }

  double const_volatile_rval_ref_mem_fn_with_var_arg_list(double i, double j, ...) const volatile && { 
    return (x+y)*i + i*j*j; 
  } 

  // CHECK: double const_volatile_rval_ref_mem_fn_with_var_arg_list_darg0(double i, double j, ...) const volatile && {
  // CHECK-NEXT:     double _d_i = 1;
  // CHECK-NEXT:     double _d_j = 0;
  // CHECK-NEXT:     const volatile SimpleFunctions _d_this_obj;
  // CHECK-NEXT:     const volatile SimpleFunctions *_d_this = &_d_this_obj;
  // CHECK-NEXT:     double _t0 = (this->x + this->y);
  // CHECK-NEXT:     double _t1 = i * j;
  // CHECK-NEXT:     return (_d_this->x + _d_this->y) * i + _t0 * _d_i + (_d_i * j + i * _d_j) * j + _t1 * _d_j;
  // CHECK-NEXT: }

  double noexcept_mem_fn(double i, double j) noexcept { 
    return (x+y)*i + i*j*j; 
  } 

  // CHECK: double noexcept_mem_fn_darg0(double i, double j) noexcept {
  // CHECK-NEXT:     double _d_i = 1;
  // CHECK-NEXT:     double _d_j = 0;
  // CHECK-NEXT:     SimpleFunctions _d_this_obj;
  // CHECK-NEXT:     SimpleFunctions *_d_this = &_d_this_obj;
  // CHECK-NEXT:     double _t0 = (this->x + this->y);
  // CHECK-NEXT:     double _t1 = i * j;
  // CHECK-NEXT:     return (_d_this->x + _d_this->y) * i + _t0 * _d_i + (_d_i * j + i * _d_j) * j + _t1 * _d_j;
  // CHECK-NEXT: }

  double noexcept_mem_fn_with_var_arg_list(double i, double j, ...) noexcept { 
    return (x+y)*i + i*j*j; 
  } 

  // CHECK: double noexcept_mem_fn_with_var_arg_list_darg0(double i, double j, ...) noexcept {
  // CHECK-NEXT:     double _d_i = 1;
  // CHECK-NEXT:     double _d_j = 0;
  // CHECK-NEXT:     SimpleFunctions _d_this_obj;
  // CHECK-NEXT:     SimpleFunctions *_d_this = &_d_this_obj;
  // CHECK-NEXT:     double _t0 = (this->x + this->y);
  // CHECK-NEXT:     double _t1 = i * j;
  // CHECK-NEXT:     return (_d_this->x + _d_this->y) * i + _t0 * _d_i + (_d_i * j + i * _d_j) * j + _t1 * _d_j;
  // CHECK-NEXT: }

  double const_noexcept_mem_fn(double i, double j) const noexcept { 
    return (x+y)*i + i*j*j; 
  } 

  // CHECK: double const_noexcept_mem_fn_darg0(double i, double j) const noexcept {
  // CHECK-NEXT:     double _d_i = 1;
  // CHECK-NEXT:     double _d_j = 0;
  // CHECK-NEXT:     const SimpleFunctions _d_this_obj;
  // CHECK-NEXT:     const SimpleFunctions *_d_this = &_d_this_obj;
  // CHECK-NEXT:     double _t0 = (this->x + this->y);
  // CHECK-NEXT:     double _t1 = i * j;
  // CHECK-NEXT:     return (_d_this->x + _d_this->y) * i + _t0 * _d_i + (_d_i * j + i * _d_j) * j + _t1 * _d_j;
  // CHECK-NEXT: }

  double const_noexcept_mem_fn_with_var_arg_list(double i, double j, ...) const noexcept { 
    return (x+y)*i + i*j*j; 
  } 

  // CHECK: double const_noexcept_mem_fn_with_var_arg_list_darg0(double i, double j, ...) const noexcept {
  // CHECK-NEXT:     double _d_i = 1;
  // CHECK-NEXT:     double _d_j = 0;
  // CHECK-NEXT:     const SimpleFunctions _d_this_obj;
  // CHECK-NEXT:     const SimpleFunctions *_d_this = &_d_this_obj;
  // CHECK-NEXT:     double _t0 = (this->x + this->y);
  // CHECK-NEXT:     double _t1 = i * j;
  // CHECK-NEXT:     return (_d_this->x + _d_this->y) * i + _t0 * _d_i + (_d_i * j + i * _d_j) * j + _t1 * _d_j;
  // CHECK-NEXT: }

  double volatile_noexcept_mem_fn(double i, double j) volatile noexcept { 
    return (x+y)*i + i*j*j; 
  } 

  // CHECK: double volatile_noexcept_mem_fn_darg0(double i, double j) volatile noexcept {
  // CHECK-NEXT:     double _d_i = 1;
  // CHECK-NEXT:     double _d_j = 0;
  // CHECK-NEXT:     volatile SimpleFunctions _d_this_obj;
  // CHECK-NEXT:     volatile SimpleFunctions *_d_this = &_d_this_obj;
  // CHECK-NEXT:     double _t0 = (this->x + this->y);
  // CHECK-NEXT:     double _t1 = i * j;
  // CHECK-NEXT:     return (_d_this->x + _d_this->y) * i + _t0 * _d_i + (_d_i * j + i * _d_j) * j + _t1 * _d_j;
  // CHECK-NEXT: }

  double volatile_noexcept_mem_fn_with_var_arg_list(double i, double j, ...) volatile noexcept { 
    return (x+y)*i + i*j*j; 
  } 

  // CHECK: double volatile_noexcept_mem_fn_with_var_arg_list_darg0(double i, double j, ...) volatile noexcept {
  // CHECK-NEXT:     double _d_i = 1;
  // CHECK-NEXT:     double _d_j = 0;
  // CHECK-NEXT:     volatile SimpleFunctions _d_this_obj;
  // CHECK-NEXT:     volatile SimpleFunctions *_d_this = &_d_this_obj;
  // CHECK-NEXT:     double _t0 = (this->x + this->y);
  // CHECK-NEXT:     double _t1 = i * j;
  // CHECK-NEXT:     return (_d_this->x + _d_this->y) * i + _t0 * _d_i + (_d_i * j + i * _d_j) * j + _t1 * _d_j;
  // CHECK-NEXT: }

  double const_volatile_noexcept_mem_fn(double i, double j) const volatile noexcept { 
    return (x+y)*i + i*j*j; 
  } 

  // CHECK: double const_volatile_noexcept_mem_fn_darg0(double i, double j) const volatile noexcept {
  // CHECK-NEXT:     double _d_i = 1;
  // CHECK-NEXT:     double _d_j = 0;
  // CHECK-NEXT:     const volatile SimpleFunctions _d_this_obj;
  // CHECK-NEXT:     const volatile SimpleFunctions *_d_this = &_d_this_obj;
  // CHECK-NEXT:     double _t0 = (this->x + this->y);
  // CHECK-NEXT:     double _t1 = i * j;
  // CHECK-NEXT:     return (_d_this->x + _d_this->y) * i + _t0 * _d_i + (_d_i * j + i * _d_j) * j + _t1 * _d_j;
  // CHECK-NEXT: }

  double const_volatile_noexcept_mem_fn_with_var_arg_list(double i, double j, ...) const volatile noexcept { 
    return (x+y)*i + i*j*j; 
  } 

  // CHECK: double const_volatile_noexcept_mem_fn_with_var_arg_list_darg0(double i, double j, ...) const volatile noexcept {
  // CHECK-NEXT:     double _d_i = 1;
  // CHECK-NEXT:     double _d_j = 0;
  // CHECK-NEXT:     const volatile SimpleFunctions _d_this_obj;
  // CHECK-NEXT:     const volatile SimpleFunctions *_d_this = &_d_this_obj;
  // CHECK-NEXT:     double _t0 = (this->x + this->y);
  // CHECK-NEXT:     double _t1 = i * j;
  // CHECK-NEXT:     return (_d_this->x + _d_this->y) * i + _t0 * _d_i + (_d_i * j + i * _d_j) * j + _t1 * _d_j;
  // CHECK-NEXT: }

  double lval_ref_noexcept_mem_fn(double i, double j) & noexcept { 
    return (x+y)*i + i*j*j; 
  } 

  // CHECK: double lval_ref_noexcept_mem_fn_darg0(double i, double j) & noexcept {
  // CHECK-NEXT:     double _d_i = 1;
  // CHECK-NEXT:     double _d_j = 0;
  // CHECK-NEXT:     SimpleFunctions _d_this_obj;
  // CHECK-NEXT:     SimpleFunctions *_d_this = &_d_this_obj;
  // CHECK-NEXT:     double _t0 = (this->x + this->y);
  // CHECK-NEXT:     double _t1 = i * j;
  // CHECK-NEXT:     return (_d_this->x + _d_this->y) * i + _t0 * _d_i + (_d_i * j + i * _d_j) * j + _t1 * _d_j;
  // CHECK-NEXT: }

  double lval_ref_noexcept_mem_fn_with_var_arg_list(double i, double j, ...) & noexcept { 
    return (x+y)*i + i*j*j; 
  } 

  // CHECK: double lval_ref_noexcept_mem_fn_with_var_arg_list_darg0(double i, double j, ...) & noexcept {
  // CHECK-NEXT:     double _d_i = 1;
  // CHECK-NEXT:     double _d_j = 0;
  // CHECK-NEXT:     SimpleFunctions _d_this_obj;
  // CHECK-NEXT:     SimpleFunctions *_d_this = &_d_this_obj;
  // CHECK-NEXT:     double _t0 = (this->x + this->y);
  // CHECK-NEXT:     double _t1 = i * j;
  // CHECK-NEXT:     return (_d_this->x + _d_this->y) * i + _t0 * _d_i + (_d_i * j + i * _d_j) * j + _t1 * _d_j;
  // CHECK-NEXT: }

  double const_lval_ref_noexcept_mem_fn(double i, double j) const & noexcept { 
    return (x+y)*i + i*j*j; 
  } 

  // CHECK: double const_lval_ref_noexcept_mem_fn_darg0(double i, double j) const & noexcept {
  // CHECK-NEXT:     double _d_i = 1;
  // CHECK-NEXT:     double _d_j = 0;
  // CHECK-NEXT:     const SimpleFunctions _d_this_obj;
  // CHECK-NEXT:     const SimpleFunctions *_d_this = &_d_this_obj;
  // CHECK-NEXT:     double _t0 = (this->x + this->y);
  // CHECK-NEXT:     double _t1 = i * j;
  // CHECK-NEXT:     return (_d_this->x + _d_this->y) * i + _t0 * _d_i + (_d_i * j + i * _d_j) * j + _t1 * _d_j;
  // CHECK-NEXT: }

  double const_lval_ref_noexcept_mem_fn_with_var_arg_list(double i, double j, ...) const & noexcept { 
    return (x+y)*i + i*j*j; 
  } 

  // CHECK: double const_lval_ref_noexcept_mem_fn_with_var_arg_list_darg0(double i, double j, ...) const & noexcept {
  // CHECK-NEXT:     double _d_i = 1;
  // CHECK-NEXT:     double _d_j = 0;
  // CHECK-NEXT:     const SimpleFunctions _d_this_obj;
  // CHECK-NEXT:     const SimpleFunctions *_d_this = &_d_this_obj;
  // CHECK-NEXT:     double _t0 = (this->x + this->y);
  // CHECK-NEXT:     double _t1 = i * j;
  // CHECK-NEXT:     return (_d_this->x + _d_this->y) * i + _t0 * _d_i + (_d_i * j + i * _d_j) * j + _t1 * _d_j;
  // CHECK-NEXT: }

  double volatile_lval_ref_noexcept_mem_fn(double i, double j) volatile & noexcept { 
    return (x+y)*i + i*j*j; 
  } 

  // CHECK: double volatile_lval_ref_noexcept_mem_fn_darg0(double i, double j) volatile & noexcept {
  // CHECK-NEXT:     double _d_i = 1;
  // CHECK-NEXT:     double _d_j = 0;
  // CHECK-NEXT:     volatile SimpleFunctions _d_this_obj;
  // CHECK-NEXT:     volatile SimpleFunctions *_d_this = &_d_this_obj;
  // CHECK-NEXT:     double _t0 = (this->x + this->y);
  // CHECK-NEXT:     double _t1 = i * j;
  // CHECK-NEXT:     return (_d_this->x + _d_this->y) * i + _t0 * _d_i + (_d_i * j + i * _d_j) * j + _t1 * _d_j;
  // CHECK-NEXT: }

  double volatile_lval_ref_noexcept_mem_fn_with_var_arg_list(double i, double j, ...) volatile & noexcept { 
    return (x+y)*i + i*j*j; 
  } 

  // CHECK: double volatile_lval_ref_noexcept_mem_fn_with_var_arg_list_darg0(double i, double j, ...) volatile & noexcept {
  // CHECK-NEXT:     double _d_i = 1;
  // CHECK-NEXT:     double _d_j = 0;
  // CHECK-NEXT:     volatile SimpleFunctions _d_this_obj;
  // CHECK-NEXT:     volatile SimpleFunctions *_d_this = &_d_this_obj;
  // CHECK-NEXT:     double _t0 = (this->x + this->y);
  // CHECK-NEXT:     double _t1 = i * j;
  // CHECK-NEXT:     return (_d_this->x + _d_this->y) * i + _t0 * _d_i + (_d_i * j + i * _d_j) * j + _t1 * _d_j;
  // CHECK-NEXT: }

  double const_volatile_lval_ref_noexcept_mem_fn(double i, double j) const volatile & noexcept { 
    return (x+y)*i + i*j*j; 
  } 

  // CHECK: double const_volatile_lval_ref_noexcept_mem_fn_darg0(double i, double j) const volatile & noexcept {
  // CHECK-NEXT:     double _d_i = 1;
  // CHECK-NEXT:     double _d_j = 0;
  // CHECK-NEXT:     const volatile SimpleFunctions _d_this_obj;
  // CHECK-NEXT:     const volatile SimpleFunctions *_d_this = &_d_this_obj;
  // CHECK-NEXT:     double _t0 = (this->x + this->y);
  // CHECK-NEXT:     double _t1 = i * j;
  // CHECK-NEXT:     return (_d_this->x + _d_this->y) * i + _t0 * _d_i + (_d_i * j + i * _d_j) * j + _t1 * _d_j;
  // CHECK-NEXT: }

  double const_volatile_lval_ref_noexcept_mem_fn_with_var_arg_list(double i, double j, ...) const volatile & noexcept { 
    return (x+y)*i + i*j*j; 
  } 

  // CHECK: double const_volatile_lval_ref_noexcept_mem_fn_with_var_arg_list_darg0(double i, double j, ...) const volatile & noexcept {
  // CHECK-NEXT:     double _d_i = 1;
  // CHECK-NEXT:     double _d_j = 0;
  // CHECK-NEXT:     const volatile SimpleFunctions _d_this_obj;
  // CHECK-NEXT:     const volatile SimpleFunctions *_d_this = &_d_this_obj;
  // CHECK-NEXT:     double _t0 = (this->x + this->y);
  // CHECK-NEXT:     double _t1 = i * j;
  // CHECK-NEXT:     return (_d_this->x + _d_this->y) * i + _t0 * _d_i + (_d_i * j + i * _d_j) * j + _t1 * _d_j;
  // CHECK-NEXT: }

  double rval_ref_noexcept_mem_fn(double i, double j) && noexcept { 
    return (x+y)*i + i*j*j; 
  } 

  // CHECK: double rval_ref_noexcept_mem_fn_darg0(double i, double j) && noexcept {
  // CHECK-NEXT:     double _d_i = 1;
  // CHECK-NEXT:     double _d_j = 0;
  // CHECK-NEXT:     SimpleFunctions _d_this_obj;
  // CHECK-NEXT:     SimpleFunctions *_d_this = &_d_this_obj;
  // CHECK-NEXT:     double _t0 = (this->x + this->y);
  // CHECK-NEXT:     double _t1 = i * j;
  // CHECK-NEXT:     return (_d_this->x + _d_this->y) * i + _t0 * _d_i + (_d_i * j + i * _d_j) * j + _t1 * _d_j;
  // CHECK-NEXT: }

  double rval_ref_noexcept_mem_fn_with_var_arg_list(double i, double j, ...) && noexcept { 
    return (x+y)*i + i*j*j; 
  } 

  // CHECK: double rval_ref_noexcept_mem_fn_with_var_arg_list_darg0(double i, double j, ...) && noexcept {
  // CHECK-NEXT:     double _d_i = 1;
  // CHECK-NEXT:     double _d_j = 0;
  // CHECK-NEXT:     SimpleFunctions _d_this_obj;
  // CHECK-NEXT:     SimpleFunctions *_d_this = &_d_this_obj;
  // CHECK-NEXT:     double _t0 = (this->x + this->y);
  // CHECK-NEXT:     double _t1 = i * j;
  // CHECK-NEXT:     return (_d_this->x + _d_this->y) * i + _t0 * _d_i + (_d_i * j + i * _d_j) * j + _t1 * _d_j;
  // CHECK-NEXT: }

  double const_rval_ref_noexcept_mem_fn(double i, double j) const && noexcept { 
    return (x+y)*i + i*j*j; 
  } 

  // CHECK: double const_rval_ref_noexcept_mem_fn_darg0(double i, double j) const && noexcept {
  // CHECK-NEXT:     double _d_i = 1;
  // CHECK-NEXT:     double _d_j = 0;
  // CHECK-NEXT:     const SimpleFunctions _d_this_obj;
  // CHECK-NEXT:     const SimpleFunctions *_d_this = &_d_this_obj;
  // CHECK-NEXT:     double _t0 = (this->x + this->y);
  // CHECK-NEXT:     double _t1 = i * j;
  // CHECK-NEXT:     return (_d_this->x + _d_this->y) * i + _t0 * _d_i + (_d_i * j + i * _d_j) * j + _t1 * _d_j;
  // CHECK-NEXT: }

  double const_rval_ref_noexcept_mem_fn_with_var_arg_list(double i, double j, ...) const && noexcept { 
    return (x+y)*i + i*j*j; 
  } 

  // CHECK: double const_rval_ref_noexcept_mem_fn_with_var_arg_list_darg0(double i, double j, ...) const && noexcept {
  // CHECK-NEXT:     double _d_i = 1;
  // CHECK-NEXT:     double _d_j = 0;
  // CHECK-NEXT:     const SimpleFunctions _d_this_obj;
  // CHECK-NEXT:     const SimpleFunctions *_d_this = &_d_this_obj;
  // CHECK-NEXT:     double _t0 = (this->x + this->y);
  // CHECK-NEXT:     double _t1 = i * j;
  // CHECK-NEXT:     return (_d_this->x + _d_this->y) * i + _t0 * _d_i + (_d_i * j + i * _d_j) * j + _t1 * _d_j;
  // CHECK-NEXT: }

  double volatile_rval_ref_noexcept_mem_fn(double i, double j) volatile && noexcept { 
    return (x+y)*i + i*j*j; 
  } 

  // CHECK: double volatile_rval_ref_noexcept_mem_fn_darg0(double i, double j) volatile && noexcept {
  // CHECK-NEXT:     double _d_i = 1;
  // CHECK-NEXT:     double _d_j = 0;
  // CHECK-NEXT:     volatile SimpleFunctions _d_this_obj;
  // CHECK-NEXT:     volatile SimpleFunctions *_d_this = &_d_this_obj;
  // CHECK-NEXT:     double _t0 = (this->x + this->y);
  // CHECK-NEXT:     double _t1 = i * j;
  // CHECK-NEXT:     return (_d_this->x + _d_this->y) * i + _t0 * _d_i + (_d_i * j + i * _d_j) * j + _t1 * _d_j;
  // CHECK-NEXT: }

  double volatile_rval_ref_noexcept_mem_fn_with_var_arg_list(double i, double j, ...) volatile && noexcept { 
    return (x+y)*i + i*j*j; 
  } 

  // CHECK: double volatile_rval_ref_noexcept_mem_fn_with_var_arg_list_darg0(double i, double j, ...) volatile && noexcept {
  // CHECK-NEXT:     double _d_i = 1;
  // CHECK-NEXT:     double _d_j = 0;
  // CHECK-NEXT:     volatile SimpleFunctions _d_this_obj;
  // CHECK-NEXT:     volatile SimpleFunctions *_d_this = &_d_this_obj;
  // CHECK-NEXT:     double _t0 = (this->x + this->y);
  // CHECK-NEXT:     double _t1 = i * j;
  // CHECK-NEXT:     return (_d_this->x + _d_this->y) * i + _t0 * _d_i + (_d_i * j + i * _d_j) * j + _t1 * _d_j;
  // CHECK-NEXT: }

  double const_volatile_rval_ref_noexcept_mem_fn(double i, double j) const volatile && noexcept { 
    return (x+y)*i + i*j*j; 
  } 

  // CHECK: double const_volatile_rval_ref_noexcept_mem_fn_darg0(double i, double j) const volatile && noexcept {
  // CHECK-NEXT:     double _d_i = 1;
  // CHECK-NEXT:     double _d_j = 0;
  // CHECK-NEXT:     const volatile SimpleFunctions _d_this_obj;
  // CHECK-NEXT:     const volatile SimpleFunctions *_d_this = &_d_this_obj;
  // CHECK-NEXT:     double _t0 = (this->x + this->y);
  // CHECK-NEXT:     double _t1 = i * j;
  // CHECK-NEXT:     return (_d_this->x + _d_this->y) * i + _t0 * _d_i + (_d_i * j + i * _d_j) * j + _t1 * _d_j;
  // CHECK-NEXT: }

  double const_volatile_rval_ref_noexcept_mem_fn_with_var_arg_list(double i, double j, ...) const volatile && noexcept { 
    return (x+y)*i + i*j*j; 
  } 

  // CHECK: double const_volatile_rval_ref_noexcept_mem_fn_with_var_arg_list_darg0(double i, double j, ...) const volatile && noexcept {
  // CHECK-NEXT:     double _d_i = 1;
  // CHECK-NEXT:     double _d_j = 0;
  // CHECK-NEXT:     const volatile SimpleFunctions _d_this_obj;
  // CHECK-NEXT:     const volatile SimpleFunctions *_d_this = &_d_this_obj;
  // CHECK-NEXT:     double _t0 = (this->x + this->y);
  // CHECK-NEXT:     double _t1 = i * j;
  // CHECK-NEXT:     return (_d_this->x + _d_this->y) * i + _t0 * _d_i + (_d_i * j + i * _d_j) * j + _t1 * _d_j;
  // CHECK-NEXT: }

  double* arr[10];

  double use_mem_var(double i, double j) {
    double *p;
    p = arr[1];
    return i;
  }

  // CHECK:   double use_mem_var_darg0(double i, double j) {
  // CHECK-NEXT:       double _d_i = 1;
  // CHECK-NEXT:       double _d_j = 0;
  // CHECK-NEXT:       SimpleFunctions _d_this_obj;
  // CHECK-NEXT:       SimpleFunctions *_d_this = &_d_this_obj;
  // CHECK-NEXT:       double *_d_p = nullptr;
  // CHECK-NEXT:       double *p;
  // CHECK-NEXT:       _d_p = _d_this->arr[1];
  // CHECK-NEXT:       p = this->arr[1];
  // CHECK-NEXT:       return _d_i;
  // CHECK-NEXT:   }

  SimpleFunctions operator+=(double value) noexcept {
      this->x += value;
      this->y += value;
      return *this;
  }

  SimpleFunctions& operator-=(double value) noexcept {
      this->x -= value;
      this->y -= value;
      return *this;
  }

  SimpleFunctions& operator*=(double value) noexcept {
    this->x *= value;
    this->y *= value;
    return *this;
  }

};

namespace clad {
namespace custom_derivatives {
namespace class_functions {
template <typename T, typename U>
clad::ValueAndPushforward<T, T> operator_plus_equal_pushforward(T* v, U value, T* d_v, U _d_value) noexcept {
    d_v->x += _d_value;
    v->x += value;
    d_v->y += _d_value;
    v->y += value;
    return {*v, *d_v};
}

template <typename T, typename U>
clad::ValueAndPushforward<T&, T&> operator_minus_equal_pushforward(T* v, U value, T* d_v, U _d_value) noexcept {
    d_v->x -= _d_value;
    v->x -= value;
    d_v->y -= _d_value;
    v->y -= value;
    return {*v, *d_v};
}

template <typename T, typename U>
clad::ValueAndPushforward<T&, T&> operator_star_equal_pushforward(T* v, U value, T* d_v, U _d_value) noexcept {
    double &_t0 = d_v->x;
    double &_t1 = v->x;
    _t0 = _t0 * value + _t1 * _d_value;
    _t1 *= value;

    double &_t2 = d_v->y;
    double &_t3 = v->y;
    _t2 = _t2 * value + _t3 * _d_value;
    _t3 *= value;

    return {*v, *d_v};
}
}
}
}

double addValueToSimpleFunction(SimpleFunctions v, double value) {
    v += value;
    return v.x;
}

double subtractValueFromSimpleFunction(SimpleFunctions v, double value) {
    v -= value;
    return v.x;
}

double multiplySimpleFunctionByValue(SimpleFunctions v, double value) {
    v *= value;
    return v.x;
}

  // CHECK:   double addValueToSimpleFunction_darg1(SimpleFunctions v, double value) {
  // CHECK-NEXT:       SimpleFunctions _d_v;
  // CHECK-NEXT:       double _d_value = 1;
  // CHECK-NEXT:       clad::ValueAndPushforward<SimpleFunctions, SimpleFunctions> _t0 = clad::custom_derivatives::class_functions::operator_plus_equal_pushforward(&v, value, &_d_v, _d_value);
  // CHECK-NEXT:       return _d_v.x;
  // CHECK-NEXT:   }

  // CHECK:   double subtractValueFromSimpleFunction_darg1(SimpleFunctions v, double value) {
  // CHECK-NEXT:       SimpleFunctions _d_v;
  // CHECK-NEXT:       double _d_value = 1;
  // CHECK-NEXT:       clad::ValueAndPushforward<SimpleFunctions &, SimpleFunctions &> _t0 = clad::custom_derivatives::class_functions::operator_minus_equal_pushforward(&v, value, &_d_v, _d_value);
  // CHECK-NEXT:       return _d_v.x;
  // CHECK-NEXT:   }

  // CHECK:   double multiplySimpleFunctionByValue_darg1(SimpleFunctions v, double value) {
  // CHECK-NEXT:       SimpleFunctions _d_v;
  // CHECK-NEXT:       double _d_value = 1;
  // CHECK-NEXT:       clad::ValueAndPushforward<SimpleFunctions &, SimpleFunctions &> _t0 = clad::custom_derivatives::class_functions::operator_star_equal_pushforward(&v, value, &_d_v, _d_value);
  // CHECK-NEXT:       return _d_v.x;
  // CHECK-NEXT:   }

  // CHECK: void mem_fn_with_void_return_pushforward(SimpleFunctions *_d_this) {
  // CHECK-NEXT:}


#define TEST(name,i,j) \
  auto d_##name = clad::differentiate(&SimpleFunctions::name,"i");\
  printf("%.2f\n", d_##name.execute(expr_1, 3, 5));\
  printf("%.2f\n", d_##name.execute(expr_2, 3, 5));\
  printf("\n");\

#define RVAL_REF_TEST(name, i, j) \
  auto d_##name = clad::differentiate(&SimpleFunctions::name,"i");\
  printf("%.2f\n", d_##name.execute(std::move(expr_1), 3, 5));\
  printf("%.2f\n", d_##name.execute(std::move(expr_2), 3, 5));\
  printf("\n");\

int main() {

  SimpleFunctions expr_1(2, 3);
  SimpleFunctions expr_2(3, 5);

  TEST(mem_fn, 3, 5)  // CHECK-EXEC: 30.00 
                      // CHECK-EXEC: 33.00 

  TEST(mem_fn_with_void_function_call, 3, 5) //CHECK-EXEC: 5.00

  TEST(mem_fn_with_var_arg_list, 3, 5)  // CHECK-EXEC: 30.00 
                                        // CHECK-EXEC: 33.00 

  TEST(const_mem_fn, 3, 5)  // CHECK-EXEC: 30.00 
                            // CHECK-EXEC: 33.00 

  TEST(const_mem_fn_with_var_arg_list, 3, 5)  // CHECK-EXEC: 30.00 
                                              // CHECK-EXEC: 33.00 

  TEST(volatile_mem_fn, 3, 5)  // CHECK-EXEC: 30.00 
                               // CHECK-EXEC: 33.00 

  TEST(volatile_mem_fn_with_var_arg_list, 3, 5)  // CHECK-EXEC: 30.00 
                                                 // CHECK-EXEC: 33.00 

  TEST(const_volatile_mem_fn, 3, 5)  // CHECK-EXEC: 30.00 
                                     // CHECK-EXEC: 33.00 

  TEST(const_volatile_mem_fn_with_var_arg_list, 3, 5)  // CHECK-EXEC: 30.00 
                                                       // CHECK-EXEC: 33.00 

  TEST(lval_ref_mem_fn, 3, 5)  // CHECK-EXEC: 30.00 
                               // CHECK-EXEC: 33.00 

  TEST(lval_ref_mem_fn_with_var_arg_list, 3, 5)  // CHECK-EXEC: 30.00 
                                                 // CHECK-EXEC: 33.00 

  TEST(const_lval_ref_mem_fn, 3, 5)  // CHECK-EXEC: 30.00 
                                     // CHECK-EXEC: 33.00 

  TEST(const_lval_ref_mem_fn_with_var_arg_list, 3, 5)  // CHECK-EXEC: 30.00 
                                                       // CHECK-EXEC: 33.00 

  TEST(volatile_lval_ref_mem_fn, 3, 5)  // CHECK-EXEC: 30.00 
                                        // CHECK-EXEC: 33.00 

  TEST(volatile_lval_ref_mem_fn_with_var_arg_list, 3, 5)  // CHECK-EXEC: 30.00 
                                                          // CHECK-EXEC: 33.00 

  TEST(const_volatile_lval_ref_mem_fn, 3, 5)  // CHECK-EXEC: 30.00 
                                              // CHECK-EXEC: 33.00 

  TEST(const_volatile_lval_ref_mem_fn_with_var_arg_list, 3, 5)  // CHECK-EXEC: 30.00 
                                                                // CHECK-EXEC: 33.00 

  RVAL_REF_TEST(rval_ref_mem_fn, 3, 5)  // CHECK-EXEC: 30.00 
                                        // CHECK-EXEC: 33.00 

  RVAL_REF_TEST(rval_ref_mem_fn_with_var_arg_list, 3, 5)  // CHECK-EXEC: 30.00 
                                                          // CHECK-EXEC: 33.00 

  RVAL_REF_TEST(const_rval_ref_mem_fn, 3, 5)  // CHECK-EXEC: 30.00 
                                              // CHECK-EXEC: 33.00 

  RVAL_REF_TEST(const_rval_ref_mem_fn_with_var_arg_list, 3, 5)  // CHECK-EXEC: 30.00 
                                                                // CHECK-EXEC: 33.00 

  RVAL_REF_TEST(volatile_rval_ref_mem_fn, 3, 5)  // CHECK-EXEC: 30.00 
                                                 // CHECK-EXEC: 33.00 

  RVAL_REF_TEST(volatile_rval_ref_mem_fn_with_var_arg_list, 3, 5)  // CHECK-EXEC: 30.00 
                                                                   // CHECK-EXEC: 33.00 

  RVAL_REF_TEST(const_volatile_rval_ref_mem_fn, 3, 5)  // CHECK-EXEC: 30.00 
                                                       // CHECK-EXEC: 33.00 

  RVAL_REF_TEST(const_volatile_rval_ref_mem_fn_with_var_arg_list, 3, 5)  // CHECK-EXEC: 30.00 
                                                                         // CHECK-EXEC: 33.00 

  TEST(noexcept_mem_fn, 3, 5)  // CHECK-EXEC: 30.00 
                               // CHECK-EXEC: 33.00 

  TEST(noexcept_mem_fn_with_var_arg_list, 3, 5)  // CHECK-EXEC: 30.00 
                                                 // CHECK-EXEC: 33.00 

  TEST(const_noexcept_mem_fn, 3, 5)  // CHECK-EXEC: 30.00 
                                     // CHECK-EXEC: 33.00 

  TEST(const_noexcept_mem_fn_with_var_arg_list, 3, 5)  // CHECK-EXEC: 30.00 
                                                       // CHECK-EXEC: 33.00 

  TEST(volatile_noexcept_mem_fn, 3, 5)  // CHECK-EXEC: 30.00 
                                        // CHECK-EXEC: 33.00 

  TEST(volatile_noexcept_mem_fn_with_var_arg_list, 3, 5)  // CHECK-EXEC: 30.00 
                                                          // CHECK-EXEC: 33.00 

  TEST(const_volatile_noexcept_mem_fn, 3, 5)  // CHECK-EXEC: 30.00 
                                              // CHECK-EXEC: 33.00 

  TEST(const_volatile_noexcept_mem_fn_with_var_arg_list, 3, 5)  // CHECK-EXEC: 30.00 
                                                                // CHECK-EXEC: 33.00 

  TEST(lval_ref_noexcept_mem_fn, 3, 5)  // CHECK-EXEC: 30.00 
                                        // CHECK-EXEC: 33.00 

  TEST(lval_ref_noexcept_mem_fn_with_var_arg_list, 3, 5)  // CHECK-EXEC: 30.00 
                                                          // CHECK-EXEC: 33.00 

  TEST(const_lval_ref_noexcept_mem_fn, 3, 5)  // CHECK-EXEC: 30.00 
                                              // CHECK-EXEC: 33.00 

  TEST(const_lval_ref_noexcept_mem_fn_with_var_arg_list, 3, 5)  // CHECK-EXEC: 30.00 
                                                                // CHECK-EXEC: 33.00 

  TEST(volatile_lval_ref_noexcept_mem_fn, 3, 5)  // CHECK-EXEC: 30.00 
                                                 // CHECK-EXEC: 33.00 

  TEST(volatile_lval_ref_noexcept_mem_fn_with_var_arg_list, 3, 5)  // CHECK-EXEC: 30.00 
                                                                   // CHECK-EXEC: 33.00 

  TEST(const_volatile_lval_ref_noexcept_mem_fn, 3, 5)  // CHECK-EXEC: 30.00 
                                                       // CHECK-EXEC: 33.00 

  TEST(const_volatile_lval_ref_noexcept_mem_fn_with_var_arg_list, 3, 5)  // CHECK-EXEC: 30.00 
                                                                         // CHECK-EXEC: 33.00 

  RVAL_REF_TEST(rval_ref_noexcept_mem_fn, 3, 5)  // CHECK-EXEC: 30.00 
                                                 // CHECK-EXEC: 33.00 

  RVAL_REF_TEST(rval_ref_noexcept_mem_fn_with_var_arg_list, 3, 5)  // CHECK-EXEC: 30.00 
                                                                   // CHECK-EXEC: 33.00 

  RVAL_REF_TEST(const_rval_ref_noexcept_mem_fn, 3, 5)  // CHECK-EXEC: 30.00 
                                                       // CHECK-EXEC: 33.00 

  RVAL_REF_TEST(const_rval_ref_noexcept_mem_fn_with_var_arg_list, 3, 5)  // CHECK-EXEC: 30.00 
                                                                         // CHECK-EXEC: 33.00 

  RVAL_REF_TEST(volatile_rval_ref_noexcept_mem_fn, 3, 5)  // CHECK-EXEC: 30.00 
                                                          // CHECK-EXEC: 33.00 

  RVAL_REF_TEST(volatile_rval_ref_noexcept_mem_fn_with_var_arg_list, 3, 5)  // CHECK-EXEC: 30.00 
                                                                            // CHECK-EXEC: 33.00 

  RVAL_REF_TEST(const_volatile_rval_ref_noexcept_mem_fn, 3, 5)  // CHECK-EXEC: 30.00 
                                                                // CHECK-EXEC: 33.00 

  RVAL_REF_TEST(const_volatile_rval_ref_noexcept_mem_fn_with_var_arg_list, 3, 5)  // CHECK-EXEC: 30.00 
                                                                                  // CHECK-EXEC: 33.00 
  d_mem_fn.setObject(&expr_1);
  printf("%.2f %.2f\n", d_mem_fn.execute(3, 5), d_mem_fn.execute(expr_2, 3, 5));  // CHECK-EXEC: 30.00
                                                                                // CHECK-EXEC: 33.00
  d_mem_fn.clearObject();
  d_mem_fn.setObject(expr_1);
  printf("%.2f %.2f\n", d_mem_fn.execute(3, 5), d_mem_fn.execute(expr_2, 3, 5));  // CHECK-EXEC: 30.00
                                                                                // CHECK-EXEC: 33.00

  auto d_use_mem_var = clad::differentiate(&SimpleFunctions::use_mem_var, "i");

  SimpleFunctions sf(2, 3);
  auto d_addValueToSimpleFunction = clad::differentiate(addValueToSimpleFunction, "value");
  auto d_subtractValueFromSimpleFunction = clad::differentiate(subtractValueFromSimpleFunction, "value");
  auto d_multiplySimpleFunctionByValue = clad::differentiate(multiplySimpleFunctionByValue, "value");

  printf("%.2f %.2f %.2f\n", d_addValueToSimpleFunction.execute(sf, 3), d_subtractValueFromSimpleFunction.execute(sf, 3), d_multiplySimpleFunctionByValue.execute(sf, 3));  // CHECK-EXEC: 1.00 -1.00 2.00
}