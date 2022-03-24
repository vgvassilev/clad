// RUN: %cladclang %s -I%S/../../include -oMemberFunctions.out 2>&1 | FileCheck %s
// RUN: ./MemberFunctions.out | FileCheck -check-prefix=CHECK-EXEC %s
// RUN: %cladclang -std=c++14 %s -I%S/../../include -oMemberFunctions-cpp14.out 2>&1 | FileCheck %s
// RUN: ./MemberFunctions-cpp14.out | FileCheck -check-prefix=CHECK-EXEC %s
// RUN: %cladclang -std=c++17 %s -I%S/../../include -oMemberFunctions-cpp17.out 2>&1 | FileCheck %s
// RUN: ./MemberFunctions-cpp17.out | FileCheck -check-prefix=CHECK-EXEC %s
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
  // CHECK-NEXT:       double *_d_p;
  // CHECK-NEXT:       double *p;
  // CHECK-NEXT:       _d_p = _d_this->arr[1];
  // CHECK-NEXT:       p = this->arr[1];
  // CHECK-NEXT:       return _d_i;
  // CHECK-NEXT:   }
};

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
}