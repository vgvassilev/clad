// RUN: %cladclang %s -I%S/../../include -oHessians.out 2>&1 | %filecheck %s
// RUN: ./Hessians.out | %filecheck_exec %s
// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -enable-tbr %s -I%S/../../include -oHessians.out
// RUN: ./Hessians.out | %filecheck_exec %s

//CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"

__attribute__((always_inline)) double f_cubed_add1(double a, double b) {
  return a * a * a + b * b * b;
}
//CHECK:{{[__attribute__((always_inline)) ]*}}double f_cubed_add1_darg0(double a, double b){{[ __attribute__((always_inline))]*}};

void f_cubed_add1_darg0_grad(double a, double b, double *_d_a, double *_d_b);
//CHECK:{{[__attribute__((always_inline)) ]*}}void f_cubed_add1_darg0_grad(double a, double b, double *_d_a, double *_d_b){{[ __attribute__((always_inline))]*}};

void f_cubed_add1_darg1_grad(double a, double b, double *_d_a, double *_d_b);
//CHECK:{{[__attribute__((always_inline)) ]*}}void f_cubed_add1_darg1_grad(double a, double b, double *_d_a, double *_d_b){{[ __attribute__((always_inline))]*}};

void f_cubed_add1_hessian(double a, double b, double *hessianMatrix);
//CHECK:{{[__attribute__((always_inline)) ]*}}void f_cubed_add1_hessian(double a, double b, double *hessianMatrix){{[ __attribute__((always_inline))]*}} {
//CHECK-NEXT:    f_cubed_add1_darg0_grad(a, b, hessianMatrix + {{0U|0UL}}, hessianMatrix + {{1U|1UL}});
//CHECK-NEXT:    f_cubed_add1_darg1_grad(a, b, hessianMatrix + {{2U|2UL}}, hessianMatrix + {{3U|3UL}});
//CHECK-NEXT: }

double f_suvat1(double u, double t) {
  return ((u * t) + ((0.5) * (9.81) * (t * t)));
}

void f_suvat1_darg0_grad(double u, double t, double *_d_u, double *_d_t);
void f_suvat1_darg1_grad(double u, double t, double *_d_u, double *_d_t);

void f_suvat1_hessian(double u, double t, double *hessianMatrix);
//CHECK:void f_suvat1_hessian(double u, double t, double *hessianMatrix) {
//CHECK-NEXT:    f_suvat1_darg0_grad(u, t, hessianMatrix + {{0U|0UL}}, hessianMatrix + {{1U|1UL}});
//CHECK-NEXT:    f_suvat1_darg1_grad(u, t, hessianMatrix + {{2U|2UL}}, hessianMatrix + {{3U|3UL}});
//CHECK-NEXT:}

double f_cond3(double x, double c) {
  if (c > 0) {
    return (x*x*x) + (c*c*c);
  }
  else {
    return (x*x*x) - (c*c*c);
  }
}

void f_cond3_darg0_grad(double x, double c, double *_d_x, double *_d_c);
void f_cond3_darg1_grad(double x, double c, double *_d_x, double *_d_c);

void f_cond3_hessian(double x, double c, double *hessianMatrix);
//CHECK:void f_cond3_hessian(double x, double c, double *hessianMatrix) {
//CHECK-NEXT:    f_cond3_darg0_grad(x, c, hessianMatrix + {{0U|0UL}}, hessianMatrix + {{1U|1UL}});
//CHECK-NEXT:    f_cond3_darg1_grad(x, c, hessianMatrix + {{2U|2UL}}, hessianMatrix + {{3U|3UL}});
//CHECK-NEXT:}

double f_power10(double x) {
  return x * x * x * x * x * x * x * x * x * x;
}

void f_power10_darg0_grad(double x, double *_d_x);

void f_power10_hessian(double x, double *hessianMatrix);
//CHECK: void f_power10_hessian(double x, double *hessianMatrix) {
//CHECK-NEXT:     f_power10_darg0_grad(x, hessianMatrix + {{0U|0UL}});
//CHECK-NEXT: }

struct Experiment {
  double x, y;
  Experiment (double p_x = 0, double p_y = 0) : x(p_x), y(p_y) {}
  double someMethod (double i, double j) {
    return i*i*j + j*j*i;
  }

  void someMethod_darg0_grad(double i,
                             double j,
                             double *_d_i,
                             double *_d_j);
  void someMethod_darg1_grad(double i,
                             double j,
                             double *_d_i,
                             double *_d_j);

  void someMethod_hessian(double x, double *hessianMatrix);
  // CHECK: void someMethod_hessian(double i, double j, double *hessianMatrix) {
  // CHECK-NEXT:     Experiment _d_this;
  // CHECK-NEXT:     this->someMethod_darg0_grad(i, j, &_d_this, hessianMatrix + {{0U|0UL}}, hessianMatrix + {{1U|1UL}});
  // CHECK-NEXT:     Experiment _d_this0;
  // CHECK-NEXT:     this->someMethod_darg1_grad(i, j, &_d_this0, hessianMatrix + {{2U|2UL}}, hessianMatrix + {{3U|3UL}});
  // CHECK-NEXT: }
};

struct Widget {
  double x, y;
  Widget(double p_x=0, double p_y=0) : x(p_x), y(p_y) {}
  double memFn_1(double i, double j) {
    return x*y*i*j;
  }

  void memFn_1_darg0_grad(double i,
                          double j,
                          double *_d_i,
                          double *_d_j);
  void memFn_1_darg1_grad(double i,
                          double j,
                          double *_d_i,
                          double *_d_j);

  void memFn_1_hessian(double i,
                       double j,
                       double *hessianMatrix);
  // CHECK: void memFn_1_hessian(double i, double j, double *hessianMatrix) {
  // CHECK-NEXT:     Widget _d_this;
  // CHECK-NEXT:     this->memFn_1_darg0_grad(i, j, &_d_this, hessianMatrix + {{0U|0UL}}, hessianMatrix + {{1U|1UL}});
  // CHECK-NEXT:     Widget _d_this0;
  // CHECK-NEXT:     this->memFn_1_darg1_grad(i, j, &_d_this0, hessianMatrix + {{2U|2UL}}, hessianMatrix + {{3U|3UL}});
  // CHECK-NEXT: }

  double memFn_2(double i, double j) {
    double a = x*i*j;
    double b = 4*a;
    return b*y*y*i;
  }

  void memFn_2_darg0_grad(double i,
                          double j,
                          double *_d_i,
                          double *_d_j);
  void memFn_2_darg1_grad(double i,
                          double j,
                          double *_d_i,
                          double *_d_j);

  void memFn_2_hessian(double i,
                       double j,
                       double *hessianMatrix);
  // CHECK: void memFn_2_hessian(double i, double j, double *hessianMatrix) {
  // CHECK-NEXT:     Widget _d_this;
  // CHECK-NEXT:     this->memFn_2_darg0_grad(i, j, &_d_this, hessianMatrix + {{0U|0UL}}, hessianMatrix + {{1U|1UL}});
  // CHECK-NEXT:     Widget _d_this0;
  // CHECK-NEXT:     this->memFn_2_darg1_grad(i, j, &_d_this0, hessianMatrix + {{2U|2UL}}, hessianMatrix + {{3U|3UL}});
  // CHECK-NEXT: }
};

double fn_def_arg(double i=0, double j=0) {
  return 2*i*j;
}

void fn_def_arg_darg0_grad(double, double, double*,
                           double*);
void fn_def_arg_darg1_grad(double, double, double*,
                           double*);
void fn_def_arg_hessian(double, double, double*);

#define TEST1(F, x) { \
  result[0] = 0;\
  auto h = clad::hessian(F);\
  h.execute(x, result);\
  printf("Result is = {%.2f}\n", result[0]); \
  F##_hessian(x, result);\
  F##_darg0_grad(x, result);\
}

#define TEST2(F, x, y)                                                         \
  {                                                                            \
    result[0] = 0;                                                             \
    result[1] = 0;                                                             \
    result[2] = 0;                                                             \
    result[3] = 0;                                                             \
    auto h = clad::hessian(F);                                                 \
    h.execute(x, y, result);                                                   \
    printf("Result is = {%.2f, %.2f, %.2f, %.2f}\n",                           \
           result[0],                                                          \
           result[1],                                                          \
           result[2],                                                          \
           result[3]);                                                         \
    F##_darg0_grad(x, y, &result[0], &result[1]);                              \
    F##_darg1_grad(x, y, &result[2], &result[3]);                              \
    F##_hessian(x, y, result);                                                 \
}

#define TEST3(F, Obj, ...) { \
  result[0]=result[1]=result[2]=result[3]=0;\
  auto h = clad::hessian(F);\
  h.execute(Obj, __VA_ARGS__, result);\
  printf("Result is = {%.2f, %.2f, %.2f, %.2f}\n", \
         result[0],          \
         result[1],          \
         result[2],          \
         result[3]);\
}


int main() {
  double result[10];

  TEST2(f_cubed_add1, 1, 2); // CHECK-EXEC: Result is = {6.00, 0.00, 0.00, 12.00}
  TEST2(f_suvat1, 1, 2); // CHECK-EXEC: Result is = {0.00, 1.00, 1.00, 9.81}
  TEST2(f_cond3, 5, -2); // CHECK-EXEC: Result is = {30.00, 0.00, 0.00, 12.00}
  TEST1(f_power10, 5); // CHECK-EXEC: Result is = {35156250.00}
  Experiment E(11, 17);
  Widget W(3, 5);
  TEST3(&Experiment::someMethod, E, 3, 5);  // CHECK-EXEC: Result is = {10.00, 16.00, 16.00, 6.00}
  TEST3(&Widget::memFn_1, W, 7, 9); // CHECK-EXEC: Result is = {0.00, 15.00, 15.00, 0.00}
  TEST3(&Widget::memFn_2, W, 7, 9); // CHECK-EXEC: Result is = {5400.00, 4200.00, 4200.00, 0.00}
  TEST2(fn_def_arg, 3, 5);  // CHECK-EXEC: Result is = {0.00, 2.00, 2.00, 0.00}

//CHECK:{{[__attribute__((always_inline)) ]*}}double f_cubed_add1_darg0(double a, double b){{[ __attribute__((always_inline))]*}} {
//CHECK-NEXT:    double _d_a = 1;
//CHECK-NEXT:    double _d_b = 0;
//CHECK-NEXT:    double _t0 = a * a;
//CHECK-NEXT:    double _t1 = b * b;
//CHECK-NEXT:    return (_d_a * a + a * _d_a) * a + _t0 * _d_a + (_d_b * b + b * _d_b) * b + _t1 * _d_b;
//CHECK-NEXT:}

//CHECK:{{[__attribute__((always_inline)) ]*}}void f_cubed_add1_darg0_grad(double a, double b, double *_d_a, double *_d_b){{[ __attribute__((always_inline))]*}} {
//CHECK-NEXT:    double _d__d_a = 0;
//CHECK-NEXT:    double _d_a0 = 1;
//CHECK-NEXT:    double _d__d_b = 0;
//CHECK-NEXT:    double _d_b0 = 0;
//CHECK-NEXT:    double _d__t0 = 0;
//CHECK-NEXT:    double _t00 = a * a;
//CHECK-NEXT:    double _d__t1 = 0;
//CHECK-NEXT:    double _t10 = b * b;
//CHECK-NEXT:    {
//CHECK-NEXT:        _d__d_a += 1 * a * a;
//CHECK-NEXT:        *_d_a += _d_a0 * 1 * a;
//CHECK-NEXT:        *_d_a += 1 * a * _d_a0;
//CHECK-NEXT:        _d__d_a += a * 1 * a;
//CHECK-NEXT:        *_d_a += (_d_a0 * a + a * _d_a0) * 1;
//CHECK-NEXT:        _d__t0 += 1 * _d_a0;
//CHECK-NEXT:        _d__d_a += _t00 * 1;
//CHECK-NEXT:        _d__d_b += 1 * b * b;
//CHECK-NEXT:        *_d_b += _d_b0 * 1 * b;
//CHECK-NEXT:        *_d_b += 1 * b * _d_b0;
//CHECK-NEXT:        _d__d_b += b * 1 * b;
//CHECK-NEXT:        *_d_b += (_d_b0 * b + b * _d_b0) * 1;
//CHECK-NEXT:        _d__t1 += 1 * _d_b0;
//CHECK-NEXT:        _d__d_b += _t10 * 1;
//CHECK-NEXT:    }
//CHECK-NEXT:    {
//CHECK-NEXT:        *_d_b += _d__t1 * b;
//CHECK-NEXT:        *_d_b += b * _d__t1;
//CHECK-NEXT:    }
//CHECK-NEXT:    {
//CHECK-NEXT:        *_d_a += _d__t0 * a;
//CHECK-NEXT:        *_d_a += a * _d__t0;
//CHECK-NEXT:    }
//CHECK-NEXT:}

//CHECK:{{[__attribute__((always_inline)) ]*}}void f_cubed_add1_darg1_grad(double a, double b, double *_d_a, double *_d_b){{[ __attribute__((always_inline))]*}} {
//CHECK-NEXT:    double _d__d_a = 0;
//CHECK-NEXT:    double _d_a0 = 0;
//CHECK-NEXT:    double _d__d_b = 0;
//CHECK-NEXT:    double _d_b0 = 1;
//CHECK-NEXT:    double _d__t0 = 0;
//CHECK-NEXT:    double _t00 = a * a;
//CHECK-NEXT:    double _d__t1 = 0;
//CHECK-NEXT:    double _t10 = b * b;
//CHECK-NEXT:    {
//CHECK-NEXT:        _d__d_a += 1 * a * a;
//CHECK-NEXT:        *_d_a += _d_a0 * 1 * a;
//CHECK-NEXT:        *_d_a += 1 * a * _d_a0;
//CHECK-NEXT:        _d__d_a += a * 1 * a;
//CHECK-NEXT:        *_d_a += (_d_a0 * a + a * _d_a0) * 1;
//CHECK-NEXT:        _d__t0 += 1 * _d_a0;
//CHECK-NEXT:        _d__d_a += _t00 * 1;
//CHECK-NEXT:        _d__d_b += 1 * b * b;
//CHECK-NEXT:        *_d_b += _d_b0 * 1 * b;
//CHECK-NEXT:        *_d_b += 1 * b * _d_b0;
//CHECK-NEXT:        _d__d_b += b * 1 * b;
//CHECK-NEXT:        *_d_b += (_d_b0 * b + b * _d_b0) * 1;
//CHECK-NEXT:        _d__t1 += 1 * _d_b0;
//CHECK-NEXT:        _d__d_b += _t10 * 1;
//CHECK-NEXT:    }
//CHECK-NEXT:    {
//CHECK-NEXT:        *_d_b += _d__t1 * b;
//CHECK-NEXT:        *_d_b += b * _d__t1;
//CHECK-NEXT:    }
//CHECK-NEXT:    {
//CHECK-NEXT:        *_d_a += _d__t0 * a;
//CHECK-NEXT:        *_d_a += a * _d__t0;
//CHECK-NEXT:    }
//CHECK-NEXT:}
}
