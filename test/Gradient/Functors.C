// RUN: %cladclang %s -I%S/../../include -oFunctors.out 2>&1 | FileCheck %s
// RUN: ./Functors.out | FileCheck -check-prefix=CHECK-EXEC %s
// CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"

struct Experiment {
  mutable double x, y;
  Experiment(double p_x = 0, double p_y = 0) : x(p_x), y(p_y) {}
  double operator()(double i, double j) { return x * i * j; }
  void setX(double val) { x = val; }

  Experiment& operator=(const Experiment& E) = default;

  // CHECK: void operator_call_grad(double i, double j, clad::array_ref<Experiment> _d_this, clad::array_ref<double> _d_i, clad::array_ref<double> _d_j) {
  // CHECK-NEXT:     double _t0;
  // CHECK-NEXT:     double _t1;
  // CHECK-NEXT:     double _t2;
  // CHECK-NEXT:     double _t3;
  // CHECK-NEXT:     _t2 = this->x;
  // CHECK-NEXT:     _t1 = i;
  // CHECK-NEXT:     _t3 = _t2 * _t1;
  // CHECK-NEXT:     _t0 = j;
  // CHECK-NEXT:     {
  // CHECK-NEXT:         double _r0 = 1 * _t0;
  // CHECK-NEXT:         double _r1 = _r0 * _t1;
  // CHECK-NEXT:         (* _d_this).x += _r1;
  // CHECK-NEXT:         double _r2 = _t2 * _r0;
  // CHECK-NEXT:         * _d_i += _r2;
  // CHECK-NEXT:         double _r3 = _t3 * 1;
  // CHECK-NEXT:         * _d_j += _r3;
  // CHECK-NEXT:     }
  // CHECK-NEXT: }
};

struct ExperimentConst {
  mutable double x, y;
  ExperimentConst(double p_x = 0, double p_y = 0) : x(p_x), y(p_y) {}
  double operator()(double i, double j) const { return x * i * j; }
  void setX(double val) const { x = val; }

  ExperimentConst& operator=(const ExperimentConst& E) = default;

  // CHECK: void operator_call_grad(double i, double j, clad::array_ref<ExperimentConst> _d_this, clad::array_ref<double> _d_i, clad::array_ref<double> _d_j) const {
  // CHECK-NEXT:     double _t0;
  // CHECK-NEXT:     double _t1;
  // CHECK-NEXT:     double _t2;
  // CHECK-NEXT:     double _t3;
  // CHECK-NEXT:     _t2 = this->x;
  // CHECK-NEXT:     _t1 = i;
  // CHECK-NEXT:     _t3 = _t2 * _t1;
  // CHECK-NEXT:     _t0 = j;
  // CHECK-NEXT:     {
  // CHECK-NEXT:         double _r0 = 1 * _t0;
  // CHECK-NEXT:         double _r1 = _r0 * _t1;
  // CHECK-NEXT:         (* _d_this).x += _r1;
  // CHECK-NEXT:         double _r2 = _t2 * _r0;
  // CHECK-NEXT:         * _d_i += _r2;
  // CHECK-NEXT:         double _r3 = _t3 * 1;
  // CHECK-NEXT:         * _d_j += _r3;
  // CHECK-NEXT:     }
  // CHECK-NEXT: }
};

struct ExperimentVolatile {
  mutable double x, y;
  ExperimentVolatile(double p_x = 0, double p_y = 0) : x(p_x), y(p_y) {}
  double operator()(double i, double j) volatile { return x * i * j; }
  void setX(double val) volatile { x = val; }

  volatile ExperimentVolatile&
  operator=(volatile ExperimentVolatile&& E) volatile {
    x = E.x;
    y = E.y;
    return (*this);
  };

  // CHECK: void operator_call_grad(double i, double j, clad::array_ref<volatile ExperimentVolatile> _d_this, clad::array_ref<double> _d_i, clad::array_ref<double> _d_j) volatile {
  // CHECK-NEXT:     double _t0;
  // CHECK-NEXT:     double _t1;
  // CHECK-NEXT:     volatile double _t2;
  // CHECK-NEXT:     double _t3;
  // CHECK-NEXT:     _t2 = this->x;
  // CHECK-NEXT:     _t1 = i;
  // CHECK-NEXT:     _t3 = _t2 * _t1;
  // CHECK-NEXT:     _t0 = j;
  // CHECK-NEXT:     {
  // CHECK-NEXT:         double _r0 = 1 * _t0;
  // CHECK-NEXT:         double _r1 = _r0 * _t1;
  // CHECK-NEXT:         (* _d_this).x += _r1;
  // CHECK-NEXT:         double _r2 = _t2 * _r0;
  // CHECK-NEXT:         * _d_i += _r2;
  // CHECK-NEXT:         double _r3 = _t3 * 1;
  // CHECK-NEXT:         * _d_j += _r3;
  // CHECK-NEXT:     }
  // CHECK-NEXT: }
};

struct ExperimentConstVolatile {
  mutable double x, y;
  ExperimentConstVolatile(double p_x = 0, double p_y = 0) : x(p_x), y(p_y) {}
  double operator()(double i, double j) const volatile { return x * i * j; }
  void setX(double val) const volatile { x = val; }

  const volatile ExperimentConstVolatile&
  operator=(const volatile ExperimentConstVolatile& E) const volatile {
    x = E.x;
    y = E.y;
    return (*this);
  };

  // CHECK: void operator_call_grad(double i, double j, clad::array_ref<volatile ExperimentConstVolatile> _d_this, clad::array_ref<double> _d_i, clad::array_ref<double> _d_j) const volatile {
  // CHECK-NEXT:     double _t0;
  // CHECK-NEXT:     double _t1;
  // CHECK-NEXT:     volatile double _t2;
  // CHECK-NEXT:     double _t3;
  // CHECK-NEXT:     _t2 = this->x;
  // CHECK-NEXT:     _t1 = i;
  // CHECK-NEXT:     _t3 = _t2 * _t1;
  // CHECK-NEXT:     _t0 = j;
  // CHECK-NEXT:     {
  // CHECK-NEXT:         double _r0 = 1 * _t0;
  // CHECK-NEXT:         double _r1 = _r0 * _t1;
  // CHECK-NEXT:         (* _d_this).x += _r1;
  // CHECK-NEXT:         double _r2 = _t2 * _r0;
  // CHECK-NEXT:         * _d_i += _r2;
  // CHECK-NEXT:         double _r3 = _t3 * 1;
  // CHECK-NEXT:         * _d_j += _r3;
  // CHECK-NEXT:     }
  // CHECK-NEXT: }
};

namespace outer {
namespace inner {
struct ExperimentNNS {
  mutable double x, y;
  ExperimentNNS(double p_x = 0, double p_y = 0) : x(p_x), y(p_y) {}
  double operator()(double i, double j) { return x * i * j; }
  void setX(double val) { x = val; }

  ExperimentNNS& operator=(const ExperimentNNS& E) = default;

  // CHECK: void operator_call_grad(double i, double j, clad::array_ref<ExperimentNNS> _d_this, clad::array_ref<double> _d_i, clad::array_ref<double> _d_j) {
  // CHECK-NEXT:     double _t0;
  // CHECK-NEXT:     double _t1;
  // CHECK-NEXT:     double _t2;
  // CHECK-NEXT:     double _t3;
  // CHECK-NEXT:     _t2 = this->x;
  // CHECK-NEXT:     _t1 = i;
  // CHECK-NEXT:     _t3 = _t2 * _t1;
  // CHECK-NEXT:     _t0 = j;
  // CHECK-NEXT:     {
  // CHECK-NEXT:         double _r0 = 1 * _t0;
  // CHECK-NEXT:         double _r1 = _r0 * _t1;
  // CHECK-NEXT:         (* _d_this).x += _r1;
  // CHECK-NEXT:         double _r2 = _t2 * _r0;
  // CHECK-NEXT:         * _d_i += _r2;
  // CHECK-NEXT:         double _r3 = _t3 * 1;
  // CHECK-NEXT:         * _d_j += _r3;
  // CHECK-NEXT:     }
  // CHECK-NEXT: }
};
} // namespace inner
} // namespace outer

#define INIT(E)                                                                \
  auto E##_grad = clad::gradient(&E);                                          \
  auto E##Ref_grad = clad::gradient(E);

#define TEST(E, d_E)                                                           \
  res[0] = res[1] = 0;                                                         \
  d_E = decltype(d_E)();                                                       \
  E##_grad.execute(7, 9, &d_E, &res[0], &res[1]);                              \
  printf("%.2f %.2f\n", res[0], res[1]);                                       \
  res[0] = res[1] = 0;                                                         \
  d_E = decltype(d_E)();                                                       \
  E##Ref_grad.execute(7, 9, &d_E, &res[0], &res[1]);                           \
  printf("%.2f %.2f\n", res[0], res[1]);

#define TEST_LAMBDA(E)                                                         \
  res[0] = res[1] = 0;                                                         \
  E##_grad.execute(7, 9, &res[0], &res[1]);                                    \
  printf("%.2f %.2f\n", res[0], res[1]);                                       \
  res[0] = res[1] = 0;                                                         \
  E##Ref_grad.execute(7, 9, &res[0], &res[1]);                                 \
  printf("%.2f %.2f\n", res[0], res[1]);

double x = 3;

int main() {
  Experiment E(3, 5), d_E, d_E_Const;
  const ExperimentConst E_Const(3, 5);
  volatile ExperimentVolatile E_Volatile(3, 5), d_E_Volatile, d_E_ConstVolatile;
  const volatile ExperimentConstVolatile E_ConstVolatile(3, 5);
  outer::inner::ExperimentNNS E_NNS(3, 5), d_E_NNS;

  auto lambda = [](double i, double j) { return i * i * j; };

  // CHECK: inline void operator_call_grad(double i, double j, clad::array_ref<double> _d_i, clad::array_ref<double> _d_j) const {
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

  auto lambdaWithCapture = [&](double ii, double j) { return x * ii * j; };

  // CHECK: inline void operator_call_grad(double ii, double j, clad::array_ref<double> _d_ii, clad::array_ref<double> _d_j) const {
  // CHECK-NEXT:     double _t0;
  // CHECK-NEXT:     double _t1;
  // CHECK-NEXT:     double _t2;
  // CHECK-NEXT:     double _t3;
  // CHECK-NEXT:     _t2 = x;
  // CHECK-NEXT:     _t1 = ii;
  // CHECK-NEXT:     _t3 = _t2 * _t1;
  // CHECK-NEXT:     _t0 = j;
  // CHECK-NEXT:     {
  // CHECK-NEXT:         double _r0 = 1 * _t0;
  // CHECK-NEXT:         double _r1 = _r0 * _t1;
  // CHECK-NEXT:         double _r2 = _t2 * _r0;
  // CHECK-NEXT:         * _d_ii += _r2;
  // CHECK-NEXT:         double _r3 = _t3 * 1;
  // CHECK-NEXT:         * _d_j += _r3;
  // CHECK-NEXT:     }
  // CHECK-NEXT: }

  double res[2];

  INIT(E);
  INIT(E_Const);
  INIT(E_Volatile);
  INIT(E_ConstVolatile);
  INIT(E_NNS);
  INIT(lambda);
  INIT(lambdaWithCapture);

  TEST(E, d_E);                               // CHECK-EXEC: 27.00 21.00
                                              // CHECK-EXEC: 27.00 21.00
  TEST(E_Const, d_E_Const);                   // CHECK-EXEC: 27.00 21.00
                                              // CHECK-EXEC: 27.00 21.00
  TEST(E_Volatile, d_E_Volatile);             // CHECK-EXEC: 27.00 21.00
                                              // CHECK-EXEC: 27.00 21.00
  TEST(E_ConstVolatile, d_E_ConstVolatile);   // CHECK-EXEC: 27.00 21.00
                                              // CHECK-EXEC: 27.00 21.00

  TEST(E_NNS, d_E_NNS);                       // CHECK-EXEC: 27.00 21.00
                                              // CHECK-EXEC: 27.00 21.00
  TEST_LAMBDA(lambda);                        // CHECK-EXEC: 126.00 49.00
                                              // CHECK-EXEC: 126.00 49.00

  TEST_LAMBDA(lambdaWithCapture);             // CHECK-EXEC: 27.00 21.00
                                              // CHECK-EXEC: 27.00 21.00

  E.setX(6);
  E_Const.setX(6);
  E_Volatile.setX(6);
  E_ConstVolatile.setX(6);
  E_NNS.setX(6);
  x = 6;

  TEST(E, d_E);                               // CHECK-EXEC: 54.00 42.00
                                              // CHECK-EXEC: 54.00 42.00
  TEST(E_Const, d_E_Const);                   // CHECK-EXEC: 54.00 42.00
                                              // CHECK-EXEC: 54.00 42.00
  TEST(E_Volatile, d_E_Volatile);             // CHECK-EXEC: 54.00 42.00
                                              // CHECK-EXEC: 54.00 42.00
  TEST(E_ConstVolatile, d_E_ConstVolatile);   // CHECK-EXEC: 54.00 42.00
                                              // CHECK-EXEC: 54.00 42.00

  TEST(E_NNS, d_E_NNS);                       // CHECK-EXEC: 54.00 42.00
                                              // CHECK-EXEC: 54.00 42.00
  TEST_LAMBDA(lambdaWithCapture);             // CHECK-EXEC: 54.00 42.00
                                              // CHECK-EXEC: 54.00 42.00
}
