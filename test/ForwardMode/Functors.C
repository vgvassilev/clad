// RUN: %cladclang %s -I%S/../../include -oFunctors.out 2>&1 | FileCheck %s
// RUN: ./Functors.out | FileCheck -check-prefix=CHECK-EXEC %s
// CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"

struct Experiment {
  mutable double x, y;
  Experiment(double p_x, double p_y) : x(p_x), y(p_y) {}
  double operator()(double i, double j) {
    return x*i*j;
  }
  void setX(double val) {
    x = val;
  }

  // CHECK: double operator_call_darg0(double i, double j) {
  // CHECK-NEXT:     double _d_i = 1;
  // CHECK-NEXT:     double _d_j = 0;
  // CHECK-NEXT:     double &_t0 = this->x;
  // CHECK-NEXT:     double _t1 = _t0 * i;
  // CHECK-NEXT:     return (0. * i + _t0 * _d_i) * j + _t1 * _d_j;
  // CHECK-NEXT: }
};

struct ExperimentConst {
  mutable double x, y;
  ExperimentConst(double p_x, double p_y) : x(p_x), y(p_y) {}
  double operator()(double i, double j) const {
    return x*i*j;
  }
  void setX(double val) const {
    x = val;
  }

  // CHECK: double operator_call_darg0(double i, double j) const {
  // CHECK-NEXT:     double _d_i = 1;
  // CHECK-NEXT:     double _d_j = 0;
  // CHECK-NEXT:     double &_t0 = this->x;
  // CHECK-NEXT:     double _t1 = _t0 * i;
  // CHECK-NEXT:     return (0. * i + _t0 * _d_i) * j + _t1 * _d_j;
  // CHECK-NEXT: }
};

struct ExperimentVolatile {
  mutable double x, y;
  ExperimentVolatile(double p_x, double p_y) : x(p_x), y(p_y) {}
  double operator()(double i, double j) volatile {
    return x*i*j;
  }
  void setX(double val) volatile {
    x = val;
  }

  // CHECK: double operator_call_darg0(double i, double j) volatile {
  // CHECK-NEXT:     double _d_i = 1;
  // CHECK-NEXT:     double _d_j = 0;
  // CHECK-NEXT:     volatile double &_t0 = this->x;
  // CHECK-NEXT:     double _t1 = _t0 * i;
  // CHECK-NEXT:     return (0. * i + _t0 * _d_i) * j + _t1 * _d_j;
  // CHECK-NEXT: }
};

struct ExperimentConstVolatile {
  mutable double x, y;
  ExperimentConstVolatile(double p_x, double p_y) : x(p_x), y(p_y) {}
  double operator()(double i, double j) const volatile {
    return x*i*j;
  }
  void setX(double val) const volatile {
    x = val;
  }

  // CHECK: double operator_call_darg0(double i, double j) const volatile {
  // CHECK-NEXT:     double _d_i = 1;
  // CHECK-NEXT:     double _d_j = 0;
  // CHECK-NEXT:     volatile double &_t0 = this->x;
  // CHECK-NEXT:     double _t1 = _t0 * i;
  // CHECK-NEXT:     return (0. * i + _t0 * _d_i) * j + _t1 * _d_j;
  // CHECK-NEXT: }
};

namespace outer {
  namespace inner {
    struct ExperimentNNS {
      mutable double x, y;
      ExperimentNNS(double p_x, double p_y) : x(p_x), y(p_y) {}
      double operator()(double i, double j) {
        return x*i*j;
      }
      void setX(double val) {
        x = val;
      }

      // CHECK: double operator_call_darg0(double i, double j) {
      // CHECK-NEXT:     double _d_i = 1;
      // CHECK-NEXT:     double _d_j = 0;
      // CHECK-NEXT:     double &_t0 = this->x;
      // CHECK-NEXT:     double _t1 = _t0 * i;
      // CHECK-NEXT:     return (0. * i + _t0 * _d_i) * j + _t1 * _d_j;
      // CHECK-NEXT: }   
    };

    auto lambdaNNS = [](double i, double j) {
      return i*i*j;
    };
  }
}

#define INIT(E)\
auto d_##E = clad::differentiate(&E, "i");\
auto d_##E##Ref = clad::differentiate(E, "i");

#define TEST(E)\
printf("%.2f %.2f\n", d_##E.execute(7, 9), d_##E##Ref.execute(7, 9));

double x=3;

int main() {
  Experiment E(3, 5);
  auto E_Again = E;
  const ExperimentConst E_Const(3, 5);
  volatile ExperimentVolatile E_Volatile(3, 5);
  const volatile ExperimentConstVolatile E_ConstVolatile(3, 5);
  outer::inner::ExperimentNNS E_NNS(3, 5);
  auto E_NNS_Again = E_NNS;
  auto lambda = [](double i, double j) {
    return i*i*j;
  };

  // CHECK: inline double operator_call_darg0(double i, double j) const {
  // CHECK-NEXT:     double _d_i = 1;
  // CHECK-NEXT:     double _d_j = 0;
  // CHECK-NEXT:     double _t0 = i * i;
  // CHECK-NEXT:     return (_d_i * i + i * _d_i) * j + _t0 * _d_j;
  // CHECK-NEXT: }

  auto lambdaWithCapture = [&](double i, double jj) {
    return x*i*jj;
  };

  // CHECK: inline double operator_call_darg0(double i, double jj) const {
  // CHECK-NEXT:     double _d_i = 1;
  // CHECK-NEXT:     double _d_jj = 0;
  // CHECK-NEXT:     double _t0 = x * i;
  // CHECK-NEXT:     return (0 * i + x * _d_i) * jj + _t0 * _d_jj;
  // CHECK-NEXT: }

  auto lambdaNNS = outer::inner::lambdaNNS;

  // CHECK: inline double operator_call_darg0(double i, double j) const {
  // CHECK-NEXT:     double _d_i = 1;
  // CHECK-NEXT:     double _d_j = 0;
  // CHECK-NEXT:     double _t0 = i * i;
  // CHECK-NEXT:     return (_d_i * i + i * _d_i) * j + _t0 * _d_j;
  // CHECK-NEXT: }

  INIT(E);
  INIT(E_Again);
  INIT(E_Const);
  INIT(E_Volatile);
  INIT(E_ConstVolatile);
  INIT(E_NNS);
  INIT(E_NNS_Again);
  INIT(lambda);
  INIT(lambdaWithCapture);
  INIT(lambdaNNS);

  TEST(E);                  // CHECK-EXEC: 27.00 27.00
  TEST(E_Again);            // CHECK-EXEC: 27.00 27.00
  TEST(E_Const);            // CHECK-EXEC: 27.00 27.00
  TEST(E_Volatile);         // CHECK-EXEC: 27.00 27.00
  TEST(E_ConstVolatile);    // CHECK-EXEC: 27.00 27.00
  TEST(E_NNS);              // CHECK-EXEC: 27.00 27.00
  TEST(E_NNS_Again);        // CHECK-EXEC: 27.00 27.00
  TEST(lambda);             // CHECK-EXEC: 126.00 126.00
  TEST(lambdaWithCapture);  // CHECK-EXEC: 27.00 27.00
  TEST(lambdaNNS);          // CHECK-EXEC: 126.00 126.00

  E.setX(6);
  E_Again.setX(6);
  E_Const.setX(6);
  E_Volatile.setX(6);
  E_ConstVolatile.setX(6);
  E_NNS.setX(6);
  E_NNS_Again.setX(6);
  x = 6;

  TEST(E);                  // CHECK-EXEC: 54.00 54.00
  TEST(E_Again);            // CHECK-EXEC: 54.00 54.00
  TEST(E_Const);            // CHECK-EXEC: 54.00 54.00
  TEST(E_Volatile);         // CHECK-EXEC: 54.00 54.00
  TEST(E_ConstVolatile);    // CHECK-EXEC: 54.00 54.00
  TEST(E_NNS);              // CHECK-EXEC: 54.00 54.00
  TEST(E_NNS_Again);        // CHECK-EXEC: 54.00 54.00
  TEST(lambdaWithCapture);  // CHECK-EXEC: 54.00 54.00
}