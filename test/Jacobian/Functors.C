// RUN: %cladclang %s -I%S/../../include -oFunctors.out 
// RUN: ./Functors.out | FileCheck -check-prefix=CHECK-EXEC %s
// CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"

struct Experiment {
  mutable double x, y;
  Experiment(double p_x, double p_y) : x(p_x), y(p_y) {}
  void operator()(double i, double j, double *output) {
    output[0] = x*i*i*j;
    output[1] = y*i*j*j;
  }
  void setX(double val) {
    x = val;
  }

  // CHECK: void operator_call_jac(double i, double j, double *output, double *jacobianMatrix) {
  // CHECK-NEXT:     double _t0;
  // CHECK-NEXT:     double _t1;
  // CHECK-NEXT:     double _t2;
  // CHECK-NEXT:     double _t3;
  // CHECK-NEXT:     double _t4;
  // CHECK-NEXT:     double _t5;
  // CHECK-NEXT:     double _t6;
  // CHECK-NEXT:     double _t7;
  // CHECK-NEXT:     double _t8;
  // CHECK-NEXT:     double _t9;
  // CHECK-NEXT:     double _t10;
  // CHECK-NEXT:     double _t11;
  // CHECK-NEXT:     _t3 = this->x;
  // CHECK-NEXT:     _t2 = i;
  // CHECK-NEXT:     _t4 = _t3 * _t2;
  // CHECK-NEXT:     _t1 = i;
  // CHECK-NEXT:     _t5 = _t4 * _t1;
  // CHECK-NEXT:     _t0 = j;
  // CHECK-NEXT:     output[0] = this->x * i * i * j;
  // CHECK-NEXT:     _t9 = this->y;
  // CHECK-NEXT:     _t8 = i;
  // CHECK-NEXT:     _t10 = _t9 * _t8;
  // CHECK-NEXT:     _t7 = j;
  // CHECK-NEXT:     _t11 = _t10 * _t7;
  // CHECK-NEXT:     _t6 = j;
  // CHECK-NEXT:     output[1] = this->y * i * j * j;
  // CHECK-NEXT:     {
  // CHECK-NEXT:         double _r6 = 1 * _t6;
  // CHECK-NEXT:         double _r7 = _r6 * _t7;
  // CHECK-NEXT:         double _r8 = _r7 * _t8;
  // CHECK-NEXT:         double _r9 = _t9 * _r7;
  // CHECK-NEXT:         jacobianMatrix[2UL] += _r9;
  // CHECK-NEXT:         double _r10 = _t10 * _r6;
  // CHECK-NEXT:         jacobianMatrix[3UL] += _r10;
  // CHECK-NEXT:         double _r11 = _t11 * 1;
  // CHECK-NEXT:         jacobianMatrix[3UL] += _r11;
  // CHECK-NEXT:     }
  // CHECK-NEXT:     {
  // CHECK-NEXT:         double _r0 = 1 * _t0;
  // CHECK-NEXT:         double _r1 = _r0 * _t1;
  // CHECK-NEXT:         double _r2 = _r1 * _t2;
  // CHECK-NEXT:         double _r3 = _t3 * _r1;
  // CHECK-NEXT:         jacobianMatrix[0UL] += _r3;
  // CHECK-NEXT:         double _r4 = _t4 * _r0;
  // CHECK-NEXT:         jacobianMatrix[0UL] += _r4;
  // CHECK-NEXT:         double _r5 = _t5 * 1;
  // CHECK-NEXT:         jacobianMatrix[1UL] += _r5;
  // CHECK-NEXT:     }
  // CHECK-NEXT: }
};

struct ExperimentConst {
  mutable double x, y;
  ExperimentConst(double p_x, double p_y) : x(p_x), y(p_y) {}
  void operator()(double i, double j, double *output) const {
    output[0] = x*i*i*j;
    output[1] = y*i*j*j;
  }
  void setX(double val) const {
    x = val;
  }

  // CHECK: void operator_call_jac(double i, double j, double *output, double *jacobianMatrix) const {
  // CHECK-NEXT:     double _t0;
  // CHECK-NEXT:     double _t1;
  // CHECK-NEXT:     double _t2;
  // CHECK-NEXT:     double _t3;
  // CHECK-NEXT:     double _t4;
  // CHECK-NEXT:     double _t5;
  // CHECK-NEXT:     double _t6;
  // CHECK-NEXT:     double _t7;
  // CHECK-NEXT:     double _t8;
  // CHECK-NEXT:     double _t9;
  // CHECK-NEXT:     double _t10;
  // CHECK-NEXT:     double _t11;
  // CHECK-NEXT:     _t3 = this->x;
  // CHECK-NEXT:     _t2 = i;
  // CHECK-NEXT:     _t4 = _t3 * _t2;
  // CHECK-NEXT:     _t1 = i;
  // CHECK-NEXT:     _t5 = _t4 * _t1;
  // CHECK-NEXT:     _t0 = j;
  // CHECK-NEXT:     output[0] = this->x * i * i * j;
  // CHECK-NEXT:     _t9 = this->y;
  // CHECK-NEXT:     _t8 = i;
  // CHECK-NEXT:     _t10 = _t9 * _t8;
  // CHECK-NEXT:     _t7 = j;
  // CHECK-NEXT:     _t11 = _t10 * _t7;
  // CHECK-NEXT:     _t6 = j;
  // CHECK-NEXT:     output[1] = this->y * i * j * j;
  // CHECK-NEXT:     {
  // CHECK-NEXT:         double _r6 = 1 * _t6;
  // CHECK-NEXT:         double _r7 = _r6 * _t7;
  // CHECK-NEXT:         double _r8 = _r7 * _t8;
  // CHECK-NEXT:         double _r9 = _t9 * _r7;
  // CHECK-NEXT:         jacobianMatrix[2UL] += _r9;
  // CHECK-NEXT:         double _r10 = _t10 * _r6;
  // CHECK-NEXT:         jacobianMatrix[3UL] += _r10;
  // CHECK-NEXT:         double _r11 = _t11 * 1;
  // CHECK-NEXT:         jacobianMatrix[3UL] += _r11;
  // CHECK-NEXT:     }
  // CHECK-NEXT:     {
  // CHECK-NEXT:         double _r0 = 1 * _t0;
  // CHECK-NEXT:         double _r1 = _r0 * _t1;
  // CHECK-NEXT:         double _r2 = _r1 * _t2;
  // CHECK-NEXT:         double _r3 = _t3 * _r1;
  // CHECK-NEXT:         jacobianMatrix[0UL] += _r3;
  // CHECK-NEXT:         double _r4 = _t4 * _r0;
  // CHECK-NEXT:         jacobianMatrix[0UL] += _r4;
  // CHECK-NEXT:         double _r5 = _t5 * 1;
  // CHECK-NEXT:         jacobianMatrix[1UL] += _r5;
  // CHECK-NEXT:     }
  // CHECK-NEXT: }
};

struct ExperimentVolatile {
  mutable double x, y;
  ExperimentVolatile(double p_x, double p_y) : x(p_x), y(p_y) {}
  void operator()(double i, double j, double *output) volatile {
    output[0] = x*i*i*j;
    output[1] = y*i*j*j;
  }
  void setX(double val) volatile {
    x = val;
  }

  // CHECK: void operator_call_jac(double i, double j, double *output, double *jacobianMatrix) volatile {
  // CHECK-NEXT:     double _t0;
  // CHECK-NEXT:     double _t1;
  // CHECK-NEXT:     double _t2;
  // CHECK-NEXT:     volatile double _t3;
  // CHECK-NEXT:     double _t4;
  // CHECK-NEXT:     double _t5;
  // CHECK-NEXT:     double _t6;
  // CHECK-NEXT:     double _t7;
  // CHECK-NEXT:     double _t8;
  // CHECK-NEXT:     volatile double _t9;
  // CHECK-NEXT:     double _t10;
  // CHECK-NEXT:     double _t11;
  // CHECK-NEXT:     _t3 = this->x;
  // CHECK-NEXT:     _t2 = i;
  // CHECK-NEXT:     _t4 = _t3 * _t2;
  // CHECK-NEXT:     _t1 = i;
  // CHECK-NEXT:     _t5 = _t4 * _t1;
  // CHECK-NEXT:     _t0 = j;
  // CHECK-NEXT:     output[0] = this->x * i * i * j;
  // CHECK-NEXT:     _t9 = this->y;
  // CHECK-NEXT:     _t8 = i;
  // CHECK-NEXT:     _t10 = _t9 * _t8;
  // CHECK-NEXT:     _t7 = j;
  // CHECK-NEXT:     _t11 = _t10 * _t7;
  // CHECK-NEXT:     _t6 = j;
  // CHECK-NEXT:     output[1] = this->y * i * j * j;
  // CHECK-NEXT:     {
  // CHECK-NEXT:         double _r6 = 1 * _t6;
  // CHECK-NEXT:         double _r7 = _r6 * _t7;
  // CHECK-NEXT:         double _r8 = _r7 * _t8;
  // CHECK-NEXT:         double _r9 = _t9 * _r7;
  // CHECK-NEXT:         jacobianMatrix[2UL] += _r9;
  // CHECK-NEXT:         double _r10 = _t10 * _r6;
  // CHECK-NEXT:         jacobianMatrix[3UL] += _r10;
  // CHECK-NEXT:         double _r11 = _t11 * 1;
  // CHECK-NEXT:         jacobianMatrix[3UL] += _r11;
  // CHECK-NEXT:     }
  // CHECK-NEXT:     {
  // CHECK-NEXT:         double _r0 = 1 * _t0;
  // CHECK-NEXT:         double _r1 = _r0 * _t1;
  // CHECK-NEXT:         double _r2 = _r1 * _t2;
  // CHECK-NEXT:         double _r3 = _t3 * _r1;
  // CHECK-NEXT:         jacobianMatrix[0UL] += _r3;
  // CHECK-NEXT:         double _r4 = _t4 * _r0;
  // CHECK-NEXT:         jacobianMatrix[0UL] += _r4;
  // CHECK-NEXT:         double _r5 = _t5 * 1;
  // CHECK-NEXT:         jacobianMatrix[1UL] += _r5;
  // CHECK-NEXT:     }
  // CHECK-NEXT: }
};

struct ExperimentConstVolatile {
  mutable double x, y;
  ExperimentConstVolatile(double p_x, double p_y) : x(p_x), y(p_y) {}
  void operator()(double i, double j, double *output) const volatile {
    output[0] = x*i*i*j;
    output[1] = y*i*j*j;
  }
  void setX(double val) const volatile {
    x = val;
  }

  // CHECK: void operator_call_jac(double i, double j, double *output, double *jacobianMatrix) const volatile {
  // CHECK-NEXT:     double _t0;
  // CHECK-NEXT:     double _t1;
  // CHECK-NEXT:     double _t2;
  // CHECK-NEXT:     volatile double _t3;
  // CHECK-NEXT:     double _t4;
  // CHECK-NEXT:     double _t5;
  // CHECK-NEXT:     double _t6;
  // CHECK-NEXT:     double _t7;
  // CHECK-NEXT:     double _t8;
  // CHECK-NEXT:     volatile double _t9;
  // CHECK-NEXT:     double _t10;
  // CHECK-NEXT:     double _t11;
  // CHECK-NEXT:     _t3 = this->x;
  // CHECK-NEXT:     _t2 = i;
  // CHECK-NEXT:     _t4 = _t3 * _t2;
  // CHECK-NEXT:     _t1 = i;
  // CHECK-NEXT:     _t5 = _t4 * _t1;
  // CHECK-NEXT:     _t0 = j;
  // CHECK-NEXT:     output[0] = this->x * i * i * j;
  // CHECK-NEXT:     _t9 = this->y;
  // CHECK-NEXT:     _t8 = i;
  // CHECK-NEXT:     _t10 = _t9 * _t8;
  // CHECK-NEXT:     _t7 = j;
  // CHECK-NEXT:     _t11 = _t10 * _t7;
  // CHECK-NEXT:     _t6 = j;
  // CHECK-NEXT:     output[1] = this->y * i * j * j;
  // CHECK-NEXT:     {
  // CHECK-NEXT:         double _r6 = 1 * _t6;
  // CHECK-NEXT:         double _r7 = _r6 * _t7;
  // CHECK-NEXT:         double _r8 = _r7 * _t8;
  // CHECK-NEXT:         double _r9 = _t9 * _r7;
  // CHECK-NEXT:         jacobianMatrix[2UL] += _r9;
  // CHECK-NEXT:         double _r10 = _t10 * _r6;
  // CHECK-NEXT:         jacobianMatrix[3UL] += _r10;
  // CHECK-NEXT:         double _r11 = _t11 * 1;
  // CHECK-NEXT:         jacobianMatrix[3UL] += _r11;
  // CHECK-NEXT:     }
  // CHECK-NEXT:     {
  // CHECK-NEXT:         double _r0 = 1 * _t0;
  // CHECK-NEXT:         double _r1 = _r0 * _t1;
  // CHECK-NEXT:         double _r2 = _r1 * _t2;
  // CHECK-NEXT:         double _r3 = _t3 * _r1;
  // CHECK-NEXT:         jacobianMatrix[0UL] += _r3;
  // CHECK-NEXT:         double _r4 = _t4 * _r0;
  // CHECK-NEXT:         jacobianMatrix[0UL] += _r4;
  // CHECK-NEXT:         double _r5 = _t5 * 1;
  // CHECK-NEXT:         jacobianMatrix[1UL] += _r5;
  // CHECK-NEXT:     }
  // CHECK-NEXT: }
};

namespace outer {
  namespace inner {
    struct ExperimentNNS {
      mutable double x, y;
      ExperimentNNS(double p_x, double p_y) : x(p_x), y(p_y) {}
      void operator()(double i, double j, double *output) {
        output[0] = x*i*i*j;
        output[1] = y*i*j*j;
      }
      void setX(double val) {
        x = val;
      }
      
      // CHECK: void operator_call_jac(double i, double j, double *output, double *jacobianMatrix) {
      // CHECK-NEXT:     double _t0;
      // CHECK-NEXT:     double _t1;
      // CHECK-NEXT:     double _t2;
      // CHECK-NEXT:     double _t3;
      // CHECK-NEXT:     double _t4;
      // CHECK-NEXT:     double _t5;
      // CHECK-NEXT:     double _t6;
      // CHECK-NEXT:     double _t7;
      // CHECK-NEXT:     double _t8;
      // CHECK-NEXT:     double _t9;
      // CHECK-NEXT:     double _t10;
      // CHECK-NEXT:     double _t11;
      // CHECK-NEXT:     _t3 = this->x;
      // CHECK-NEXT:     _t2 = i;
      // CHECK-NEXT:     _t4 = _t3 * _t2;
      // CHECK-NEXT:     _t1 = i;
      // CHECK-NEXT:     _t5 = _t4 * _t1;
      // CHECK-NEXT:     _t0 = j;
      // CHECK-NEXT:     output[0] = this->x * i * i * j;
      // CHECK-NEXT:     _t9 = this->y;
      // CHECK-NEXT:     _t8 = i;
      // CHECK-NEXT:     _t10 = _t9 * _t8;
      // CHECK-NEXT:     _t7 = j;
      // CHECK-NEXT:     _t11 = _t10 * _t7;
      // CHECK-NEXT:     _t6 = j;
      // CHECK-NEXT:     output[1] = this->y * i * j * j;
      // CHECK-NEXT:     {
      // CHECK-NEXT:         double _r6 = 1 * _t6;
      // CHECK-NEXT:         double _r7 = _r6 * _t7;
      // CHECK-NEXT:         double _r8 = _r7 * _t8;
      // CHECK-NEXT:         double _r9 = _t9 * _r7;
      // CHECK-NEXT:         jacobianMatrix[2UL] += _r9;
      // CHECK-NEXT:         double _r10 = _t10 * _r6;
      // CHECK-NEXT:         jacobianMatrix[3UL] += _r10;
      // CHECK-NEXT:         double _r11 = _t11 * 1;
      // CHECK-NEXT:         jacobianMatrix[3UL] += _r11;
      // CHECK-NEXT:     }
      // CHECK-NEXT:     {
      // CHECK-NEXT:         double _r0 = 1 * _t0;
      // CHECK-NEXT:         double _r1 = _r0 * _t1;
      // CHECK-NEXT:         double _r2 = _r1 * _t2;
      // CHECK-NEXT:         double _r3 = _t3 * _r1;
      // CHECK-NEXT:         jacobianMatrix[0UL] += _r3;
      // CHECK-NEXT:         double _r4 = _t4 * _r0;
      // CHECK-NEXT:         jacobianMatrix[0UL] += _r4;
      // CHECK-NEXT:         double _r5 = _t5 * 1;
      // CHECK-NEXT:         jacobianMatrix[1UL] += _r5;
      // CHECK-NEXT:     }
      // CHECK-NEXT: }
    };

    auto lambdaNNS = [](double i, double j, double *output) {
      output[0] = i*i*j;
      output[1] = i*j*j;
    };

    // CHECK: inline void operator_call_jac(double i, double j, double *output, double *jacobianMatrix) const {
    // CHECK-NEXT:     double _t0;
    // CHECK-NEXT:     double _t1;
    // CHECK-NEXT:     double _t2;
    // CHECK-NEXT:     double _t3;
    // CHECK-NEXT:     double _t4;
    // CHECK-NEXT:     double _t5;
    // CHECK-NEXT:     double _t6;
    // CHECK-NEXT:     double _t7;
    // CHECK-NEXT:     _t2 = i;
    // CHECK-NEXT:     _t1 = i;
    // CHECK-NEXT:     _t3 = _t2 * _t1;
    // CHECK-NEXT:     _t0 = j;
    // CHECK-NEXT:     output[0] = i * i * j;
    // CHECK-NEXT:     _t6 = i;
    // CHECK-NEXT:     _t5 = j;
    // CHECK-NEXT:     _t7 = _t6 * _t5;
    // CHECK-NEXT:     _t4 = j;
    // CHECK-NEXT:     output[1] = i * j * j;
    // CHECK-NEXT:     {
    // CHECK-NEXT:         double _r4 = 1 * _t4;
    // CHECK-NEXT:         double _r5 = _r4 * _t5;
    // CHECK-NEXT:         jacobianMatrix[2UL] += _r5;
    // CHECK-NEXT:         double _r6 = _t6 * _r4;
    // CHECK-NEXT:         jacobianMatrix[3UL] += _r6;
    // CHECK-NEXT:         double _r7 = _t7 * 1;
    // CHECK-NEXT:         jacobianMatrix[3UL] += _r7;
    // CHECK-NEXT:     }
    // CHECK-NEXT:     {
    // CHECK-NEXT:         double _r0 = 1 * _t0;
    // CHECK-NEXT:         double _r1 = _r0 * _t1;
    // CHECK-NEXT:         jacobianMatrix[0UL] += _r1;
    // CHECK-NEXT:         double _r2 = _t2 * _r0;
    // CHECK-NEXT:         jacobianMatrix[0UL] += _r2;
    // CHECK-NEXT:         double _r3 = _t3 * 1;
    // CHECK-NEXT:         jacobianMatrix[1UL] += _r3;
    // CHECK-NEXT:     }
    // CHECK-NEXT: }
  }
}

#define INIT(E)                                                                \
  auto d_##E = clad::jacobian(&E);                                             \
  auto d_##E##Ref = clad::jacobian(E);

#define TEST(E)                                                                \
  result[0] = result[1] = result[2] = result[3] = 0;                           \
  output[0] = output[1] = 0;                                                   \
  d_##E.execute(7, 9, output, result);                                         \
  printf("{%.2f, %.2f, %.2f, %.2f}, ", result[0], result[1], result[2],        \
         result[3]);                                                           \
  result[0] = result[1] = result[2] = result[3] = 0;                           \
  output[0] = output[1] = 0;                                                   \
  d_##E##Ref.execute(7, 9, output, result);                                    \
  printf("{%.2f, %.2f, %.2f, %.2f}\n", result[0], result[1], result[2],        \
         result[3]);

double x = 3;
double y = 5;
int main() {
  double output[2], result[4];
  Experiment E(3, 5);
  auto E_Again = E;
  const ExperimentConst E_Const(3, 5);
  volatile ExperimentVolatile E_Volatile(3, 5);
  const volatile ExperimentConstVolatile E_ConstVolatile(3, 5);
  outer::inner::ExperimentNNS E_NNS(3, 5);
  auto E_NNS_Again = E_NNS;
  auto lambda = [](double i, double j, double *output) {
    output[0] = i*i*j;
    output[1] = i*j*j;
  };

  // CHECK: inline void operator_call_jac(double i, double j, double *output, double *jacobianMatrix) const {
  // CHECK-NEXT:     double _t0;
  // CHECK-NEXT:     double _t1;
  // CHECK-NEXT:     double _t2;
  // CHECK-NEXT:     double _t3;
  // CHECK-NEXT:     double _t4;
  // CHECK-NEXT:     double _t5;
  // CHECK-NEXT:     double _t6;
  // CHECK-NEXT:     double _t7;
  // CHECK-NEXT:     _t2 = i;
  // CHECK-NEXT:     _t1 = i;
  // CHECK-NEXT:     _t3 = _t2 * _t1;
  // CHECK-NEXT:     _t0 = j;
  // CHECK-NEXT:     output[0] = i * i * j;
  // CHECK-NEXT:     _t6 = i;
  // CHECK-NEXT:     _t5 = j;
  // CHECK-NEXT:     _t7 = _t6 * _t5;
  // CHECK-NEXT:     _t4 = j;
  // CHECK-NEXT:     output[1] = i * j * j;
  // CHECK-NEXT:     {
  // CHECK-NEXT:         double _r4 = 1 * _t4;
  // CHECK-NEXT:         double _r5 = _r4 * _t5;
  // CHECK-NEXT:         jacobianMatrix[2UL] += _r5;
  // CHECK-NEXT:         double _r6 = _t6 * _r4;
  // CHECK-NEXT:         jacobianMatrix[3UL] += _r6;
  // CHECK-NEXT:         double _r7 = _t7 * 1;
  // CHECK-NEXT:         jacobianMatrix[3UL] += _r7;
  // CHECK-NEXT:     }
  // CHECK-NEXT:     {
  // CHECK-NEXT:         double _r0 = 1 * _t0;
  // CHECK-NEXT:         double _r1 = _r0 * _t1;
  // CHECK-NEXT:         jacobianMatrix[0UL] += _r1;
  // CHECK-NEXT:         double _r2 = _t2 * _r0;
  // CHECK-NEXT:         jacobianMatrix[0UL] += _r2;
  // CHECK-NEXT:         double _r3 = _t3 * 1;
  // CHECK-NEXT:         jacobianMatrix[1UL] += _r3;
  // CHECK-NEXT:     }
  // CHECK-NEXT: }

  auto lambdaWithCapture = [&](double i, double jj, double *output) {
    output[0] = x*i*i*jj;
    output[1] = y*i*jj*jj;
  };

  // CHECK: inline void operator_call_jac(double i, double jj, double *output, double *jacobianMatrix) const {
  // CHECK-NEXT:     double _t0;
  // CHECK-NEXT:     double _t1;
  // CHECK-NEXT:     double _t2;
  // CHECK-NEXT:     double _t3;
  // CHECK-NEXT:     double _t4;
  // CHECK-NEXT:     double _t5;
  // CHECK-NEXT:     double _t6;
  // CHECK-NEXT:     double _t7;
  // CHECK-NEXT:     double _t8;
  // CHECK-NEXT:     double _t9;
  // CHECK-NEXT:     double _t10;
  // CHECK-NEXT:     double _t11;
  // CHECK-NEXT:     _t3 = x;
  // CHECK-NEXT:     _t2 = i;
  // CHECK-NEXT:     _t4 = _t3 * _t2;
  // CHECK-NEXT:     _t1 = i;
  // CHECK-NEXT:     _t5 = _t4 * _t1;
  // CHECK-NEXT:     _t0 = jj;
  // CHECK-NEXT:     output[0] = x * i * i * jj;
  // CHECK-NEXT:     _t9 = y;
  // CHECK-NEXT:     _t8 = i;
  // CHECK-NEXT:     _t10 = _t9 * _t8;
  // CHECK-NEXT:     _t7 = jj;
  // CHECK-NEXT:     _t11 = _t10 * _t7;
  // CHECK-NEXT:     _t6 = jj;
  // CHECK-NEXT:     output[1] = y * i * jj * jj;
  // CHECK-NEXT:     {
  // CHECK-NEXT:         double _r6 = 1 * _t6;
  // CHECK-NEXT:         double _r7 = _r6 * _t7;
  // CHECK-NEXT:         double _r8 = _r7 * _t8;
  // CHECK-NEXT:         double _r9 = _t9 * _r7;
  // CHECK-NEXT:         jacobianMatrix[2UL] += _r9;
  // CHECK-NEXT:         double _r10 = _t10 * _r6;
  // CHECK-NEXT:         jacobianMatrix[3UL] += _r10;
  // CHECK-NEXT:         double _r11 = _t11 * 1;
  // CHECK-NEXT:         jacobianMatrix[3UL] += _r11;
  // CHECK-NEXT:     }
  // CHECK-NEXT:     {
  // CHECK-NEXT:         double _r0 = 1 * _t0;
  // CHECK-NEXT:         double _r1 = _r0 * _t1;
  // CHECK-NEXT:         double _r2 = _r1 * _t2;
  // CHECK-NEXT:         double _r3 = _t3 * _r1;
  // CHECK-NEXT:         jacobianMatrix[0UL] += _r3;
  // CHECK-NEXT:         double _r4 = _t4 * _r0;
  // CHECK-NEXT:         jacobianMatrix[0UL] += _r4;
  // CHECK-NEXT:         double _r5 = _t5 * 1;
  // CHECK-NEXT:         jacobianMatrix[1UL] += _r5;
  // CHECK-NEXT:     }
  // CHECK-NEXT: }

  auto lambdaNNS = outer::inner::lambdaNNS;

  INIT(E);
  INIT(E_Again);
  INIT(E_Const);
  INIT(E_Volatile);
  INIT(E_ConstVolatile);
  INIT(E_NNS);
  INIT(E_NNS_Again);
  INIT(lambdaNNS);
  INIT(lambda);
  INIT(lambdaWithCapture);

  TEST(E);                  // CHECK-EXEC: {378.00, 147.00, 405.00, 630.00}, {378.00, 147.00, 405.00, 630.00}
  TEST(E_Again);            // CHECK-EXEC: {378.00, 147.00, 405.00, 630.00}, {378.00, 147.00, 405.00, 630.00}
  TEST(E_Const);            // CHECK-EXEC: {378.00, 147.00, 405.00, 630.00}, {378.00, 147.00, 405.00, 630.00}
  TEST(E_Volatile);         // CHECK-EXEC: {378.00, 147.00, 405.00, 630.00}, {378.00, 147.00, 405.00, 630.00}
  TEST(E_ConstVolatile);    // CHECK-EXEC: {378.00, 147.00, 405.00, 630.00}, {378.00, 147.00, 405.00, 630.00}
  TEST(E_NNS);              // CHECK-EXEC: {378.00, 147.00, 405.00, 630.00}, {378.00, 147.00, 405.00, 630.00}
  TEST(E_NNS_Again);        // CHECK-EXEC: {378.00, 147.00, 405.00, 630.00}, {378.00, 147.00, 405.00, 630.00}
  TEST(lambdaWithCapture);  // CHECK-EXEC: {378.00, 147.00, 405.00, 630.00}, {378.00, 147.00, 405.00, 630.00}
  TEST(lambda);             // CHECK-EXEC: {126.00, 49.00, 81.00, 126.00}, {126.00, 49.00, 81.00, 126.00}
  TEST(lambdaNNS);          // CHECK-EXEC: {126.00, 49.00, 81.00, 126.00}, {126.00, 49.00, 81.00, 126.00}

  E.setX(6);
  E_Again.setX(6);
  E_Const.setX(6);
  E_Volatile.setX(6);
  E_ConstVolatile.setX(6);
  E_NNS.setX(6);
  E_NNS_Again.setX(6);
  x = 6;

  TEST(E);                  // CHECK-EXEC: {756.00, 294.00, 405.00, 630.00}, {756.00, 294.00, 405.00, 630.00}
  TEST(E_Again);            // CHECK-EXEC: {756.00, 294.00, 405.00, 630.00}, {756.00, 294.00, 405.00, 630.00}
  TEST(E_Const);            // CHECK-EXEC: {756.00, 294.00, 405.00, 630.00}, {756.00, 294.00, 405.00, 630.00}
  TEST(E_Volatile);         // CHECK-EXEC: {756.00, 294.00, 405.00, 630.00}, {756.00, 294.00, 405.00, 630.00}
  TEST(E_ConstVolatile);    // CHECK-EXEC: {756.00, 294.00, 405.00, 630.00}, {756.00, 294.00, 405.00, 630.00}
  TEST(E_NNS);              // CHECK-EXEC: {756.00, 294.00, 405.00, 630.00}, {756.00, 294.00, 405.00, 630.00}
  TEST(E_NNS_Again);        // CHECK-EXEC: {756.00, 294.00, 405.00, 630.00}, {756.00, 294.00, 405.00, 630.00}
  TEST(lambdaWithCapture);  // CHECK-EXEC: {756.00, 294.00, 405.00, 630.00}, {756.00, 294.00, 405.00, 630.00}
}