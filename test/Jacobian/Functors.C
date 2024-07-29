// RUN: %cladclang %s -I%S/../../include -oFunctors.out 2>&1 | %filecheck %s
// RUN: ./Functors.out | %filecheck_exec %s
// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -enable-tbr %s -I%S/../../include -oFunctors.out
// RUN: ./Functors.out | %filecheck_exec %s
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
  // CHECK-NEXT:     output[0] = this->x * i * i * j;
  // CHECK-NEXT:     output[1] = this->y * i * j * j;
  // CHECK-NEXT:     {
  // CHECK-NEXT:         jacobianMatrix[{{2U|2UL}}] += this->y * 1 * j * j;
  // CHECK-NEXT:         jacobianMatrix[{{3U|3UL}}] += this->y * i * 1 * j;
  // CHECK-NEXT:         jacobianMatrix[{{3U|3UL}}] += this->y * i * j * 1;
  // CHECK-NEXT:     }
  // CHECK-NEXT:     {
  // CHECK-NEXT:         jacobianMatrix[{{0U|0UL}}] += this->x * 1 * j * i;
  // CHECK-NEXT:         jacobianMatrix[{{0U|0UL}}] += this->x * i * 1 * j;
  // CHECK-NEXT:         jacobianMatrix[{{1U|1UL}}] += this->x * i * i * 1;
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
  // CHECK-NEXT:     output[0] = this->x * i * i * j;
  // CHECK-NEXT:     output[1] = this->y * i * j * j;
  // CHECK-NEXT:     {
  // CHECK-NEXT:         jacobianMatrix[{{2U|2UL}}] += this->y * 1 * j * j;
  // CHECK-NEXT:         jacobianMatrix[{{3U|3UL}}] += this->y * i * 1 * j;
  // CHECK-NEXT:         jacobianMatrix[{{3U|3UL}}] += this->y * i * j * 1;
  // CHECK-NEXT:     }
  // CHECK-NEXT:     {
  // CHECK-NEXT:         jacobianMatrix[{{0U|0UL}}] += this->x * 1 * j * i;
  // CHECK-NEXT:         jacobianMatrix[{{0U|0UL}}] += this->x * i * 1 * j;
  // CHECK-NEXT:         jacobianMatrix[{{1U|1UL}}] += this->x * i * i * 1;
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
  // CHECK-NEXT:     double _t0 = this->x * i;
  // CHECK-NEXT:     output[0] = this->x * i * i * j;
  // CHECK-NEXT:     double _t1 = this->y * i;
  // CHECK-NEXT:     output[1] = this->y * i * j * j;
  // CHECK-NEXT:     {
  // CHECK-NEXT:         jacobianMatrix[{{2U|2UL}}] += this->y * 1 * j * j;
  // CHECK-NEXT:         jacobianMatrix[{{3U|3UL}}] += _t1 * 1 * j;
  // CHECK-NEXT:         jacobianMatrix[{{3U|3UL}}] += _t1 * j * 1;
  // CHECK-NEXT:     }
  // CHECK-NEXT:     {
  // CHECK-NEXT:         jacobianMatrix[{{0U|0UL}}] += this->x * 1 * j * i;
  // CHECK-NEXT:         jacobianMatrix[{{0U|0UL}}] += _t0 * 1 * j;
  // CHECK-NEXT:         jacobianMatrix[{{1U|1UL}}] += _t0 * i * 1;
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
  // CHECK-NEXT:     double _t0 = this->x * i;
  // CHECK-NEXT:     output[0] = this->x * i * i * j;
  // CHECK-NEXT:     double _t1 = this->y * i;
  // CHECK-NEXT:     output[1] = this->y * i * j * j;
  // CHECK-NEXT:     {
  // CHECK-NEXT:         jacobianMatrix[{{2U|2UL}}] += this->y * 1 * j * j;
  // CHECK-NEXT:         jacobianMatrix[{{3U|3UL}}] += _t1 * 1 * j;
  // CHECK-NEXT:         jacobianMatrix[{{3U|3UL}}] += _t1 * j * 1;
  // CHECK-NEXT:     }
  // CHECK-NEXT:     {
  // CHECK-NEXT:         jacobianMatrix[{{0U|0UL}}] += this->x * 1 * j * i;
  // CHECK-NEXT:         jacobianMatrix[{{0U|0UL}}] += _t0 * 1 * j;
  // CHECK-NEXT:         jacobianMatrix[{{1U|1UL}}] += _t0 * i * 1;
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
      // CHECK-NEXT:     output[0] = this->x * i * i * j;
      // CHECK-NEXT:     output[1] = this->y * i * j * j;
      // CHECK-NEXT:     {
      // CHECK-NEXT:         jacobianMatrix[{{2U|2UL}}] += this->y * 1 * j * j;
      // CHECK-NEXT:         jacobianMatrix[{{3U|3UL}}] += this->y * i * 1 * j;
      // CHECK-NEXT:         jacobianMatrix[{{3U|3UL}}] += this->y * i * j * 1;
      // CHECK-NEXT:     }
      // CHECK-NEXT:     {
      // CHECK-NEXT:         jacobianMatrix[{{0U|0UL}}] += this->x * 1 * j * i;
      // CHECK-NEXT:         jacobianMatrix[{{0U|0UL}}] += this->x * i * 1 * j;
      // CHECK-NEXT:         jacobianMatrix[{{1U|1UL}}] += this->x * i * i * 1;
      // CHECK-NEXT:     }
      // CHECK-NEXT: }
    };

    auto lambdaNNS = [](double i, double j, double *output) {
      output[0] = i*i*j;
      output[1] = i*j*j;
    };

    // CHECK: inline void operator_call_jac(double i, double j, double *output, double *jacobianMatrix) const {
    // CHECK-NEXT:     output[0] = i * i * j;
    // CHECK-NEXT:     output[1] = i * j * j;
    // CHECK-NEXT:     {
    // CHECK-NEXT:         jacobianMatrix[{{2U|2UL}}] += 1 * j * j;
    // CHECK-NEXT:         jacobianMatrix[{{3U|3UL}}] += i * 1 * j;
    // CHECK-NEXT:         jacobianMatrix[{{3U|3UL}}] += i * j * 1;
    // CHECK-NEXT:     }
    // CHECK-NEXT:     {
    // CHECK-NEXT:         jacobianMatrix[{{0U|0UL}}] += 1 * j * i;
    // CHECK-NEXT:         jacobianMatrix[{{0U|0UL}}] += i * 1 * j;
    // CHECK-NEXT:         jacobianMatrix[{{1U|1UL}}] += i * i * 1;
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
  // CHECK-NEXT:     output[0] = i * i * j;
  // CHECK-NEXT:     output[1] = i * j * j;
  // CHECK-NEXT:     {
  // CHECK-NEXT:         jacobianMatrix[{{2U|2UL}}] += 1 * j * j;
  // CHECK-NEXT:         jacobianMatrix[{{3U|3UL}}] += i * 1 * j;
  // CHECK-NEXT:         jacobianMatrix[{{3U|3UL}}] += i * j * 1;
  // CHECK-NEXT:     }
  // CHECK-NEXT:     {
  // CHECK-NEXT:         jacobianMatrix[{{0U|0UL}}] += 1 * j * i;
  // CHECK-NEXT:         jacobianMatrix[{{0U|0UL}}] += i * 1 * j;
  // CHECK-NEXT:         jacobianMatrix[{{1U|1UL}}] += i * i * 1;
  // CHECK-NEXT:     }
  // CHECK-NEXT: }

  auto lambdaWithCapture = [&](double i, double jj, double *output) {
    output[0] = x*i*i*jj;
    output[1] = y*i*jj*jj;
  };

  // CHECK: inline void operator_call_jac(double i, double jj, double *output, double *jacobianMatrix) const {
  // CHECK-NEXT:     output[0] = x * i * i * jj;
  // CHECK-NEXT:     output[1] = y * i * jj * jj;
  // CHECK-NEXT:     {
  // CHECK-NEXT:         jacobianMatrix[{{2U|2UL}}] += y * 1 * jj * jj;
  // CHECK-NEXT:         jacobianMatrix[{{3U|3UL}}] += y * i * 1 * jj;
  // CHECK-NEXT:         jacobianMatrix[{{3U|3UL}}] += y * i * jj * 1;
  // CHECK-NEXT:     }
  // CHECK-NEXT:     {
  // CHECK-NEXT:         jacobianMatrix[{{0U|0UL}}] += x * 1 * jj * i;
  // CHECK-NEXT:         jacobianMatrix[{{0U|0UL}}] += x * i * 1 * jj;
  // CHECK-NEXT:         jacobianMatrix[{{1U|1UL}}] += x * i * i * 1;
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
