// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -disable-tbr %s -I%S/../../include -oFunctors.out 2>&1 | %filecheck %s
// RUN: ./Functors.out | %filecheck_exec %s
// RUN: %cladclang %s -I%S/../../include -oFunctors.out
// RUN: ./Functors.out | %filecheck_exec %s

#include "clad/Differentiator/Differentiator.h"

struct Experiment {
  mutable double x, y;
  Experiment(double p_x, double p_y) : x(p_x), y(p_y) {}
  void operator()(double i, double j, double *_clad_out_output) {
    _clad_out_output[0] = x*i*i*j;
    _clad_out_output[1] = y*i*j*j;
  }
  void setX(double val) {
    x = val;
  }

  // CHECK: void operator_call_jac(double i, double j, double *_clad_out_output, clad::matrix<double> *_d_vector__clad_out_output) {
  // CHECK-NEXT:     unsigned long indepVarCount = {{2U|2UL|2ULL}};
  // CHECK-NEXT:     clad::array<double> _d_vector_i = clad::one_hot_vector(indepVarCount, {{0U|0UL|0ULL}});
  // CHECK-NEXT:     clad::array<double> _d_vector_j = clad::one_hot_vector(indepVarCount, {{1U|1UL|1ULL}});
  // CHECK-NEXT:     *_d_vector__clad_out_output = clad::identity_matrix(_d_vector__clad_out_output->rows(), indepVarCount, {{2U|2UL|2ULL}});
  // CHECK-NEXT:     double &_t0 = this->x;
  // CHECK-NEXT:     double _t1 = _t0 * i;
  // CHECK-NEXT:     double _t2 = _t1 * i;
  // CHECK-NEXT:     (*_d_vector__clad_out_output)[0] = ((0 * i + _t0 * _d_vector_i) * i + _t1 * _d_vector_i) * j + _t2 * _d_vector_j;
  // CHECK-NEXT:     _clad_out_output[0] = _t2 * j;
  // CHECK-NEXT:     double &_t3 = this->y;
  // CHECK-NEXT:     double _t4 = _t3 * i;
  // CHECK-NEXT:     double _t5 = _t4 * j;
  // CHECK-NEXT:     (*_d_vector__clad_out_output)[1] = ((0 * i + _t3 * _d_vector_i) * j + _t4 * _d_vector_j) * j + _t5 * _d_vector_j;
  // CHECK-NEXT:     _clad_out_output[1] = _t5 * j;
  // CHECK-NEXT: }

};

struct ExperimentConst {
  mutable double x, y;
  ExperimentConst(double p_x, double p_y) : x(p_x), y(p_y) {}
  void operator()(double i, double j, double *_clad_out_output) const {
    _clad_out_output[0] = x*i*i*j;
    _clad_out_output[1] = y*i*j*j;
  }
  void setX(double val) const {
    x = val;
  }

  // CHECK: void operator_call_jac(double i, double j, double *_clad_out_output, clad::matrix<double> *_d_vector__clad_out_output) const {
  // CHECK-NEXT:     unsigned long indepVarCount = {{2U|2UL|2ULL}};
  // CHECK-NEXT:     clad::array<double> _d_vector_i = clad::one_hot_vector(indepVarCount, {{0U|0UL|0ULL}});
  // CHECK-NEXT:     clad::array<double> _d_vector_j = clad::one_hot_vector(indepVarCount, {{1U|1UL|1ULL}});
  // CHECK-NEXT:     *_d_vector__clad_out_output = clad::identity_matrix(_d_vector__clad_out_output->rows(), indepVarCount, {{2U|2UL|2ULL}});
  // CHECK-NEXT:     double &_t0 = this->x;
  // CHECK-NEXT:     double _t1 = _t0 * i;
  // CHECK-NEXT:     double _t2 = _t1 * i;
  // CHECK-NEXT:     (*_d_vector__clad_out_output)[0] = ((0 * i + _t0 * _d_vector_i) * i + _t1 * _d_vector_i) * j + _t2 * _d_vector_j;
  // CHECK-NEXT:     _clad_out_output[0] = _t2 * j;
  // CHECK-NEXT:     double &_t3 = this->y;
  // CHECK-NEXT:     double _t4 = _t3 * i;
  // CHECK-NEXT:     double _t5 = _t4 * j;
  // CHECK-NEXT:     (*_d_vector__clad_out_output)[1] = ((0 * i + _t3 * _d_vector_i) * j + _t4 * _d_vector_j) * j + _t5 * _d_vector_j;
  // CHECK-NEXT:     _clad_out_output[1] = _t5 * j;
  // CHECK-NEXT: }

};

struct ExperimentVolatile {
  mutable double x, y;
  ExperimentVolatile(double p_x, double p_y) : x(p_x), y(p_y) {}
  void operator()(double i, double j, double *_clad_out_output) volatile {
    _clad_out_output[0] = x*i*i*j;
    _clad_out_output[1] = y*i*j*j;
  }
  void setX(double val) volatile {
    x = val;
  }

  // CHECK: void operator_call_jac(double i, double j, double *_clad_out_output, clad::matrix<double> *_d_vector__clad_out_output) volatile {
  // CHECK-NEXT:     unsigned long indepVarCount = {{2U|2UL|2ULL}};
  // CHECK-NEXT:     clad::array<double> _d_vector_i = clad::one_hot_vector(indepVarCount, {{0U|0UL|0ULL}});
  // CHECK-NEXT:     clad::array<double> _d_vector_j = clad::one_hot_vector(indepVarCount, {{1U|1UL|1ULL}});
  // CHECK-NEXT:     *_d_vector__clad_out_output = clad::identity_matrix(_d_vector__clad_out_output->rows(), indepVarCount, {{2U|2UL|2ULL}});
  // CHECK-NEXT:     volatile double &_t0 = this->x;
  // CHECK-NEXT:     double _t1 = _t0 * i;
  // CHECK-NEXT:     double _t2 = _t1 * i;
  // CHECK-NEXT:     (*_d_vector__clad_out_output)[0] = ((0 * i + _t0 * _d_vector_i) * i + _t1 * _d_vector_i) * j + _t2 * _d_vector_j;
  // CHECK-NEXT:     _clad_out_output[0] = _t2 * j;
  // CHECK-NEXT:     volatile double &_t3 = this->y;
  // CHECK-NEXT:     double _t4 = _t3 * i;
  // CHECK-NEXT:     double _t5 = _t4 * j;
  // CHECK-NEXT:     (*_d_vector__clad_out_output)[1] = ((0 * i + _t3 * _d_vector_i) * j + _t4 * _d_vector_j) * j + _t5 * _d_vector_j;
  // CHECK-NEXT:     _clad_out_output[1] = _t5 * j;
  // CHECK-NEXT: }

};

struct ExperimentConstVolatile {
  mutable double x, y;
  ExperimentConstVolatile(double p_x, double p_y) : x(p_x), y(p_y) {}
  void operator()(double i, double j, double *_clad_out_output) const volatile {
    _clad_out_output[0] = x*i*i*j;
    _clad_out_output[1] = y*i*j*j;
  }
  void setX(double val) const volatile {
    x = val;
  }

  // CHECK: void operator_call_jac(double i, double j, double *_clad_out_output, clad::matrix<double> *_d_vector__clad_out_output) const volatile {
  // CHECK-NEXT:     unsigned long indepVarCount = {{2U|2UL|2ULL}};
  // CHECK-NEXT:     clad::array<double> _d_vector_i = clad::one_hot_vector(indepVarCount, {{0U|0UL|0ULL}});
  // CHECK-NEXT:     clad::array<double> _d_vector_j = clad::one_hot_vector(indepVarCount, {{1U|1UL|1ULL}});
  // CHECK-NEXT:     *_d_vector__clad_out_output = clad::identity_matrix(_d_vector__clad_out_output->rows(), indepVarCount, {{2U|2UL|2ULL}});
  // CHECK-NEXT:     volatile double &_t0 = this->x;
  // CHECK-NEXT:     double _t1 = _t0 * i;
  // CHECK-NEXT:     double _t2 = _t1 * i;
  // CHECK-NEXT:     (*_d_vector__clad_out_output)[0] = ((0 * i + _t0 * _d_vector_i) * i + _t1 * _d_vector_i) * j + _t2 * _d_vector_j;
  // CHECK-NEXT:     _clad_out_output[0] = _t2 * j;
  // CHECK-NEXT:     volatile double &_t3 = this->y;
  // CHECK-NEXT:     double _t4 = _t3 * i;
  // CHECK-NEXT:     double _t5 = _t4 * j;
  // CHECK-NEXT:     (*_d_vector__clad_out_output)[1] = ((0 * i + _t3 * _d_vector_i) * j + _t4 * _d_vector_j) * j + _t5 * _d_vector_j;
  // CHECK-NEXT:     _clad_out_output[1] = _t5 * j;
  // CHECK-NEXT: }

};

namespace outer {
  namespace inner {
    struct ExperimentNNS {
      mutable double x, y;
      ExperimentNNS(double p_x, double p_y) : x(p_x), y(p_y) {}
      void operator()(double i, double j, double *_clad_out_output) {
        _clad_out_output[0] = x*i*i*j;
        _clad_out_output[1] = y*i*j*j;
      }
      void setX(double val) {
        x = val;
      }
      
  // CHECK: void operator_call_jac(double i, double j, double *_clad_out_output, clad::matrix<double> *_d_vector__clad_out_output) {
  // CHECK-NEXT:     unsigned long indepVarCount = {{2U|2UL|2ULL}};
  // CHECK-NEXT:     clad::array<double> _d_vector_i = clad::one_hot_vector(indepVarCount, {{0U|0UL|0ULL}});
  // CHECK-NEXT:     clad::array<double> _d_vector_j = clad::one_hot_vector(indepVarCount, {{1U|1UL|1ULL}});
  // CHECK-NEXT:     *_d_vector__clad_out_output = clad::identity_matrix(_d_vector__clad_out_output->rows(), indepVarCount, {{2U|2UL|2ULL}});
  // CHECK-NEXT:     double &_t0 = this->x;
  // CHECK-NEXT:     double _t1 = _t0 * i;
  // CHECK-NEXT:     double _t2 = _t1 * i;
  // CHECK-NEXT:     (*_d_vector__clad_out_output)[0] = ((0 * i + _t0 * _d_vector_i) * i + _t1 * _d_vector_i) * j + _t2 * _d_vector_j;
  // CHECK-NEXT:     _clad_out_output[0] = _t2 * j;
  // CHECK-NEXT:     double &_t3 = this->y;
  // CHECK-NEXT:     double _t4 = _t3 * i;
  // CHECK-NEXT:     double _t5 = _t4 * j;
  // CHECK-NEXT:     (*_d_vector__clad_out_output)[1] = ((0 * i + _t3 * _d_vector_i) * j + _t4 * _d_vector_j) * j + _t5 * _d_vector_j;
  // CHECK-NEXT:     _clad_out_output[1] = _t5 * j;
  // CHECK-NEXT: }

    };

    auto lambdaNNS = [](double i, double j, double *_clad_out_output) {
      _clad_out_output[0] = i*i*j;
      _clad_out_output[1] = i*j*j;
    };

  // CHECK: inline void operator_call_jac(double i, double j, double *_clad_out_output, clad::matrix<double> *_d_vector__clad_out_output) const {
  // CHECK-NEXT:     unsigned long indepVarCount = {{2U|2UL|2ULL}};
  // CHECK-NEXT:     clad::array<double> _d_vector_i = clad::one_hot_vector(indepVarCount, {{0U|0UL|0ULL}});
  // CHECK-NEXT:     clad::array<double> _d_vector_j = clad::one_hot_vector(indepVarCount, {{1U|1UL|1ULL}});
  // CHECK-NEXT:     *_d_vector__clad_out_output = clad::identity_matrix(_d_vector__clad_out_output->rows(), indepVarCount, {{2U|2UL|2ULL}});
  // CHECK-NEXT:     double _t0 = i * i;
  // CHECK-NEXT:     (*_d_vector__clad_out_output)[0] = (_d_vector_i * i + i * _d_vector_i) * j + _t0 * _d_vector_j;
  // CHECK-NEXT:     _clad_out_output[0] = _t0 * j;
  // CHECK-NEXT:     double _t1 = i * j;
  // CHECK-NEXT:     (*_d_vector__clad_out_output)[1] = (_d_vector_i * j + i * _d_vector_j) * j + _t1 * _d_vector_j;
  // CHECK-NEXT:     _clad_out_output[1] = _t1 * j;
  // CHECK-NEXT: }

  }
}

#define INIT(E)                                                                \
  auto d_##E = clad::jacobian(&E);                                             \
  auto d_##E##Ref = clad::jacobian(E);

#define TEST(E)                                                                \
  _clad_out_output[0] = _clad_out_output[1] = 0;                                                   \
  d_##E.execute(7, 9, _clad_out_output, &result);                                        \
  printf("{%.2f, %.2f, %.2f, %.2f}, ", result[0][0], result[0][1],             \
                                       result[1][0], result[1][1]);            \
  _clad_out_output[0] = _clad_out_output[1] = 0;                                                   \
  d_##E##Ref.execute(7, 9, _clad_out_output, &result);                                   \
  printf("{%.2f, %.2f, %.2f, %.2f}, ", result[0][0], result[0][1],             \
                                       result[1][0], result[1][1]);

double x = 3;
double y = 5;
int main() {
  double _clad_out_output[2];
  clad::matrix<double> result(2, 2);
  Experiment E(3, 5);
  auto E_Again = E;
  const ExperimentConst E_Const(3, 5);
  volatile ExperimentVolatile E_Volatile(3, 5);
  const volatile ExperimentConstVolatile E_ConstVolatile(3, 5);
  outer::inner::ExperimentNNS E_NNS(3, 5);
  auto E_NNS_Again = E_NNS;
  auto lambda = [](double i, double j, double *_clad_out_output) {
    _clad_out_output[0] = i*i*j;
    _clad_out_output[1] = i*j*j;
  };

  // CHECK-NEXT: inline void operator_call_jac(double i, double j, double *_clad_out_output, clad::matrix<double> *_d_vector__clad_out_output) const {
  // CHECK-NEXT:     unsigned long indepVarCount = {{2U|2UL|2ULL}};
  // CHECK-NEXT:     clad::array<double> _d_vector_i = clad::one_hot_vector(indepVarCount, {{0U|0UL|0ULL}});
  // CHECK-NEXT:     clad::array<double> _d_vector_j = clad::one_hot_vector(indepVarCount, {{1U|1UL|1ULL}});
  // CHECK-NEXT:     *_d_vector__clad_out_output = clad::identity_matrix(_d_vector__clad_out_output->rows(), indepVarCount, {{2U|2UL|2ULL}});
  // CHECK-NEXT:     double _t0 = i * i;
  // CHECK-NEXT:     (*_d_vector__clad_out_output)[0] = (_d_vector_i * i + i * _d_vector_i) * j + _t0 * _d_vector_j;
  // CHECK-NEXT:     _clad_out_output[0] = _t0 * j;
  // CHECK-NEXT:     double _t1 = i * j;
  // CHECK-NEXT:     (*_d_vector__clad_out_output)[1] = (_d_vector_i * j + i * _d_vector_j) * j + _t1 * _d_vector_j;
  // CHECK-NEXT:     _clad_out_output[1] = _t1 * j;
  // CHECK-NEXT: }

  auto lambdaWithCapture = [&](double i, double jj, double *_clad_out_output) {
    _clad_out_output[0] = x*i*i*jj;
    _clad_out_output[1] = y*i*jj*jj;
  };

// CHECK: inline void operator_call_jac(double i, double jj, double *_clad_out_output, clad::matrix<double> *_d_vector__clad_out_output) const {
// CHECK-NEXT:     unsigned long indepVarCount = {{2U|2UL|2ULL}};
// CHECK-NEXT:     clad::array<double> _d_vector_i = clad::one_hot_vector(indepVarCount, {{0U|0UL|0ULL}});
// CHECK-NEXT:     clad::array<double> _d_vector_jj = clad::one_hot_vector(indepVarCount, {{1U|1UL|1ULL}});
// CHECK-NEXT:     *_d_vector__clad_out_output = clad::identity_matrix(_d_vector__clad_out_output->rows(), indepVarCount, {{2U|2UL|2ULL}});
// CHECK-NEXT:     double _t0 = x * i;
// CHECK-NEXT:     double _t1 = _t0 * i;
// CHECK-NEXT:     (*_d_vector__clad_out_output)[0] = ((0. * i + x * _d_vector_i) * i + _t0 * _d_vector_i) * jj + _t1 * _d_vector_jj;
// CHECK-NEXT:     _clad_out_output[0] = _t1 * jj;
// CHECK-NEXT:     double _t2 = y * i;
// CHECK-NEXT:     double _t3 = _t2 * jj;
// CHECK-NEXT:     (*_d_vector__clad_out_output)[1] = ((0. * i + y * _d_vector_i) * jj + _t2 * _d_vector_jj) * jj + _t3 * _d_vector_jj;
// CHECK-NEXT:     _clad_out_output[1] = _t3 * jj;
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

