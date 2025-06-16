// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -disable-tbr %s -I%S/../../include -oTemplateFunctors.out 2>&1 | %filecheck %s
// RUN: ./TemplateFunctors.out | %filecheck_exec %s
// RUN: %cladclang %s -I%S/../../include -oTemplateFunctors.out
// RUN: ./TemplateFunctors.out | %filecheck_exec %s

#include "clad/Differentiator/Differentiator.h"

template <typename T> struct Experiment {
  mutable T x, y;
  Experiment(T p_x, T p_y) : x(p_x), y(p_y) {}
  void operator()(T i, T j, T *_clad_out_output) {
    _clad_out_output[0] = x*y*i*j;
    _clad_out_output[1] = 2*x*y*i*j;
  }
  void setX(T val) { x = val; }
};

// CHECK: void operator_call_jac(double i, double j, double *_clad_out_output, clad::matrix<double> *_d_vector__clad_out_output) {
// CHECK-NEXT:     unsigned long indepVarCount = {{2U|2UL|2ULL}};
// CHECK-NEXT:     clad::array<double> _d_vector_i = clad::one_hot_vector(indepVarCount, {{0U|0UL|0ULL}});
// CHECK-NEXT:     clad::array<double> _d_vector_j = clad::one_hot_vector(indepVarCount, {{1U|1UL|1ULL}});
// CHECK-NEXT:     *_d_vector__clad_out_output = clad::identity_matrix(_d_vector__clad_out_output->rows(), indepVarCount, {{2U|2UL|2ULL}});
// CHECK-NEXT:     double &_t0 = this->x;
// CHECK-NEXT:     double &_t1 = this->y;
// CHECK-NEXT:     double _t2 = _t0 * _t1;
// CHECK-NEXT:     double _t3 = _t2 * i;
// CHECK-NEXT:     (*_d_vector__clad_out_output)[0] = ((0 * _t1 + _t0 * 0) * i + _t2 * _d_vector_i) * j + _t3 * _d_vector_j;
// CHECK-NEXT:     _clad_out_output[0] = _t3 * j;
// CHECK-NEXT:     double &_t4 = this->x;
// CHECK-NEXT:     double _t5 = 2 * _t4;
// CHECK-NEXT:     double &_t6 = this->y;
// CHECK-NEXT:     double _t7 = _t5 * _t6;
// CHECK-NEXT:     double _t8 = _t7 * i;
// CHECK-NEXT:     (*_d_vector__clad_out_output)[1] = ((((clad::zero_vector(indepVarCount)) * _t4 + 2 * 0) * _t6 + _t5 * 0) * i + _t7 * _d_vector_i) * j + _t8 * _d_vector_j;
// CHECK-NEXT:     _clad_out_output[1] = _t8 * j;
// CHECK-NEXT: }

template <> struct Experiment<long double> {
  mutable long double x, y;
  Experiment(long double p_x, long double p_y) : x(p_x), y(p_y) {}
  void operator()(long double i, long double j, long double *_clad_out_output) {
    _clad_out_output[0] = x*y*i*i*j;
    _clad_out_output[1] = 2*x*y*i*i*j;
  }
  void setX(long double val) { x = val; }
};

// CHECK: void operator_call_jac(long double i, long double j, long double *_clad_out_output, clad::matrix<long double> *_d_vector__clad_out_output) {
// CHECK-NEXT:     unsigned long indepVarCount = {{2U|2UL|2ULL}};
// CHECK-NEXT:     clad::array<long double> _d_vector_i = clad::one_hot_vector(indepVarCount, {{0U|0UL|0ULL}});
// CHECK-NEXT:     clad::array<long double> _d_vector_j = clad::one_hot_vector(indepVarCount, {{1U|1UL|1ULL}});
// CHECK-NEXT:     *_d_vector__clad_out_output = clad::identity_matrix(_d_vector__clad_out_output->rows(), indepVarCount, {{2U|2UL|2ULL}});
// CHECK-NEXT:     long double &_t0 = this->x;
// CHECK-NEXT:     long double &_t1 = this->y;
// CHECK-NEXT:     long double _t2 = _t0 * _t1;
// CHECK-NEXT:     long double _t3 = _t2 * i;
// CHECK-NEXT:     long double _t4 = _t3 * i;
// CHECK-NEXT:     (*_d_vector__clad_out_output)[0] = (((0 * _t1 + _t0 * 0) * i + _t2 * _d_vector_i) * i + _t3 * _d_vector_i) * j + _t4 * _d_vector_j;
// CHECK-NEXT:     _clad_out_output[0] = _t4 * j;
// CHECK-NEXT:     long double &_t5 = this->x;
// CHECK-NEXT:     long double _t6 = 2 * _t5;
// CHECK-NEXT:     long double &_t7 = this->y;
// CHECK-NEXT:     long double _t8 = _t6 * _t7;
// CHECK-NEXT:     long double _t9 = _t8 * i;
// CHECK-NEXT:     long double _t10 = _t9 * i;
// CHECK-NEXT:     (*_d_vector__clad_out_output)[1] = (((((clad::zero_vector(indepVarCount)) * _t5 + 2 * 0) * _t7 + _t6 * 0) * i + _t8 * _d_vector_i) * i + _t9 * _d_vector_i) * j + _t10 * _d_vector_j;
// CHECK-NEXT:     _clad_out_output[1] = _t10 * j;
// CHECK-NEXT: }

#define INIT(E)                                                                \
  auto d_##E = clad::jacobian(&E);                                             \
  auto d_##E##Ref = clad::jacobian(E);

#define TEST_DOUBLE(E, ...)                                                    \
  _clad_out_output[0] = _clad_out_output[1] = 0;                                                   \
  d_##E.execute(__VA_ARGS__, _clad_out_output, &result);                                 \
  printf("{%.2f, %.2f, %.2f, %.2f} ", result[0][0], result[0][1],              \
                                      result[1][0], result[1][1]);             \
  _clad_out_output[0] = _clad_out_output[1] = 0;                                                   \
  d_##E##Ref.execute(__VA_ARGS__, _clad_out_output, &result);                            \
  printf("{%.2f, %.2f, %.2f, %.2f} ", result[0][0], result[0][1],              \
                                      result[1][0], result[1][1]);             

#define TEST_LONG_DOUBLE(E, ...)                                               \
  output_ld[0] = output_ld[1] = 0;                                             \
  d_##E.execute(__VA_ARGS__, output_ld, &result_ld);                           \
  printf("{%.2Lf, %.2Lf, %.2Lf, %.2Lf} ", result_ld[0][0], result_ld[0][1],    \
                                      result_ld[1][0], result_ld[1][1]);       \
  output_ld[0] = output_ld[1] = 0;                                             \
  d_##E##Ref.execute(__VA_ARGS__, output_ld, &result_ld);                      \
  printf("{%.2Lf, %.2Lf, %.2Lf, %.2Lf} ", result_ld[0][0], result_ld[0][1],    \
                                      result_ld[1][0], result_ld[1][1]);             

int main() {
  double _clad_out_output[2];
  clad::matrix<double> result(2, 2);
  long double output_ld[2];
  clad::matrix<long double> result_ld(2, 2);
  Experiment<double> E(3, 5);
  Experiment<long double> E_ld(3, 5);

  INIT(E);
  INIT(E_ld);

  TEST_DOUBLE(E, 7, 9);           // CHECK-EXEC: {135.00, 105.00, 270.00, 210.00} {135.00, 105.00, 270.00, 210.00}
  TEST_LONG_DOUBLE(E_ld, 7, 9);   // CHECK-EXEC: {1890.00, 735.00, 3780.00, 1470.00} {1890.00, 735.00, 3780.00, 1470.00}

  E.setX(5);
  E_ld.setX(5);

  TEST_DOUBLE(E, 7, 9);           // CHECK-EXEC: {225.00, 175.00, 450.00, 350.00} {225.00, 175.00, 450.00, 350.00}
  TEST_LONG_DOUBLE(E_ld, 7, 9);   // CHECK-EXEC: {3150.00, 1225.00, 6300.00, 2450.00} {3150.00, 1225.00, 6300.00, 2450.00}
}

