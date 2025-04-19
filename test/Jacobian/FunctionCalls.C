// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -disable-tbr %s -I%S/../../include -oFunctionCalls.out 2>&1 | %filecheck %s
// RUN: ./FunctionCalls.out | %filecheck_exec %s
// RUN: %cladclang %s -I%S/../../include -oFunctionCalls.out
// RUN: ./FunctionCalls.out | %filecheck_exec %s

#include <cmath>
#include "clad/Differentiator/Differentiator.h"

double outputs[4];
clad::matrix<double> results(2, 2);

void fn1(double i, double j, double* _clad_out_output) {
  _clad_out_output[0] = std::pow(i, j);
  _clad_out_output[1] = std::pow(j, i);
}

// CHECK: void fn1_jac(double i, double j, double *_clad_out_output, clad::matrix<double> *_d_vector__clad_out_output) {
// CHECK-NEXT:     unsigned long indepVarCount = {{2U|2UL|2ULL}};
// CHECK-NEXT:     clad::array<double> _d_vector_i = clad::one_hot_vector(indepVarCount, {{0U|0UL|0ULL}});
// CHECK-NEXT:     clad::array<double> _d_vector_j = clad::one_hot_vector(indepVarCount, {{1U|1UL|1ULL}});
// CHECK-NEXT:     *_d_vector__clad_out_output = clad::identity_matrix(_d_vector__clad_out_output->rows(), indepVarCount, {{2U|2UL|2ULL}});
// CHECK-NEXT:     {{.*}} _t0 = clad::custom_derivatives::std::pow_pushforward(i, j, _d_vector_i, _d_vector_j);
// CHECK-NEXT:     (*_d_vector__clad_out_output)[0] = _t0.pushforward;
// CHECK-NEXT:     _clad_out_output[0] = _t0.value;
// CHECK-NEXT:     {{.*}} _t1 = clad::custom_derivatives::std::pow_pushforward(j, i, _d_vector_j, _d_vector_i);
// CHECK-NEXT:     (*_d_vector__clad_out_output)[1] = _t1.pushforward;
// CHECK-NEXT:     _clad_out_output[1] = _t1.value;
// CHECK-NEXT: }

double add(double a, double b) { return a + b ;}

// CHECK: clad::ValueAndPushforward<double, clad::array<double> > add_vector_pushforward(double a, double b, clad::array<double> _d_a, clad::array<double> _d_b) {
// CHECK-NEXT:     unsigned long indepVarCount = _d_b.size();
// CHECK-NEXT:     return {a + b, _d_a + _d_b};
// CHECK-NEXT: }

void fn2(double a, double b, double* _clad_out_res){ 
    _clad_out_res[0] = a*10 + b*b*9;
    _clad_out_res[0] = add(_clad_out_res[0], 10);
    _clad_out_res[1] = a*a*9 + b*b*10;
}

// CHECK: void fn2_jac(double a, double b, double *_clad_out_res, clad::matrix<double> *_d_vector__clad_out_res) {
// CHECK-NEXT:     unsigned long indepVarCount = {{2U|2UL|2ULL}};
// CHECK-NEXT:     clad::array<double> _d_vector_a = clad::one_hot_vector(indepVarCount, {{0U|0UL|0ULL}});
// CHECK-NEXT:     clad::array<double> _d_vector_b = clad::one_hot_vector(indepVarCount, {{1U|1UL|1ULL}});
// CHECK-NEXT:     *_d_vector__clad_out_res = clad::identity_matrix(_d_vector__clad_out_res->rows(), indepVarCount, {{2U|2UL|2ULL}});
// CHECK-NEXT:     double _t0 = b * b;
// CHECK-NEXT:     (*_d_vector__clad_out_res)[0] = _d_vector_a * 10 + a * (clad::zero_vector(indepVarCount)) + (_d_vector_b * b + b * _d_vector_b) * 9 + _t0 * (clad::zero_vector(indepVarCount));
// CHECK-NEXT:     _clad_out_res[0] = a * 10 + _t0 * 9;
// CHECK-NEXT:     clad::ValueAndPushforward<double, clad::array<double> > _t1 = add_vector_pushforward(_clad_out_res[0], 10, (*_d_vector__clad_out_res)[0], clad::zero_vector(indepVarCount));
// CHECK-NEXT:     (*_d_vector__clad_out_res)[0] = _t1.pushforward;
// CHECK-NEXT:     _clad_out_res[0] = _t1.value;
// CHECK-NEXT:     double _t2 = a * a;
// CHECK-NEXT:     double _t3 = b * b;
// CHECK-NEXT:     (*_d_vector__clad_out_res)[1] = (_d_vector_a * a + a * _d_vector_a) * 9 + _t2 * (clad::zero_vector(indepVarCount)) + (_d_vector_b * b + b * _d_vector_b) * 10 + _t3 * (clad::zero_vector(indepVarCount));
// CHECK-NEXT:     _clad_out_res[1] = _t2 * 9 + _t3 * 10;
// CHECK-NEXT: }

#define INIT(F) auto d_##F = clad::jacobian(F);

#define DERIVED_FN(F) d_##F

template <unsigned numOfOutputs, typename Fn, typename... Args>
void test(Fn derivedFn, Args... args) {
  unsigned numOfParameters = sizeof...(args);
  for (unsigned i = 0; i < numOfOutputs; ++i)
    outputs[i] = 0;
  derivedFn.execute(args..., outputs, &results);
  printf("{");
  for (unsigned i = 0; i < numOfOutputs; ++i) {
    for (unsigned j = 0; j < numOfParameters; ++j) {
      printf("%.2f", results[i][j]);
      if (i != numOfOutputs - 1 || j != numOfParameters - 1)
        printf(", ");
    }
  }
  printf("}\n");
}

int main() {
  INIT(fn1);
  test<2>(DERIVED_FN(fn1), 3, 5); // CHECK-EXEC: {405.00, 266.96, 201.18, 75.00}

  INIT(fn2);
  test<2>(DERIVED_FN(fn2), 3, 5); // CHECK-EXEC: {10.00, 90.00, 54.00, 100.00}
}

