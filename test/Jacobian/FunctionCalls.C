// RUN: %cladclang %s -I%S/../../include -oFunctionCalls.out 2>&1 | %filecheck %s
// RUN: ./FunctionCalls.out | %filecheck_exec %s
// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -enable-tbr %s -I%S/../../include -oFunctionCalls.out
// RUN: ./FunctionCalls.out | %filecheck_exec %s

#include <cmath>
#include "clad/Differentiator/Differentiator.h"

double outputs[4];
clad::matrix<double> results(2, 2);

void fn1(double i, double j, double* output) {
  output[0] = std::pow(i, j);
  output[1] = std::pow(j, i);
}

// CHECK: void fn1_jac(double i, double j, double *output, clad::matrix<double> *_d_vector_output) {
// CHECK-NEXT:     unsigned long indepVarCount = _d_vector_output->rows() + {{2U|2UL|2ULL}};
// CHECK-NEXT:     clad::array<double> _d_vector_i = clad::one_hot_vector(indepVarCount, {{0U|0UL|0ULL}});
// CHECK-NEXT:     clad::array<double> _d_vector_j = clad::one_hot_vector(indepVarCount, {{1U|1UL|1ULL}});
// CHECK-NEXT:     *_d_vector_output = clad::identity_matrix(_d_vector_output->rows(), indepVarCount, {{2U|2UL|2ULL}});
// CHECK-NEXT:     {{.*}} _t0 = clad::custom_derivatives::pow_pushforward(i, j, _d_vector_i, _d_vector_j);
// CHECK-NEXT:     *_d_vector_output[0] = _t0.pushforward;
// CHECK-NEXT:     output[0] = _t0.value;
// CHECK-NEXT:     {{.*}} _t1 = clad::custom_derivatives::pow_pushforward(j, i, _d_vector_j, _d_vector_i);
// CHECK-NEXT:     *_d_vector_output[1] = _t1.pushforward;
// CHECK-NEXT:     output[1] = _t1.value;
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
}

