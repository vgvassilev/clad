// RUN: %cladclang %s -I%S/../../include -oVectorModeDereference.out 2>&1 | %filecheck %s
// RUN: ./VectorModeDereference.out | %filecheck_exec %s

#include "clad/Differentiator/Differentiator.h"
#include <cstdio>

double f(const double* arr, double w) { return *arr * w + *(arr + 1) * w * w; }

// CHECK: void f_dvec(const double *arr, double w, clad::array_ref<double> _d_arr, double *_d_w) {
// CHECK-NEXT:     unsigned {{int|long|long long}} indepVarCount = _d_arr.size() + {{1U|1UL|1ULL}};
// CHECK-NEXT:     clad::matrix<double> _d_vector_arr = clad::identity_matrix(_d_arr.size(), indepVarCount, {{0U|0UL|0ULL}});
// CHECK-NEXT:     clad::array<double> _d_vector_w = clad::one_hot_vector(indepVarCount, _d_arr.size());
// CHECK-NEXT:     double _t0 = *(arr + 1) * w;
// CHECK-NEXT:     {
// CHECK-NEXT:         clad::array<double> _d_vector_return(_d_vector_arr[0] * w + *arr * _d_vector_w + (_d_vector_arr[1] * w + *(arr + 1) * _d_vector_w) * w + _t0 * _d_vector_w);
// CHECK-NEXT:         _d_arr = _d_vector_return.slice({{0U|0UL|0ULL}}, _d_arr.size());
// CHECK-NEXT:         *_d_w = _d_vector_return[_d_arr.size()];
// CHECK-NEXT:         return;
// CHECK-NEXT:     }
// CHECK-NEXT: }

// No tangent matrix for y: dereferences keep the scalar path.
double g(double x, const double* y) {
  int k = 0;
  return *y * x + *(y - 1) * x + *(y + (k = 1)) * x;
}

// CHECK: void g_dvec_0(double x, const double *y, double *_d_x) {
// CHECK-NEXT:     unsigned {{int|long|long long}} indepVarCount = {{1U|1UL|1ULL}};
// CHECK-NEXT:     clad::array<double> _d_vector_x = clad::one_hot_vector(indepVarCount, {{0U|0UL|0ULL}});
// CHECK-NEXT:     clad::array<int> _d_vector_k(clad::zero_vector(indepVarCount));
// CHECK-NEXT:     int k = 0;
// CHECK-NEXT:     {
// CHECK-NEXT:         clad::array<double> _d_vector_return(*y * _d_vector_x + *(y - 1) * _d_vector_x + *(y + (k = 1)) * _d_vector_x);
// CHECK-NEXT:         *_d_x = _d_vector_return[{{0U|0UL|0ULL}}];
// CHECK-NEXT:         return;
// CHECK-NEXT:     }
// CHECK-NEXT: }

int main() {
  auto f_dvec = clad::differentiate<clad::opts::vector_mode>(f, "arr, w");
  double arr[2] = {2, 3}, darr[2] = {0, 0}, dw = 0;
  clad::array_ref<double> darr_ref(darr, 2);
  f_dvec.execute(arr, 1.5, darr_ref, &dw);
  printf("{%.2f, %.2f, %.2f}\n", darr[0], darr[1], dw); // CHECK-EXEC: {1.50, 2.25, 11.00}

  auto g_dvec = clad::differentiate<clad::opts::vector_mode>(g, "x");
  double y[3] = {1, 2, 3}, dx = 0;
  g_dvec.execute(1.5, y + 1, &dx);
  printf("{%.2f}\n", dx); // CHECK-EXEC: {6.00}
}
