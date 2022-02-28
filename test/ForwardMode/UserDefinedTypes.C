// RUN: %cladclang %s -I%S/../../include -oUserDefinedTypes.out 2>&1 | FileCheck %s
// RUN: ./UserDefinedTypes.out | FileCheck -check-prefix=CHECK-EXEC %s

// CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"

std::pair<double, double> fn1(double i, double j) {
  std::pair<double, double> c(3, 5), d({7.00, 9.00});
  std::pair<double, double> e = d;
  c.first += i;
  c.second += 2 * i + j;
  d.first += 2 * i;
  d.second += i;
  c.first += d.first;
  c.second += d.second;
  return c;
}

// CHECK: std::pair<double, double> fn1_darg0(double i, double j) {
// CHECK-NEXT:     double _d_i = 1;
// CHECK-NEXT:     double _d_j = 0;
// CHECK-NEXT:     std::pair<double, double> _d_c(0, 0), _d_d({0., 0.});
// CHECK-NEXT:     std::pair<double, double> c(3, 5), d({7., 9.});
// CHECK-NEXT:     std::pair<double, double> _d_e = _d_d;
// CHECK-NEXT:     std::pair<double, double> e = d;
// CHECK-NEXT:     _d_c.first += _d_i;
// CHECK-NEXT:     c.first += i;
// CHECK-NEXT:     _d_c.second += 0 * i + 2 * _d_i + _d_j;
// CHECK-NEXT:     c.second += 2 * i + j;
// CHECK-NEXT:     _d_d.first += 0 * i + 2 * _d_i;
// CHECK-NEXT:     d.first += 2 * i;
// CHECK-NEXT:     _d_d.second += _d_i;
// CHECK-NEXT:     d.second += i;
// CHECK-NEXT:     _d_c.first += _d_d.first;
// CHECK-NEXT:     c.first += d.first;
// CHECK-NEXT:     _d_c.second += _d_d.second;
// CHECK-NEXT:     c.second += d.second;
// CHECK-NEXT:     return _d_c;
// CHECK-NEXT: }

std::pair<double, double> fn2(double i, double j) {
  std::pair<double, double> c, d;
  std::pair<double, double> e = d;
  return std::pair<double, double>();
}

// CHECK: std::pair<double, double> fn2_darg0(double i, double j) {
// CHECK-NEXT:     double _d_i = 1;
// CHECK-NEXT:     double _d_j = 0;
// CHECK-NEXT:     std::pair<double, double> _d_c, _d_d;
// CHECK-NEXT:     std::pair<double, double> c, d;
// CHECK-NEXT:     std::pair<double, double> _d_e = _d_d;
// CHECK-NEXT:     std::pair<double, double> e = d;
// CHECK-NEXT:     return std::pair<double, double>();
// CHECK-NEXT: }

std::pair<double, double> fn3(double i, double j) {
  std::pair<double, double> c, d;
  std::pair<double, double> e = d;
  return {};
}

// CHECK: std::pair<double, double> fn3_darg0(double i, double j) {
// CHECK-NEXT:     double _d_i = 1;
// CHECK-NEXT:     double _d_j = 0;
// CHECK-NEXT:     std::pair<double, double> _d_c, _d_d;
// CHECK-NEXT:     std::pair<double, double> c, d;
// CHECK-NEXT:     std::pair<double, double> _d_e = _d_d;
// CHECK-NEXT:     std::pair<double, double> e = d;
// CHECK-NEXT:     return {};
// CHECK-NEXT: }

std::pair<double, double> fn4(double i, double j) {
  std::pair<double, double> c, d;
  std::pair<double, double> e = d;
  return std::pair<double, double>(1, 3);
}

// CHECK: std::pair<double, double> fn4_darg0(double i, double j) {
// CHECK-NEXT:     double _d_i = 1;
// CHECK-NEXT:     double _d_j = 0;
// CHECK-NEXT:     std::pair<double, double> _d_c, _d_d;
// CHECK-NEXT:     std::pair<double, double> c, d;
// CHECK-NEXT:     std::pair<double, double> _d_e = _d_d;
// CHECK-NEXT:     std::pair<double, double> e = d;
// CHECK-NEXT:     return std::pair<double, double>(0, 0);
// CHECK-NEXT: }

template<typename T, unsigned N>
struct Tensor {
  T data[N] = {};
};

Tensor<double, 5> fn5(double i, double j) {
  Tensor<double, 5> T;
  for (int l=0; l<5; ++l) {
    T.data[l] = (l+1)*i;
  }
  return T;
} 

// CHECK: Tensor<double, 5> fn5_darg0(double i, double j) {
// CHECK-NEXT:     double _d_i = 1;
// CHECK-NEXT:     double _d_j = 0;
// CHECK-NEXT:     Tensor<double, 5> _d_T;
// CHECK-NEXT:     Tensor<double, 5> T;
// CHECK-NEXT:     {
// CHECK-NEXT:         int _d_l = 0;
// CHECK-NEXT:         for (int l = 0; l < 5; ++l) {
// CHECK-NEXT:             int _t0 = (l + 1);
// CHECK-NEXT:             _d_T.data[l] = (_d_l + 0) * i + _t0 * _d_i;
// CHECK-NEXT:             T.data[l] = _t0 * i;
// CHECK-NEXT:         }
// CHECK-NEXT:     }
// CHECK-NEXT:     return _d_T;
// CHECK-NEXT: }

#define INIT(fn, ...) auto d_##fn = clad::differentiate(fn, __VA_ARGS__);

#define TEST_PAIR_OF_DOUBLES(fn, ...)                                          \
  {                                                                            \
    auto res = d_##fn.execute(__VA_ARGS__);                                    \
    printf("{%.2f, %.2f}\n", res.first, res.second);                           \
  }

#define TEST_TENSOR_DOUBLE_5(fn, ...)                                          \
  {                                                                            \
    auto res = d_##fn.execute(__VA_ARGS__);                                    \
    printf("{%.2f, %.2f, %.2f, %.2f, %.2f}\n", res.data[0], res.data[1],       \
           res.data[2], res.data[3], res.data[4]);                             \
  }

int main() {
  INIT(fn1, "i");
  INIT(fn2, "i");
  INIT(fn3, "i");
  INIT(fn4, "i");
  INIT(fn5, "i");

  TEST_PAIR_OF_DOUBLES(fn1, 3, 5);  // CHECK-EXEC: {3.00, 3.00}
  TEST_PAIR_OF_DOUBLES(fn2, 3, 5);  // CHECK-EXEC: {0.00, 0.00}
  TEST_PAIR_OF_DOUBLES(fn3, 3, 5);  // CHECK-EXEC: {0.00, 0.00}
  TEST_PAIR_OF_DOUBLES(fn4, 3, 5);  // CHECK-EXEC: {0.00, 0.00}
  TEST_TENSOR_DOUBLE_5(fn5, 3, 5);  // CHECK-EXEC: {1.00, 2.00, 3.00, 4.00, 5.00}
}