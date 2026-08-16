// RUN: %cladclang %s -I%S/../../include -o %t 2>&1 | %filecheck %s
// RUN: %t | %filecheck_exec %s
// RUN: %cladclang -std=c++14 %s -I%S/../../include -o %t14
// RUN: %t14 | %filecheck_exec %s
// RUN: %cladclang -std=c++20 %s -I%S/../../include -o %t20
// RUN: %t20 | %filecheck_exec %s

#include "clad/Differentiator/Differentiator.h"
#include "clad/Differentiator/STLBuiltins.h"

#include <cstdio>
#include <vector>

double replace_and_square(std::vector<double> values, double x) {
  values[0] = x;
  return values[0] * values[0];
}

// CHECK: void replace_and_square_grad_1(std::vector<double> values, double x, double *_d_x) {
// CHECK-NEXT:     std::vector<double> _d_values(clad::zero_like(values));

struct PointerView {
  explicit PointerView(std::size_t size) : storage(size), data(storage.data()) {}

  std::vector<double> storage;
  double* data;
};

namespace clad {
PointerView zero_like(const PointerView& value) {
  return PointerView(value.storage.size());
}
} // namespace clad

double scale_view(const PointerView& values, double x) {
  return values.data[0] * x;
}

// CHECK: void scale_view_grad_1(const PointerView &values, double x, double *_d_x) {
// CHECK-NEXT:     PointerView _d_values(clad::zero_like(values));

struct CopyFallback {
  explicit CopyFallback(double value) : value(value) {}

  double value;
};

double replace_fallback(CopyFallback values, double x) {
  values.value = x;
  return values.value * values.value;
}

// CHECK: void replace_fallback_grad_1(CopyFallback values, double x, double *_d_x) {
// CHECK-NEXT:     CopyFallback _d_values(values);
// CHECK-NEXT:     clad::zero_init(_d_values);

int main() {
  std::vector<double> values{10.0};
  double d_x = 0.0;
  auto gradient = clad::gradient(replace_and_square, "x");
  gradient.execute(values, 3.0, &d_x);
  std::printf("%.1f\n", d_x);

  PointerView view(1);
  view.storage[0] = 4.0;
  double d_scale = 0.0;
  auto scale_gradient = clad::gradient(scale_view, "x");
  scale_gradient.execute(view, 3.0, &d_scale);
  std::printf("%.1f\n", d_scale);

  CopyFallback fallback(10.0);
  double d_fallback = 0.0;
  auto fallback_gradient = clad::gradient(replace_fallback, "x");
  fallback_gradient.execute(fallback, 3.0, &d_fallback);
  std::printf("%.1f\n", d_fallback);
}

// CHECK-EXEC: 6.0
// CHECK-EXEC-NEXT: 4.0
// CHECK-EXEC-NEXT: 6.0
