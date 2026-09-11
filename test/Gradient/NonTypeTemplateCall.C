// RUN: %cladclang %s -std=c++17 -I%S/../../include -o %t
// RUN: %t | %filecheck_exec %s

#include "clad/Differentiator/Differentiator.h"

extern "C" int printf(const char*, ...);

double product(double x, double y) { return x * y; }

struct Arithmetic {
  template <typename T> static T multiply(T x, T y) { return x * y; }
};

int custom_calls = 0;

namespace clad {
namespace custom_derivatives {
void product_pullback(double x, double y, double d_output, double* d_x,
                      double* d_y) {
  *d_x += d_output * y;
  *d_y += d_output * x;
}

namespace class_functions {
template <typename T>
void multiply_pullback(T x, T y, T d_output, T* d_x, T* d_y) {
  ++custom_calls;
  *d_x += d_output * y;
  *d_y += d_output * x;
}
} // namespace class_functions
} // namespace custom_derivatives
} // namespace clad

template <auto Function> double invoke(double x, double y) {
  return Function(x, y);
}

double indirect_product(double x, double y) { return invoke<product>(x, y); }

double static_product(double x, double y) { return Arithmetic::multiply(x, y); }

int main() {
  auto gradient = clad::gradient(indirect_product);
  double dx = 0, dy = 0;
  gradient.execute(3, 4, &dx, &dy);
  printf("%.1f %.1f\n", dx, dy);
  // CHECK-EXEC: 4.0 3.0
  auto static_gradient = clad::gradient(static_product);
  dx = dy = 0;
  static_gradient.execute(3, 4, &dx, &dy);
  printf("%.1f %.1f %d\n", dx, dy, custom_calls);
  // CHECK-EXEC-NEXT: 4.0 3.0 1
}
