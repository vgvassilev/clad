// RUN: %cladclang -I%S/../../include -DCLAD_NO_NUM_DIFF %s 2>&1 | %filecheck %s

#include "clad/Differentiator/Differentiator.h"

double sum(float a, float b) { 
  return a + b; 
}

namespace clad {
  namespace custom_derivatives {
    template <typename T>
    void sum_pullback(T a, T b, double d_y, T *d_a, T *d_b) {
      if (d_a) *d_a += d_y;
      if (d_b) *d_b += d_y;
    }
  }
}

double fn(double u) {
  //Passing '7' (int) to 'float b' triggers the ImplicitCastExpr.
  //CHECK: clad::custom_derivatives::sum_pullback
  return sum(u, 7);
}

int main() {
  auto fn_grad = clad::gradient(fn);
  double d_u = 0;
  fn_grad.execute(1.0, &d_u);
  
  return 0;
}
