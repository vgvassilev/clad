// RUN: %cladclang %s -I%S/../../include -oCladtorch.out 2>&1 | %filecheck %s
// RUN: ./Cladtorch.out | %filecheck_exec %s
// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -disable-tbr %s -I%S/../../include -oCladtorch.out
// RUN: ./Cladtorch.out | %filecheck_exec %s

#include "clad/Differentiator/Differentiator.h"

namespace cladtorch {
struct Tensor {
  Tensor(): data(0) {};
  float data;
};
}

float fn1(const cladtorch::Tensor& t, const cladtorch::Tensor& u) {
  auto b = t;
  return b.data;
}
// CHECK: void fn1_grad_0(const cladtorch::Tensor &t, const cladtorch::Tensor &u, cladtorch::Tensor *_d_t) {
// CHECK-NEXT:     {{.*}}cladtorch::Tensor _d_u(u);
// CHECK-NEXT:     {{.*}}cladtorch::Tensor b = t;
// CHECK-NEXT:     {{.*}}cladtorch::Tensor _d_b(t);
// CHECK-NEXT:     clad::zero_init(_d_b);
// CHECK-NEXT:     _d_b.data += 1;
// CHECK-NEXT:     Tensor::constructor_pullback(t, &_d_b, &(*_d_t));
// CHECK-NEXT: }

int main() {
  cladtorch::Tensor t, u, d_t;
  t.data = 5; u.data = 3; d_t.data = 0;
  auto dfn1 = clad::gradient(fn1, "t");
  dfn1.execute(t, u, &d_t);
  printf("%.2f\n", d_t.data); // CHECK-EXEC: 1.00
}
