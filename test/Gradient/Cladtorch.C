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

float fn1(const cladtorch::Tensor& t) {
  auto b = t;
  return b.data;
}
// CHECK: void fn1_grad(const cladtorch::Tensor &t, cladtorch::Tensor *_d_t) {
// CHECK-NEXT:     {{.*}}cladtorch::Tensor b = t;
// CHECK-NEXT:     {{.*}}cladtorch::Tensor _d_b(t);
// CHECK-NEXT:     clad::zero_init(_d_b);
// CHECK-NEXT:     _d_b.data += 1;
// CHECK-NEXT:     Tensor::constructor_pullback(t, &_d_b, &(*_d_t));
// CHECK-NEXT: }

int main() {
  cladtorch::Tensor t, d_t;
  t.data = 5; d_t.data = 0;
  auto dfn1 = clad::gradient(fn1);
  dfn1.execute(t, &d_t);
  printf("%.2f\n", d_t.data); // CHECK-EXEC: 1.00
}
