// RUN: %cladclang %s -I%S/../../include -oCladtorch.out 2>&1 | %filecheck %s
// RUN: ./Cladtorch.out | %filecheck_exec %s
// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -disable-tbr %s -I%S/../../include -oCladtorch.out
// RUN: ./Cladtorch.out | %filecheck_exec %s

#include "clad/Differentiator/Differentiator.h"
#include "clad/Differentiator/STLBuiltins.h"
#include <vector>

namespace clad {
namespace tensor_like {
namespace cladtorch {
struct Tensor;
}
}
}

namespace cladtorch {
struct Tensor {
  Tensor(): data(0) {};
  Tensor(float val) : data(val) {}
  float data;
  Tensor& operator+=(const Tensor& other) {
    data = other.data;
    return *this;
  }
};
}

struct other_struct {
  float x;
};

float fn1(const cladtorch::Tensor& t, const cladtorch::Tensor& u, other_struct o) {
  auto b = t;
  std::vector<cladtorch::Tensor> v{{u, b}};
  return v[1].data;
}
// CHECK: void fn1_grad_0(const cladtorch::Tensor &t, const cladtorch::Tensor &u, other_struct o, cladtorch::Tensor *_d_t) {
// CHECK-NEXT:     {{.*}}cladtorch::Tensor _d_u(u);
// CHECK-NEXT:     other_struct _d_o = {0.F};
// CHECK-NEXT:     {{.*}}cladtorch::Tensor b = t;
// CHECK-NEXT:     {{.*}}cladtorch::Tensor _d_b(b);
// CHECK-NEXT:     clad::zero_init(_d_b);
// CHECK-NEXT:     {{.*}}std::vector<cladtorch::Tensor> v{u, b};
// CHECK-NEXT:     {{.*}}std::vector<cladtorch::Tensor> _d_v(v);
// CHECK-NEXT:     clad::zero_init(_d_v);
// CHECK-NEXT:     {{.*}}std::vector<cladtorch::Tensor> _t0 = v;
// CHECK-NEXT:     clad::ValueAndAdjoint<Tensor &, Tensor &> _t1 = clad::custom_derivatives::class_functions::operator_subscript_reverse_forw(&v, 1, &_d_v, 0UL);
// CHECK-NEXT:     {
// CHECK-NEXT:         size_type _r1 = 0UL;
// CHECK-NEXT:         v = _t0;
// CHECK-NEXT:         clad::custom_derivatives::class_functions::operator_subscript_pullback(&v, 1, {}, &_d_v, &_r1);
// CHECK-NEXT:         _t1.adjoint.data += 1;
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         clad::array<Tensor> _r0 = 2UL;
// CHECK-NEXT:         clad::custom_derivatives::class_functions::constructor_pullback({u, b}, &_d_v, &_r0);
// CHECK-NEXT:         Tensor::constructor_pullback(u, &_r0[0], &_d_u);
// CHECK-NEXT:         Tensor::constructor_pullback(b, &_r0[1], &_d_b);
// CHECK-NEXT:     }
// CHECK-NEXT:     Tensor::constructor_pullback(t, &_d_b, &(*_d_t));
// CHECK-NEXT: }

int main() {
  cladtorch::Tensor t, u, d_t;
  t.data = 5; u.data = 3; d_t.data = 0;
  auto dfn1 = clad::gradient(fn1, "t");
  other_struct o{9.0};
  dfn1.execute(t, u, o, &d_t);
  printf("%.2f\n", d_t.data); // CHECK-EXEC: 1.00
}
