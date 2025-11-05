// RUN: %cladclang %s -I%S/../../include -oCladtorch.out 2>&1 | %filecheck %s
// RUN: ./Cladtorch.out | %filecheck_exec %s
// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -disable-tbr -Xclang -plugin-arg-clad -Xclang -enable-va %s -I%S/../../include -oCladtorch.out
// RUN: ./Cladtorch.out | %filecheck_exec %s
// XFAIL: valgrind

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

struct no_namespace {
  float x;
};

const static struct {
  float z = 2.F;
} anon;

namespace {
struct anon_namespace {
  float x;
};
}

namespace not_found {
struct Tensor {
  float x;
};
}


float fn1(
  const cladtorch::Tensor& t, const cladtorch::Tensor& u, no_namespace o, 
  decltype(anon) a, anon_namespace z, not_found::Tensor w
) {
  auto b = t;
  std::vector<cladtorch::Tensor> v{{u, b}};
  return v[1].data;
}

// CHECK: static inline constexpr void constructor_pullback(const {{.*}}Tensor &arg, cladtorch::Tensor *_d_this, {{.*}}Tensor *_d_arg) noexcept {
// CHECK-NEXT:     {
// CHECK-NEXT:         (*_d_arg).data += _d_this->data;
// CHECK-NEXT:         _d_this->data = 0.F;
// CHECK-NEXT:     }
// CHECK-NEXT: }

// CHECK: void fn1_grad_0(const cladtorch::Tensor &t, const cladtorch::Tensor &u, no_namespace o, decltype(anon) a, {{.*}}anon_namespace z, not_found::Tensor w, cladtorch::Tensor *_d_t) {
// CHECK-NEXT:     {{.*}}cladtorch::Tensor _d_u(u);
// CHECK-NEXT:     no_namespace _d_o = {0.F};
// CHECK-NEXT:     {{.*}} _d_a = {0.F};
// CHECK-NEXT:     anon_namespace _d_z = {0.F};
// CHECK-NEXT:     not_found::Tensor _d_w = {0.F};
// CHECK-NEXT:     {{.*}}cladtorch::Tensor b = t;
// CHECK-NEXT:     {{.*}}cladtorch::Tensor _d_b = (*_d_t);
// CHECK-NEXT:     {{.*}}std::vector<cladtorch::Tensor> v{{.*}}{u, b}{{.*}};
// CHECK-NEXT:     {{.*}}std::vector<cladtorch::Tensor> _d_v{{.*}}{_d_u, _d_b}{{.*}};
// CHECK-NEXT:     _d_v[1].data += 1;
// CHECK-NEXT:     {
// CHECK-NEXT:         clad::array<{{cladtorch::Tensor|std::vector<cladtorch::Tensor, std::allocator<cladtorch::Tensor> >::value_type}}> _r0 = {{2U|2UL}};
// CHECK:              {{.*}}clad::custom_derivatives::class_functions::constructor_pullback({u, b}, {{.*}}&_d_v, &_r0{{.*}});
// CHECK-NEXT:         Tensor::constructor_pullback(u, &_r0[0], &_d_u);
// CHECK-NEXT:         Tensor::constructor_pullback(b, &_r0[1], &_d_b);
// CHECK-NEXT:     }
// CHECK-NEXT:     Tensor::constructor_pullback(t, &_d_b, _d_t);
// CHECK-NEXT: }

int main() {
  cladtorch::Tensor t, u, d_t;
  t.data = 5; u.data = 3; d_t.data = 0;
  auto dfn1 = clad::gradient(fn1, "t");
  no_namespace o{9.0}; auto a = anon;
  anon_namespace z{4.0}; not_found::Tensor w{6.0};
  dfn1.execute(t, u, o, a, z, w, &d_t);
  printf("%.2f\n", d_t.data); // CHECK-EXEC: 1.00
}
