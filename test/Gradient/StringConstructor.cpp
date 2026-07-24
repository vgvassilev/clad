// RUN: %cladclang %s -I%S/../../include -oStringConstructor.out -Xclang -verify 2>&1 | %filecheck %s
// RUN: ./StringConstructor.out | %filecheck_exec %s
//
// Constructing a std::string inside a differentiated function used to crash
// clad: it synthesized a reverse-forward propagator for the constructor by
// cloning the body, which delegates to a private member (`this->__init(...)`),
// but the propagator is a static function with no `this`, so CodeGen
// dereferenced a null `this`. Str reproduces that shape without <string> -- a
// constructor that delegates to a member function. Removing the marker below
// makes clang segfault in CodeGen generating Str::constructor_reverse_forw
// (verified against this clang). The defaulted constructor matters: without it
// clad's generated adjoint fails earlier at Sema ("no matching constructor"),
// masking the CodeGen crash the marker actually prevents. With the marker the
// construction is opaque, so the body is never cloned.

#include "clad/Differentiator/Differentiator.h"

struct Str {
  const char* Data = nullptr;
  Str() = default; // reach CodeGen, not an earlier Sema error
  Str(const char* P) { init(P); } // body needs `this`; unsafe as a static clone
private:
  void init(const char* P) { Data = P; }
};
CLAD_NONDIFFERENTIABLE_TYPE(Str);

double f(double x) {
  Str s("a"); // opaque -> not differentiated, no `init` clone
  return x * x;
}

// CHECK: void f_grad(double x, double *_d_x) {
// CHECK-NEXT: Str s("a");
// CHECK-NEXT: Str _d_s({{.*}});
// CHECK-NOT: init
// CHECK: *_d_x += 1 * x;
// CHECK-NEXT: *_d_x += x * 1;

int main() {
  auto g = clad::gradient(f);
  double x = 3, d_x = 0;
  g.execute(x, &d_x);
  printf("%.2f\n", d_x); // CHECK-EXEC: 6.00
}

// expected-no-diagnostics
