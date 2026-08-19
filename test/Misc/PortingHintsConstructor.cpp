// A library constructor clad would clone triggers -fclad-porting-hints for BOTH
// its reverse-forward pass and its pullback. The reverse-forward one, in
// addition to the custom-derivative signature and the non-differentiable
// marker, offers the elidable_reverse_forw route -- for a constructor whose
// reverse pass is a no-op (e.g. a shallow copy that shares its adjoint).
// RUN: clang -std=c++17 -fsyntax-only -fplugin=%cladlib -Xclang -plugin-arg-clad \
// RUN:   -Xclang -fclad-porting-hints %s -I%S/../../include 2>&1 | %filecheck %s

#include "clad/Differentiator/Differentiator.h"
#include "PortingHintsGadget.h"

double f(double x) {
  Gadget g(x);
  return g.v; // clones Gadget's constructor -> reverse_forw + pullback hints
}

int main() {
  auto grad = clad::gradient(f);
  double dx = 0;
  grad.execute(2, &dx);
}

// CHECK: remark: clad has no custom derivative for 'Gadget' and is differentiating its definition, descending into library internals
// CHECK: note: to differentiate it, provide clad::custom_derivatives::constructor_reverse_forw with signature
// CHECK: note: or mark it non-differentiable with CLAD_NONDIFFERENTIABLE_TYPE(Gadget)
// CHECK: note: or, if its reverse-forward pass is a no-op {{.*}}mark it elidable_reverse_forw
// CHECK: remark: clad has no custom derivative for 'Gadget' and is differentiating its definition, descending into library internals
// CHECK: note: to differentiate it, provide clad::custom_derivatives::constructor_pullback with signature
// CHECK: note: or mark it non-differentiable with CLAD_NONDIFFERENTIABLE_TYPE(Gadget)
