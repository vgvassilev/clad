// A boundary method on a TEMPLATE INSTANTIATION. The marker suggestion must
// carry the template arguments -- CLAD_NONDIFFERENTIABLE_TYPE(Boxed<double>),
// not Boxed, which getQualifiedNameAsString drops and which would expand to an
// ill-formed Tag<Boxed>. clad additionally notes that a full specialization
// marks only this instantiation; the whole family needs a partial one.
// RUN: clang -std=c++17 -fsyntax-only -fplugin=%cladlib -Xclang -plugin-arg-clad \
// RUN:   -Xclang -fclad-porting-hints %s -I%S/../../include 2>&1 | %filecheck %s

#include "clad/Differentiator/Differentiator.h"
#include "PortingHintsBoxed.h"

double f(double x) {
  Boxed<double> b{2.0};
  return b.scale(x);
}

int main() {
  auto g = clad::gradient(f);
  double dx = 0;
  g.execute(2, &dx);
}

// CHECK: remark: clad has no custom derivative for 'scale' and is differentiating its definition, descending into library internals
// CHECK: note: to differentiate it, provide clad::custom_derivatives::scale_pullback with signature {{.*}}Boxed<double>
// CHECK: note: or mark it non-differentiable with CLAD_NONDIFFERENTIABLE_TYPE(Boxed<double>)
// CHECK: note: this marks only this specialization; to mark the whole template, {{.*}}CLAD_NONDIFFERENTIABLE
