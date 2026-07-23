// With -fclad-porting-hints, clad emits a remark for every function defined
// outside the main file that it differentiates by cloning (i.e. that has no
// custom derivative), naming the expected custom-derivative signature and the
// non-differentiable marker.
// RUN: clang -std=c++17 -fsyntax-only -fplugin=%cladlib -Xclang -plugin-arg-clad -Xclang \
// RUN:   -fclad-porting-hints %s -I%S/../../include 2>&1 | %filecheck %s
//
// Without the flag, clad is silent about the clone.
// RUN: clang -std=c++17 -fsyntax-only -fplugin=%cladlib %s -I%S/../../include 2>&1 \
// RUN:   | %filecheck --check-prefix=CHECK-QUIET --allow-empty %s
//
// The flag is listed in -help.
// RUN: clang -std=c++17 -fsyntax-only -fplugin=%cladlib -Xclang -plugin-arg-clad -Xclang \
// RUN:   -help %s -I%S/../../include 2>&1 | %filecheck --check-prefix=CHECK-HELP %s
//
#include "clad/Differentiator/Differentiator.h"
#include "PortingHintsLib.h"

double f(double x) {
  Widget w{2.0};
  return w.scale(x);
}

int main() {
  auto g = clad::gradient(f);
  double dx = 0;
  g.execute(2, &dx);
}

// The main-file function under differentiation is never reported as a boundary
// -- only functions defined outside it are. 'f' is differentiated before it
// descends into 'scale', so a broken main-file guard would emit this first.
// CHECK-NOT: no custom derivative for 'f'
// CHECK: remark: clad has no custom derivative for 'scale' and is differentiating its definition, descending into library internals
// CHECK: note: to differentiate it, provide clad::custom_derivatives::scale_{{.*}} with signature
// CHECK: note: or declare it non-differentiable with clad::custom_derivatives::nondifferentiable(clad::Tag<Widget>{})

// CHECK-QUIET-NOT: remark:

// CHECK-HELP: -fclad-porting-hints
