// Tests that a custom derivative that lives in a C++ module is found and used:
// the module decl is only deserialized by the lookup the planning traversal
// issues, i.e. while the traversal is on the stack. This mirrors the setup in
// which ROOT's runtime C++ modules re-enter CladPlugin::HandleTopLevelDecl
// mid-traversal (there the deserialized decls are handed straight to the
// consumers, so the delivered groups must be parked until the active walk
// finishes). Under a plain clang plugin invocation the reader delivers them
// through HandleInterestingDecl instead, which clad only queues -- so this
// test cannot reproduce the re-entrant Walk itself; it pins down the lazy
// module-deserialization path that leads up to it.
//
// The module map is passed explicitly, and -fno-implicit-module-maps is
// required on top of that because -fmodules implies -fimplicit-module-maps:
// implicit module-map discovery would also modularize the standard library on
// some hosts (macOS), whose module builds emit diagnostics that this test's
// FileCheck run forbids.
//
// RUN: rm -rf %t
// RUN: %cladclang -fmodules -fimplicit-modules -fno-implicit-module-maps \
// RUN:   -fmodule-map-file=%S/Inputs/CustomDerivativeFromModule/module.modulemap \
// RUN:   -fmodules-cache-path=%t/cache -I%S/Inputs/CustomDerivativeFromModule \
// RUN:   %s -I%S/../../include -o%t/CustomDerivativeFromModule.out 2>&1 | %filecheck %s
// RUN: %t/CustomDerivativeFromModule.out | %filecheck_exec %s

#include "clad/Differentiator/Differentiator.h"
#include "custom.h"

double opaque(double& x) { return 2. * x; }

double fn(double x) { return opaque(x); }

// CHECK: void fn_grad(double x, double *_d_x) {
// CHECK-NEXT:     clad::custom_derivatives::opaque_pullback(x, 1, _d_x);
// CHECK-NEXT: }

int main() {
  auto grad = clad::gradient(fn);
  double dx = 0;
  grad.execute(3.0, &dx);
  printf("Result is:%.2f\n", dx); // CHECK-EXEC: Result is:2.00
}
