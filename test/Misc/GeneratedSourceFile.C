// Getting at the code clad generated, from a debugger.
//
// The derivative's statements have locations in a buffer that holds their
// code, which is enough for a diagnostic but not for a debugger: lldb can
// read the text out of the object with -gembed-source, and gdb cannot read
// that extension at all -- it resolves the name in the line table against the
// compilation directory and opens a file. Write the code to a file and name
// the buffer after it and both are served.
//
// RUN: rm -rf %t.dir && mkdir -p %t.dir
// RUN: %cladclang -g -gdwarf-5 -Xclang -plugin-arg-clad \
// RUN:   -Xclang -fgenerated-source-dir=%t.dir -c -o %t.o %s \
// RUN:   -I%S/../../include 2>&1 | %filecheck --allow-empty %s
// RUN: cat %t.dir/*.clad.cpp | FileCheck --check-prefix=CHECK-FILE %s
//
// One file per translation unit, named after it, holding what clad wrote.
// CHECK-FILE: void f_grad(double x, double *_d_x)
// CHECK-FILE: double _t0 = t;
// CHECK-FILE: t = _t0;
//
// Asking for it says nothing: the advice below is for the case where nothing
// can read the code, and this is not that case.
// CHECK-NOT: warning:

#include "clad/Differentiator/Differentiator.h"

double f(double x) {
  double t = x * x;
  t = t * t;
  return t;
}

int main() {
  auto g = clad::gradient(f);
  double d = 0;
  g.execute(2, &d);
  return 0;
}

// Debug information was asked for and nothing will be able to show the code,
// so clad says which flag to add. What it suggests depends on the debugger
// being tuned for, because -gembed-source only serves one of them. Plain
// FileCheck: the run is about a warning, which the shared %filecheck rejects.
//
// RUN: %cladclang -glldb -gdwarf-5 -c -o %t.o %s -I%S/../../include 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-LLDB %s
// CHECK-LLDB: warning: debug information for the code clad generated points at no source a debugger can open; add -gembed-source, or -plugin-arg-clad -fgenerated-source-dir=<dir>
//
// gdb has no support for the embedded-source extension, so it is not offered.
// RUN: %cladclang -ggdb -gdwarf-5 -c -o %t.o %s -I%S/../../include 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-GDB %s
// CHECK-GDB: warning: debug information for the code clad generated points at no source a debugger can open; add -plugin-arg-clad -fgenerated-source-dir=<dir>
// CHECK-GDB-NOT: -gembed-source
//
// With the code in the object and a debugger that reads it from there, there
// is nothing to say.
// RUN: %cladclang -glldb -gdwarf-5 -gembed-source -c -o %t.o %s \
// RUN:   -I%S/../../include 2>&1 | %filecheck --check-prefix=CHECK-QUIET \
// RUN:   --allow-empty %s
// CHECK-QUIET-NOT: warning:
//
// And nothing at all without debug information, which is the common build.
// RUN: %cladclang -c -o %t.o %s -I%S/../../include 2>&1 \
// RUN:   | %filecheck --check-prefix=CHECK-QUIET --allow-empty %s
//
// A directory that cannot be written to is worth a word and not worth
// failing over: the line table still has the right file and line, and only
// the text is missing. Plain FileCheck, since this is about a warning.
// RUN: %cladclang -g -Xclang -plugin-arg-clad \
// RUN:   -Xclang -fgenerated-source-dir=%t.dir/missing -c -o %t.o %s \
// RUN:   -I%S/../../include 2>&1 | FileCheck --check-prefix=CHECK-UNWRITABLE %s
// CHECK-UNWRITABLE: warning: could not write the generated source to '{{.*}}missing{{.*}}.clad.cpp':
//
// Naming the flag with no directory is an answer too: nothing is written and
// nothing is said. A plugin's diagnostic belongs to no -W group, so this is
// the only way to turn the advice off.
// RUN: %cladclang -g -gdwarf-5 -Xclang -plugin-arg-clad \
// RUN:   -Xclang -fgenerated-source-dir= -c -o %t.o %s -I%S/../../include \
// RUN:   2>&1 | %filecheck --check-prefix=CHECK-QUIET --allow-empty %s
