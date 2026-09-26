// The three pages the tables describe are rendered where the documentation is
// built, which no other test reaches. Render them here too, so an emitter
// that stops working fails in the suite rather than in a readthedocs build
// nobody watches until it is red.

// RUN: mkdir -p %t.d
// RUN: %clad_tblgen -gen-analyses-docs %S/../../lib/Differentiator/Analyses.td \
// RUN:   -o %t.d/Analyses.rst
// RUN: FileCheck --check-prefix=ANALYSES --input-file=%t.d/Analyses.rst %s
// Each construct is a section a report's code links to.
// ANALYSES: What the analyses look for
// ANALYSES: .. _clad1001:
// ANALYSES: CLAD1001

// RUN: %clad_tblgen -gen-diagnostics-docs %S/../../lib/Differentiator/Analyses.td \
// RUN:   -o %t.d/Diagnostics.rst
// RUN: FileCheck --check-prefix=DIAGS --input-file=%t.d/Diagnostics.rst %s
// DIAGS: What clad says
// DIAGS: clad keeps this value for the reverse sweep

// RUN: %clad_tblgen -gen-options-docs %S/../../lib/Differentiator/Options.td \
// RUN:   -o %t.d/Options.rst
// RUN: FileCheck --check-prefix=OPTS --input-file=%t.d/Options.rst %s
// OPTS: Options
// OPTS: ``-fdump-derived-fn``
