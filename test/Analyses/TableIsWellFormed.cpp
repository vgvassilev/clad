// The table is the only place clad's analyses, constructs and messages are
// written down, so a mistake in it has to stop the build rather than quietly
// render less than it says. Each input below trips one check.

// A code is what a report prints and a reader searches for, so no two
// constructs may own one. Every construct is looked at, listed or not: a
// retired one still owns its number.
// RUN: not %clad_tblgen -I %S/../../lib/Differentiator -gen-analysis-descs \
// RUN:   %S/Inputs/duplicate-code.td -o %t.def 2>&1 | FileCheck %s
// CHECK: CLAD1001 is already CountedLoop

// Everything the table defines has to be reachable from Clad, or it is written
// down and never rendered.
// RUN: not %clad_tblgen -I %S/../../lib/Differentiator -gen-analysis-descs \
// RUN:   %S/Inputs/unlisted-diagnostic.td -o %t.def 2>&1 \
// RUN:   | FileCheck --check-prefix=DIAGNOSTIC %s
// DIAGNOSTIC: Clad's Diagnostics does not list note_unlisted

// RUN: not %clad_tblgen -I %S/../../lib/Differentiator -gen-analysis-descs \
// RUN:   %S/Inputs/unlisted-desc.td -o %t.def 2>&1 \
// RUN:   | FileCheck --check-prefix=DESC %s
// DESC: Clad's Descs does not list UnlistedConstruct

// A way to miss a construct belongs to a construct, or nothing prints it.
// RUN: not %clad_tblgen -I %S/../../lib/Differentiator -gen-analysis-descs \
// RUN:   %S/Inputs/orphan-miss.td -o %t.def 2>&1 \
// RUN:   | FileCheck --check-prefix=ORPHAN %s
// ORPHAN: no construct lists OrphanMiss
