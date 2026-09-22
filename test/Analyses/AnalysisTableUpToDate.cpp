// Analyses.td is the one place clad's analyses, the constructs they look for,
// the ways an input can miss one and the wording of every report are written
// down. The headers rendered from it are committed, so building clad never
// runs tblgen, and this checks that what is committed is what the table
// says. The pages are rendered when the documentation is built and not
// committed, so they are only rendered here, which keeps their emitters
// honest.
//
// RUN: mkdir -p %t.d
// RUN: %clad_tblgen -gen-analyses %S/../../lib/Differentiator/Analyses.td -o %t.d/Analyses.def
// RUN: %clad_tblgen -gen-analysis-descs %S/../../lib/Differentiator/Analyses.td -o %t.d/AnalysisDescs.def
// RUN: %clad_tblgen -gen-analyses-docs %S/../../lib/Differentiator/Analyses.td -o %t.d/Analyses.rst
// RUN: %clad_tblgen -gen-diagnostics %S/../../lib/Differentiator/Analyses.td -o %t.d/Diagnostics.def
// RUN: %clad_tblgen -gen-diagnostics-docs %S/../../lib/Differentiator/Analyses.td -o %t.d/Diagnostics.rst
// RUN: diff -u --strip-trailing-cr %S/../../include/clad/Differentiator/Analyses.def %t.d/Analyses.def
// RUN: diff -u --strip-trailing-cr %S/../../lib/Differentiator/AnalysisDescs.def %t.d/AnalysisDescs.def
// RUN: diff -u --strip-trailing-cr %S/../../lib/Differentiator/Diagnostics.def %t.d/Diagnostics.def
