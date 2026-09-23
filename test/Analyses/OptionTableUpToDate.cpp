// Options.td is the one place every option clad takes is written down: its
// spelling, what -help says about it, and which field of clad::Options it
// sets. The header rendered from it is committed, so building clad never runs
// tblgen, and this checks that what is committed is what the table says.
//
// The reference page is rendered when the documentation is built and is not
// committed, so it is only rendered here, which keeps its emitter honest.
//
// --strip-trailing-cr: the committed file arrives through git, which on
// Windows may hand it over with CRLF, while tblgen always writes LF.
//
// RUN: mkdir -p %t.d
// RUN: %clad_tblgen -gen-options %S/../../lib/Differentiator/Options.td -o %t.d/Options.def
// RUN: %clad_tblgen -gen-options-docs %S/../../lib/Differentiator/Options.td -o %t.d/Options.rst
// RUN: diff -u --strip-trailing-cr %S/../../include/clad/Differentiator/Options.def %t.d/Options.def
