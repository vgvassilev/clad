// RUN: %cladclang %s -I%S/../../include -fsyntax-only -Xclang -verify

#include "clad/Differentiator/Differentiator.h"

// The four kinds issue #2088 named each got a clone, but StmtClone has a case
// for a fraction of clang's statement kinds and the rest reached the same
// fallback -- `assert(0); return 0;`, which under NDEBUG is a null the caller
// walks. The fallback now says what it cannot do instead of handing that back,
// so a kind nobody has written a clone for fails the compile with its name
// rather than crashing. GCCAsmStmt is one such kind today; if it ever gains a
// clone, this test wants a different kind rather than deleting.

// expected-warning@+2 {{statement kind 'GCCAsmStmt' is not supported}}
// expected-error@+1 {{clad cannot differentiate this function: no clone exists for statement kind 'GCCAsmStmt'}}
double with_asm(double x) { asm("nop"); return x * x; }

int main() { clad::differentiate(with_asm, "x"); }
