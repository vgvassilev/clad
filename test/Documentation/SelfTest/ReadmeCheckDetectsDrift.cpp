// RUN: ! %python %S/../README/check-readme.py %S/Inputs/drifted-readme.md \
// RUN:     %S/Inputs
//
// The README's examples are copies, kept in step with the tests by
// check-readme.py. This feeds it a README whose block has drifted from the
// example it claims to show, and requires it to say so.

int main() { return 0; }
