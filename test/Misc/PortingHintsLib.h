// Companion "library" header for PortingHints.C. Its functions live outside the
// test's main file, so clad treats them as a library boundary and (under
// -fclad-porting-hints) emits porting remarks when it differentiates them.
#pragma once

struct Widget {
  double value;
  double scale(double x) const { return x * value; }
};
