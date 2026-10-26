// Companion "library" header for PortingHintsConstructor.cpp. Its constructor is
// non-elidable (a computing member initializer), so clad schedules and clones a
// reverse-forward propagator for it -- the case that offers the
// CLAD_ELIDABLE_REVERSE_FORW porting route.
#pragma once

struct Gadget {
  double v;
  Gadget(double s) : v(s * 2.0) {}
};
