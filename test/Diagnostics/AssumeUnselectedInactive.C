// RUN: %cladclang %s -I%S/../../include -fsyntax-only -Xclang -verify

#include "clad/Differentiator/Differentiator.h"

double product(double x, double y) { return x * y; }

void check_options() {
  clad::differentiate<clad::opts::assume_unselected_inactive>(product, "x"); // expected-error {{assume_unselected_inactive requires clad::gradient with the Clad backend}}
  clad::hessian<clad::opts::assume_unselected_inactive>(product); // expected-error {{assume_unselected_inactive requires clad::gradient with the Clad backend}}
  clad::hessian<clad::opts::diagonal_only | clad::opts::assume_unselected_inactive>(product); // expected-error {{assume_unselected_inactive requires clad::gradient with the Clad backend}}
  clad::gradient<clad::opts::use_enzyme | clad::opts::assume_unselected_inactive>(product, "x"); // expected-error {{assume_unselected_inactive requires clad::gradient with the Clad backend}}
}
