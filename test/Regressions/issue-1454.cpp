// RUN: %cladclang -I%S/../../include %s

#include "clad/Differentiator/Differentiator.h"

double __cdecl myfn(double x) {return 0;}
namespace clad {
  namespace custom_derivatives {
    void myfn_pullback(double x, double dy, double *dx) {
      *dx += dy;
    }
  }
}

double f (double x) { return myfn(x); }

int main() {
   auto df = clad::gradient(f);
}
