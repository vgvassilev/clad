// RUN: %cladclang -I%S/../../include %s

#include "clad/Differentiator/Differentiator.h"

double __cdecl myfn(double x) {return 0;}
namespace clad {
  namespace custom_derivatives {
    ValueAndPushforward<double, double> myfn_pushforward(double x, double dx) {
      return { x, 2 * dx };
    }
  }
}

double f (double x) { return myfn(x); }

int main() {
   auto df = clad::gradient(f);
}
