// RUN: %cladclang %s -I%S/../../include -oIntegerMemberTangent.out
// RUN: ./IntegerMemberTangent.out | %filecheck_exec %s

// An integer member of a tangent object is not a derivative. The custom
// constructor_pushforward of a class whose constructor refuses a zero slice
// count, like a beamline element of ImpactX, gives the tangent a valid one.

#include "clad/Differentiator/Differentiator.h"
#include <cstdio>

struct E {
  double ds;
  int n;
  E(double ds_, int n_) : ds(ds_), n(n_) {}
  int nslice() const { return n; }
};

namespace clad {
namespace custom_derivatives {
namespace class_functions {
inline clad::ValueAndPushforward<E, E>
constructor_pushforward(clad::Tag<E>, double ds, int n, double d_ds, int) {
  return {E(ds, n), E(d_ds, n)};
}
} // namespace class_functions
} // namespace custom_derivatives
} // namespace clad

double accessor(double k) {
  E e(k, 4);
  return e.ds / e.nslice();
}

double member(double k) {
  E e(k, 4);
  return e.ds * e.n;
}

int main() {
  printf("%.2f\n", clad::differentiate(accessor, "k").execute(3)); // CHECK-EXEC: 0.25
  printf("%.2f\n", clad::differentiate(member, "k").execute(3));   // CHECK-EXEC: 4.00
}
