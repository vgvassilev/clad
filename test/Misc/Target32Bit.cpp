// Differentiating for a 32-bit target.
//
// A pointer is half as wide there and std::size_t with it, which is enough to
// change what clad's headers mean. clad::array once took its subscript as
// std::size_t while also converting implicitly to T*: on a 64-bit target both
// the member and the built-in subscript need a conversion on the index, so
// the member wins on the object argument; on a 32-bit one std::ptrdiff_t is
// int, so the built-in matches the index exactly, neither candidate is better
// in every argument, and `a[i]` for an int i stops compiling. That was fixed
// in 262025e0 by taking std::ptrdiff_t; nothing had caught it until a build
// on a 32-bit machine did.
//
// The target is what matters here, not the host: an x86_64 machine emits
// 32-bit code perfectly well, and this is the half of 32-bit coverage that a
// user meets -- they compile clad's headers, they do not build clad itself.
//
// This does not replace the Arch job, which builds clad itself for i386 and
// runs the whole suite there. It is the part that costs nothing and so can
// run everywhere the suite does, macOS and Windows included, rather than on
// the one Linux job: nothing is linked, so no 32-bit runtime is needed.
//
// REQUIRES: target-i386
// RUN: %cladclang -m32 -fsyntax-only %s -I%S/../../include

#include "clad/Differentiator/Differentiator.h"

double f(double x, double y) {
  double t = x * y;
  return t * x;
}

int main() {
  auto g = clad::gradient(f);
  double dx = 0, dy = 0;
  g.execute(3, 4, &dx, &dy);

  // The shape that regressed: a subscript whose type is neither size_t nor
  // ptrdiff_t, so overload resolution has to choose.
  clad::array<double> a(4);
  int i = 2;
  a[i] = 1.0;
  return 0;
}
