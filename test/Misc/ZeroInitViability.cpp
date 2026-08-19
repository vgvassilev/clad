// RUN: %cladclang %s -I%S/../../include -fsyntax-only -Xclang -verify
//
// Nothing is ignored: every diagnostic is spelled out below. In particular a
// -Wnontrivial-* warning must not appear, as it would mean the rejected type
// still reaches the memcpy, which is the thing being prevented.

// clad::zero_init's fallback zeroes an object with a byte-wise memcpy, which is
// not well-defined for every type. Its guard used to be compiled only into CUDA
// translation units, so a direct clad::zero_init call in an ordinary host build
// went unchecked. Check that a host build is guarded too.

#include "clad/Differentiator/Differentiator.h"

#include <vector>

struct TrivialPod {
  double a, b;
};

struct WithCtorTrivialCopy { // a user ctor does not stop trivial copyability
  double a, b;
  WithCtorTrivialCopy(double x, double y) : a(x), b(y) {}
};

struct Owning { // owns a heap handle: memcpy-zeroing it would corrupt it
  double* p;
  Owning() : p(new double(0)) {}
  Owning(const Owning& o) : p(new double(*o.p)) {}
  ~Owning() { delete p; }
};

struct HasOverload {
  double* p;
};
namespace clad {
inline void zero_init(HasOverload& h) { *h.p = 0; }
} // namespace clad

void accepted() {
  double d = 1;
  clad::zero_init(d);

  TrivialPod p{1, 2};
  clad::zero_init(p);

  WithCtorTrivialCopy w(1, 2);
  clad::zero_init(w);

  std::vector<double> v(3);
  clad::zero_init(v);

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays)
  double arr[2]{}; // a C array is iterable, and is what clad zeroes in a loop
  clad::zero_init(arr);

  HasOverload h{&d}; // an overload still wins over the fallback
  clad::zero_init(h);
}

void rejected() {
  Owning o;
  // expected-error@clad/Differentiator/Differentiator.h:* {{Clad device fallback zero_init requires trivially destructible types}}
  // expected-note@clad/Differentiator/Differentiator.h:* {{in instantiation of function template specialization 'clad::zero_impl<Owning, 0>' requested here}}
  clad::zero_init(o); // expected-note {{in instantiation of function template specialization 'clad::zero_init<Owning>' requested here}}
}

int main() { return 0; }
