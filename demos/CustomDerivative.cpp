//--------------------------------------------------------------------*- C++ -*-
// clad - The C++ Clang-based Automatic Differentiator
//
// Tells clad the derivative of a function whose code is not worth
// differentiating.
//
//----------------------------------------------------------------------------//

// Necessary for clad to work include
#include "clad/Differentiator/Differentiator.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>

// An old trick: raise a number to a power by treating its bit pattern as a
// logarithm. A few integer operations instead of a call to pow, accurate to
// within a few percent.
//
// Differentiating this code would be pointless. Shifting a bit pattern has no
// useful derivative, and the derivative of an approximation is not the one
// you want. This is what custom derivatives are for. It is also what you do
// for a library function whose source you do not have.
float fast_pow(float base, float exponent) {
  std::uint32_t bits;
  std::memcpy(&bits, &base, sizeof bits);
  float as_log = static_cast<float>(bits) * 1.1920929e-7f - 127.f;
  bits = static_cast<std::uint32_t>((exponent * as_log + 127.f) * 8388608.f);
  float result;
  std::memcpy(&result, &bits, sizeof result);
  return result;
}

// So tell clad what the derivative is. It finds this by name and stops
// looking at the body.
//
// A function of one argument needs a pushforward instead, even in reverse
// mode. There the pullback is only a multiplication by f'(x), so clad asks
// for ValueAndPushforward<T, T> f_pushforward(T x, T d_x). A one-argument
// pullback is never found.
//
// Write ::std, not std. Clad declares its own namespace called std inside
// this one, and the unqualified name finds that.
// docs-begin-custom
namespace clad {
namespace custom_derivatives {
void fast_pow_pullback(float base, float exponent, float d_result,
                       float* d_base, float* d_exponent) {
  *d_base += d_result * exponent * ::std::pow(base, exponent - 1.f);
  *d_exponent += d_result * ::std::pow(base, exponent) * ::std::log(base);
}
} // namespace custom_derivatives
} // namespace clad
// docs-end-custom

float model(float base, float exponent) { return fast_pow(base, exponent); }

int main() {
  // docs-begin-custom-call
  auto grad = clad::gradient(model);
  // docs-end-custom-call

  float base = 3.f, exponent = 2.5f, d_base = 0.f, d_exponent = 0.f;
  grad.execute(base, exponent, &d_base, &d_exponent);

  // The value is a few percent out, because that is what the approximation
  // gives. The derivatives are exact, because they come from the pullback and
  // not from the code. Clad believes what it is told, and nothing checks a
  // pullback against the function it claims to differentiate.
  printf("fast_pow(%g, %g) = %.4f, where pow gives %.4f\n", base, exponent,
         fast_pow(base, exponent), std::pow(base, exponent));
  printf("d/dbase     = %.4f   exact %.4f\n", d_base,
         exponent * std::pow(base, exponent - 1.f));
  printf("d/dexponent = %.4f   exact %.4f\n", d_exponent,
         std::pow(base, exponent) * std::log(base));

  return 0;
}
