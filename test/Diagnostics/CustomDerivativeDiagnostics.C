// RUN: %cladclang %s -I%S/../../include -fsyntax-only -Xclang -verify 2>&1 | %filecheck %s

#include "clad/Differentiator/Differentiator.h"

namespace helpers {
    // test using shadow decls
    int g_pullback(int) { // expected-note {{candidate 'g_pullback' has different number of parameters (expected 5 but has 1)}}
    return true;
    }
}

namespace clad {
namespace custom_derivatives {
  // parameter `x` must be of type `double`
  void g_pullback(float x, double y, double _d_y0, float *_d_x, double *_d_y) { // expected-note {{candidate 'g_pullback' has type mismatch at 1st parameter (expected 'double' but has 'float')}}
    *_d_x += _d_y0;
  }
  // no pulback parameter `_d_y` is provided
  template <typename T, typename U>
  void g_pullback(T x, U y, T* _d_x, U* _d_y) { // expected-note-re {{candidate template ignored: could not match '{{.*}} *' against 'double'}}
    *_d_x += 1;
  }
  // test using shadow decls
  using ::helpers::g_pullback;
}
}

double g(double x, double y) {
  return x + y;
}

double f1(double x, double y) {
  return g(x, y); // expected-error {{user-defined derivative for 'g' was provided but not used}}
}

namespace clad {
namespace custom_derivatives {
  // the return type should be `clad::ValueAndPushforward`
  double h_pushforward(const double x, double y, const double _d_x, double _d_y) { //expected-note{{candidate 'h_pushforward' has different return type}}
    return _d_x - _d_y;
  }

  // the template does not account for const-ness
  template <typename T>
  T h_pushforward(T x, T y, T _d_x, T _d_y) { // expected-note {{candidate template ignored: deduced conflicting types for parameter 'T' ('clad::ValueAndPushforward<double, double>' vs. 'double')}}
    return _d_x - _d_y;
  }
}
}

double h(const double x, double y) {
  return x - y;
}

double f2(double x, double y) {
  return h(x, y); // expected-error {{user-defined derivative for 'h' was provided but not used;}}
}

int main() {
    clad::gradient(f1);
    clad::differentiate(f2, "x");
}
