// RUN: %cladclang %s -I%S/../../include -Xclang -verify 2>&1 | %filecheck %s

#include "clad/Differentiator/Differentiator.h"

namespace helpers {
    // test using shadow decls
    int g_pullback(int) { // expected-note {{Candidate not viable: cannot match the requested signature with 'int (int)'}}
    return true;
    }
}

namespace clad {
namespace custom_derivatives {
  // parameter `x` must be of type `double`
  void g_pullback(float x, double y, double _d_y0, float *_d_x, double *_d_y) { // expected-note {{Candidate not viable: cannot match the requested signature with 'void (float, double, double, float *, double *)'}}
    *_d_x += _d_y0;
  }
  // no pulback parameter `_d_y` is provided
  template <typename T, typename U>
  void g_pullback(T x, U y, T* _d_x, U* _d_y) { // expected-note {{Candidate not viable: cannot match the requested signature with 'void (T, U, T *, U *)'}}
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
  return g(x, y); // expected-warning {{A custom derivative for 'g_pullback' was found but not used because its signature does not match the expected signature 'void (double, double, double, double *, double *)'}}
}

namespace clad {
namespace custom_derivatives {
  // the return type should be `clad::ValueAndPushforward`
  double h_pushforward(const double x, double y, const double _d_x, double _d_y) { // expected-note {{Candidate not viable: cannot match the requested signature with 'double (const double, double, const double, double)'}}
    return _d_x - _d_y;
  }

  // the template does not account for const-ness
  template <typename T>
  T h_pushforward(T x, T y, T _d_x, T _d_y) { // expected-note {{Candidate not viable: cannot match the requested signature with 'T (T, T, T, T)'}}
    return _d_x - _d_y;
  }
}
}

double h(const double x, double y) {
  return x - y;
}

double f2(double x, double y) {
  return h(x, y); // expected-warning {{A custom derivative for 'h_pushforward' was found but not used because its signature does not match the expected signature 'clad::ValueAndPushforward<double, double> (const double, double, const double, double)'}}
}

int main() {
    clad::gradient(f1);
    clad::differentiate(f2, "x");
}
