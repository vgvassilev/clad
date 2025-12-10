// RUN: %cladclang %s -I%S/../../include -fsyntax-only -Xclang -verify

#include "clad/Differentiator/Differentiator.h"

#include <string>

namespace clad {
    double getErrorVal(double dx, double x) { // expected-note {{candidate 'getErrorVal' has different number of parameters (expected 3 but has 2)}}
        return dx * x; 
    }
    template<typename T>
    T getErrorVal(T dx, T x, T name) { // expected-note {{candidate template ignored: deduced conflicting types for parameter 'T' ('double' vs. 'const char *')}}
        return dx * x; 
    }
    float getErrorVal(double dx, double x, const char* name) { // expected-note {{candidate 'getErrorVal' has different return type ('double' expected but has 'float')}}
        return dx * x; 
    }
    double getErrorVal(double dx, double x, std::string name) { // expected-note {{candidate 'getErrorVal' has type mismatch at 3rd parameter (expected 'const char *' but has 'std::string' (aka 'basic_string<char>'))}}
        return dx * x; 
    }
} // namespace clad

// Add/Sub operations
float f1(float x, float y) { // expected-error {{user-defined derivative error function was provided but not used; expected signature 'double (double, double, const char *)' does not match}}
  return x + y;
}

int main() {
  clad::estimate_error(f1);
}
